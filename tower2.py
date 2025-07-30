import torch, math
import torch.nn.functional as fc
from torch import nn, Tensor
from typing import Optional, Tuple
from torch.utils.tensorboard import SummaryWriter   # type: ignore

"""
第二代Tower架构
"""

class FeedForward(nn.Module):
    """全连接层"""
    def __init__(self, d, dff, use_dropout: bool=False):
        """
        参数:
        - d: 输入/输出的维度
        - dff: 前馈网络内部层的维度
        - use_dropout: 是否使用dropout
        """
        super().__init__()
        self.ffn = nn.Sequential(            # 前馈网络
            nn.Linear(d, dff),               # 维度变换
            nn.Dropout(0.05, use_dropout),   # Dropout
            nn.GELU(),                       # 激活函数
            nn.Linear(dff, d),               # 维度变换
            nn.Dropout(0.05, use_dropout)    # Dropout
        )

    def forward(self, inputs: Tensor):
        return self.ffn(inputs)


class RoPE_Emb(nn.Module):
    """RoPE位置编码"""
    def __init__(self, d: int, max_len: int=8192, device: Optional[str]=None):
        """
        RoPE位置编码
        - d: 模型维度
        - max_len: 最大序列长度
        """
        super().__init__()

        self.d = d
        self.max_len = max_len
        self.device = device

        inv_freq = 1.0 / (10000 ** (torch.arange(0, d, 2).float().to(device) / d))
        # 计算频率

        self.register_buffer('inv_freq', inv_freq, persistent=False)
        # 注册频率

        self._get_embedding(inv_freq)
        # 预计算

    def _get_embedding(self, inv_freq):
        """预计算位置编码"""
        len_ids = torch.arange(self.max_len, device=self.device)
        # 序列索引

        freqs = torch.outer(len_ids, inv_freq)
        # 计算频率

        emb = torch.cat((freqs, freqs), dim=-1)
        # 复制频率参数, 使复数对共享相同的频率

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        # 频率缓存

    def forward(self) -> Tuple:
        """
        生成RoPE位置编码
        """

        self.cos_cached: Tensor
        self.sin_cached: Tensor

        return (
            self.cos_cached,
            self.sin_cached,
        )

def RoPE_rotate(x: Tensor) -> Tensor:
    """
    RoPE旋转操作
    - x: 输入张量
    """
    x1 = x[..., : x.shape[-1] // 2]   # 取前一半维度
    x2 = x[..., x.shape[-1] // 2 :]   # 取后一半维度
    return torch.cat((-x2, x1), dim=-1)   # 拼接

def RoPE_reshape(x: Tensor) -> Tensor:
    """重塑张量形状"""
    batch, head_num, seq_len, dim = x.shape
    x = x.view(batch, head_num, seq_len, dim//2, 2).transpose(4, 3).reshape(batch, head_num, seq_len, dim)

    return x

def RoPE_apply(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, pos_ids: Tensor):
    """
    应用RoPE编码
    - q: query
    - k: key
    - cos: RoPE cos
    - sin: RoPE sin
    - pos_ids: 位置索引
    """
    cos = cos[pos_ids].unsqueeze(1)   # 按位置索引选择cos值
    sin = sin[pos_ids].unsqueeze(1)   # 按位置索引选择sin值

    q = RoPE_reshape(q)
    # 重塑 Query

    k = RoPE_reshape(k)
    # 重塑 Key

    q_embed = (q * cos) + (RoPE_rotate(q) * sin)
    k_embed = (k * cos) + (RoPE_rotate(k) * sin)
    # 应用旋转位置编码

    return q_embed, k_embed

def RoPE_apply_with_cache(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, q_pos_ids: Tensor, k_pos_ids: Tensor):
    """
    在使用 KV cache 下应用 RoPE 编码
    - q: query
    - k: key
    - cos: RoPE cos
    - sin: RoPE sin
    - q_pos_ids: query 位置索引
    - k_pos_ids: key 位置索引
    """
    q_sin = sin[q_pos_ids].unsqueeze(1)   # 按位置索引选择sin值
    q_cos = cos[q_pos_ids].unsqueeze(1)   # 按位置索引选择cos值
    k_sin = sin[k_pos_ids].unsqueeze(1)   # 按位置索引选择sin值
    k_cos = cos[k_pos_ids].unsqueeze(1)   # 按位置索引选择cos值

    k_sin = k_sin[:, :, :k.size(2)]   # 仅保留最新的 k_pos_ids
    k_cos = k_cos[:, :, :k.size(2)]   # 仅保留最新的 k_pos_ids

    q = RoPE_reshape(q)
    # 重塑 Query

    k = RoPE_reshape(k)
    # 重塑 Key

    q_embed = (q * q_cos) + (RoPE_rotate(q) * q_sin)
    k_embed = (k * k_cos) + (RoPE_rotate(k) * k_sin)
    # 应用旋转位置编码

    return q_embed, k_embed

def CrossRoPE_apply(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, q_pos_ids: Tensor, k_pos_ids: Tensor):
    """
    交叉注意力应用RoPE编码
    - q: query
    - k: key
    - cos: RoPE cos
    - sin: RoPE sin
    - q_pos_ids: query位置索引
    - k_pos_ids: key位置索引
    """
    q_cos = cos[q_pos_ids].unsqueeze(1)
    k_cos = cos[k_pos_ids].unsqueeze(1)
    # 按位置索引选择cos值

    q_sin = sin[q_pos_ids].unsqueeze(1)
    k_sin = sin[k_pos_ids].unsqueeze(1)
    # 按位置索引选择sin值

    q = RoPE_reshape(q)
    # 重塑 Query

    k = RoPE_reshape(k)
    # 重塑 Key

    q_embed = (q * q_cos) + (RoPE_rotate(q) * q_sin)
    k_embed = (k * k_cos) + (RoPE_rotate(k) * k_sin)
    # 应用旋转位置编码

    return q_embed, k_embed


class RMS_norm(nn.Module):
    """均方根层归一化, 相比传统 LayerNorm 有助于梯度稳定性和模型泛化"""
    def __init__(self, hidden_size):
        """
        均方根层归一化 <br>
        hidden_size: 可学习的缩放参数
        """
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        # 定义可学习参数, 初始化

        self.variance_epsilon = 1e-7
        # 防止除零错误

    def forward(self, hidden_states: Tensor):
        input_dtype = hidden_states.dtype                 # 获得原始数据类型
        hidden_states = hidden_states.to(torch.float32)   # 转换成FP32

        variance = hidden_states.pow(2).mean(-1, keepdim=True)   # 沿最后一维计算均方
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # RMS_Norm计算

        return self.weight * hidden_states.to(input_dtype)
        # 还原原始数据类型


def cache_update(cache: Tensor, inputs: Tensor, cache_pos: int | None, base_len: int):
    """
    更新缓存, 以适应新的输入

    参数:
    - cache: 缓存张量   [batch, cache_len, d]
    - inputs: 新输入张量  [batch, seq_len, d]
    - cache_pos: 当前缓存位置
    - base_len: 保留的长度 (通常是全局提示词长度)
    """
    new_token = inputs.size(1)   # 新输入的长度
    if cache_pos is None:   # 如果没有缓存位置, 则初始化为0
        cache_pos = 0

    # 情况1: 直接追加
    if cache_pos + new_token <= cache.size(1):
        cache[:, cache_pos:cache_pos + new_token] = inputs
        new_cache_pos = cache_pos + new_token
        return cache, new_cache_pos

    # 情况2: 输入过大, 舍弃原缓存
    elif base_len + new_token >= cache.size(1):
        save_len = cache.size(1) - base_len
        # 用的时候别整的 base_len > max_len
        # 这情况是真救不了

        cache = torch.cat(
            [
                cache[:, :base_len, :],   # 保留的基础缓存
                inputs[:, -save_len:, :],   # 保留未丢弃的部分
            ], dim=1
        )   # 更新缓存

        new_cache_pos = cache.size(1)
        return cache, new_cache_pos

    # 情况3: 需要丢弃旧数据
    else:   # 缓存长度不足时, 丢弃最早的缓存
        cache = torch.cat(
            [
                cache[:, :base_len, :],   # 保留的基础缓存
                cache[:, base_len:(cache.size(1)-new_token), :],   # 保留未丢弃的部分
                inputs,   # 新输入
            ], dim=1
        )   # 更新缓存

        new_cache_pos = cache.size(1)
        print("cache_pos:", new_cache_pos)
        return cache, new_cache_pos


class MLAframe(nn.Module):
    """MLA前置框架"""
    def __init__(self, d, d_pre_head, head_num, max_len: int=8192,
        use_dropout: bool=False, device: Optional[str]=None):
        """
        MLA前置框架
        用于MLA初始化

        参数:
        - d: 输入/输出的维度
        - d_pre_head: 每个头的隐藏层维度 (非RoPE维度)
        - head_num: 头的数量
        - use_dropout: 是否使用dropout
        - device: 计算设备
        """
        super().__init__()

        self.d = d
        self.d_pre_head = d_pre_head
        self.head_num = head_num
        self.use_dropout = use_dropout
        self.device = device
        # 前置传入参数

        self.d_rope = d_pre_head // 2   # 0.5 d_head
        # 计算位置编码维度

        self.dc_v = self.d_pre_head   # 1.0 d_head
        # value维度

        self.dc_kv = self.d_pre_head // 8   # kv Lora维度   0.125 d_head
        self.dc_q = d // 4         # quary Lora维度   0.25 d
        # 低秩压缩的维度
        # DeepSeek 此处 kv 压缩比为 1/14 , q 压缩比为 1/4.7

        self.out_proj = nn.Linear(
            self.head_num * self.dc_v,   # d_head * head_num
            self.d,   # 1.0 d
            bias=False,
        )   # 输出投影

        # ================ quary Lora ================
        self.q_head_dim = self.d_pre_head + self.d_rope   # 1.5 d_head
        # 每个头的quary维度

        self.attn_scale = self.q_head_dim ** -0.5
        # 注意力缩放

        self.q_up = nn.Linear(
            self.dc_q,   # 0.25 d
            self.head_num * self.q_head_dim,   # 1.5 d_head * head_num
            bias=False,
        )   # 升维矩阵

        self.q_down = nn.Linear(
            self.d,   # d
            self.dc_q,   # 0.25 d
            bias=False,
        )   # 降维矩阵

        self.q_down_norm = RMS_norm(self.dc_q)

        # =========== key & value Lora ===========
        self.meg_d = self.d_pre_head + self.dc_v   # 1.0 d_head + 0.125 d
        # 合并投影, 便于实现单次完成两者升维

        self.kv_up = nn.Linear(
            self.dc_kv,   # 0.125 d
            self.head_num * self.meg_d,   # 1.0 d_head * head_num + 0.125 d * head_num
            bias=False,
        )   # 升维矩阵

        self.kv_down = nn.Linear(
            self.d,   # 1.0 d
            self.dc_kv + self.d_rope,   # 0.125 d + 0.5 d_head
            bias=False,
        )   # 降维矩阵

        self.kv_norm = RMS_norm(self.dc_kv)

        # ============ RoPE ============
        self.rope = RoPE_Emb(
            self.d_rope,
            max_len=2 * max_len if 2 * max_len > 8192 else 8192,   # 留足余量
            device=device,
        )


class MLA(MLAframe):
    """多头潜在注意力, 通过低秩投影 (LoRA) 压缩 Q/K/V 维度"""
    def __init__(self, d: int, d_pre_head: int, head_num: int, max_len: int, use_cache: bool=False,
        cache_type: Optional[str]=None, use_dropout: bool=False, device: Optional[str]=None):
        """
        我的想法是尽量减少亢余参数;  
        所以相比于主流实现而言自由度更小, 相应的传参更少

        参数:
        - d: 输入/输出的维度
        - dk_pre_head: 每个头的隐藏层维度 (非RoPE维度)
        - head_num: 头的数量
        - max_len: 最大序列长度
        - use_cache: 是否使用 KV 缓存
        - cache_type: KV 缓存类型  (legacy / absorb)
        - use_dropout: 是否使用dropout
        - device: 计算设备
        """
        super().__init__(d, d_pre_head, head_num, max_len, use_dropout, device)

        # ====== kv cache ======
        self.max_kv_cache = max_len
        self.use_cache = use_cache
        self.cache_type = cache_type
        self.cache_pos = None   # 缓存位置

        if use_cache:
            assert cache_type in ['legacy', 'absorb'],   \
                "cache_type 必须是 'legacy' 或 'absorb' | cache_type must be 'legacy' or 'absorb'"

            if cache_type == 'legacy':   # 传统缓存
                self.register_buffer("kv_cache", torch.zeros(1, max_len, self.dc_kv + self.d_rope), persistent=False)

            elif cache_type == 'absorb':   # 矩阵吸收缓存
                self.register_buffer("kv_cache", torch.zeros(1, max_len, self.dc_kv), persistent=False)
                self.register_buffer("pe_cache", torch.zeros(1, max_len, self.d_rope), persistent=False)

        else:   # 不使用缓存
            pass

    def clean_cache(self):
        """清除 KV 缓存"""
        if self.cache_type == 'legacy':
            self.kv_cache = torch.zeros(1, self.max_kv_cache, self.dc_kv + self.d_rope)
        elif self.cache_type == 'absorb':
            self.kv_cache = torch.zeros(1, self.max_kv_cache, self.dc_kv)
            self.pe_cache = torch.zeros(1, self.max_kv_cache, self.d_rope)

        else:
            print("无缓存, 无需清除 | No cache to clear")

        self.cache_pos = None

    def without_cache_forward(self, inputs: Tensor, pos_ids: Tensor, mask: Optional[Tensor]=None) -> Tensor:
        """
        不使用缓存的前向传播

        参数:
        - inputs: 输入序列 [batch, seq_len, d]
        - pos_ids: 位置索引
        - mask: 掩码 [seq_len, seq_len]
        - base_len: 基础长度 (通常是全局提示词长度)
        """
        batch_size, seq_len, _ = inputs.size()   # 获得批次与长度

        # ===== quary 计算 =====
        q = self.q_down(inputs)
        q = self.q_down_norm(q)
        q = self.q_up(q)
        # 低秩投影

        q: Tensor = q.view(batch_size, seq_len, self.head_num, self.q_head_dim).transpose(1, 2)
        q_nope, q_rope = torch.split(q, [self.d_pre_head, self.d_rope], dim=-1)
        # 多头拆分 & 维度分割

        # ========= KV 处理 =========
        c_kv = self.kv_down(inputs)
        c_kv, k_rope = torch.split(
            c_kv, [self.dc_kv, self.d_rope], dim=-1
        )   # 分割维度

        c_kv: Tensor = self.kv_norm(c_kv)
        # 归一化 KV

        kv: Tensor = self.kv_up(c_kv)
        # 升维处理

        kv = kv.view(batch_size, c_kv.size(1), self.head_num, self.d_pre_head + self.dc_v).transpose(1, 2)
        k_nope, value = torch.split(kv, [self.d_pre_head, self.dc_v], dim=-1)
        k_rope = k_rope.view(batch_size, c_kv.size(1), 1, self.d_rope).transpose(1, 2)
        # 形状转换 & 矩阵分割

        # ============ RoPE 应用 ============
        cos, sin = self.rope()
        q_rope, k_rope = RoPE_apply(
            q_rope, k_rope, cos, sin, pos_ids,
        )   # 应用 RoPE 编码

        # ============ attention ============
        query = torch.concat(
            [q_nope, q_rope], dim=-1
        )   # 拼接 Query

        key = torch.concat(
            [k_nope, k_rope.expand(-1, self.head_num, -1, -1)], dim=-1
        )   # 拼接 Key

        attn_output = fc.scaled_dot_product_attention(
            query, key, value, attn_mask=mask,
            dropout_p=0.05 if self.use_dropout else 0.0,
        )   # 注意力计算

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        output = self.out_proj(attn_output)
        # 变换形状, 输出投影

        return output

    def legacy_forward(self, inputs: Tensor, all_pos_ids: Tensor, inputs_pos_ids: Tensor,
        mask: Optional[Tensor]=None, base_len: int=0) -> Tensor:
        """
        传统缓存的前向传播, 使用 flash attention 优化

        参数:
        - inputs: 输入序列 [batch, seq_len, d]
        - all_pos_ids: 全序列位置索引
        - inputs_pos_ids: 输入位置索引
        - mask: 掩码 [seq_len, seq_len]
        - base_len: 基础长度 (通常是全局提示词长度)
        """
        batch_size, seq_len, _ = inputs.size()   # 获得批次与长度
        new_token = inputs.size(1)

        # ===== quary 计算 =====
        q = self.q_down(inputs)
        q = self.q_down_norm(q)
        q = self.q_up(q)
        # 低秩投影

        q: Tensor = q.view(batch_size, seq_len, self.head_num, self.q_head_dim).transpose(1, 2).contiguous()
        q_nope, q_rope = torch.split(q, [self.d_pre_head, self.d_rope], dim=-1)
        # 多头拆分 & 维度分割

        # ======== 缓存处理 ========
        c_kv = self.kv_down(inputs)
        # 新输入的 token

        self.kv_cache: Tensor = self.kv_cache.expand(batch_size, -1, -1).contiguous()
        # 缓存本体, 扩展维度

        self.kv_cache, self.cache_pos = cache_update(self.kv_cache, c_kv, self.cache_pos, base_len)
        # 更新 KV 缓存

        all_c_kv, k_rope = torch.split(self.kv_cache[:, :self.cache_pos, :], [self.dc_kv, self.d_rope], dim=-1)
        # 分割 nope & rope

        mask = None if mask is None else mask[-new_token:, :].contiguous()
        # 调整掩码形状

         # ======== KV 合并处理 ========
        kv: Tensor = self.kv_up(all_c_kv)
        kv = kv.view(batch_size, all_c_kv.size(1), self.head_num, self.d_pre_head + self.dc_v).transpose(1, 2)
        k_nope, value = torch.split(kv, [self.d_pre_head, self.dc_v], dim=-1)
        k_rope = k_rope.view(batch_size, all_c_kv.size(1), 1, self.d_rope).transpose(1, 2).contiguous()
        # 形状转换 & 矩阵分割

        # ============ RoPE 应用 ============
        cos, sin = self.rope()
        q_rope, k_rope = RoPE_apply_with_cache(
            q_rope, k_rope, cos, sin, inputs_pos_ids, all_pos_ids,
        )   # 应用 RoPE 编码

        # ============ attention ============
        query = torch.concat(
            [q_nope, q_rope], dim=-1
        )   # 拼接 Query

        key = torch.concat(
            [k_nope, k_rope.expand(-1, self.head_num, -1, -1)], dim=-1
        )   # 拼接 Key

        attn_output = fc.scaled_dot_product_attention(
            query.contiguous(), key.contiguous(), value.contiguous(), attn_mask=mask,
            dropout_p=0.05 if self.use_dropout else 0.0,
        )   # 注意力计算

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        output = self.out_proj(attn_output)
        # 变换形状, 输出投影

        return output

    def absorb_forward(self, inputs: Tensor, all_pos_ids: Tensor, inputs_pos_ids: Tensor,
            mask: Optional[Tensor]=None, base_len: int=0) -> Tensor:
        """
        矩阵吸收优化缓存的前向传播
        参数:
        - inputs: 输入序列 [batch, seq_len, d]
        - all_pos_ids: 全序列位置索引
        - inputs_pos_ids: 输入位置索引
        - mask: 掩码 [seq_len, seq_len]
        - base_len: 基础长度 (通常是全局提示词长度)
        """
        batch_size, seq_len, _ = inputs.size()   # 获得批次与长度
        new_token = inputs.size(1)

        # ===== quary 计算 =====
        q = self.q_down(inputs)
        q = self.q_down_norm(q)
        q = self.q_up(q)
        # 低秩投影

        q: Tensor = q.view(batch_size, seq_len, self.head_num, self.q_head_dim).transpose(1, 2)
        q_nope, q_rope = torch.split(q, [self.d_pre_head, self.d_rope], dim=-1)
        # 多头拆分 & 维度分割

        # ======= KV 处理 =======
        kv = self.kv_down(inputs)
        # 新输入的 token

        kv_nope, k_rope = torch.split(kv, [self.dc_kv, self.d_rope], dim=-1)
        # 分割维度

        # ============ RoPE 应用 ============
        cos, sin = self.rope()
        q_rope, k_rope = RoPE_apply_with_cache(
            q_rope, k_rope.unsqueeze(1), cos, sin, inputs_pos_ids, inputs_pos_ids,
        )   # 应用 RoPE 编码

        k_rope = k_rope.squeeze(1)   # 去除多余维度

        # ============ KV 缓存处理 ============
        self.kv_cache: Tensor = self.kv_cache.expand(batch_size, -1, -1).contiguous()
        self.pe_cache: Tensor = self.pe_cache.expand(batch_size, -1, -1).contiguous()
        # 扩展 KV 缓存

        self.kv_cache, _ = cache_update(self.kv_cache, kv_nope, self.cache_pos, base_len)
        self.pe_cache, self.cache_pos = cache_update(self.pe_cache, k_rope, self.cache_pos, base_len)
        # 更新 KV 缓存

        # ============ attention ============
        kv_up = self.kv_up.weight   # 升维矩阵
        kv_up = kv_up.view(self.head_num, -1, self.dc_kv)
        # 重塑 kv_up 矩阵

        q_nope, q_rope = q_nope.transpose(1, 2), q_rope.transpose(1, 2)
        q_nope = torch.einsum("bshd,hdc->bshc", q_nope, kv_up[:, :self.d_pre_head])
        # 矩阵吸收

        attn_scores: Tensor = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:, :self.cache_pos]) +
                torch.einsum("bshr,btr->bsht", q_rope, self.pe_cache[:, :self.cache_pos])) * self.attn_scale
        # 注意力计算

        if mask is not None:   # 掩码
            attn_scores += mask.transpose(1, 2)

        attn_scores = attn_scores.softmax(dim=-1, dtype=torch.float32)
        # softmax

        outputs = torch.einsum("bsht,btc->bshc", attn_scores, self.kv_cache[:, :self.cache_pos])
        outputs = torch.einsum("bshc,hdc->bshd", outputs, kv_up[:, -self.dc_v:])
        # value 计算 (矩阵吸收)

        outputs = outputs.reshape(batch_size, seq_len, self.head_num * self.dc_v)
        outputs = self.out_proj(outputs)   # 输出投影

        return outputs

    def forward(self, inputs: Tensor, all_pos_ids: Tensor, inputs_pos_ids: Optional[Tensor]=None,
                mask: Optional[Tensor]=None, base_len: int=0) -> Tensor:
        """
        - input: 输入序列 [batch, seq_len, d]
        - all_pos_ids: 全序列位置索引
        - inputs_pos_ids: 输入位置索引
        - mask: 掩码 [seq_len, seq_len]
        - base_len: 基础长度 (通常是全局提示词长度)
        """
        if not self.use_cache:   # 不使用缓存时
            return self.without_cache_forward(inputs, all_pos_ids, mask)

        elif self.cache_type == 'legacy':   # 常规缓存
            return self.legacy_forward(inputs, all_pos_ids, inputs_pos_ids, mask, base_len)   # type: ignore

        elif self.cache_type == 'absorb':   # 矩阵吸收优化缓存
            return self.absorb_forward(inputs, all_pos_ids, inputs_pos_ids, mask, base_len)   # type: ignore

        else:   # 其他情况
            return self.without_cache_forward(inputs, all_pos_ids, mask)


class Vit_MLA(MLAframe):
    """多头潜在注意力, 通过低秩投影 (LoRA) 压缩 Q/K/V 维度"""
    def __init__(self, d, d_pre_head, head_num,
        use_dropout: bool=False, device: Optional[str]=None):
        """
        Vit 没办法使用 KV Cache

        参数:
        - d: 输入/输出的维度
        - dk_pre_head: 每个头的隐藏层维度 (非RoPE维度)
        - head_num: 头的数量
        - use_dropout: 是否使用dropout
        - device: 计算设备
        """
        super().__init__(d, d_pre_head, head_num, 8192, use_dropout, device)

    def forward(self, inputs: Tensor, pos_ids, mask=None):
        """
        - input: 输入序列 [batch, seq_len, d]
        - pos_ids: 位置索引
        - mask: 掩码
        """

        batch_size, seq_len, _ = inputs.size()
        # 获得批次与长度

        # ===== quary 计算 =====
        q = self.q_down(inputs)
        q = self.q_down_norm(q)
        q = self.q_up(q)
        # 低秩投影

        q: Tensor = q.view(batch_size, seq_len, self.head_num, self.q_head_dim).transpose(1, 2)
        q_nope, q_rope = torch.split(q, [self.d_pre_head, self.d_rope], dim=-1)
        # 多头拆分 & 维度分割

        # ===== key & value 计算 ======
        c_kv = self.kv_down(inputs)
        c_kv, k_rope = torch.split(c_kv, [self.dc_kv, self.d_rope], dim=-1)
        kv = self.kv_norm(c_kv)
        kv = self.kv_up(kv)
        # 低秩投影

        kv: Tensor = kv.view(batch_size, seq_len, self.head_num, self.d_pre_head + self.dc_v).transpose(1, 2)
        k_nope, value = torch.split(kv, [self.d_pre_head, self.dc_v], dim=-1)
        # 升维 & 多头拆分

        k_rope = k_rope.view(batch_size, seq_len, 1, self.d_rope).transpose(1, 2)

        # ============ RoPE 应用 ============
        cos, sin = self.rope()
        q_rope, k_rope = RoPE_apply(
            q_rope, k_rope, cos, sin, pos_ids,
        )

        # ============ attention ============
        query = torch.concat(
            [q_nope, q_rope], dim=-1
        )   # 拼接 Query

        key = torch.concat(
            [k_nope, k_rope.expand(-1, self.head_num, -1, -1)], dim=-1
        )   # 拼接 Key

        attn_output = fc.scaled_dot_product_attention(
            query, key, value, attn_mask=mask,
            dropout_p=0.05 if self.use_dropout else 0.0,
        )   # 注意力计算

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        output = self.out_proj(attn_output)
        # 变换形状, 输出投影

        return output


class CrossMLA(MLAframe):
    """交叉多头潜在注意力 支持kv cache"""
    def __init__(self, d, d_pre_head, head_num, use_dropout=False, device=None):
        """
        参数:
        - d: 输入/输出的维度
        - d_pre_head: 每个头的隐藏层维度 (非RoPE维度)
        - head_num: 头的数量
        - max_len: 最大序列长度
        - use_dropout: 是否使用 dropout
        - device: 设备
        """
        super().__init__(d, d_pre_head, head_num, use_dropout, use_dropout)
        self.text_cache_len = 0

        self.register_buffer("kv_cache", torch.zeros(1, 1, self.dc_kv), persistent=False)

    def forward(self, q_inputs: Tensor, kv_input: Tensor, q_pos_ids, kv_pos_ids):
        """
        不使用缓存的前向传播

        - q_input: decoder 输入序列 [batch, text_seq_len, d]
        - kv_input: encoder 输入序列 [batch, encoder_seq_len, d]
        - q_pos_ids: 解码器位置ID [batch, text_seq_len]
        - kv_pos_ids: 编码器位置ID [batch, encoder_seq_len]
        """
        batch_size, text_seq_len, _ = q_inputs.size()

        # ===== quary 计算 =====
        q = self.q_down(q_inputs)
        q = self.q_down_norm(q)
        q = self.q_up(q)
        # 低秩投影

        q: Tensor = q.view(batch_size, text_seq_len, self.head_num, self.q_head_dim).transpose(1, 2)
        q_nope, q_rope = torch.split(q, [self.d_pre_head, self.d_rope], dim=-1)
        # 多头拆分 & 维度分割

        # ========= KV 处理 =========
        c_kv = self.kv_down(kv_input)
        c_kv, k_rope = torch.split(
            c_kv, [self.dc_kv, self.d_rope], dim=-1
        )   # 分割维度

        c_kv: Tensor = self.kv_norm(c_kv)
        # 归一化 KV

        kv: Tensor = self.kv_up(c_kv)
        # 升维处理

        kv = kv.view(batch_size, c_kv.size(1), self.head_num, self.d_pre_head + self.dc_v).transpose(1, 2)
        k_nope, value = torch.split(kv, [self.d_pre_head, self.dc_v], dim=-1)
        k_rope = k_rope.view(batch_size, c_kv.size(1), 1, self.d_rope).transpose(1, 2)
        # 形状转换 & 矩阵分割

        # ============ RoPE 应用 ============
        cos, sin = self.rope()
        q_rope, k_rope = CrossRoPE_apply(
            q_rope, k_rope, cos, sin, q_pos_ids, kv_pos_ids,
        )   # 应用 RoPE 编码

        # ============ attention ============
        query = torch.concat(
            [q_nope, q_rope], dim=-1
        )   # 拼接 Query

        key = torch.concat(
            [k_nope, k_rope.expand(-1, self.head_num, -1, -1)], dim=-1
        )   # 拼接 Key

        attn_output = fc.scaled_dot_product_attention(
            query, key, value, attn_mask=None,
            dropout_p=0.05 if self.use_dropout else 0.0,
        )   # 注意力计算

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, text_seq_len, -1)
        output = self.out_proj(attn_output)
        # 变换形状, 输出投影

        return output


class Expert(nn.Module):
    """专家头"""
    def __init__(self, d, dff):
        """标准的 SwiGLU 结构"""
        super().__init__()
        self.Wx = nn.Linear(d, dff, bias=False)
        # 映射线性层
        
        self.Vx = nn.Linear(d, dff, bias=False)
        # 门控机制

        self.last_linear = nn.Linear(dff, d, bias=False)
        # 输出线性层

    def forward(self, inputs: Tensor):
        """
        - inputs: 输入序列
        """
        Wx = self.Wx(inputs)
        Vx = self.Vx(inputs)
        # 线性映射

        gate = fc.silu(Wx)
        output = gate * Vx
        # 门控机制

        output = self.last_linear(output)
        # 输出线性层

        return output


class MOERouter(nn.Module):
    """路由门逻辑"""
    def __init__(self, d, expert_num, top_k):
        """
        参数:
        - d: 输入维度
        - expert_num: 专家数量
        - top_k: 激活专家数
        """
        super().__init__()
        self.gate = nn.Linear(d, expert_num)   # 路由门
        self.expert_num = expert_num
        self.top_k = top_k

    def forward(self, hidden_states):

        router_logits = self.gate(hidden_states)
        # 计算路由 logits

        routing_probs = fc.softmax(router_logits, dim=-1)
        # softmax
        
        router_weights, selected_experts = torch.topk(
            routing_probs, self.top_k, dim=-1
        )   # 返回 top_k 权重及其专家

        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)
        # 权重归一化

        expert_mask = fc.one_hot(selected_experts, num_classes=self.expert_num)
        expert_mask = expert_mask.permute(2, 1, 0)
        # 生成专家掩码, 降低时间复杂度

        return router_logits, router_weights, selected_experts, expert_mask


class SparseMOE(nn.Module):
    def __init__(self, d, dff, expert_num, top_k, init_weights: bool=False):
        """
        稀疏混合专家模型

        参数:
        - d: 输入维度
        - dff: 映射维度
        - expert_num: 专家数量
        - top_k: 激活专家数
        - init_weights: 是否初始化权重
        """
        super().__init__()
        self.d = d
        self.dff = dff
        self.expert_num = expert_num
        self.top_k = top_k
        # 初始化参数

        self.experts = nn.ModuleList([
            Expert(self.d, self.dff) 
            for _ in range(self.expert_num)
        ])  # 添加专家头 

        self.router = MOERouter(self.d, self.expert_num, self.top_k)
        # 路由模块

        if init_weights:   # 初始化权重
            self.apply(generate_init_weights)

    def forward(self, x: Tensor):
        batch_size, seq_len, d = x.shape
        d_states = x.view(-1, d)

        router_logits, router_weights, _, expert_mask = self.router(d_states)
        # 获取路由信息

        router_weights: Tensor   # 方便IDE工作
        final_hidden_states = torch.zeros(
            (batch_size * seq_len, d),
            dtype=d_states.dtype,
            device=d_states.device
        )   # 初始化输出张量

        for expert_idx in range(self.expert_num):   # 遍历每个专家, 检查是否有 token 被分配
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            # 找到需要当前专家处理的 token

            if top_x.shape[0] == 0:
                continue
            # 专家未被 token 选择时, 跳过计算以节省资源

            current_state = d_states[top_x, :]
            # 获取当前专家处理的 token 的输入

            current_hidden_states = expert_layer(current_state) * \
                router_weights[top_x, idx].unsqueeze(-1)
                # 计算加权输出

            final_hidden_states.index_add_(0, top_x, current_hidden_states)
            # 累加最终输出

        final_hidden_states = final_hidden_states.view(batch_size, seq_len, d)
        # 恢复原始形状

        return final_hidden_states, router_logits


class MOE(nn.Module):
    """混合专家模型"""
    def __init__(self, d, dff, share_num, expert_num, top_k, init_weights: bool=False):
        """
        参数:
        - d: 每个专家的输入维度
        - dff: 映射维度
        - share_num: 共享专家数量
        - expert_num: 专家数量
        - top_k: 激活专家数
        - init_weights: 是否初始化权重
        """
        super().__init__()
        self.moe = SparseMOE(d, dff, expert_num, top_k, init_weights)
        # 稀疏混合专家模型

        self.share_experts = nn.ModuleList([
            Expert(d, dff) for _ in range(share_num)
        ])

        if init_weights:   # 初始化权重
            self.apply(generate_init_weights)

    def forward(self, x: Tensor):
        """
        - x: 输入序列 [batch, seq_len, d]
        """

        moe_output, router_logits = self.moe(x)
        # MoE计算

        share_out = [
            expert(x) for expert in self.share_experts
        ]   # 共享专家计算

        share_out = torch.stack(share_out, dim=0).sum(dim=0)
        # 累加共享计算结果
    
        output = share_out + moe_output   # 累加共享与MoE计算结果
        return output                     # 返回输出


class Get_Pos_ids(nn.Module):
    """获得 pos_ids"""
    def __init__(self):
        """创建并获得 pos_ids"""
        super().__init__()

    def forward(self, x: Tensor, _: int=0) -> tuple[Tensor, None]:
        """
        - x: 输入序列 [batch, seq_len, d]
        - _ : 缓存位置 (未使用)
        """
        batch_size, seq_len = x.size(0), x.size(1)
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return pos_ids, None  # [batch_size, seq_len]

class Get_Pos_ids_with_cache(nn.Module):
    """获得带缓存的 pos_ids"""
    def __init__(self):
        """创建并获得 pos_ids"""
        super().__init__()

    def forward(self, x: Tensor, cache_pos: int) -> tuple[Tensor, Tensor]:
        """
        - x: 输入序列 [batch, seq_len, d]
        - cache_pos: 当前缓存位置
        """
        batch_size, seq_len = x.size(0), x.size(1)

        all_pos_ids = torch.arange(cache_pos + seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        inputs_pos_ids = torch.arange(cache_pos, cache_pos + seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        return all_pos_ids, inputs_pos_ids   # 返回所有位置 ID 和输入位置 ID


class Padding_Mask(nn.Module):
    """填充索引掩码"""
    def __init__(self, padding_idx: int):
        """
        padding_idx: 填充索引
        """
        super().__init__()
        self.padding_idx = padding_idx

    def forward(self, x: Tensor) -> Tensor:
        """
        - x: 输入序列 [batch, seq_len]
        """
        batch_size, seq_len = x.size(0), x.size(1)
        padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=x.device)
        padding_mask[x == self.padding_idx] = float('-inf')
        padding_mask = padding_mask.reshape(batch_size, 1, 1, seq_len)
        # 创建一个与 x 形状相同的全零矩阵
        # 把 padding 位置设置为-inf
        # 扩展维度

        return padding_mask

class Causal_Mask(nn.Module):
    """因果掩码"""
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        - x: 输入序列 [batch, seq_len]
        """
        batch_size, seq_len = x.size(0), x.size(1)
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1)
        # 创建上三角掩码, 设置掩码遮掩

        causal_mask = causal_mask.reshape(1, 1, seq_len, seq_len)
        # 扩展维度

        return causal_mask
    
class Visual_Mask(nn.Module):
    """视觉掩码"""
    def __init__(self):
        super().__init__()

    def forward(self, patch_mask: Tensor) -> Tensor:
        """
        - patch_mask: 块掩码 [batch, num_patches]
        """
        visual_mask = patch_mask.unsqueeze(1).unsqueeze(2)   # [batch, 1, 1, num_patches]
        visual_mask = torch.where(visual_mask == 0, float('-inf'), 0.0)   # 无效位置设为-inf
        return visual_mask


class Decoder(nn.Module):
    """解码器"""
    def __init__(self, head_num: int, share_num: int, exp_num: int, top_k: int, d: int, dk: int, max_len: int,
        use_cache: bool=False, cache_type: Optional[str]=None, use_dropout: bool=False, init_weights: bool=False, ffn_type: str='ffn'):
        """
        参数:
        - head_num: 注意力头数
        - share_num: 共享专家数
        - exp_num: 专家数量
        - top_k: 激活专家数
        - d: 输入/输出维度
        - dk: 每个头的维度
        - max_len: 最大序列长度
        - use_cache: 是否使用缓存
        - cache_type: 缓存类型 (legacy / absorb)
        - use_dropout: 是否使用dropout
        - init_weights: 是否初始化模型
        - ffn_type: 前馈网络类型 (ffn / moe)
        """
        super().__init__()
        self.self_attn = MLA(d, dk, head_num, max_len, use_cache, cache_type, use_dropout)
        # 多头自注意力层

        self.cross_attn = CrossMLA(d, dk, head_num, use_dropout)
        # 交叉多头自注意力层

        self.get_pos_ids = Get_Pos_ids_with_cache() if use_cache else Get_Pos_ids()
        self.cross_pos_ids = Get_Pos_ids()
        # 获得 pos_ids

        self.self_attn_norm = nn.LayerNorm(d)
        self.cross_attn_norm = nn.LayerNorm(d)
        self.ffn_norm = nn.LayerNorm(d)
        # 层归一化

        assert ffn_type in ['ffn', 'moe'], '请选择正确的前馈网络类型 (ffn / moe)'

        if ffn_type == 'ffn':
            self.ffn = FeedForward(d, 4*d, use_dropout)
            # 前馈网络

        elif ffn_type == 'moe':
            self.ffn = MOE(d, d//2, share_num, exp_num, top_k, init_weights)
            # 混合专家网络
            # 因为有很多专家, dff的维度较小

        if init_weights:   # 初始化权重
            self.apply(generate_init_weights)

    def forward(
            self, inputs_tuple: Tuple[Tensor, Tensor], enc_inputs: Optional[Tensor]=None, base_len: int=0
        ) -> Tuple:
        """
        layer_norm 使用 Pre-LN 结构

        参数:
        - inputs_tuple: 输入元组 [input, mask]
        - input: 输入序列 [batch, seq_len, model_dim]
        - mask: 目标序列的掩码 [batch, seq_len]
        - enc_inputs: 编码器输入序列 [batch, seq_len, model_dim]
        - base_len: 基础长度 (通常是全局提示词长度)
        """

        inputs, mask = inputs_tuple
        # 解包元组

        # ======= 自注意力阶段 =======
        norm_input = self.self_attn_norm(inputs)   # 归一化输入
        all_pos_ids, inputs_pos_ids = self.get_pos_ids(inputs,
            self.self_attn.cache_pos if self.self_attn.cache_pos is not None else 0)
        # 获取位置 ID

        self_attn_output = self.self_attn(
            norm_input,
            all_pos_ids=all_pos_ids,
            inputs_pos_ids=inputs_pos_ids,
            mask=mask,
            base_len=base_len,
        )   # 自注意力计算

        self_attn_output = inputs + self_attn_output
        attn_output = self_attn_output
        # 残差连接

        # ======= 交叉注意力阶段 =======
        if enc_inputs is not None:
            q_pos_ids, _ = self.cross_pos_ids(inputs)
            kv_pos_ids, _ = self.cross_pos_ids(enc_inputs)
            # 获得 pos_ids

            norm_cross = self.cross_attn_norm(self_attn_output)
            cross_attn_output = self.cross_attn(
                norm_cross, 
                enc_inputs,
                q_pos_ids=q_pos_ids,
                kv_pos_ids=kv_pos_ids,
            )   # 交叉注意力计算

            attn_output = self_attn_output + cross_attn_output
            # 残差连接

        # ======= 前馈阶段 =======
        norm_output = self.ffn_norm(attn_output)
        final_output = attn_output + self.ffn(norm_output)
        # 残差连接

        return (final_output, mask)


class VisionPretreat(nn.Module):
    """视觉预处理"""
    def __init__(self, img_size):
        """
        提供图像大小与视觉掩码处理 <br>
        img_size: 图像大小 (size * size)
        """
        super().__init__()
        self.img_size = img_size

    def forward(self, imgs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        输入:
            imgs (Tensor): 图像批次 [batch, in_chans, height, width]
        返回:
            processed (Tensor): 处理后的图像 [batch, in_chans, img_size, img_size]
            mask_ids (Tensor): 掩码位置信息 [batch, 1, img_size, img_size]
        """
        batch_size, in_chans, height, width = imgs.size()
        # 获取图像的高度和宽度

        max_side = max(height, width)
        scale = self.img_size / max_side
        new_H, new_W = int(height * scale), int(width * scale)
        # 确定长边和缩放比例

        resized = fc.interpolate(
            imgs, 
            size=(new_H, new_W), 
            mode="bilinear", 
            align_corners=False
        )   # 缩放图像, 保留长宽比

        pad_h = (self.img_size - new_H) // 2
        pad_w = (self.img_size - new_W) // 2
        padding = [pad_w,                            # 左填充
                   self.img_size - new_W - pad_w,    # 右填充
                   pad_h,                            # 上填充
                   self.img_size - new_H - pad_h]    # 下填充
        processed = fc.pad(resized, padding, value=0)
        # 填充图像 (居中填充)

        mask = torch.zeros((batch_size, 1, self.img_size, self.img_size), 
            dtype=torch.float32, device=imgs.device)
        mask[:, :, pad_h:pad_h+new_H, pad_w:pad_w+new_W] = 1
        # 生成掩码

        return processed, mask


class PatchEmbedding(nn.Module):
    """分割patches"""
    def __init__(self, img_size=1024, patch_size=28, in_chans=3, embed_dim=64):
        """
        参数:
        - img_size: 图像大小 (size * size)
        - patch_size: 分割大小 (正方形)
        - in_chans: 输入通道数
        - embed_dim: 嵌入维度
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        # 初始化

        self.conv = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 卷积层

        self.pr_imgs = VisionPretreat(img_size)
        # 预处理图像

    def forward(self, imgs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        参数:
        - x: 输入图像 [batch, in_chans, img_size, img_size]
        """
        imgs = imgs.to(torch.float32)   # 转换为浮点数
        imgs, pixel_mask = self.pr_imgs(imgs)   # [batch, in_chans, img_size, img_size]
        # 预处理图像

        conv_imgs: Tensor = self.conv(imgs)   # [embed_dim, num_patches, num_patches]
        # 通过卷积层分割图像

        output = conv_imgs.flatten(2)     # [batch, embed_dim, num_patches^2]
        output = output.transpose(1, 2)   # [batch, num_patches^2, embed_dim]
        # 调整尺寸以适应Transformer输入格式

        mask_patch = fc.max_pool2d(
            pixel_mask, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        ).flatten(1)
        # [batch, num_patches]
        # 生成patch级掩码

        return output, mask_patch   # 返回处理后的图像


class VisionEncoder(nn.Module):
    """视觉编码器"""
    def __init__(self, head_num: int, d: int, dk: int,
        use_dropout: bool=False, init_weights: bool=False):
        """
        应该没人会在这里用MOE吧?

        参数:
        - head_num: 注意力头数
        - d: 输入/输出维度
        - dk: 每个头的维度
        - use_dropout: 是否使用dropout
        - init_weights: 是否初始化模型
        """
        super().__init__()
        self.head_num = head_num
        self.d = d
        self.dk = dk
        self.use_dropout = use_dropout
        self.init_weights = init_weights
        # 初始化参数

        self.visual_mask = Visual_Mask()

        self.self_attn = Vit_MLA(d, dk, head_num, use_dropout)
        # 多头自注意力层

        self.get_pos_ids = Get_Pos_ids()
        # 获得 pos_ids

        self.self_attn_norm = nn.LayerNorm(d)
        self.ffn_norm = nn.LayerNorm(d)
        # 层归一化

        self.ffn = FeedForward(d, 4*d, use_dropout)
        # 前馈网络

        if init_weights:   # 初始化权重
            self.apply(generate_init_weights)

    def forward(self, imgs: Tensor, patch_mask: Tensor) -> Tensor:
        """
        图像编码器前向传播
        - imgs: 输入图像 [batch, num_patch^2, d]
        - patch_mask: 块掩码 [batch, num_patches]
        """
        # ===== 自注意力阶段 =====
        residual = imgs
        norm_input = self.self_attn_norm(imgs)
        pos_ids, _ = self.get_pos_ids(imgs)

        self_attn_output = self.self_attn(
            norm_input,
            pos_ids=pos_ids,
            mask=self.visual_mask(patch_mask),
        )   # 自注意力计算

        self_attn_output = residual + self_attn_output
        # 残差连接

        # ===== 前馈阶段 =====
        norm_output = self.ffn_norm(self_attn_output)
        final_output = self_attn_output + self.ffn(norm_output)
        # 残差连接

        return final_output


def generate_init_weights(module: nn.Module):
    """初始化模型权重"""
    if isinstance(module, nn.Linear):   # 线性层
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:   # 线性层偏置
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):   # 嵌入层
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.padding_idx is not None:
            nn.init.zeros_(module.weight[module.padding_idx])

    elif isinstance(module, (nn.LayerNorm, RMS_norm)):   # 层归一化
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.ones_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)


class Tower2_Model(nn.Module):
    """Tower2"""
    def __init__(
        self,
        vocab_size: int,
        dk: int,
        head_num: int,
        share_num: int,
        exp_num: int,
        top_k: int,
        decoder_num: int,
        vit_num: int,
        pad_idx: int,
        img_size: int,
        patch_size: int,
        in_chans: int,
        max_len: int,
        use_cache: bool,
        cache_type: str | None,
        device: str,
        use_dropout: bool,
        init_weights: bool,
        ffn_type: str,
        train: bool,
    ):
        """
        总模型实现

        参数:
        - vocab_size: 词汇表大小
        - dk: 每个头的维度
        - head_num: 注意力头数
        - share_num: 共享专家数
        - exp_num: 专家数量
        - top_k: 激活专家数
        - decoder_num: 解码器数量
        - vit_num: vision encoder模型数量
        - pad_idx: 填充索引
        - img_size: 图像大小 (size * size)
        - patch_size: 分割大小 (正方形)
        - in_chans: 输入通道数
        - max_len: 最大序列长度
        - use_cache: 是否使用缓存
        - cache_type: 缓存类型 (legacy / absorb)
        - device: 计算设备
        - use_dropout: 是否使用dropout
        - init_weights: 是否初始化模型
        - ffn_type: 前馈网络类型 (ffn / moe)
        - train: 是否处于训练模式 (独立参数, 并非 torch.module.train)
        """

        super().__init__()
        self.device = device
        # 初始化设备类型

        self.training = train
        # 是否处于训练模式

        d = dk * head_num
        # 计算输入维度

        self.pad_mask = Padding_Mask(pad_idx).to(device)
        self.casual_mask = Causal_Mask().to(device)
        # 填充索引掩码 / 因果掩码

        self.embed = nn.Embedding(vocab_size, d, padding_idx=pad_idx).to(device)
        # 词表映射

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=d
        ).to(device)
        # 预处理 & 图像分割

        self.vit_num = vit_num
        if vit_num > 0:
            self.visual_encoders = nn.ModuleList([
                VisionEncoder(
                    head_num=head_num,
                    d=d,
                    dk=dk,
                    use_dropout=use_dropout,
                ) for _ in range(vit_num)
            ])   # 视觉编码器

        self.decoders = nn.ModuleList([
            Decoder(
                head_num=head_num,
                share_num=share_num,
                exp_num=exp_num,
                top_k=top_k,
                d=d,
                dk=dk,
                max_len=max_len,
                use_cache=use_cache,
                cache_type=cache_type,
                use_dropout=use_dropout,
                ffn_type=ffn_type
            ) for _ in range(decoder_num)
        ])   # 解码器

        self.final_norm = RMS_norm(d).to(self.device)
        # 输出归一化, 有利于稳定输出分布

        self.last_linear = nn.Linear(d, vocab_size, bias=False).to(self.device)
        # 输出线性层, 将解码器的输出映射到词汇表的大小

        self.last_linear.weight = self.embed.weight
        # 嵌入层与输出线性层共享权重

        if init_weights:   # 初始化权重
            self.apply(generate_init_weights)

    def forward(self, text_inputs: Tensor, imgs: Optional[Tensor]=None, base_len: int=0) -> Tensor:
        """
        前向传播
        - text_inputs: text输入序列 [batch, seq_len]
        - imgs: 图像输入序列 [batch, in_chans, img_size, img_size]
        - base_len: 基础长度 (用于缓存位置, 默认为0)
        """
        embed_output = self.embed(text_inputs)
        # 嵌入层

        if self.training:   # 训练阶段
            padding_mask = self.pad_mask(text_inputs)
            # 填充掩码

            causal_mask = self.casual_mask(text_inputs)
            # 因果掩码

            text_mask = padding_mask + causal_mask
            # 组合掩码

        else:   # 推理阶段
            text_mask = None

        dec_input = embed_output

        if imgs is not None and self.vit_num > 0:
            img_emb, patch_mask = self.patch_embed(imgs)
            # 图像嵌入

            residual_base = img_emb.clone()
            for layer_idx, encoder in enumerate(self.visual_encoders):   # 编码器计算
                img_emb: Tensor = encoder(img_emb, patch_mask)

                if (layer_idx + 1) % 4 == 0:   # 每隔4层格外做一次残差, 参见 llama 3
                    img_emb = img_emb + residual_base
                    residual_base = img_emb.clone()   # 更新残差基准

            img_output = img_emb
            # 编码器输出

        else:
            img_output = None
            # 没有图像输入时

        for decoder in self.decoders:   # 解码器计算
            dec_input, _ = decoder((dec_input, text_mask), img_output, base_len)

        decoder_output = dec_input
        # 解码器输出

        outputs = self.final_norm(decoder_output)
        outputs = self.last_linear(outputs)
        # 输出线性层

        return outputs


if __name__ == '__main__':

    inputs = torch.randint(
        low=0,
        high=10000,
        size=(1, 64),
        dtype=torch.long
    ).to('cuda')

    imgs = torch.randint(
         low=0,
         high=255,
         size=(1, 3, 1024, 1024),
         dtype=torch.long
    ).to('cuda')

    tower = Tower2_Model(
        vocab_size=10000,
        dk=64,
        head_num=6,
        share_num=4,
        exp_num=8,
        top_k=2,
        decoder_num=6,
        vit_num=6,
        pad_idx=0,
        img_size=1024,
        patch_size=28,
        in_chans=3,
        max_len=256,
        use_cache=True,
        cache_type='absorb',# / legacy
        device='cuda',
        use_dropout=False,
        init_weights=True,
        ffn_type='moe',
        train=True,
    ).to('cuda')

    # tower = torch.compile(tower)  # Compile the model for optimization
    output: Tensor = tower(inputs, imgs)
    print(output.shape)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())

    print("参数量:", count_parameters(tower))

    # writer = SummaryWriter('tr_logs/tower')  # 日志目录
    # writer.add_graph(tower, (inputs, imgs))
    # writer.close()
