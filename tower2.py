import torch
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
        - seq_len: 序列长度
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


class MLAframe(nn.Module):
    """MLA前置框架"""
    def __init__(self, d, d_pre_head, head_num,
        use_dropout: bool=False, device: Optional[str]=None):
        """
        MLA前置框架
        用于MLA初始化

        参数:
        - d: 输入/输出的维度
        - dk_pre_head: 每个头的隐藏层维度 (非RoPE维度)
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

        self.d_rope = d_pre_head // 2
        # 计算位置编码维度

        self.dc_v = self.d_pre_head
        # value维度

        self.dc_kv = 4 * self.d_pre_head   # kv Lora维度
        self.dc_q = 3 * self.dc_kv         # quary Lora维度
        # 低秩压缩的维度
        # 为什么是这个比例关系? 我也不知道, 反正ds是这么干的

        self.out_proj = nn.Linear(
            self.head_num * self.dc_v,
            self.d,
            bias=False,
        )   # 输出投影

        # ================ quary Lora ================
        self.q_head_dim = self.d_pre_head + self.d_rope
        # 每个头的quary维度

        self.q_up = nn.Linear(
            self.dc_q,
            self.head_num * self.q_head_dim,
            bias=False,
        )   # 升维矩阵

        self.q_down = nn.Linear(
            self.d,
            self.dc_q,
            bias=False,
        )   # 降维矩阵

        self.q_down_norm = RMS_norm(self.dc_q)

        # =========== key & value Lora ===========
        self.meg_d = self.d_pre_head + self.dc_v
        # 合并投影, 便于实现单次完成两者升维

        self.kv_up = nn.Linear(
            self.dc_kv,
            self.head_num * self.meg_d,
            bias=False,
        )   # 升维矩阵

        self.kv_down = nn.Linear(
            self.d,
            self.dc_kv + self.d_rope,
            bias=False,
        )   # 降维矩阵

        self.kv_down_norm = RMS_norm(self.dc_kv)

        # ============ RoPE ============
        self.rope = RoPE_Emb(
            self.d_rope,
            max_len=8192,
            device=device,
        )


class MLA(MLAframe):
    """多头潜在注意力, 通过低秩投影 (LoRA) 压缩 Q/K/V 维度"""
    def __init__(self, d, d_pre_head, head_num, max_len: int, max_cache: int,
        use_dropout: bool=False, device: Optional[str]=None):
        """
        我的想法是尽量减少亢余参数;  
        所以相比于主流实现而言自由度更小, 相应的传参更少

        参数:
        - d: 输入/输出的维度
        - dk_pre_head: 每个头的隐藏层维度 (非RoPE维度)
        - head_num: 头的数量
        - max_len: 最大序列长度
        - max_cache: 最大 KV Cache 长度
        - use_dropout: 是否使用dropout
        - device: 计算设备
        """
        super().__init__(d, d_pre_head, head_num, use_dropout, device)
        assert max_cache <= max_len, "最大序列长度要大于最大缓存长度 | max len must be greater than max cache"
        # 检查最大长度

        self.max_cache = max_cache
        # 初始化

        # ====== kv cache ======
        self.register_buffer("kv_cache", torch.zeros(1, max_cache, self.dc_kv + self.d_rope), persistent=False)
        self.register_buffer("cache_pos", torch.tensor([]), persistent=False)   # 用维度大小记录
        self.register_buffer("max_len", torch.zeros(max_len), persistent=False)
        # 初始化缓存

    def forward(self, inputs: Tensor, pos_ids, mask: Optional[Tensor]=None):   # type: ignore
        """
        - input: 输入序列 [batch, seq_len, d]
        - pos_ids: 位置索引
        - mask: 掩码
        """
        batch_size, seq_len, _ = inputs.size()
        self.kv_cache: Tensor = self.kv_cache.expand(batch_size, -1, -1).contiguous()
        kv_cache_len = self.cache_pos.size(0)   # kv 缓存长度
        new_token = inputs.size(1) - kv_cache_len
        self.max_len: Tensor

        # ===== quary 计算 =====
        q = self.q_down(inputs)
        q = self.q_down_norm(q)
        q = self.q_up(q)
        # 低秩投影

        q: Tensor = q.view(batch_size, seq_len, self.head_num, self.q_head_dim).transpose(1, 2)
        q_nope, q_rope = torch.split(q, [self.d_pre_head, self.d_rope], dim=-1)
        # 多头拆分 & 维度分割

        # ========== 处理当前步 KV 投影 ==========
        qk_inputs = inputs[:, -new_token:, :]

        current_c_kv_all = self.kv_down(qk_inputs)
        c_kv, k_rope = torch.split(
            current_c_kv_all, [self.dc_kv, self.d_rope], dim=-1
        )   # 分割维度

        c_kv = self.kv_down_norm(c_kv)

        if not Tower2_Model.train and self.max_cache > 0:
            # ======== 历史缓存处理 ========
            past_c_kv, past_k_rope = torch.split(
                self.kv_cache[:, :kv_cache_len, :], [self.dc_kv, self.d_rope], dim=-1
            )   # 拆分历史缓存

            c_kv = torch.cat([past_c_kv, c_kv], dim=1)
            k_rope = torch.cat([past_k_rope, k_rope], dim=1)
            # 拼接历史缓存与当前步 KV

            c_kv = c_kv[:, -min(c_kv.size(1), self.max_len.size(0)):, :]
            k_rope = k_rope[:, -min(k_rope.size(1), self.max_len.size(0)):, :]
            # 截断至最大长度

            # ======= 生成新缓存 =======
            self.cache_pos = torch.zeros(min(seq_len, self.max_cache), dtype=torch.long)   # 更新缓存长度
            self.kv_cache[:, :self.cache_pos.size(0), :] = torch.cat(
                [
                    c_kv[:, :self.cache_pos.size(0), :],
                    k_rope[:, :self.cache_pos.size(0), :],
                ], dim=-1
            )   # 更新缓存

            mask = None if mask is None else mask[-new_token:, :]
            # 调整掩码形状

        # ======== KV 合并处理 ========
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
        )

        # ============ attention ============
        query = torch.concat(
            [q_nope, q_rope], dim=-1
        )[:, -new_token: ,:]
        # 拼接 Query, 只保留当前步

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
        super().__init__(d, d_pre_head, head_num, use_dropout, device)

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
        kv = self.kv_down_norm(c_kv)
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


class CrossMLA(MLA):
    """交叉多头潜在注意力 支持kv cache"""
    def __init__(self, d, d_pre_head, head_num, max_len: int, max_cache: int
        , use_dropout=False, device=None):
        """
        参数:
        - d: 输入/输出的维度
        - d_pre_head: 每个头的隐藏层维度 (非RoPE维度)
        - head_num: 头的数量
        - max_len: 最大序列长度
        - max_cache: 最大 KV Cache 长度
        - use_dropout: 是否使用 dropout
        - device: 设备
        """
        super().__init__(d, d_pre_head, head_num, max_len, max_cache, use_dropout, device)
        self.text_cache_len = 0

    def forward(self, q_inputs: Tensor, kv_input: Tensor, q_pos_ids, kv_pos_ids):
        """
        - q_input: decoder 输入序列 [batch, text_seq_len, d]
        - kv_input: encoder 输入序列 [batch, encoder_seq_len, d]
        - q_pos_ids: 解码器位置ID [batch, text_seq_len]
        - kv_pos_ids: 编码器位置ID [batch, encoder_seq_len]
        """

        batch_size, text_seq_len, _ = q_inputs.size()
        _, enc_seq_len, _ = kv_input.size()
        # 获得批次与长度

        self.kv_cache: Tensor = self.kv_cache.expand(batch_size, -1, -1).contiguous()
        kv_cache_len = self.cache_pos.size(0)    # kv 缓存长度
        new_cache = enc_seq_len - kv_cache_len   # 未缓存的编码器长度
        self.max_len: Tensor
        # 这里只限制文本长度
        # 图像不管, 否则会丢失信息

        new_text = text_seq_len - min(self.text_cache_len, self.max_len.size(0))
        # 未缓存的文本长度

        # ===== quary 计算 =====
        q = self.q_down(q_inputs)
        q = self.q_down_norm(q)
        q = self.q_up(q)
        # 低秩投影

        q: Tensor = q.view(batch_size, text_seq_len, self.head_num, self.q_head_dim).transpose(1, 2)
        q_nope, q_rope = torch.split(q, [self.d_pre_head, self.d_rope], dim=-1)
        # 多头拆分 & 维度分割


        # ========== 处理当前步 KV 投影 ==========
        kv_inputs = kv_input[:, -new_cache:, :]
        # 未缓存大小

        current_c_kv_all = self.kv_down(kv_inputs)
        c_kv, k_rope = torch.split(
            current_c_kv_all, [self.dc_kv, self.d_rope], dim=-1
        )   # 分割维度

        c_kv = self.kv_down_norm(c_kv)

        if not Tower2_Model.train and self.max_cache > 0:
            # ======== 历史缓存处理 ========
            past_c_kv, past_k_rope = torch.split(
                self.kv_cache[:, :kv_cache_len, :], [self.dc_kv, self.d_rope], dim=-1
            )   # 拆分历史缓存

            c_kv = torch.cat([past_c_kv, c_kv], dim=1)
            k_rope = torch.cat([past_k_rope, k_rope], dim=1)
            # 拼接历史缓存与当前步 KV

            # ======= 生成新缓存 =======
            self.cache_pos = torch.zeros(min(enc_seq_len, self.max_cache), dtype=torch.long)   # 更新缓存长度
            self.kv_cache[:, :self.cache_pos.size(0), :] = torch.cat(
                [
                    c_kv[:, :self.cache_pos.size(0), :],
                    k_rope[:, :self.cache_pos.size(0), :],
                ], dim=-1
            )   # 更新缓存

        # ======== KV 合并处理 ========
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
        )

        # ============ attention ============
        query = torch.concat(
            [q_nope, q_rope], dim=-1
        )[:, -new_text: ,:]   # 拼接 Query, 只保留当前步
        self.text_cache_len = query.size(1)
        # 获得文本缓存长度

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
        self.Wx = nn.Linear(d, dff)
        # 映射线性层
        
        self.Vx = nn.Linear(d, dff)
        # 门控机制

        self.last_linear = nn.Linear(dff, d)
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
    def __init__(self, d, dff, expert_num, top_k):
        """
        稀疏混合专家模型

        参数:
        - d: 输入维度
        - dff: 映射维度
        - expert_num: 专家数量
        - top_k: 激活专家数
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
    def __init__(self, d, dff, share_num, expert_num, top_k):
        """
        参数:
        - d: 每个专家的输入维度
        - dff: 映射维度
        - share_num: 共享专家数量
        - expert_num: 专家数量
        """
        super().__init__()
        self.moe = SparseMOE(d, dff, expert_num, top_k)
        # 稀疏混合专家模型

        self.share_experts = nn.ModuleList([
            Expert(d, dff) for _ in range(share_num)
        ])

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

    def forward(self, x: Tensor) -> Tensor:
        """
        - x: 输入序列 [batch, seq_len, d]
        """
        batch_size, seq_len = x.size(0), x.size(1)
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return pos_ids  # [batch_size, seq_len]


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
        visual_mask = patch_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, num_patches]
        visual_mask = torch.where(visual_mask == 0, float('-inf'), 0.0)  # 无效位置设为-inf
        return visual_mask


class Decoder(nn.Module):
    """解码器"""
    def __init__(self, head_num: int, share_num: int, exp_num: int, top_k: int, d: int, dk: int,
            max_len: int, max_cache: int, use_dropout: bool=False, init_weights: bool=False, ffn_type: str='ffn'):
        """
        参数:
        - head_num: 注意力头数
        - share_num: 共享专家数
        - exp_num: 专家数量
        - top_k: 激活专家数
        - d: 输入/输出维度
        - dk: 每个头的维度
        - max_len: 最大序列长度
        - max_cache: 最大缓存长度
        - use_dropout: 是否使用dropout
        - init_weights: 是否初始化模型
        - ffn_type: 前馈网络类型 (ffn / moe)
        """
        super().__init__()
        self.self_attn = MLA(d, dk, head_num, max_len, max_cache, use_dropout)
        # 多头自注意力层

        self.cross_attn = CrossMLA(d, dk, head_num, max_len, max_cache, use_dropout)
        # 交叉多头自注意力层

        self.get_pos_ids = Get_Pos_ids()
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
            self.ffn = MOE(d, d//2, share_num, exp_num, top_k)
            # 混合专家网络
            # 因为有很多专家, dff的维度较小

    def forward(
            self, inputs_tuple: Tuple[Tensor, Tensor], enc_inputs: Optional[Tensor]=None
        ) -> Tuple:
        """
        layer_norm 使用 Pre-LN 结构

        参数:
        - inputs_tuple: 输入元组 [input, mask]
        - input: 输入序列 [batch, seq_len, model_dim]
        - mask: 目标序列的掩码 [batch, seq_len]
        - enc_inputs: 编码器输入序列 [batch, seq_len, model_dim]
        """

        inputs, mask = inputs_tuple
        # 解包元组

        # ======= 自注意力阶段 =======
        norm_input = self.self_attn_norm(inputs)

        self_attn_output = self.self_attn(
            norm_input,
            pos_ids=self.get_pos_ids(inputs),
            mask=mask,
        )   # 自注意力计算

        self_attn_output = inputs + self_attn_output
        attn_output = self_attn_output
        # 残差连接

        # ======= 交叉注意力阶段 =======
        if enc_inputs is not None:
            norm_cross = self.cross_attn_norm(self_attn_output)
            cross_attn_output = self.cross_attn(
                norm_cross, 
                enc_inputs,
                q_pos_ids=self.get_pos_ids(inputs),
                kv_pos_ids=self.get_pos_ids(enc_inputs),
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

    def forward(self, imgs: Tensor, patch_mask: Tensor) -> Tensor:
        """
        图像编码器前向传播
        - imgs: 输入图像 [batch, num_patch^2, d]
        - patch_mask: 块掩码 [batch, num_patches]
        """
        # ===== 自注意力阶段 =====
        residual = imgs
        norm_input = self.self_attn_norm(imgs)

        self_attn_output = self.self_attn(
            norm_input,
            pos_ids=self.get_pos_ids(imgs),
            mask=self.visual_mask(patch_mask),
        )   # 自注意力计算

        self_attn_output = residual + self_attn_output
        # 残差连接

        # ===== 前馈阶段 =====
        norm_output = self.ffn_norm(self_attn_output)
        final_output = self_attn_output + self.ffn(norm_output)
        # 残差连接

        return final_output


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
        max_cache: int,
        device: str,
        use_dropout: bool,
        init_weights: bool,
        ffn_type: str,
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
        - max_cache: 最大缓存长度
        - device: 计算设备
        - use_dropout: 是否使用dropout
        - init_weights: 是否初始化模型
        - ffn_type: 前馈网络类型 (ffn / moe)
        """

        super().__init__()
        self.device = device
        # 初始化设备类型

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
                max_cache=max_cache,
                use_dropout=use_dropout,
                ffn_type=ffn_type
            ) for _ in range(decoder_num)
        ])   # 解码器

        self.final_norm = RMS_norm(d).to(self.device)
        # 输出归一化, 有利于稳定输出分布

        self.last_linear = nn.Linear(d, vocab_size).to(self.device)
        # 输出线性层, 将解码器的输出映射到词汇表的大小

    def forward(self, text_inputs: Tensor, imgs: Optional[Tensor]=None) -> Tensor:
        """
        前向传播
        - text_inputs: text输入序列 [batch, seq_len]
        - imgs: 图像输入序列 [batch, in_chans, img_size, img_size]
        """
        embed_output = self.embed(text_inputs)
        # 嵌入层

        padding_mask = self.pad_mask(text_inputs)
        # 填充掩码

        causal_mask = self.casual_mask(text_inputs)
        # 因果掩码

        text_mask = padding_mask + causal_mask
        # 组合掩码

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
            dec_input, _ = decoder((dec_input, text_mask), img_output)

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
        size=(4, 128),
        dtype=torch.long
    ).to('cuda')

    imgs = torch.randint(
         low=0,
         high=255,
         size=(4, 3, 1024, 1024),
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
        max_cache=0,
        device='cuda',
        use_dropout=False,
        init_weights=False,
        ffn_type='moe'
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
