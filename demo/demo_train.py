import torch
from tower2 import *
import os
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import sentencepiece as spm
from galore_torch.adamw8bit import AdamW8bit

class DemoDataset(Dataset):
    def __init__(self, data_dir, metadata_file, sp_model_path, img_size=1024, max_seq_len=256):
        super().__init__()
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len

        # 加载元数据
        with open(os.path.join(data_dir, metadata_file), 'r',encoding='utf-8') as f:
            self.metadata = json.load(f)

        # 初始化SentencePiece处理器
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)   # type: ignore

        # 定义特殊标记
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.pad_id = self.sp.pad_id()
        
        # 图像预处理管道
        self.img_transform = T.Compose([
            T.Lambda(lambda x: x.convert("RGB")),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.metadata)

    def _process_text(self, question, answer):
        """处理问答文本，格式: <bos>问题<sep>答案<eos>"""
        # 编码文本
        question_ids = self.sp.encode(question, add_bos=False, add_eos=False)   # type: ignore
        answer_ids = self.sp.encode(answer, add_bos=False, add_eos=False)   # type: ignore

        # 添加特殊标记并拼接
        input_ids = [self.bos_id] + question_ids + [self.sp.unk_id()] + answer_ids + [self.eos_id]

        # 截断序列
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len-1] + [self.eos_id]

        # 创建注意力掩码和标签
        seq_len = len(input_ids)
        labels = [-100] * (len(question_ids)+2) + answer_ids + [self.eos_id]  # 仅计算答案部分的损失

        # 填充序列
        pad_len = self.max_seq_len - seq_len
        input_ids += [self.pad_id] * pad_len
        labels += [-100] * pad_len  # 忽略填充部分的损失

        return {
            "input_ids": torch.LongTensor(input_ids),
            "labels": torch.LongTensor(labels)
        }

    def __getitem__(self, idx):
        item = self.metadata[idx]

        # 处理图像
        img_path = os.path.join(self.data_dir, item["image"])
        img = Image.open(img_path)
        img_tensor = self.img_transform(img)  # [3, 512, 512]

        # 处理文本
        text_data = self._process_text(item["ask"], item["answer"])

        return {
            "image": img_tensor,
            "input_ids": text_data["input_ids"],
            "labels": text_data["labels"]
        }

def collate_fn(batch):
    """自定义批处理"""
    return {
        "images": torch.stack([item["image"] for item in batch]),
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch])
    }

# 训练示例 ---------------------------------------------------------------
def DemoTrain():
    # 配置参数
    data_dir = "demo"
    metadata_file = "test.json"
    sp_model_path = "tokenizer/tower_dict_v1.0_32768.model"
    batch_size = 1
    
    # 初始化数据集和数据加载器
    dataset = DemoDataset(data_dir, metadata_file, sp_model_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    # 初始化模型
    model = Tower2_Model(   # type: ignore
        vocab_size=dataset.sp.vocab_size(),  # 使用SentencePiece的词汇表大小
        dk=64,
        head_num=6,
        share_num=4,
        exp_num=8,
        top_k=4,
        decoder_num=6,
        vit_num=6,
        pad_idx=dataset.pad_id,  # 使用SentencePiece的pad_id
        img_size=512,
        patch_size=28,
        in_chans=3,
        max_len=256,
        use_cache=False,
        device="cuda",
        use_dropout=False,
        init_weights=True,
        ffn_type='moe'
    ).cuda()
    
    # 训练配置
    optimizer = AdamW8bit(model.parameters(), lr=1e-4)
    
    for epoch in range(150):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            # 数据迁移到GPU
            images = batch["images"].cuda()
            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()
            
            # 前向传播
            outputs = model(
                text_inputs=input_ids,
                imgs=images
            )  # [batch, seq_len, vocab_size]
            
            # 计算损失 (仅计算答案部分)
            shift_logits = outputs[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()

        
        print(f"Epoch {epoch+1} | Avg Loss: {total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "tower2_demo.pth")

if __name__ == "__main__":
    DemoTrain()