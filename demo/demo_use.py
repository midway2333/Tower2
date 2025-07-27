import os
import json
import torch
from torchvision import transforms
from PIL import Image
import sentencepiece as spm
import argparse
from tower2 import *
import torchvision.transforms as T

# 加载SentencePiece处理器
sp = spm.SentencePieceProcessor()
sp.load('tokenizer\\tower_dict_v1.0_32768.model')  # type: ignore

vocab_size = sp.GetPieceSize()
bos_id = sp.bos_id()
eos_id = sp.eos_id()
pad_id = sp.pad_id()

class demo_use():
    def __init__(self, model_path, use_cache=False, cache_type=None):
        self.model_path = model_path
        self.use_cache = use_cache
        self.cache_type = cache_type

        # 图像预处理管道
        self.img_transform = T.Compose([
            T.Lambda(lambda x: x.convert("RGB")),
            T.ToTensor(),
        ])

    def _init_model(self):
        # 构建模型并加载权重
        model = Tower2_Model(
        vocab_size=sp.vocab_size(),   # 使用SentencePiece的词汇表大小
        dk=64,
        head_num=6,
        share_num=4,
        exp_num=8,
        top_k=4,
        decoder_num=6,
        vit_num=6,
        pad_idx=pad_id,   # 使用SentencePiece的pad_id
        img_size=512,
        patch_size=28,
        in_chans=3,
        max_len=256,
        use_cache=self.use_cache,
        cache_type=self.cache_type,
        device="cuda",
        use_dropout=False,
        init_weights=True,
        ffn_type='moe',
        train=False,   # 设置为推理模式
    ).cuda()
        
        return model
        
    def load_model(self):
        self.model = self._init_model()
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

    def infer_without_cache(self, image_path):
        image = Image.open(image_path)
        img_tensor = self.img_transform(image)   # type: ignore
        img_tensor: Tensor = img_tensor.unsqueeze(0).to("cuda")

        user_input = input("\nUser: ")
        user_input_ids: list = [bos_id] + sp.encode(user_input) + [sp.unk_id()]   # type: ignore

        while True:
            with torch.no_grad():
                x = torch.tensor(user_input_ids).unsqueeze(0).to("cuda")
                logits = self.model(x, img_tensor)
                last_logits = logits[:, -1, :]
                # 获取最后一个token的概率

            probabilities = torch.nn.functional.softmax(last_logits, dim=-1).squeeze()
            predicted_index = torch.argmax(probabilities).item()

            # 终止条件检测
            if predicted_index in [sp.eos_id()]:
                print("\n", end='')
                break

            # 更新序列
            user_input_ids.append(predicted_index)

            # 解码输出
            word = sp.decode([predicted_index])   # type: ignore
            print(word, end='', flush=True)

    def infer_with_cache(self, image_path):
        image = Image.open(image_path)
        img_tensor = self.img_transform(image)   # type: ignore
        img_tensor: Tensor = img_tensor.unsqueeze(0).to("cuda")

        user_input = input("\nUser: ")
        user_input_ids = [bos_id] + sp.encode(user_input) + [sp.unk_id()]   # type: ignore

        while True:
            with torch.no_grad():
                x = torch.tensor(user_input_ids).unsqueeze(0).to("cuda")
                logits = self.model(x, img_tensor)
                last_logits = logits[:, -1, :]
                # 获取最后一个token的概率

            probabilities = torch.nn.functional.softmax(last_logits, dim=-1).squeeze()
            predicted_index = torch.argmax(probabilities).item()

            # 终止条件检测
            if predicted_index in [sp.eos_id()]:
                print("\n", end='')
                break

            # 更新序列
            user_input_ids = [predicted_index]

            # 解码输出
            word = sp.decode([predicted_index])   # type: ignore
            print(word, end='', flush=True)

# 运行示例
if __name__ == "__main__":
    demo = demo_use("tower2_demo.pth", use_cache=True, cache_type='absorb')
    demo.load_model()
    demo.infer_with_cache("demo\\images\\img1.jpg")
