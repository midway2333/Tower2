import os
import json
import torch
from torchvision import transforms
from PIL import Image
import sentencepiece as spm
import argparse
from tower2 import *

class MultimodalInferencer:
    def __init__(self, config):
        # 初始化配置
        self.device = torch.device('cuda')#torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(config['sp_model_path'])   # type: ignore
        self.img_size = config['img_size']
        self.max_seq_len = config['max_seq_len']
        
        # 初始化模型
        self.model = self._init_model(config['model_config'])
        self.model.load_state_dict(torch.load(config['model_path'], map_location=self.device, weights_only=True))
        self.model.eval()
        
        # 图像预处理
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def _init_model(self, model_config):
        # 模型结构与训练代码保持一致
        return Tower2_Model(   # type: ignore
            vocab_size=model_config['vocab_size'],
            dk=model_config['dk'],
            head_num=model_config['head_num'],
            share_num=model_config['share_num'],
            exp_num=model_config['exp_num'],
            top_k=model_config['top_k'],
            coder_num=model_config['coder_num'],
            pad_idx=self.sp_model.pad_id(),
            img_size=model_config['img_size'],
            patch_size=model_config['patch_size'],
            in_chans=model_config['in_chans'],
            device=self.device.type,
            max_len=20000,
            max_cache=5,
            use_dropout=False,
            init_weights=False,
            ffn_type=model_config['ffn_type']
        ).to(self.device)

    def preprocess_text(self, question):
        # 编码问题并添加控制标记
        question_ids = self.sp_model.encode(question, add_bos=False, add_eos=False)   # type: ignore
        input_ids = [self.sp_model.bos_id()] + question_ids + [self.sp_model.unk_id()]
        
        # 填充/截断
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
        else:
            input_ids += [self.sp_model.pad_id()] * (self.max_seq_len - len(input_ids))
        
        return torch.LongTensor([input_ids]).to(self.device)

    def preprocess_image(self, image_path):
        img = Image.open(image_path).convert("RGB")
        return self.img_transform(img).unsqueeze(0).to(self.device)  # [1, 3, H, W]   # type: ignore

    def generate(self, image_path, question, max_new_tokens=50):
        # 预处理输入
        img_tensor = self.preprocess_image(image_path)
        text_tensor = self.preprocess_text(question)

        # 生成回答
        generated_ids = []
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(text_tensor, img_tensor)
                
                # 获取最后一个token的logits
                next_token_logits = outputs[0, -1, :]
                
                # 采样
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # 终止条件
                if next_token == self.sp_model.eos_id():
                    break
                
                generated_ids.append(next_token.item())
                text_tensor = torch.cat([
                    text_tensor, 
                    next_token.unsqueeze(1)
                ], dim=1)
        
        # 解码文本
        answer = self.sp_model.decode(generated_ids)   # type: ignore
        return self._clean_answer(answer)

    def _clean_answer(self, text):
        # 移除特殊标记和多余空格
        return text.replace('<unk>', '').replace('<s>', '').replace('</s>', '').strip()

if __name__ == "__main__":
    # 配置参数（需与训练时一致）
    config = {
        "model_path": "tower2_demo.pth",
        "model_config": {
            "vocab_size": 32768,
            "dk": 64,
            "head_num": 6,
            "share_num": 4,
            "exp_num": 8,
            "top_k": 4,
            "coder_num": 6,
            "img_size": 512,
            "patch_size": 28,
            "in_chans": 3,
            "ffn_type": "moe"
        },
        "sp_model_path": "tower2\\tokenizer\\tower_dict_v1.0_32768.model",
        "img_size": 512,
        "max_seq_len": 256
    }

    # 初始化推理器
    inferencer = MultimodalInferencer(config)

    # 命令行接口
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--question", type=str, required=True, help="问题文本")
    args = parser.parse_args()

    # 执行推理
    answer = inferencer.generate(
        image_path=args.image,
        question=args.question,
        max_new_tokens=100,
    )

    # python tower2\demo_use.py --image "tower2\demo\images\img1.jpg" --question "请简单描述这张照片的具体内容"
    # 输出结果
    print("\n=== 推理结果 ===")
    print(f"图像路径: {args.image}")
    print(f"问题: {args.question}")
    print(f"回答: {answer}")
    print("=================\n")