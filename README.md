# 10-Day Multimodal Sprint (4090D + conda)

## 目标
- Demo A：Qwen2-VL 图像问答（可选 LoRA/QLoRA 微调）
- Demo B：文档/图表问答（OCR + RAG + 引用回答）
- 训练：迷你 CLIP（Flickr30k/COCO 子集），输出检索 Demo + 消融报告

## 环境
- GPU：4090D（24GB）
- Python 3.10, conda
- PyTorch CUDA 12.1

## 快速开始
1) 创建环境  
   ```bash
   conda env create -f environment.yml
   conda activate mmlm-10days
   bash scripts/setup.sh
   ```

2) 跑通通用图像问答 Demo（Qwen2-VL）  
   ```bash
   python scripts/run_vlm_demo.py
   # 浏览器打开 http://127.0.0.1:7860
   ```

3) 跑通图文检索（零样例 + 训练迷你 CLIP）  
   ```bash
   # 零样例检索
   python scripts/zero_shot_retrieval.py --data flickr30k --root ./data
   # 训练迷你 CLIP
   python scripts/train_mini_clip.py --data flickr30k --root ./data --epochs 3 --batch 128
   ```

4) 文档/图表问答（OCR + RAG + VLM）  
   ```bash
   python scripts/docqa_demo.py
   # 浏览器打开 http://127.0.0.1:7861
   ```

5) 可选：LLaVA 轻量指令微调（LoRA/QLoRA）  
   ```bash
   bash scripts/finetune_llava_lora.sh
   ```

## 每天建议与里程碑
- Day 1: 跑通 Qwen2-VL Demo + 零样例检索，准备数据
- Day 2-3: 迷你 CLIP 训练 + Recall 提升 + 可视化（t-SNE）
- Day 4-5: LoRA/QLoRA 微调（小样本 SFT）+ 手工评测
- Day 6: 文档/图表问答（OCR+RAG+VLM），实现引用回答
- Day 7: FastAPI/Gradio 服务化 + 小集评测脚本（EM/F1/Recall@K）
- Day 8: 推理优化（4-bit、KV Cache、批处理），加缓存
- Day 9: README/报告/动图 Demo、错误分析
- Day 10: 打磨与复现脚本，准备作品集

## 数据与参考
- Flickr30k: https://huggingface.co/datasets/flickr30k
- COCO Captions: https://huggingface.co/datasets/coco_captions
- DocVQA: https://huggingface.co/datasets/docvqa
- ChartQA: https://huggingface.co/datasets/ChartQA
- LLaVA Instruct: https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K
- 模型：Qwen/Qwen2-VL-7B-Instruct, laion/CLIP-ViT-B-16 (open-clip-torch)

## 算力与超参建议（4090D 24GB）
- Qwen2-VL 推理：bf16 或 4-bit 量化；max_new_tokens 128-256
- LoRA：rank 16-32，bf16，分辨率 336/448，梯度累积凑全局 batch 64-128
- 迷你 CLIP：224 分辨率，batch 128（累积到 512），LR 1e-4，温度可学习

## 注意
- 首次加载模型会较慢；建议提前预热
- 如显存紧张：降低分辨率、开启 load_in_4bit、减少 max_length、使用梯度检查点