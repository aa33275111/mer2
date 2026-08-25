# v1_sft_mcplus_attention — SFT 基线(MC+)

> **状态:待跑**。对应历史实验表 **SFT 行**(复现目标 EW-F1 **61.3%**,P=58.1 R=65.1)。本文是方法说明书,跑通后把结果填到 [`../result.md`](../result.md),日志/评测/图放 `logs/` `eval/` `vis/`。

## 1. 方法概述

AffectGPT 多模态大语言模型:以 **Qwen2.5-7B-Instruct** 为 LLM,
融合 **HuBERT** 音频编码、**OpenFace** 人脸特征(**不使用文本/字幕模态**),
通过 Q-Former 将多模态特征对齐到 LLM 语义空间,采用 **LoRA-SFT** 用 MERCaption+ 去重后全量指令微调。

## 2. 模型结构

| 组件 | 配置 |
|---|---|
| LLM | Qwen2.5-7B-Instruct (`PATH_TO_LLM['Qwen25']`) |
| 音频编码器 | HuBERT-large 中文 (`PATH_TO_AUDIO['HUBERT_LARGE']`),冻结 |
| 视觉编码器 | CLIP-ViT-Large (`PATH_TO_VISUAL['CLIP_VIT_LARGE']`),编码人脸帧 |
| 人脸特征 | OpenFace 逐帧 npy |
| 特征融合 | `multi_fusion_type: attention`(audio+face,无文本) |
| 微调 | LoRA-SFT(**trainable params ≈ 40,370,176**,~40M) |

## 3. 训练配置(历史已记录配置,即本版复现配置)

| 超参 | 值 |
|---|---|
| 训练数据 | mc+ 去重后 **31,276** 条(历史 31,327,重跑剔除 51 条泄漏样本后略少) |
| 模态 | **Audio + Face**(无 subtitle/文本) |
| max_epoch / iters_per_epoch | 60 / 5000 |
| init_lr / min_lr | **1e-5 / 1e-5**(cosine decay 实际未下降,与历史一致) |
| batch_size_train | 3 × gpu_num(4 卡 = 12,与历史一致) |
| weight_decay | 0.05 |
| amp | FP16 混合精度 |
| seed | 42 |

## 4. 复现命令

```bash
cd /root/MER2026_Track2

# 训练 (4 卡 A100: torchrun 自动设 RANK/WORLD_SIZE, yaml world_size 会被覆盖)
torchrun --nproc_per_node=4 --master_port=29500 train.py \
  --cfg-path=train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz.yaml

# 推理 (生成 reason)
CUDA_VISIBLE_DEVICES=0 python -u inference_hybird.py --zeroshot \
  --dataset='MER2026OV' \
  --cfg-path=train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz.yaml \
  --options "inference.test_epochs=10-60" "inference.skip_epoch=5"

# OV 标签抽取 + EW-F1 评测
CUDA_VISIBLE_DEVICES=0 python ovlabel_extraction.py
CUDA_VISIBLE_DEVICES=0 python evaluation.py
```

## 5. 产物落盘

| 产物 | 路径 |
|---|---|
| ckpt(不复制进 result) | `output/{cfg_name}/{job_id}/...` |
| 训练日志 | `logs/`(或软链 output 下日志) |
| reason | `output/results-mer2026ov/.../{epoch}.npz` |
| OV 标签 | `output/results-mer2026ov/.../{epoch}-openset.npz` |
| 可视化 | `vis/` |

## 6. 结果(跑完后填写)

- 最佳 epoch:`____`
- **EW-F1 (level1)= `____`%**(复现目标 ≈ 61.3%)
- EW-F1 (level2)= `____`% | P = `____`% | R = `____`%
- 训练 loss 收敛于:`____`
- 备注:`____`
