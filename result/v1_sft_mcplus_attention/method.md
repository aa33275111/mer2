# v1_sft_mcplus_attention — SFT 基线(MC+ 全量)

> **状态:待跑**。本文是方法说明书,跑通后把实际结果填到 [`../result.md`](../result.md) 总览表,日志/评测/图放到 `logs/` `eval/` `vis/`。

## 1. 方法概述

AffectGPT 多模态大语言模型:以 **Qwen2.5-7B-Instruct** 为 LLM,
融合 **HuBERT** 音频编码、**CLIP** 视频编码、**OpenFace** 人脸特征、字幕文本,
通过 Q-Former 将多模态特征对齐到 LLM 语义空间,采用 **LoRA-SFT** 用 MERCaption+ 全量指令微调。

## 2. 模型结构

| 组件 | 配置 |
|---|---|
| LLM | Qwen2.5-7B-Instruct (`PATH_TO_LLM['Qwen25']`) |
| 音频编码器 | HuBERT-large 中文 (`PATH_TO_AUDIO['HUBERT_LARGE']`) |
| 视频编码器 | CLIP-ViT-Large (`PATH_TO_VISUAL['CLIP_VIT_LARGE']`) |
| 人脸特征 | OpenFace 逐帧 npy |
| 特征融合 | `multi_fusion_type: attention`(audio+video+face+text) |
| 微调 | LoRA-SFT(所有 projection / Q-Former / LoRA 可训) |

## 3. 训练配置(等价于 `train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz.yaml`)

| 超参 | 值 |
|---|---|
| 训练数据 | MERCaption+ 全量 31,327 条 |
| max_epoch / iters_per_epoch | 60 / 5000 |
| init_lr / min_lr / warmup_lr | 1e-5 / 1e-5 / 1e-6,linear_warmup_cosine |
| batch_size_train | 3 × gpu_num |
| weight_decay | 0.05 |
| amp | FP16 混合精度 |
| seed | 42 |

## 4. 复现命令

```bash
cd /root/MER2026_Track2

# 训练
CUDA_VISIBLE_DEVICES=0 python -u train.py \
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
