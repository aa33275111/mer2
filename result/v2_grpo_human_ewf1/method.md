# v2_grpo_human_ewf1 — GRPO(EW_F1 奖励,历史 v7 配置)

> **状态:待跑**(依赖 v1 SFT checkpoint)。对应历史实验表 **v7 行**(复现目标 EW-F1 **62.1%**)。

## 1. 方法概述

在 v1 SFT 基础上,用 **human 90%**(1,379 条)数据做 **GRPO** 强化学习。
奖励函数 = **纯 EW_F1**(官方 5 情感轮 level1 F1 平均)。只跑第一版,不引入其它 reward。

## 2. 超参(历史 v7 记录配置,本版默认)

| 超参 | 值 |
|---|---|
| Reward | **EW_F1**(纯) |
| G(组大小) | **8** |
| Temperature | **1.0 固定** |
| LR | **5e-7** |
| epsilon(裁剪) | **0.1** |
| grad_accum | **4** |
| max_steps | **500** |
| KL 系数 beta | 0.04(可调) |
| 训练数据 | human train90(1,379 条) |
| 起点 | v1 SFT checkpoint |
| SFT baseline | 61.3 |

## 3. 算法要点

- 每个 prompt 采样 **G=8** 条响应 → 组内奖励归一化得到 advantage
- 策略梯度:per-token importance ratio = exp(logp_policy − logp_ref),带 **clip(ε=0.1)**
- **KL 惩罚**:对冻结的参考模型(SFT 权重)逐 token 算 KL,β=0.04
- 只更新 **LoRA 参数**(~40M)

## 4. 复现命令

```bash
cd /root/MER2026_Track2
CUDA_VISIBLE_DEVICES=0 python -u grpo/train_grpo.py \
  --cfg-path train_configs/grpo_human_ewf1.yaml \
  --sft-ckpt output/{cfg_name}/{job_id}/checkpoint_006000_loss_0.909.pth \
  --reward ewf1 \
  --G 8 --temp 1.0 --lr 5e-7 --eps 0.1 --grad-accum 4 --max-steps 500
```

## 5. 产物落盘

| 产物 | 路径 |
|---|---|
| GRPO ckpt | `output/grpo/grpo_step{step:06d}_ewf1_{ewf1:.4f}.pth` |
| 训练日志 | `logs/` |
| 验证 EW-F1 曲线 | `vis/ewf1_step_curve.png` |
| 最终评测 | 按 `evaluation.md` 流程,用 v1/v2 对比登记 |

## 6. 结果(跑完后填写)

- 最佳验证 EW-F1 (level1)= `____`%(复现目标 **62.1%**)
- 训练 loss / KL 收敛于:`____`
- 相对 v1 的提升:`____`
- 备注:`____`
