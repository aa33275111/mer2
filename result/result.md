# MER2026 Track2 — 实验结果记录

> 本文件是 MER2026 Track2 开放词汇多模态情感识别项目**重跑实验的唯一结果登记入口**。
> 每跑完一个版本,必须:① 在下方「结果总览」登记一行;② 在本目录建 `v{xxx}/` 子文件夹,放该版的方法说明、日志、评测输出与可视化。
> 评测协议(数据划分 / 流程 / 指标 / 可视化)见 [`my_affectgpt/evaluation/evaluation.md`](../my_affectgpt/evaluation/evaluation.md)。

---

## 1. 版本命名规范

命名格式(文件夹名 + 结果表「版本」列一致):

```
v{序号}_{阶段}_{数据}_{变体}
```

| 字段 | 取值 | 含义 |
|---|---|---|
| `v{序号}` | v1, v2, ... | 主版本号,同一方法链递增 |
| `{阶段}` | `baseline` / `sft` / `grpo` | 训练阶段:zero-shot 基线 / LoRA-SFT / GRPO 强化学习 |
| `{数据}` | `mcplus` / `human` | 训练数据:MERCaption+ 全量 / human 人工标注 |
| `{变体}` | `attention` / `qformer` / `ewf1` / `pr` ... | 关键方法变体:融合方式、GRPO 奖励函数等,可多段 `_` 连接 |

示例:

| 版本 | 说明 |
|---|---|
| `v1_sft_mcplus_attention` | SFT,MC+ 全量训练,多模态 attention 融合 |
| `v2_grpo_human_ewf1` | GRPO,human 90% 训练,EW-F1 奖励 |
| `v2_grpo_human_pr` | GRPO,human 90% 训练,Precision/Recall 约束奖励 |

**规则:**
- 同一次训练从头到尾算一个版本;换数据、换融合方式、换奖励函数即开新版本。
- 同一版本内部超参微调,用 `method.md` 里的「子实验」记录,不改版本号。
- 版本号按创建顺序递增,删除的版本号不重用。

---

## 2. 数据与评测协议摘要

| 项目 | 规定 |
|---|---|
| **SFT 训练数据** | `track2_train_mercaptionplus_dedup.csv` (31,276 条,已剔除与测试集重叠的 51 条) |
| **GRPO 训练数据** | `track2_train_human.csv` 的 90% (≈1,379 条) |
| **测试集(统一)** | `track2_train_human.csv` 按 9:1 固定 seed 划出的 10% (≈153 条),SFT / GRPO 统一用它评测 |
| **划分方式** | `result/scripts/split_human_data.py`(9:1,seed=42)+ `result/scripts/dedup_mcplus.py`(防泄漏) |
| **输入模态** | 音频(HuBERT)+ 人脸(OpenFace),**无文本/字幕** |
| **主指标** | **EW-F1 (level1)**:5 个情感轮 level1 F1 的平均 |
| **辅助指标** | EW-F1 (level2)、Precision / Recall |

详见 [`evaluation.md`](../my_affectgpt/evaluation/evaluation.md)。

---

## 3. 结果总览

> EW-F1 单位为 %。`待跑` = 尚未开始;`进行中` = 训练/评测未完成。

| 版本 | 阶段 | 训练数据 | 方法要点 | 测试集 | **EW-F1 (level1)** | EW-F1 (level2) | P / R | 备注 |
|---|---|---|---|---|---|---|---|---|
| `v1_sft_mcplus_attention` | SFT | mc+ dedup (31,276) | AffectGPT + LoRA-SFT,音频+人脸(无文本),attention 融合 | human 10% | 待跑 | 待跑 | 待跑 | 对应历史 SFT 配置 → 61.3% |
| `v2_grpo_human_ewf1` | GRPO | human 90% | 在 v1 上 GRPO,EW_F1 奖励 | human 10% | 待跑 | 待跑 | 待跑 | 对应历史 v7 配置 → 62.1% |
| `v2_grpo_human_p3` | GRPO | human 90% | EW_F1 × P³ 奖励(惩罚低精度/过度预测) | human 10% | 待跑 | 待跑 | 待跑 | 对应历史 P3 配置 → 61.8% |

### 3.1 历史实验参照(原代码删前记录,复现目标)

> 用户 2026-08-25 提供的原始实验表。重跑以 **SFT 与 v7 GRPO 为第一版**复现目标,其余作为奖励策略对比实验。

| 实验 | Reward | G | Temp | LR | Steps | Best EW-F1 |
|---|---|---|---|---|---|---|
| **SFT** | — | — | — | 1e-5 | — | **61.3** |
| **v7** | **EW_F1** | 8 | **1.0** | **5e-7** | 500 | **62.1** |
| EW+format | EW_F1 + 0.1·format | 8 | 0.7–1.1 | 8e-7 | 600 | 61.0 |
| μ_F1 | 0.85·EW + 0.15·μ_F1 | 8 | 0.8–1.25 | — | — | 60.6 |
| **P3** | **EW_F1 × P³** | 8 | **1.0** | **1e-6** | 1500 | **61.8** |

奖励策略分析(面试可讲):v7(纯 EW_F1)最高 62.1;EW+format 反而降(0.1·format 干扰);μ_F1 略降(0.85/0.15 权重把 P/R 往中间拉);P3 用 P³ 强惩罚低精度(抑制过度预测),61.8 但更稳。→ 反映 GRPO 中 reward 设计对"预测偏置与 P/R 平衡"的敏感度。

---

## 4. 各版本详情

### v1_sft_mcplus_attention — SFT 基线(MC+)

- **状态**:待跑
- **方法**:AffectGPT 多模态大模型,融合 HuBERT 音频 / OpenFace 人脸(**无文本**),Q-Former 对齐到 Qwen2.5-7B,LoRA-SFT 用 mc+ 去重后全量指令微调
- **子目录**:[`v1_sft_mcplus_attention/`](v1_sft_mcplus_attention/)
- **复现命令**:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python -u train.py \
    --cfg-path=train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz.yaml
  ```
- **复现目标**:测试集 EW-F1 (level1) ≈ **61.3%**(历史 SFT 配置:init_lr 1e-5,cosine)

### v2_grpo_human_ewf1 — GRPO(EW_F1 奖励,历史 v7 配置)

- **状态**:待跑(依赖 v1)
- **方法**:在 v1 SFT 基础上,用 human 90% 数据 GRPO,奖励函数 = 官方 **EW_F1**
- **超参(对应历史 v7)**:G=8、temperature=1.0、lr=5e-7、eps=0.1、grad_accum=4、500 steps
- **入口**:`grpo/train_grpo.py --cfg-path=train_configs/grpo_human_ewf1.yaml --reward ewf1`
- **子目录**:`v2_grpo_human_ewf1/`(已建)
- **复现目标**:测试集 EW-F1 (level1) ≈ **62.1%**

### v2_grpo_human_p3 — GRPO(EW_F1 × P³ 奖励,历史 P3 配置)

- **状态**:待跑(依赖 v1)
- **方法**:奖励 = EW_F1 × Precision³,强惩罚低精度(抑制过度预测)
- **超参(对应历史 P3)**:G=8、temperature=1.0、lr=1e-6、1500 steps
- **入口**:`grpo/train_grpo.py --cfg-path=train_configs/grpo_human_ewf1.yaml --reward p3 --lr 1e-6 --steps 1500`
- **复现目标**:测试集 EW-F1 (level1) ≈ **61.8%**

### v2_grpo_human_pr — GRPO(Precision/Recall 约束奖励)

- **状态**:待跑(依赖 v1)
- **方法**:同 v2_ewf1,但奖励函数改为 Precision/Recall 约束,用于对比分析 GRPO 过程中的预测偏置与 P/R 平衡问题
- **子目录**:待创建

---

## 5. 复现命令速查

```bash
cd /root/MER2026_Track2

# 0) 配置数据/模型路径
#    编辑 config.py: DATA_DIR['MER2026']、PATH_TO_LLM 等由 'xxx' 改为实际路径

# 1) 9:1 划分 human 数据 (只需一次)
python result/scripts/split_human_data.py

# 2) 训练 SFT (MC+ 去重后全量; 4 卡 A100, 与历史配置一致)
torchrun --nproc_per_node=4 --master_port=29500 train.py \
  --cfg-path=train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz.yaml

# 3) 推理 (生成 reason, 保存到 output/results-mer2026ov/)
CUDA_VISIBLE_DEVICES=0 python -u inference_hybird.py --zeroshot \
  --dataset='MER2026OV' \
  --cfg-path=train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz.yaml \
  --options "inference.test_epochs=10-60" "inference.skip_epoch=5"

# 4) reason -> OV 标签 (Qwen 批量抽取)
python ovlabel_extraction.py

# 5) EW-F1 评测
python evaluation.py
```

---

## 6. 变更记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-08-25 | — | 初始化 result 仓库,建立命名规范与评测协议(9:1 划分 / 统一测试集 / EW-F1 指标) |
