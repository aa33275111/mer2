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
| `v1_sft_mcplus_attention` | SFT | MC+ 全量 | AffectGPT + LoRA-SFT,audio+video+face+text,attention 融合 | human 10% | 待跑 | 待跑 | 待跑 | 简历历史值参照:61.3% |
| `v2_grpo_human_ewf1` | GRPO | human 90% | 在 v1 上 GRPO,EW-F1 奖励 | human 10% | 待跑 | 待跑 | 待跑 | 简历历史值参照:62.1% |
| `v2_grpo_human_pr` | GRPO | human 90% | 在 v1 上 GRPO,Precision/Recall 约束奖励 | human 10% | 待跑 | 待跑 | 待跑 | 与 ewf1 版对比 |

---

## 4. 各版本详情

### v1_sft_mcplus_attention — SFT 基线(MC+)

- **状态**:待跑
- **方法**:AffectGPT 多模态大模型,融合 HuBERT 音频 / CLIP 视频 / OpenFace 人脸 / 字幕文本,Q-Former 对齐到 Qwen2.5-7B,LoRA-SFT 用 MERCaption+ 全量指令微调
- **子目录**:[`v1_sft_mcplus_attention/`](v1_sft_mcplus_attention/)
- **复现命令**:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python -u train.py \
    --cfg-path=train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz.yaml
  ```
- **复现目标**:测试集 EW-F1 (level1) ≈ **61.3%**(简历历史值,作为对勾)

### v2_grpo_human_ewf1 — GRPO(EW-F1 奖励)

- **状态**:待跑(依赖 v1)
- **方法**:在 v1 SFT 基础上,用 human 90% 数据做 GRPO 强化学习,奖励函数 = 官方 EW-F1
- **子目录**:待创建
- **复现目标**:测试集 EW-F1 (level1) ≈ **62.1%**(简历历史值)

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

# 2) 训练 SFT (MC+ 全量)
CUDA_VISIBLE_DEVICES=0 python -u train.py \
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
