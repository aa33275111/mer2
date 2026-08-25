# MER2026 Track2 — 评测规范 (Evaluation Protocol)

> 本文档**规定**本项目重跑实验的完整评测口径:用什么数据训练、用什么数据测试、用什么指标、做什么可视化。
> 所有版本实验必须严格遵循本规范,结果统一登记到 [`../../result/result.md`](../../result/result.md)。
> 各阶段的详细输入/输出规格见 [`pipeline_io.md`](pipeline_io.md)。

---

## 1. 评测目标

开放词汇多模态情感识别 (Open-Vocabulary Multimodal Emotion Recognition, MER-FG):
模型输入 **音频 + 人脸** 两种模态(**不使用文本/字幕**,`face_or_frame: multiface_audio_face`),
输出任意数量、任意类别的情感标签(开放词表),
用 **Emotion Wheel (情感轮) 的 F1** 衡量预测标签与人工标注之间的重叠程度。

---

## 2. 数据规范

### 2.1 数据源 (MER2026 数据集,存放于 `config.py` 的 `DATA_DIR['MER2026']`)

> **真实数据路径**:`/root/fsas/AffectGPT_dataset`。完整数据按分组存放在 `*_7z/{group}/` 下
> (扁平 `audio/ video/ openface_face/` 是失效符号链接,勿用)。已核验每个 csv 的样本在各分组内
> 音频/视频/人脸覆盖率 100%。

| 数据 | 文件 | 样本数 | 用途 |
|---|---|---|---|
| 人工标注 (Human-OV) | `track2_train_human.csv` | 1,532 | 划分 train90 / test10 |
| 自动标注 (MER-Caption+) | `track2_train_mercaptionplus.csv` | 31,327 | SFT 训练(去重后 31,276) |
| 音/视频/人脸 | `*_7z/{group}/audio video openface_face` | 各分组自洽 | 音频+人脸模态输入 |

CSV 关键字段:`name`(样本名,与音视频文件名一致)、`openset`(情感标签列表,逗号分隔)。
**字幕文件不使用**(`subtitle_chieng.csv` 本地仅 2 万行且 english 全空,且本方案不用文本模态)。

### 2.2 数据划分(唯一协议)

从 `track2_train_human.csv` 按 **9:1 固定 seed 划分**:

| 划分 | 比例 | 数量 | 文件 |
|---|---|---|---|
| train90(GRPO 训练) | 90% | ≈1,379 | `track2_train_human_train90.csv` |
| **test10(统一测试集)** | 10% | ≈153 | `track2_train_human_test10.csv` |

- 用 [`../../result/scripts/split_human_data.py`](../../result/scripts/split_human_data.py) 生成,`seed=42`,**任何重跑结果完全一致**;脚本会校验 train/test 无交集。
- **所有阶段(SFT / GRPO / 消融)统一用 `track2_train_human_test10.csv` 评测**,保证可比性。

**去重(防泄漏)**:test10 的 153 条里有 51 条(33%)同时出现在 mc+ 里。用
[`../../result/scripts/dedup_mcplus.py`](../../result/scripts/dedup_mcplus.py) 剔除这 51 条,
产出 `track2_train_mercaptionplus_dedup.csv`(31,276 条)作为 SFT 训练集。脚本会校验去重后与
test10 残留重叠为 0。

### 2.3 各阶段训练/测试数据速查

| 阶段 | 训练数据 | 验证数据(选最优 epoch) | 测试数据 |
|---|---|---|---|
| LoRA-SFT | `track2_train_mercaptionplus_train.csv` (≈30,976) | `track2_train_mercaptionplus_val.csv` (300) | `track2_train_human_test10.csv` (≈153) |
| GRPO | `track2_train_human_train90.csv` (≈1,379) | 训练中在 test10 采样算 EW-F1 | `track2_train_human_test10.csv` (≈153) |

> SFT 训练时每 `eval_interval`(默认 5)轮做一次**生成式 EW-F1 验证**(在验证集上生成标签,
> 用 `grpo/rewards.compute_pair_scores` 算 EW-F1),按 **EW-F1 选最优 epoch**(yaml `valid_splits: ["val"]` + `eval_interval: 5`,
> runner 存 `checkpoint_best.pth`);每轮权重都保留(`checkpoint_{epoch}_loss_{loss}.pth`,不覆盖)。
> 最终报告指标仍是 test10 的 EW-F1。GRPO 训练中 `train_grpo.py` 每 `--eval-every` 步在 test10 上算 EW-F1。
>
> 测试时把 `config.py` 中 `PATH_TO_LABEL['MER2026OV']` 指向 `track2_train_human_test10.csv`
> (该文件含 `openset` 字段,evaluation.py 读取它作为 ground-truth)。

---

## 3. 评测流程(命令级)

```
train.py ──► inference_hybird.py ──► ovlabel_extraction.py ──► evaluation.py
 (训练)        (生成 reason 描述)       (reason→OV 标签)          (算 EW-F1)
```

### Step 1:训练

```bash
cd /root/MER2026_Track2
# SFT (MC+ 全量)
CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --cfg-path=train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz.yaml
# GRPO (human 90%) — 对应 yaml 待建
```

### Step 2:推理(生成每条样本的情感推理描述 reason)

```bash
CUDA_VISIBLE_DEVICES=0 python -u inference_hybird.py --zeroshot \
  --dataset='MER2026OV' \
  --cfg-path=train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz.yaml \
  --options "inference.test_epochs=10-60" "inference.skip_epoch=5"
# 输出: output/results-mer2026ov/{ckpt_root_name}/{epoch}.npz  (name2reason)
```

### Step 3:reason → 开放词表情感标签

```bash
CUDA_VISIBLE_DEVICES=0 python ovlabel_extraction.py
# 用 Qwen2.5-7B 把 reason 转成 OV 标签列表
# 输出: output/results-mer2026ov/{ckpt_root_name}/{epoch}-openset.npz  (name2pred)
```

### Step 4:EW-F1 评测

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation.py
# 读取 {epoch}-openset.npz 作为预测, track2_train_human_test10.csv 作为 GT, 计算 EW-F1
```

> 相关代码位置:
> - 评测指标实现:`my_affectgpt/evaluation/wheel.py`
> - reason→OV 抽取:`toolkit/utils/qwen.py` 的 `reason_to_openset_qwen`
> - 评测入口:`evaluation.py`(`wheel_metric_calculation` + `calculate_openset_overlap_rate`)

---

## 4. 指标定义

### 4.1 主指标:EW-F1 (Emotion Wheel F1)

官方指标,解决开放词表的同义词/标签变体问题,采用 **3 级层级映射**(见 `emotion_wheel/wheel_mapping.npz`):

| 层级 | 映射 | 说明 |
|---|---|---|
| L1 → L2 | `format_mapping` (7,356 词) | 情感词归一为词根(如 angered → anger) |
| L2 → L1 | `raw_mapping` (1,255 词) | 同义词归并(如 happy/joyful → happy) |
| L1 → L0 | `wheel_map_whole` (5 轮 × level1/level2) | 映射到 5 个情感轮各自的内圈基本类别 |

对每个样本计算 openset 重叠率:

```
precision = |GT ∩ Pred| / |Pred|
recall    = |GT ∩ Pred| / |GT|
F1        = 2·P·R / (P+R)
```

最终得分 = **5 个情感轮 (`case3_wheel1..wheel5`) F1 的平均**:
- `wheel_metric_calculation(level='level1')` → **EW-F1 (level1)(主指标)**
- `wheel_metric_calculation(level='level2')` → EW-F1 (level2)(辅助指标)

指标代码对应:`calculate_openset_overlap_rate(metric='case3_wheel{k}_{level}')`,即
`func_backward_case3` → 按 `format_mapping → raw_mapping → wheel_map` 逐级向后归并,再算重叠率。

> **统一口径:不在词表中的词直接剔除**(`func_map_label_to_synonym` 中 `label==''` 跳过)。

### 4.2 辅助指标

- **Precision / Recall**:openset 重叠率的精度与召回(随 EW-F1 一起输出)。
- **训练过程监控**:loss(CE)、learning rate,用于判断收敛与调参。

### 4.3 报告格式

登记到 `result/result.md` 时统一:

```
EW-F1 (level1) = 62.1%   |   EW-F1 (level2) = xx.x%   |   P = xx.x%  R = xx.x%
```

---

## 5. 可视化规范

所有可视化统一输出到 `result/v{xxx}/vis/`,命名 `{版本}_{图名}.png`。规定以下 4 类,按需增减:

| # | 图 | 内容 | 用途 |
|---|---|---|---|
| 1 | `loss_lr_curve.png` | 训练 loss / lr 随 epoch(或 iter)变化曲线,训练集折线 + 平滑 | 判断收敛、过拟合、调 lr |
| 2 | `ewf1_epoch_curve.png` | 测试集 EW-F1 (level1) 随 epoch 变化(在 `inference.test_epochs` 范围内每个评测 epoch 打分) | 选择最佳 epoch,观察过拟合 |
| 3 | `wheel_bar.png` | 5 个情感轮 × (level1 / level2) 的 F1 柱状图,标注平均值 | 看哪轮好/差,定位类别薄弱点 |
| 4 | `wheel_radar.png` | 情感轮雷达图:GT 标签分布 vs 预测标签分布(归一化后叠加) | 看预测偏置(哪些类别被过度/不足预测) |
| 5 | `qualitative_examples.md` | 定性示例表:样本名 + 视频关键帧 + reason + GT labels vs Pred labels + 是否命中 | 面试展示、错误分析 |

> 可视化脚本可复用 `wheel.py` 的分轮 F1 输出(逐轮 `calculate_openset_overlap_rate` 已由
> `wheel_metric_calculation` 给出),画图用 matplotlib。

---

## 6. 结果登记规范

每跑完一个版本,按顺序执行:

1. **建子文件夹** `result/v{序号}_{阶段}_{数据}_{变体}/`,内含:
   - `method.md` — 方法说明(模型结构、融合方式、数据、超参、复现命令、ckpt 在 `output/` 的实际路径)
   - `logs/` — 训练日志(或软链到 `output/`)
   - `eval/` — `{epoch}-openset.npz`、EW-F1 结果文本/表格
   - `vis/` — 第 5 节可视化产物
2. **在 `result/result.md` 结果总览表登记一行**(版本 / 阶段 / 训练数据 / 方法要点 / 测试集 / EW-F1(level1) / EW-F1(level2) / P·R / 备注)。
3. 在 `result/result.md`「变更记录」追加一行。

---

## 7. 复现检查清单

- [ ] `config.py` 中 `DATA_DIR['MER2026']` 已指向 `/root/fsas/AffectGPT_dataset`
- [ ] `models/` 下已放置 Qwen2.5-7B / chinese-hubert-large / clip-vit-large-patch14
- [ ] 已运行 `split_human_data.py` 生成 train90 / test10 两份 csv
- [ ] 已运行 `dedup_mcplus.py` 生成去重后 mc+ 训练集(31,276)
- [ ] `PATH_TO_LABEL['MERCaptionPlus']` 指向 dedup csv,`PATH_TO_LABEL['Human']` 指向 train90 csv
- [ ] 测试用 csv 已指向 `track2_train_human_test10.csv`(改 `PATH_TO_LABEL['MER2026OV']`)
- [ ] 模态配置 `face_or_frame: multiface_audio_face`(音频+人脸,无文本)
- [ ] SFT / GRPO 两阶段均用同一 test10 评测
- [ ] 结果与图已按第 6 节登记进 `result/`
