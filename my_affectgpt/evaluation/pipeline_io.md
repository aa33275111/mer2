# MER2026 Track2 — 全流程 输入/输出 规格

> 本文档按**位置**(pipeline 每个阶段)详细记录输入与输出:文件格式、数据结构、张量形状、落盘位置。
> 面试讲解项目时按本文档逐段讲即可。配套协议见 [`evaluation.md`](evaluation.md),结果登记见 [`../../result/result.md`](../../result/result.md)。

整体链路:

```
数据准备 → 数据读取 → 模型 forward → 训练(runner)
                                          ↓
                                    checkpoint
                                          ↓
推理(inference_hybird) → reason ─→ OV标签抽取(ovlabel_extraction) → 评测(evaluation/wheel) → EW-F1
```

---

## 位置 0 — 数据准备(一次性)

| | 内容 |
|---|---|
| **输入** | `/root/fsas/AffectGPT_dataset/` 下 csv + 分组音视频/人脸 |
| **脚本** | `result/scripts/split_human_data.py`(9:1 划分)、`result/scripts/dedup_mcplus.py`(剔除泄漏) |
| **输出** | `track2_train_human_train90.csv`(**1,379** 条)、`track2_train_human_test10.csv`(**153** 条)、`track2_train_mercaptionplus_dedup.csv`(**31,276** 条) |

数据流细节:
- 原始人工标注 `track2_train_human.csv`(1,532)→ 按 seed=42 9:1 划分 → train90 / test10(无交集)
- 原始 mc+ `track2_train_mercaptionplus.csv`(31,327)→ 剔除与 test10 重叠的 51 条 → dedup(31,276),消除 SFT 数据泄漏
- **模态** = 音频 + 人脸(**无文本**,`face_or_frame: multiface_audio_face`);每组数据在 `*_7z/{group}/` 下完整自洽(已核验覆盖率 100%)

---

## 位置 1 — 数据读取(dataset)

入口:`my_affectgpt/datasets/datasets/base_dataset.py` 的 `__getitem__` / `collater`;子类 `human_dataset.py` / `mercaptionplus_dataset.py`。

### 单样本输出(`__getitem__` 返回 dict)

| key | 形状 | 说明 |
|---|---|---|
| `face` | `[3, 8, 224, 224]` | 人脸,经 vis_processor 变换(训练增强) |
| `raw_face` | `[3, 8, 224, 224]` | 人脸原始帧(均匀采样 8 帧) |
| `frame` | `[3, 8, 224, 224]` | 视频帧(本方案未用,因 face_or_frame 只加载 face+audio) |
| `raw_frame` | `[3, 8, 224, 224]` | 同上 |
| `audio` | `[8, 1, 128, 204]` | 8 个 2s 片段的对数梅尔谱 |
| `raw_audio` | `[8, 1, 32000]` | 8 个 2s 片段原始波形(16kHz × 2s) |
| `label` | `[seq_len]` | 训练目标,非回答部分为 `-100`(IGNORE_INDEX) |
| `text_input` | `[seq_len]` | prompt(含 `<AudioHere>` `<FaceHere>` `<MultiHere>` 占位)+ answer |
| `dataset` / `face_or_frame` | str | 标记用 |

> prompt 模板(`get_prompt_for_multimodal` 的 `multiface_audio_face` 分支):
> `###Human: The audio and video merged info is: <Multi><MultiHere></Multi>. The audio content is as follows: <Audio><AudioHere></Audio>. Meanwhile, we uniformly sample raw frames from the video and extract faces from these frames: <Video><FaceHere></Video>. Now, please answer my question based on all the provided information. {question} ###Assistant:`
> (无 subtitle 文本。)

### batch 输出(`collater` 返回 dict)

| key | 形状 | 说明 |
|---|---|---|
| `input_ids` | `[B, L]` | `<bos>` + 文本 + `<eos>` + `<pad>`,含 patch token |
| `labels` | `[B, L]` | 对齐 input_ids,非答案区为 -100 |
| `attention_masks` | `[B, L]` | `input_ids != pad_token_id` |
| `faces` / `raw_faces` | `[B, 3, 8, 224, 224]` | 仅当该模态被加载时存在 |
| `audios` / `raw_audios` | `[B, 8, 1, 128, 204]` / `[B, 8, 1, 32000]` | 同上 |
| `dataset` / `face_or_frame` | str | 标记 |

---

## 位置 2 — 模型 forward(`models/affectgpt.py`)

### 输入(batch samples)
`input_ids`(含 5 类 patch token)、`labels`、`attention_masks`、`faces/raw_faces`、`audios/raw_audios`。

### 内部特征流(以 attention 融合为例)

```
face:  [B,3,8,224,224] --CLIP(ViT-L/14)--> [B,8,768] --attention融合--> [B,768]
      --affectgpt_proj--> [B,1,llmdim]                     (num_video_query_token=1)
audio: [B,8,1,128,204] --HuBERT-large--> [B,T,1024] --attention融合--> [B,1024]
      --audio_llama_proj--> [B,1,llmdim]                    (num_audio_query_token=1)
multi: 对 face/audio 均值 → multi_video/audio_embs 升到同一维 → concat
      --attention_mlp + fc_att(2权重)--> 加权融合 [B,maxdim] --multi_llama_proj--> [B,1,llmdim]
```

- llmdim = `llama_model.config.hidden_size`(Qwen2.5-7B = 3584)
- 把 `inputs_embeds` 里 `<FaceHere>/<AudioHere>/<MultiHere>` 位置替换为对应特征 → 送入 Qwen2.5-7B(**LoRA**,r 可配)自回归

### 输出

```python
{"loss": loss}   # 标量 CE loss(仅训练用, 面试常问"为什么只有 loss"→ 因为评测走外部 EW-F1)
```

---

## 位置 3 — 训练(runner:`runners/runner_base.py` + `tasks/base_task.py`)

| | 内容 |
|---|---|
| **输入** | cfg(yaml)+ datasets + model |
| **输出** | checkpoint 文件 + 训练日志 |

- checkpoint:`output/{cfg_name}/{job_id}/checkpoint_{epoch:06d}_loss_{loss}.pth`
  - `cfg_name` = yaml 文件名;`job_id` = `{cfg_name}_{YYYYMMDDHHmm}`(train.py 生成)
  - 内容:`torch.save(save_obj)`,`save_obj['model']` 为 state_dict
- 日志:每个 epoch 记录 `loss` / `lr`(`MetricLogger` 输出到 stdout + 日志)
- 训练协议(见 [`evaluation.md`](evaluation.md)):
  - SFT:mc+ dedup 31,276 条,60 epoch,init_lr 1e-5,cosine,batch=3×gpu,AMP
  - GRPO:human train90 1,379 条(待实现)

---

## 位置 4 — 推理(`inference_hybird.py`)

| | 内容 |
|---|---|
| **输入** | `--cfg-path` + `--dataset=MER2026OV` + checkpoint(自动/指定 epoch)+ 测试名单 |
| **测试名单** | `MER2026OV_Dataset.read_test_names()` → 读 `PATH_TO_LABEL['MER2026OV']` 的 `name` 列(开发期指向 test10 的 153 条) |
| **输出** | `output/results-mer2026ov/{ckpt_root_name}/{epoch}.npz` |

每一样本:
```
sample_data → chat.postprocess_audio/face/multi → img_list{audio,face,multi,...}
prompt(question_only 的 ovlabel 提问) → chat.answer_sample()
  (num_beams=1, do_sample=True, top_p=0.9, max_new_tokens=1200)
→ name2reason[name] = reason 文本
```

npz 结构(`np.savez_compressed`):
```python
{"name2reason": {name: reason_str}}   # np.load(...)['name2reason'].tolist()
```

---

## 位置 5 — reason → OV 标签(`ovlabel_extraction.py`)

| | 内容 |
|---|---|
| **输入** | `{epoch}.npz`(name2reason)+ `Qwen2.5-7B`(vLLM 批量,`reason_to_openset_qwen`) |
| **输出** | `{epoch}-openset.npz` |

```python
{"filenames": [name, ...], "fileitems": [[label1, label2, ...], ...]}
# 同一行 name → 对应的 OV 标签列表; np.load(allow_pickle=True)
```

> 这一步用 Qwen 把"情感推理描述"翻译成开放词表情感标签,解决 reason 与 GT 标签格式不对齐的问题。

---

## 位置 6 — 评测(`evaluation.py` + `evaluation/wheel.py`)

| | 内容 |
|---|---|
| **输入** | `{epoch}-openset.npz`(预测)+ `track2_train_human_test10.csv`(GT,openset 列)+ `emotion_wheel/wheel_mapping.npz` |
| **输出** | EW-F1(level1/level2)+ Precision / Recall |

计算链路(`wheel_metric_calculation` → `calculate_openset_overlap_rate`):

```
GT/Pred 每个标签:
  L3→L2  format_mapping(7,356 词, 归一词根)
  L2→L1  raw_mapping(1,255 词, 同义词归并)
  L1→L0  wheel_map_whole(5 轮 × level1/level2, 映射到内圈基本类别)
  不在词表 → 剔除

per-sample:  precision = |GT∩Pred| / |Pred|
             recall    = |GT∩Pred| / |GT|
             F1        = 2·P·R / (P+R)

EW-F1 = 5 个轮 (case3_wheel1..5) F1 的平均  ← 官方主指标
```

- `wheel_metric_calculation(level='level1')` → **EW-F1 (level1)**(主指标)
- `wheel_metric_calculation(level='level2')` → EW-F1 (level2)(辅助)
- 登记格式:`EW-F1(level1) = 62.1% | EW-F1(level2) = xx.x% | P = xx.x% R = xx.x%`

---

## 常用命令(串起全流程)

```bash
cd /root/MER2026_Track2

# 0) 一次性: 划分 + 去重
python result/scripts/split_human_data.py
python result/scripts/dedup_mcplus.py

# 1) SFT 训练 (mc+ dedup)
CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --cfg-path=train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz.yaml

# 2) 推理 (开发期: 把 config.PATH_TO_LABEL['MER2026OV'] 指向 test10 csv)
CUDA_VISIBLE_DEVICES=0 python -u inference_hybird.py --zeroshot \
  --dataset='MER2026OV' \
  --cfg-path=train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz.yaml \
  --options "inference.test_epochs=10-60" "inference.skip_epoch=5"

# 3) reason → OV 标签
CUDA_VISIBLE_DEVICES=0 python ovlabel_extraction.py

# 4) EW-F1 评测
CUDA_VISIBLE_DEVICES=0 python evaluation.py
```

> 各阶段产物都登记到 `result/result.md`,可视化统一放 `result/v{xxx}/vis/`(规范见 [`evaluation.md`](evaluation.md) 第 5 节)。
