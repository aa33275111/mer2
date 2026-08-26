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

### 内部特征流(**Q-Former 融合**,与配置一致)

```
face:  [B,3,8,224,224] --CLIP(ViT-L/14)--> [B,8,768]
      --Video Q-Former(查询向量交叉注意力)--> [B, num_video_query_token, 768]
      --affectgpt_proj--> [B,1,llmdim]                          (num_video_query_token=1)
audio: [B,8,1,128,204] --HuBERT-large--> [B,T,1024]
      --Audio Q-Former--> [B, num_audio_query_token, 1024]
      --audio_llama_proj--> [B,1,llmdim]                        (num_audio_query_token=1)
multi: face_hidden + audio_hidden --multi_video/audio_embs 升到同一维--> concat [B, Ta+Tv, maxdim]
      --Multi Q-Former--> [B, num_multi_query_token, maxdim]
      --multi_llama_proj--> [B,1,llmdim]                        (num_multi_query_token=1)
```

- **Video/Audio/Multi Q-Former**: 可学习的 query token 对编码器特征做**交叉注意力**,把变长的序列压缩成固定 `num_*_query_token` 个 token(本配置各 =1)
- llmdim = `llama_model.config.hidden_size`(Qwen2.5-7B = 3584)
- 把 `inputs_embeds` 里 `<FaceHere>/<AudioHere>/<MultiHere>` 位置替换为对应特征 → 送入 Qwen2.5-7B(**LoRA**,r 可配)自回归

### 输出

```python
{"loss": loss}   # 标量 CE loss(仅训练用, 面试常问"为什么只有 loss"→ 因为评测走外部 EW-F1)
```

---

## 位置 2.5 — Prompt 设计与训练目标(loss mask)

### 训练时的完整 prompt(`get_prompt_for_multimodal` 的 `multiface_audio_face` 分支)

```
###Human: The audio and video merged info is: <Multi><MultiHere></Multi>.
The audio content is as follows: <Audio><AudioHere></Audio>.
Meanwhile, we uniformly sample raw frames from the video and extract faces from these frames: <Video><FaceHere></Video>.
Now, please answer my question based on all the provided information.
Please recognize all possible emotional states of the character.
###Assistant: The character's emotional state is {labels}.###
```

- `<AudioHere>` / `<FaceHere>` / `<MultiHere>` 是占位 token,forward 时被替换成 HuBERT/CLIP 提取的特征向量(不再是文本)
- `{labels}` 是训练目标(如 `happy, sad`)

### 哪部分参与计算(loss 只算 answer 段)

token 序列(以单样本为例):

```
input_ids:  [<bos> ###Human: ...问题 ###Assistant: | happy, sad |###<eos> <pad>...]
labels:      [-100  -100...             -100        | happy, sad |###  <eos>  -100...]
                                                       ↑ 只有这段参与 CE loss
```

- prompt 段、pad 段都是 `-100`(`config.IGNORE_INDEX`),被 mask 掉
- 自回归:loss = 对 `labels != -100` 的每个 answer token 算 `-log P(next | 之前全部)`,平均

### 为什么 prompt/pad 段 -100、不参与梯度

1. **prompt 段是"条件",不是"预测目标"**:因果 LM 是 `P(next | 之前所有 token)`。prompt(指令 + 多模态特征 + 问题)是**给定输入**,模型要"读"它而不是"复述"它。对 prompt 算 loss 等于逼模型预测自己的输入,无意义还会干扰条件学习。
2. **基座模型本来就能读懂 prompt**:Qwen2.5-7B 预训练已具备语言/指令理解;多模态 token 的对齐由 **Q-Former + 投影层**完成(这些是训练重点)。prompt 段不学,是"把模型当条件生成器,从 prompt 的 hidden state 出发预测后续"。
3. **pad 段只是填充 batch 长度**,无信息,算 loss 是纯噪声。

### 固定输出格式是怎么学到的?

- **格式就在 answer 段里,而 answer 段是算 loss 的**。每个训练样本的 target 都是
  `The character's emotional state is {labels}.###`(监督信号)
- 模型被训练成:看到 `###Assistant:` 之后,继续输出这种固定格式的句子
- 格式 delimiter(`###Human:` / `###Assistant:` / `###`)是 answer 段的一部分,模型通过海量样本模仿学会"该在 ###Assistant: 后输出情感标签、遇到 ### 结束"
- 一句话:**prompt 格式是"给的"(masked),输出格式是"学的"(lossed)**

### 推理时

- 只输入 prompt(到 `###Assistant:` 为止),占位 token 换成特征
- `llama_model.generate(...)` 自回归生成 answer(采样或贪心)
- 生成文本(如 `The character's emotional state is happy, sad`)→ 后续 OV 标签抽取 → 与 GT 算 EW-F1

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
