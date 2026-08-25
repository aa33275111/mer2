"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging

import numpy as np
import torch
import torch.distributed as dist

import config
from my_affectgpt.common.dist_utils import is_dist_avail_and_initialized
from my_affectgpt.common.registry import registry
from my_affectgpt.tasks.base_task import BaseTask
from grpo.rewards import compute_pair_scores, parse_pred_labels

logger = logging.getLogger(__name__)


@registry.register_task("video_text_pretrain")
class VideoTextPretrainTask(BaseTask):
    def __init__(self):
        super().__init__()
        self._eval_dataset = None

    def before_evaluation(self, model, dataset, **kwargs):
        self._eval_dataset = dataset
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def valid_step(self, model, samples):
        loss = model(samples)["loss"]
        return [loss.item()]

    # ------------------------------------------------------------------ #
    # 生成式 EW-F1 验证 (2026-08-25):
    #   对验证集样本生成情感标签, 用 grpo.rewards 算 EW-F1 (真实任务指标)
    #   用于选最优 epoch; val loss 同时记录, 只作参考。
    # ------------------------------------------------------------------ #
    def _generate_labels(self, model, ds, sample, device):
        qa = ds.get_qa_pairs(ds.dataset, 'ovlabel', sample)
        prompt = ds.get_prompt_for_multimodal(ds.face_or_frame, sample.get('subtitle'), qa['question'])
        prompt = ds.replace_token_for_multimodal(prompt)
        prompt_ids = ds.to_token_ids(prompt, ds.max_length).to(device)

        video_path = ds._get_video_path(sample) if hasattr(ds, '_get_video_path') else None
        audio_path = ds._get_audio_path(sample) if hasattr(ds, '_get_audio_path') else None
        face_npy = ds._get_face_path(sample) if hasattr(ds, '_get_face_path') else None
        sd = ds.read_frame_face_audio_text(video_path, face_npy, audio_path, None)

        feats = {}
        for k, pk in [('face', 'faces'), ('raw_face', 'raw_faces'),
                      ('audio', 'audios'), ('raw_audio', 'raw_audios')]:
            feats[pk] = None if sd[k] is None else sd[k].unsqueeze(0).to(device)

        input_ids = prompt_ids.unsqueeze(0)
        attn = input_ids.ne(model.llama_tokenizer.pad_token_id)
        samples = {
            'input_ids': input_ids,
            'attention_masks': attn,
            'face_or_frame': ds.face_or_frame,
            **feats,
        }
        with torch.no_grad():
            embeds = model.prepare_inputs_embeds(samples)
            gen = model.llama_model.generate(
                inputs_embeds=embeds,
                attention_mask=attn.to(device),
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=model.llama_tokenizer.pad_token_id,
            )
        rids = gen[0][len(prompt_ids):]
        text = model.llama_tokenizer.decode(rids, skip_special_tokens=True)
        return text

    def evaluation(self, model, data_loader, cuda_enabled=True):
        ds = self._eval_dataset
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.eval()

        ann = ds.annotation
        n_samples = int(getattr(config, 'EVAL_N_SAMPLES', 200))
        if len(ann) > n_samples:
            rng = np.random.RandomState(42)  # 固定采样, 保证每次验证同一批
            idx = rng.choice(len(ann), n_samples, replace=False)
        else:
            idx = range(len(ann))

        f1s = []
        for i in idx:
            sample = ann[i]
            try:
                pred_text = self._generate_labels(model, ds, sample, device)
                gt = [x.strip() for x in sample['ovlabel'].split(',') if x.strip()]
                f1s.append(compute_pair_scores(gt, parse_pred_labels(pred_text))['f1'])
            except Exception as e:
                logger.warning(f'[val] skip {sample["name"]}: {e}')

        if is_dist_avail_and_initialized():
            dist.barrier()
        return {'f1_list': f1s}

    def after_evaluation(self, val_result, split_name, epoch):
        f1s = val_result.get('f1_list', [])
        ewf1 = float(np.mean(f1s)) if f1s else 0.0
        return {
            'agg_metrics': ewf1,   # 越高越好: 直接用 EW-F1 选最优
            'val_ewf1': ewf1,
            'epoch': epoch,
        }
