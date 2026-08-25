# *_*coding:utf-8 *_*
"""
GRPO 训练器 (Group Relative Policy Optimization)

算法要点 (与 v7 历史配置一致):
  - 每个 prompt 采样 G=8 条响应, 组成一组
  - 组内对奖励做均值/方差归一化得到 advantage (a_i = (r_i - mean) / (std + eps))
  - 策略梯度损失:  per-token importance ratio = exp(logp_policy - logp_ref),
                   带裁剪 (clip epsilon=0.1) 的 min(ratio*adv, clip(ratio)*adv)
  - KL 惩罚: 对参考模型逐 token 计算 KL, beta 加权
  - 温度固定 1.0 采样

接口:
  from grpo.grpo_trainer import GRPOTrainer
  trainer = GRPOTrainer(policy, ref, tokenizer, dataset, reward_fn, **cfg)
  trainer.train()
"""
import logging
import numpy as np
import torch
import torch.nn.functional as F

import config

logger = logging.getLogger(__name__)


class GRPOTrainer:
    def __init__(
        self,
        policy,                # AffectGPT, LoRA 可训练 (策略模型)
        ref,                   # AffectGPT, 全冻结 (参考模型)
        tokenizer,             # llama_tokenizer (含 patch tokens)
        dataset,               # human train90 数据集
        reward_fn,             # (pred_text, gt_labels) -> dict (含 'reward')
        device='cuda',
        G=8,                   # 组大小
        temperature=1.0,       # 采样温度 (v7: 固定 1.0)
        lr=5e-7,               # 学习率 (v7)
        beta=0.04,             # KL 惩罚系数
        eps=0.1,               # 裁剪 epsilon (v7: 0.1)
        grad_accum=4,          # 梯度累积步数 (v7: 4)
        max_steps=500,         # 总更新步数 (v7: 500)
        max_new_tokens=64,     # 生成的响应最大 token 数
        seed=42,
    ):
        self.policy = policy.to(device)
        self.ref = ref.to(device)
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.reward_fn = reward_fn
        self.device = device
        self.G = G
        self.temperature = temperature
        self.beta = beta
        self.eps = eps
        self.grad_accum = grad_accum
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.seed = seed

        # 只更新 LoRA 可训练参数
        trainable = [p for p in self.policy.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in trainable)
        logger.info(f'[GRPO] trainable params: {n_params:,} ({n_params / 1e6:.1f}M)')
        self.optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)

        self._rng = np.random.RandomState(seed)
        self._step = 0
        self._data_idx = None  # lazy 洗牌迭代
        self._make_iterator()

    # ------------------------------------------------------------------ #
    # 数据迭代: 按 sample 粒度洗牌
    # ------------------------------------------------------------------ #
    def _make_iterator(self):
        indices = np.arange(len(self.dataset))
        while True:
            self._rng.shuffle(indices)
            for i in indices:
                yield self.dataset.annotation[i]

    def _next_sample(self):
        if self._data_idx is None:
            self._data_iter = self._make_iterator()
        return next(self._data_iter)

    # ------------------------------------------------------------------ #
    # 输入构建: prompt + 多模态特征 + GT
    # ------------------------------------------------------------------ #
    def _build_input(self, sample):
        ds = self.dataset
        qa = ds.get_qa_pairs(ds.dataset, 'ovlabel', sample)
        prompt = ds.get_prompt_for_multimodal(ds.face_or_frame, sample.get('subtitle'), qa['question'])
        prompt = ds.replace_token_for_multimodal(prompt)
        prompt_ids = ds.to_token_ids(prompt, ds.max_length).to(self.device)

        video_path = ds._get_video_path(sample) if hasattr(ds, '_get_video_path') else None
        audio_path = ds._get_audio_path(sample) if hasattr(ds, '_get_audio_path') else None
        face_npy = ds._get_face_path(sample) if hasattr(ds, '_get_face_path') else None
        sd = ds.read_frame_face_audio_text(video_path, face_npy, audio_path, None)

        feats = {}
        for k, pk in [('face', 'faces'), ('raw_face', 'raw_faces'),
                      ('audio', 'audios'), ('raw_audio', 'raw_audios')]:
            feats[pk] = None if sd[k] is None else sd[k].to(self.device)

        gt = [x.strip() for x in sample['ovlabel'].split(',') if x.strip()]
        return prompt_ids, feats, gt

    def _feature_dict(self, feats, n):
        """把单样本特征重复 n 份, 构成 [n, ...] 的 features dict。"""
        out = {}
        for k, v in feats.items():
            if v is None:
                out[k] = None
            else:
                out[k] = v.unsqueeze(0).repeat(n, *([1] * v.dim()))
        return out

    # ------------------------------------------------------------------ #
    # 生成: 一个样本采 G 条响应
    # ------------------------------------------------------------------ #
    def _generate(self, prompt_ids, feats, n):
        input_ids = prompt_ids.unsqueeze(0).repeat(n, 1)
        attn = input_ids.ne(self.tokenizer.pad_token_id)
        samples = {
            'input_ids': input_ids,
            'attention_masks': attn,
            'face_or_frame': self.dataset.face_or_frame,
            **self._feature_dict(feats, n),
        }
        with torch.no_grad():
            embeds = self.policy.prepare_inputs_embeds(samples)
            gen = self.policy.llama_model.generate(
                inputs_embeds=embeds,
                attention_mask=attn.to(self.device),
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.9,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        texts, resp_ids = [], []
        for row in gen:
            rids = row[len(prompt_ids):]
            rids = rids[rids != self.tokenizer.pad_token_id]
            texts.append(self.tokenizer.decode(rids, skip_special_tokens=True))
            resp_ids.append(rids.clone())
        return texts, resp_ids

    # ------------------------------------------------------------------ #
    # logprob: (prompt + response) 序列, 只对 response 计
    # ------------------------------------------------------------------ #
    def _logprobs(self, model, prompt_ids, resp_ids_list, feats):
        B = len(resp_ids_list)
        seqs = [torch.cat([prompt_ids, rids]) for rids in resp_ids_list]
        seqs = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True,
                                               padding_value=self.tokenizer.pad_token_id).to(self.device)
        labels = torch.full_like(seqs, config.IGNORE_INDEX)
        for i, rids in enumerate(resp_ids_list):
            start = len(prompt_ids)
            labels[i, start:start + len(rids)] = rids
        attn = seqs.ne(self.tokenizer.pad_token_id)
        samples = {
            'input_ids': seqs,
            'labels': labels,
            'attention_masks': attn,
            'face_or_frame': self.dataset.face_or_frame,
            **self._feature_dict(feats, B),
        }
        token_lp, mask = model.compute_response_logprobs(samples, return_tokens=True)
        return token_lp, mask  # [B, L-1] each

    # ------------------------------------------------------------------ #
    # GRPO 损失
    # ------------------------------------------------------------------ #
    def _grpo_loss(self, policy_lp, ref_lp, mask, rewards):
        B, G = rewards.shape
        r = torch.tensor(rewards, dtype=torch.float32, device=self.device)  # [B, G]
        adv = (r - r.mean(dim=1, keepdim=True)) / (r.std(dim=1, keepdim=True) + 1e-8)  # 组内归一化
        adv = adv.unsqueeze(-1)  # [B, G, 1]

        ratio = torch.exp(policy_lp - ref_lp)                 # [B, G, L]
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - self.eps, 1.0 + self.eps) * adv
        pg_loss = -(torch.min(unclipped, clipped) * mask.float())

        # per-token KL (GRPO 常用无偏估计)
        kl = torch.exp(ref_lp - policy_lp) - (ref_lp - policy_lp) - 1.0
        kl = kl * mask.float()

        n = mask.float().sum().clamp(min=1.0)
        loss = pg_loss.sum() / n + self.beta * kl.sum() / n
        return loss, float(kl.sum() / n)

    # ------------------------------------------------------------------ #
    # 训练
    # ------------------------------------------------------------------ #
    def train(self, eval_fn=None, eval_every=50, save_root=None):
        self.policy.train()
        self.optimizer.zero_grad()
        accum_loss = 0.0
        accum_kl = 0.0

        for step in range(1, self.max_steps + 1):
            sample = self._next_sample()
            prompt_ids, feats, gt = self._build_input(sample)

            # 1) 采样 G 条响应 + 算奖励
            resp_texts, resp_ids = self._generate(prompt_ids, feats, self.G)
            rewards = [self.reward_fn(t, gt)['reward'] for t in resp_texts]

            # 2) policy / ref logprob
            policy_lp, mask = self._logprobs(self.policy, prompt_ids, resp_ids, feats)
            with torch.no_grad():
                ref_lp, _ = self._logprobs(self.ref, prompt_ids, resp_ids, feats)

            # 3) GRPO loss (单样本组: B=1, G)
            rewards_arr = np.array(rewards).reshape(1, self.G)
            loss, kl_val = self._grpo_loss(
                policy_lp.unsqueeze(0), ref_lp.unsqueeze(0), mask.unsqueeze(0), rewards_arr)
            loss = loss / self.grad_accum
            loss.backward()
            accum_loss += loss.item() * self.grad_accum
            accum_kl += kl_val

            # 4) 梯度累积更新
            if step % self.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.policy.parameters() if p.requires_grad], max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                logger.info(f'[GRPO] step={step}/{self.max_steps} loss={accum_loss / self.grad_accum:.4f} '
                            f'kl={accum_kl / self.grad_accum:.4f} avg_reward={rewards_arr.mean():.4f} '
                            f'best_reward={rewards_arr.max():.4f}')
                accum_loss, accum_kl = 0.0, 0.0

            # 5) 周期验证 + 存档
            if eval_fn is not None and step % eval_every == 0:
                ewf1 = eval_fn(self.policy)
                logger.info(f'[GRPO] step={step} validation EW-F1 = {ewf1 * 100:.2f}%')
                if save_root:
                    torch.save({'model': self.policy.state_dict(),
                                'step': step, 'ewf1': ewf1},
                               f'{save_root}/grpo_step{step:06d}_ewf1_{ewf1:.4f}.pth')

        # 最后一次存档
        if save_root:
            torch.save({'model': self.policy.state_dict(), 'step': self.max_steps},
                       f'{save_root}/grpo_final_step{self.max_steps:06d}.pth')
        logger.info('[GRPO] training finished.')
