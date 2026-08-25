# *_*coding:utf-8 *_*
"""
GRPO 训练入口 (第一版: 纯 EW_F1 奖励, 对齐历史 v7 配置)

用法 (GPU 机器):
    CUDA_VISIBLE_DEVICES=0 python -u grpo/train_grpo.py \
        --cfg-path train_configs/grpo_human_ewf1.yaml \
        --sft-ckpt output/{cfg_name}/{job_id}/checkpoint_006000_loss_0.909.pth \
        --reward ewf1 \
        --G 8 --temp 1.0 --lr 5e-7 --eps 0.1 --grad-accum 4 --max-steps 500

关键超参 (v7 历史配置):
    G=8, temperature=1.0, lr=5e-7, eps=0.1, grad_accum=4, max_steps=500, reward=EW_F1
"""
import argparse
import copy
import logging
import os
import time

import numpy as np
import torch

import config
import my_affectgpt.tasks as tasks
from my_affectgpt.common.config import Config
from my_affectgpt.common.registry import registry
from grpo.rewards import get_reward_fn
from grpo.grpo_trainer import GRPOTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='AffectGPT GRPO Training')
    parser.add_argument('--cfg-path', required=True, help='GRPO yaml (模型+数据集配置)')
    parser.add_argument('--sft-ckpt', required=True, help='SFT checkpoint (GRPO 起点)')
    parser.add_argument('--reward', default='ewf1', help='奖励函数: ewf1 (第一版只用它)')
    parser.add_argument('--G', type=int, default=8)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=5e-7)
    parser.add_argument('--eps', type=float, default=0.1, help='GRPO 裁剪 epsilon')
    parser.add_argument('--beta', type=float, default=0.04, help='KL 惩罚系数')
    parser.add_argument('--grad-accum', type=int, default=4)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--eval-every', type=int, default=50)
    parser.add_argument('--eval-samples', type=int, default=50, help='验证集采样条数')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save-root', default='output/grpo', help='GRPO ckpt 输出目录')
    return parser.parse_args()


def build_model_from_cfg(cfg, sft_ckpt):
    """加载 AffectGPT 并载入 SFT checkpoint (ckpt_3 优先级最高)。"""
    model_cfg = copy.deepcopy(cfg.model_cfg)
    model_cfg.ckpt_3 = sft_ckpt
    model_cls = registry.get_model_class(model_cfg.arch)
    model = model_cls.from_config(model_cfg)
    return model


def freeze_model(model):
    """把模型所有参数冻结, 用于参考模型。"""
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def build_human_dataset(cfg, label_csv=None):
    """构建 human 数据集 (默认 train90; 传 label_csv 可指向 test10 做验证)。"""
    old = config.PATH_TO_LABEL['Human']
    if label_csv is not None:
        config.PATH_TO_LABEL['Human'] = label_csv
    try:
        task = tasks.setup_task(cfg)
        datasets = task.build_datasets(cfg)   # 只建数据, 不建模型
    finally:
        config.PATH_TO_LABEL['Human'] = old
    return datasets['human']['train']


def make_eval_fn(trainer, test_dataset, reward_fn, n_samples, gpu):
    """返回 eval_fn(policy) -> 在 test10 上采 n 条, 生成 1 条响应, 算平均 EW-F1。"""
    rng = np.random.RandomState(0)
    def eval_fn(policy):
        policy.eval()
        scores = []
        indices = rng.choice(len(test_dataset), size=min(n_samples, len(test_dataset)), replace=False)
        for i in indices:
            sample = test_dataset.annotation[i]
            try:
                prompt_ids, feats, gt = trainer._build_input(sample)
                texts, _ = trainer._generate(prompt_ids, feats, n=1)
                r = reward_fn(texts[0], gt)
                scores.append(r['reward'])
            except Exception as e:
                logger.warning(f'[eval] skip {sample["name"]}: {e}')
        policy.train()
        return float(np.mean(scores)) if scores else 0.0
    return eval_fn


def main():
    args = parse_args()
    torch.manual_seed(42)
    device = f'cuda:{args.gpu}'
    assert torch.cuda.is_available(), 'GRPO 必须运行在 GPU 上'

    cfg = Config(args)

    # 1) 策略模型 (载入 SFT ckpt) + 参考模型 (同权重, 冻结)
    logger.info('Loading policy model (with SFT checkpoint)...')
    policy = build_model_from_cfg(cfg, args.sft_ckpt).to(device)
    logger.info('Loading reference model (frozen copy)...')
    ref = build_model_from_cfg(cfg, args.sft_ckpt)
    freeze_model(ref)
    ref.to(device)

    # 2) 数据集: GRPO 训练用 human train90; 验证用 test10
    train_dataset = build_human_dataset(cfg)   # config.PATH_TO_LABEL['Human'] = train90
    test_csv = os.path.join(config.DATA_DIR['MER2026'], 'track2_train_human_test10.csv')
    test_dataset = build_human_dataset(cfg, label_csv=test_csv)

    tokenizer = policy.llama_tokenizer
    reward_fn = get_reward_fn(args.reward)
    logger.info(f'Reward function: {args.reward}')

    # 3) GRPO trainer
    trainer = GRPOTrainer(
        policy=policy, ref=ref, tokenizer=tokenizer, dataset=train_dataset,
        reward_fn=reward_fn, device=device,
        G=args.G, temperature=args.temp, lr=args.lr, beta=args.beta, eps=args.eps,
        grad_accum=args.grad_accum, max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
    )
    os.makedirs(args.save_root, exist_ok=True)
    eval_fn = make_eval_fn(trainer, test_dataset, reward_fn, args.eval_samples, args.gpu)

    # 4) 训练
    logger.info(f'Starting GRPO: G={args.G} temp={args.temp} lr={args.lr} eps={args.eps} '
                f'grad_accum={args.grad_accum} max_steps={args.max_steps}')
    t0 = time.time()
    trainer.train(eval_fn=eval_fn, eval_every=args.eval_every, save_root=args.save_root)
    logger.info(f'GRPO done in {time.time() - t0:.0f}s. ckpts -> {args.save_root}')


if __name__ == '__main__':
    main()
