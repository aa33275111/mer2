# *_*coding:utf-8 *_*
"""
SFT 小样本冒烟测试: 验证 数据 → 模型加载 → 训练步 → 生成式 EW-F1 验证 → 存 checkpoint 全链路。

用 mini csv (12 训练 + 6 验证样本), 2 epoch × 3 iter, 跑通即代表改动无误。

用法 (GPU 机器, 本机即 2×A100 80G):
    CUDA_VISIBLE_DEVICES=0 python result/scripts/sft_smoke_test.py
"""
import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg-path', default='train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz.yaml')
    parser.add_argument('--n-train', type=int, default=12)
    parser.add_argument('--n-val', type=int, default=6)
    args = parser.parse_args()

    # 1) 造 mini csv, 并临时把 config 指向它们
    data_dir = config.DATA_DIR['MER2026']
    train_full = pd.read_csv(config.PATH_TO_LABEL['MERCaptionPlus'])
    val_full = pd.read_csv(config.PATH_TO_LABEL['MERCaptionPlusVal'])
    mini_train = train_full.iloc[:args.n_train].reset_index(drop=True)
    mini_val = val_full.iloc[:args.n_val].reset_index(drop=True)
    smoke_train = os.path.join(data_dir, '_smoke_train.csv')
    smoke_val = os.path.join(data_dir, '_smoke_val.csv')
    mini_train.to_csv(smoke_train, index=False)
    mini_val.to_csv(smoke_val, index=False)
    config.PATH_TO_LABEL['MERCaptionPlus'] = smoke_train
    config.PATH_TO_LABEL['MERCaptionPlusVal'] = smoke_val
    config.EVAL_N_SAMPLES = args.n_val   # 验证只生成 n_val 条, 冒烟够用
    print(f'[1/5] mini csv 就绪: train={args.n_train} val={args.n_val}')

    # 2) 加载配置并覆盖为小样本训练参数
    from my_affectgpt.common.config import Config
    options = [
        'run.max_epoch=2', 'run.iters_per_epoch=3',
        'run.batch_size_train=2', 'run.batch_size_eval=2',
        'run.eval_interval=2', 'run.num_workers=0',
        'run.world_size=1', 'run.distributed=False', 'run.device=cuda:0',
        'run.resume_ckpt_path=null',
    ]
    ns = argparse.Namespace(cfg_path=args.cfg_path, options=options)
    cfg = Config(ns)

    # 3) 构建 task / datasets / model
    import my_affectgpt.tasks as tasks
    from my_affectgpt.common.logger import setup_logger
    from my_affectgpt.runners.runner_base import RunnerBase
    setup_logger()  # 让 runner 的 logging 输出到 stdout, 便于看到验证日志
    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    print(f'[2/5] 数据集构建 OK: {list(datasets.keys())} -> splits {list(datasets["mercaptionplus"].keys())}')
    model = task.build_model(cfg)   # 注意: 不要提前 .to(device), runner 的 model 属性会在需要时移动
    print('[3/5] 模型加载 OK (含 LoRA)')

    # 4) 跑 2 个 epoch
    runner = RunnerBase(cfg=cfg, job_id='smoke_test', task=task, model=model, datasets=datasets)
    runner.train()
    print('[5/5] SFT 冒烟测试跑通!')


if __name__ == '__main__':
    main()
