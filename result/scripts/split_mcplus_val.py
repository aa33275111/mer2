# *_*coding:utf-8 *_*
"""
从 mc+ 去重训练集里划出一小块验证集 (用于 SFT 训练中选择最优 epoch / 监控)。

协议 (2026-08-25):
  - 输入: track2_train_mercaptionplus_dedup.csv (31,276, 已剔除与 test10 重叠样本)
  - 从中按 seed=42 随机划 ~1% 作验证集 (默认 300 条)
  - 输出:
      track2_train_mercaptionplus_train.csv  (训练集, 剩余 ≈30,976)
      track2_train_mercaptionplus_val.csv    (验证集, ≈300)
  - 保证: 验证集与 test10 无重叠 (来源 dedup 已排除), 与 mc+ 训练集无重叠

用法:
    python split_mcplus_val.py [--val-size 300] [--seed 42]
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # MER2026_Track2
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="mc+ 划出验证集")
    parser.add_argument("--input", default=None,
                        help="mc+ 去重 csv; 默认 config.PATH_TO_LABEL['MERCaptionPlus'](去重版)")
    parser.add_argument("--val-size", type=int, default=300, help="验证集条数, 默认 300")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_csv = args.input or config.PATH_TO_LABEL["MERCaptionPlus"]
    assert os.path.exists(input_csv), f"找不到输入: {input_csv} (请先跑 dedup_mcplus.py)"
    out_dir = os.path.dirname(input_csv)

    df = pd.read_csv(input_csv)
    assert len(df) > args.val_size, "mc+ 去重集太小, 无法划分"
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(df))
    val_idx, train_idx = perm[:args.val_size], perm[args.val_size:]
    train_df, val_df = df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)

    train_csv = os.path.join(out_dir, "track2_train_mercaptionplus_train.csv")
    val_csv = os.path.join(out_dir, "track2_train_mercaptionplus_val.csv")
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    # 无重叠检查
    inter = set(train_df["name"]) & set(val_df["name"])
    test10 = set(pd.read_csv(os.path.join(out_dir, "track2_train_human_test10.csv"))["name"])
    inter_test = set(val_df["name"]) & test10
    print("=" * 60)
    print(f"mc+ 去重集: {len(df)} 条")
    print(f"训练集: {len(train_df)}  -> {train_csv}")
    print(f"验证集: {len(val_df)}  -> {val_csv}")
    print(f"train∩val: {len(inter)} | val∩test10: {len(inter_test)} (都应=0)")
    print("=" * 60)


if __name__ == "__main__":
    main()
