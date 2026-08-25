# *_*coding:utf-8 *_*
"""
从 track2_train_human.csv (1532 条人工标注) 按 9:1 固定 seed 划分 训练 / 测试 集。

划分协议 (与 evaluation/evaluation.md 保持一致):
  - 90% -> track2_train_human_train90.csv  (GRPO 阶段训练数据)
  - 10% -> track2_train_human_test10.csv    (统一测试集, SFT / GRPO 都用它评测)

用法:
    python split_human_data.py                           # 默认读 config.PATH_TO_LABEL['Human']
    python split_human_data.py --input /path/track2_train_human.csv
    python split_human_data.py --ratio 0.1 --seed 42 --output-dir /some/dir

说明:
  - 固定 seed=42, 保证可复现; 任何一次重跑都得到完全相同的一对 csv。
  - 基于 name 去重后按样本(行)随机打乱, 保证 train/test 无交集。
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
    parser = argparse.ArgumentParser(description="MER2026 Track2 human 数据 9:1 划分")
    parser.add_argument("--input", default=None,
                        help="track2_train_human.csv 路径; 默认取 config.PATH_TO_LABEL['Human']")
    parser.add_argument("--ratio", type=float, default=0.1,
                        help="测试集比例, 默认 0.1 (即 9:1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子, 默认 42")
    parser.add_argument("--output-dir", default=None,
                        help="输出目录; 默认与输入文件同目录")
    args = parser.parse_args()

    # ---------- 读取输入 ----------
    input_csv = args.input or config.PATH_TO_LABEL["Human"]
    if input_csv.startswith("xxx") or not os.path.exists(input_csv):
        print(f"[WARN] 找不到输入文件: {input_csv}")
        print("       请在 config.py 中把 DATA_DIR['MER2026'] 改为实际数据集路径, 或用 --input 指定。")
        sys.exit(1)

    df = pd.read_csv(input_csv)
    assert "name" in df.columns and "openset" in df.columns, \
        f"csv 必须包含 name / openset 字段, 实际列: {list(df.columns)}"
    print(f"[INFO] 读取 {input_csv}")
    print(f"[INFO] 总样本数: {len(df)}")

    # ---------- 按 name 去重检查 ----------
    dup = df[df.duplicated("name", keep=False)]
    if len(dup) > 0:
        print(f"[WARN] 发现 {dup['name'].nunique()} 个重复 name, 共 {len(dup)} 行 (按首次出现保留)")
        df = df.drop_duplicates("name", keep="first")

    # ---------- 固定 seed 随机划分 ----------
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(df))
    n_test = int(round(len(df) * args.ratio))
    n_test = max(1, min(n_test, len(df) - 1))  # 至少留 1 条做测试, 至少留 1 条做训练
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # ---------- 无重叠检查 ----------
    inter = set(train_df["name"]) & set(test_df["name"])
    assert len(inter) == 0, f"train/test 存在 {len(inter)} 个重叠 name, 划分失败!"

    # ---------- 保存 ----------
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(input_csv))
    os.makedirs(output_dir, exist_ok=True)
    train_csv = os.path.join(output_dir, "track2_train_human_train90.csv")
    test_csv = os.path.join(output_dir, "track2_train_human_test10.csv")
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print("=" * 60)
    print(f"[DONE] 划分完成 (seed={args.seed}, test_ratio={args.ratio}):")
    print(f"  train: {len(train_df)} 条  -> {train_csv}")
    print(f"  test : {len(test_df)} 条  -> {test_csv}")
    print(f"  重叠样本: {len(inter)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
