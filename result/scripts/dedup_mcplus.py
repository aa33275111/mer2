# *_*coding:utf-8 *_*
"""
从 MERCaption+ 训练集剔除与测试集 (human 10%) 重叠的样本, 保证 SFT 训练无数据泄漏。

背景 (2026-08-25):
  track2_train_human.csv 按 9:1 划出的 test10 里, 有 51 条 (33%) 同时出现在
  track2_train_mercaptionplus.csv 中 (同一视频被人工标注集与自动标注集重复收录)。
  若 SFT 直接在全量 mc+ 上训练, 模型训练时已"看过"这 51 条测试样本, EW-F1 会偏乐观。

本脚本: 剔除与 test10 重叠的 mc+ 样本, 输出泄漏无关的训练集。

用法:
    python dedup_mcplus.py
    python dedup_mcplus.py --mcplus /path/...csv --test10 /path/...csv --output /path/...csv
"""
import os
import sys
import argparse

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # MER2026_Track2
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="mc+ 训练集剔除与 test10 重叠的样本")
    parser.add_argument("--mcplus", default=None, help="mc+ 原始 csv; 默认 config.PATH_TO_LABEL['MERCaptionPlus']")
    parser.add_argument("--test10", default=None,
                        help="测试集 csv; 默认 {DATA_DIR}/track2_train_human_test10.csv")
    parser.add_argument("--output", default=None,
                        help="输出去重后 csv; 默认 {DATA_DIR}/track2_train_mercaptionplus_dedup.csv")
    args = parser.parse_args()

    mcplus_csv = args.mcplus or config.PATH_TO_LABEL["MERCaptionPlus"]
    test10_csv = args.test10 or os.path.join(config.DATA_DIR["MER2026"], "track2_train_human_test10.csv")
    out_csv = args.output or os.path.join(config.DATA_DIR["MER2026"], "track2_train_mercaptionplus_dedup.csv")

    for p, tag in [(mcplus_csv, "mc+"), (test10_csv, "test10")]:
        assert os.path.exists(p), f"[ERROR] 找不到 {tag}: {p}"

    mc = pd.read_csv(mcplus_csv)
    test10 = set(pd.read_csv(test10_csv)["name"])
    before = len(mc)

    overlap = mc[mc["name"].isin(test10)]
    mc_dedup = mc[~mc["name"].isin(test10)].reset_index(drop=True)
    mc_dedup.to_csv(out_csv, index=False)

    print("=" * 60)
    print(f"mc+ 原始样本:  {before}")
    print(f"与 test10 重叠: {len(overlap)} 条 (已剔除)")
    print(f"去重后训练集:  {len(mc_dedup)} 条")
    print(f"输出: {out_csv}")
    # 交叉验证
    again = set(mc_dedup["name"]) & test10
    print(f"去重后残留重叠: {len(again)} (应为 0)")
    print("=" * 60)


if __name__ == "__main__":
    main()
