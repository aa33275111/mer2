# *_*coding:utf-8 *_*
"""
GRPO 奖励函数模块。

对应简历: "围绕官方 EW-F1 指标设计并实现多种 Reward Function, 对比 EW-F1、
Precision/Recall 约束等奖励策略, 分析强化学习过程中模型预测偏置及 Precision-Recall 平衡问题。"

统一接口:  reward_fn(pred_text: str, gt_labels: list[str]) -> dict
返回 dict 至少含 'reward' (标量, 越大越好), 并附带 precision/recall/f1 等可分析字段。

实现指标与官方一致 (见 my_affectgpt/evaluation/evaluation.md):
  3 级映射 format_mapping → raw_mapping → wheel_map_whole (5 情感轮),
  openset 重叠率 P/R, F1 = 2PR/(P+R), EW-F1 = 5 轮 F1 平均。
"""
import re

import numpy as np

import config

# ------------------------------------------------------------------ #
# 一次性加载情感轮映射 (词表较大, 全局缓存)
# ------------------------------------------------------------------ #
_MAPPING = None


def get_mapping():
    global _MAPPING
    if _MAPPING is None:
        mp = np.load(config.OUTSIDE_WHEEL_MAPPING, allow_pickle=True)
        _MAPPING = {
            'format_mapping': mp['format_mapping'].tolist(),       # L3 -> L2
            'raw_mapping':    mp['raw_mapping'].tolist(),          # L2 -> L1
            'wheel_map_whole': mp['wheel_map_whole'].tolist(),     # L1 -> 每轮基本类别
        }
    return _MAPPING


def _backward(label, format_mapping, raw_mapping, wheel_map):
    """case3 反向映射: label -> 最内圈情感轮基本类别 (对齐 wheel.py func_backward_case3)。"""
    if label not in format_mapping:
        return ""
    level1_whole = []
    for fmt in format_mapping[label]:
        for raw in raw_mapping.get(fmt, []):
            level1_whole.append(raw)
    for level1 in sorted(level1_whole):  # 保证唯一性
        if level1 in wheel_map:
            return wheel_map[level1]
    return ""


def map_label_list(labels, format_mapping, raw_mapping, wheel_map):
    """把标签列表逐级归并到某个情感轮的基本类别; 不在词表的直接剔除。"""
    out = []
    for lb in labels:
        lb = (lb or '').lower().strip()
        if not lb:
            continue
        mapped = _backward(lb, format_mapping, raw_mapping, wheel_map)
        if mapped:
            out.append(mapped)
    return out


# ------------------------------------------------------------------ #
# 标签解析: 从模型生成的文本里抽出情感标签列表
# ------------------------------------------------------------------ #
def parse_pred_labels(text):
    """兼容多种生成格式:
       "The character's emotional state is happy, sad."
       "[happy, sad]" / "['happy', 'sad']"
       "happy, sad, worried"
    """
    if not text:
        return []
    t = text.strip()
    t = re.sub(r"^The\s+character['’]?s\s+emotional\s+state\s+is\s*:?\s*", '', t, flags=re.I)
    t = re.sub(r"^[Aa]ssistant\s*:?\s*", '', t, flags=re.I)
    t = t.strip()
    if t.startswith('[') and t.endswith(']'):
        t = t[1:-1]
    parts = re.split(r'[,，;；\n|]+', t)
    labels, seen = [], set()
    for p in parts:
        p = p.strip().strip('\'"`。.！!? ')
        p = re.sub(r'^[-*•\d.]+\s*', '', p)
        p = re.sub(r'\s+', ' ', p)
        if p and p.lower() not in seen:
            seen.add(p.lower())
            labels.append(p)
    return labels


# ------------------------------------------------------------------ #
# 单样本打分
# ------------------------------------------------------------------ #
def compute_pair_scores(gt_list, pred_list, level='level1'):
    """单样本在 5 个情感轮上的平均 P/R/F1。"""
    mp = get_mapping()
    fmt, raw = mp['format_mapping'], mp['raw_mapping']
    wheel_map_whole = mp['wheel_map_whole']
    f1s, ps, rs = [], [], []
    for wheel_name in ['wheel1', 'wheel2', 'wheel3', 'wheel4', 'wheel5']:
        wheel_map = wheel_map_whole[wheel_name][level]
        g = set(map_label_list(gt_list, fmt, raw, wheel_map))
        p = set(map_label_list(pred_list, fmt, raw, wheel_map))
        if not g:                       # GT 映射后为空则跳过该轮
            continue
        if not p:                       # 预测无命中 -> P/R/F1 全 0
            f1s.append(0.0); ps.append(0.0); rs.append(0.0)
            continue
        inter = len(g & p)
        precision = inter / len(p)
        recall = inter / len(g)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1); ps.append(precision); rs.append(recall)
    return {
        'f1': float(np.mean(f1s)) if f1s else 0.0,
        'precision': float(np.mean(ps)) if ps else 0.0,
        'recall': float(np.mean(rs)) if rs else 0.0,
        'n_wheels': len(f1s),
    }


# ------------------------------------------------------------------ #
# 多种 Reward Function (面试可讲的核心: 对比不同奖励策略)
# ------------------------------------------------------------------ #
def ewf1_reward(pred_text, gt_labels, level='level1'):
    """官方指标 EW-F1: 5 轮 F1 平均 (level1 为主指标)。"""
    sc = compute_pair_scores(gt_labels, parse_pred_labels(pred_text), level=level)
    return {**sc, 'reward': sc['f1'], 'name': f'ewf1_{level}'}


def ewf1_l2_reward(pred_text, gt_labels):
    """EW-F1 level2 (辅助, 更粗粒度)。"""
    return ewf1_reward(pred_text, gt_labels, level='level2')


def pr_weighted_reward(pred_text, gt_labels, alpha=0.5):
    """P/R 加权: alpha*P + (1-alpha)*R, 可调 P/R 权重观察偏置。"""
    sc = compute_pair_scores(gt_labels, parse_pred_labels(pred_text), level='level1')
    reward = alpha * sc['precision'] + (1 - alpha) * sc['recall']
    return {**sc, 'reward': reward, 'name': f'pr_weighted_a{alpha}'}


def pr_constraint_reward(pred_text, gt_labels):
    """P/R 约束奖励: F1 乘上 P/R 均衡度惩罚项 (1 - |P-R|)。
       鼓励模型兼顾 Precision 与 Recall, 惩罚只堆标签/只报少标签的偏置。
    """
    sc = compute_pair_scores(gt_labels, parse_pred_labels(pred_text), level='level1')
    balance = 1.0 - abs(sc['precision'] - sc['recall'])
    reward = sc['f1'] * balance
    return {**sc, 'reward': reward, 'name': 'pr_constraint'}


def format_reward(pred_text, gt_labels):
    """格式奖励: 能解析出有效标签即有基础分, 有轮命中再加分 (防模型生成废话)。"""
    sc = compute_pair_scores(gt_labels, parse_pred_labels(pred_text), level='level1')
    n_pred = len(parse_pred_labels(pred_text))
    reward = 0.0
    if n_pred > 0:
        reward += 0.2                      # 格式正确基础分
    reward += sc['f1'] * 0.8               # 内容质量分
    return {**sc, 'reward': reward, 'name': 'format'}


def micro_f1_score(pred_text, gt_labels, level='level1'):
    """micro-F1: 5 轮 TP/FP/FN 池化后算一个全局 F1 (区别于 EW-F1 的 macro 平均)。"""
    mp = get_mapping()
    fmt, raw = mp['format_mapping'], mp['raw_mapping']
    wheel_map_whole = mp['wheel_map_whole']
    tp = fp = fn = 0
    for wheel_name in ['wheel1', 'wheel2', 'wheel3', 'wheel4', 'wheel5']:
        wheel_map = wheel_map_whole[wheel_name][level]
        g = set(map_label_list(gt_labels, fmt, raw, wheel_map))
        p = set(map_label_list(parse_pred_labels(pred_text), fmt, raw, wheel_map))
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


def ewf1_format_reward(pred_text, gt_labels, w=0.1):
    """历史 'EW+format': EW_F1 + 0.1 * format(输出可解析即给 format 分)。"""
    sc = compute_pair_scores(gt_labels, parse_pred_labels(pred_text), level='level1')
    fmt = 1.0 if len(parse_pred_labels(pred_text)) > 0 else 0.0
    reward = sc['f1'] + w * fmt
    return {**sc, 'reward': reward, 'format_score': fmt, 'name': f'ewf1_format_w{w}'}


def mu_f1_reward(pred_text, gt_labels, alpha=0.85):
    """历史 'μ_F1': 0.85 * EW_F1 + 0.15 * micro_F1 (macro/micro 平衡)。"""
    sc = compute_pair_scores(gt_labels, parse_pred_labels(pred_text), level='level1')
    mf1 = micro_f1_score(pred_text, gt_labels)
    reward = alpha * sc['f1'] + (1 - alpha) * mf1
    return {**sc, 'reward': reward, 'micro_f1': mf1, 'name': f'mu_f1_a{alpha}'}


def p3_reward(pred_text, gt_labels, power=3):
    """历史 'P3': EW_F1 × P³ (Precision 的三次方), 强惩罚低精度/过度预测。"""
    sc = compute_pair_scores(gt_labels, parse_pred_labels(pred_text), level='level1')
    reward = sc['f1'] * (sc['precision'] ** power)
    return {**sc, 'reward': reward, 'name': f'ewf1_p{power}'}


def get_reward_fn(name):
    """奖励函数工厂。
    name: ewf1 / ewf1_l2 / pr_weighted / pr_constraint / format /
          ewf1_format / mu_f1 / p3
    """
    factory = {
        'ewf1': lambda: ewf1_reward,
        'ewf1_l2': lambda: ewf1_l2_reward,
        'pr_weighted': lambda: pr_weighted_reward,
        'pr_constraint': lambda: pr_constraint_reward,
        'format': lambda: format_reward,
        'ewf1_format': lambda: ewf1_format_reward,
        'mu_f1': lambda: mu_f1_reward,
        'p3': lambda: p3_reward,
    }
    if name not in factory:
        raise ValueError(f'未知 reward: {name}, 可选 {list(factory)}')
    return factory[name]()
