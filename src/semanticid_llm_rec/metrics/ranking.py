from __future__ import annotations

import math


def hit_at_k(ranked_items: list[int], target: int, k: int) -> float:
    """Hit@K：真实目标是否出现在推荐列表前 K 个。"""
    return 1.0 if int(target) in [int(x) for x in ranked_items[:k]] else 0.0


def ndcg_at_k(ranked_items: list[int], target: int, k: int) -> float:
    """NDCG@K：命中了也区分排名位置，越靠前得分越高。"""
    for idx, item in enumerate(ranked_items[:k]):
        if int(item) == int(target):
            return 1.0 / math.log2(idx + 2)
    return 0.0


def reciprocal_rank(ranked_items: list[int], target: int) -> float:
    """MRR 的单样本分数：真实目标排名的倒数。"""
    for idx, item in enumerate(ranked_items):
        if int(item) == int(target):
            return 1.0 / float(idx + 1)
    return 0.0


def aggregate_ranking_metrics(
    ranked_lists: list[list[int]],
    targets: list[int],
    ks: list[int] | tuple[int, ...] = (5, 10, 20),
) -> dict[str, float]:
    """聚合一批用户样本的 Hit/NDCG/MRR。"""
    if len(ranked_lists) != len(targets):
        raise ValueError("ranked_lists and targets must have the same length")
    total = max(len(targets), 1)
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"Hit@{k}"] = sum(hit_at_k(r, t, k) for r, t in zip(ranked_lists, targets)) / total
        metrics[f"NDCG@{k}"] = sum(ndcg_at_k(r, t, k) for r, t in zip(ranked_lists, targets)) / total
    metrics["MRR"] = sum(reciprocal_rank(r, t) for r, t in zip(ranked_lists, targets)) / total
    return metrics
