from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd


class CooccurrenceRecommender:
    """本地轻量推荐器：用共现统计模拟“根据历史找相似物品”的推荐过程。

    它不是最终生成式模型，而是为了在没有服务器 LoRA/RQ-VAE checkpoint 时，
    本地仍然可以跑 inference/evaluate，便于面试演示完整链路。
    """

    def __init__(self, train_interactions: pd.DataFrame):
        self.popularity = Counter(train_interactions["movie_id"].astype(int).tolist())
        self.cooccur: dict[int, Counter] = defaultdict(Counter)
        for _, group in train_interactions.groupby("user_id"):
            items = group.sort_values("position")["movie_id"].astype(int).tolist()
            # 去重后统计同一个用户历史中任意两个电影的共现次数。
            unique_items = list(dict.fromkeys(items))
            for src in unique_items:
                for dst in unique_items:
                    if src != dst:
                        self.cooccur[src][dst] += 1

    def rank(
        self,
        history: list[int],
        candidates: list[int] | None = None,
        top_k: int = 10,
        exclude_seen: bool = True,
    ) -> list[int]:
        """对候选物品打分排序，默认排除用户已经看过的电影。"""
        seen = set(int(x) for x in history)
        scores: Counter = Counter()
        # 最近行为通常更能代表当前兴趣，这里只取最后 10 个历史 item 做共现扩展。
        for item in history[-10:]:
            scores.update(self.cooccur.get(int(item), Counter()))
        # 加一小部分全局热度，避免冷启动候选完全没有分数。
        for item, count in self.popularity.items():
            scores[int(item)] += 0.01 * count

        if candidates is None:
            pool = set(scores.keys())
        else:
            pool = set(int(x) for x in candidates)
            for item in pool:
                scores[item] += 0.0
        if exclude_seen:
            pool -= seen
        ranked = sorted(pool, key=lambda x: (-scores[x], x))
        return ranked[:top_k]
