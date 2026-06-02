from __future__ import annotations

import random


def sample_negatives(
    all_item_ids: list[int],
    positive_id: int,
    seen_ids: set[int],
    n: int = 99,
    seed: int = 2026,
) -> list[int]:
    """为 leave-one-out 评估采样负例。

    每个测试样本由 1 个真实 target + n 个负例组成，模型需要把真实 target 排到前面。
    """
    rng = random.Random(seed + int(positive_id))
    blocked = set(seen_ids)
    blocked.add(int(positive_id))
    pool = [int(i) for i in all_item_ids if int(i) not in blocked]
    if len(pool) <= n:
        return pool
    return rng.sample(pool, n)
