from __future__ import annotations

from semanticid_llm_rec.metrics.ranking import aggregate_ranking_metrics, hit_at_k, ndcg_at_k, reciprocal_rank


def test_ranking_metrics_single_list():
    ranked = [10, 20, 30, 40]
    assert hit_at_k(ranked, 30, 2) == 0.0
    assert hit_at_k(ranked, 30, 3) == 1.0
    assert round(ndcg_at_k(ranked, 30, 3), 6) == round(1 / 2.0, 6)
    assert reciprocal_rank(ranked, 30) == 1 / 3


def test_aggregate_ranking_metrics():
    metrics = aggregate_ranking_metrics([[1, 2, 3], [4, 5, 6]], [1, 6], ks=[1, 3])
    assert metrics["Hit@1"] == 0.5
    assert metrics["Hit@3"] == 1.0
    assert metrics["MRR"] == (1.0 + 1 / 3) / 2
