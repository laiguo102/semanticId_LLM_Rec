from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semanticid_llm_rec.data.negative_sampling import sample_negatives  # noqa: E402
from semanticid_llm_rec.data.preprocess import read_jsonl  # noqa: E402
from semanticid_llm_rec.metrics.ranking import aggregate_ranking_metrics  # noqa: E402
from semanticid_llm_rec.models.heuristic import CooccurrenceRecommender  # noqa: E402
from semanticid_llm_rec.utils.config import ensure_dirs, load_config  # noqa: E402


def main() -> None:
    """本地评估入口：用 leave-one-out 样本计算 Hit/NDCG/MRR。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/local.yaml")
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs()
    processed_dir = ROOT / "data" / "processed"
    movies = pd.read_csv(processed_dir / "movies.csv")
    train = pd.read_csv(processed_dir / "train_interactions.csv")
    samples = read_jsonl(processed_dir / f"{args.split}_samples.jsonl")
    max_users = cfg["eval"].get("max_users")
    if max_users is not None:
        samples = samples[: int(max_users)]

    all_item_ids = movies["movie_id"].astype(int).tolist()
    recommender = CooccurrenceRecommender(train)
    ranked_lists = []
    targets = []
    for row in samples:
        history = [int(x) for x in row["history"]]
        target = int(row["target"])
        # 评估协议：1 个真实目标 + negative_num 个未看过负例，比较目标的排序位置。
        negatives = sample_negatives(
            all_item_ids,
            positive_id=target,
            seen_ids=set(history),
            n=int(cfg["eval"]["negative_num"]),
            seed=int(cfg["project"]["seed"]) + int(row["user_id"]),
        )
        candidates = negatives + [target]
        # 对同一候选集合排序，保证不同模型/策略的指标可比。
        ranked = recommender.rank(history, candidates=candidates, top_k=len(candidates), exclude_seen=True)
        ranked_lists.append(ranked)
        targets.append(target)

    metrics = aggregate_ranking_metrics(ranked_lists, targets, ks=cfg["eval"]["ks"])
    out = {"split": args.split, "users": len(samples), "metrics": metrics, "backend": "local-cooccurrence-fallback"}
    out_path = ROOT / "outputs" / "metrics.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Saved metrics -> {out_path}")


if __name__ == "__main__":
    main()
