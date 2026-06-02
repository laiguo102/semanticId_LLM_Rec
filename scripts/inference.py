from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semanticid_llm_rec.data.preprocess import read_jsonl  # noqa: E402
from semanticid_llm_rec.models.heuristic import CooccurrenceRecommender  # noqa: E402
from semanticid_llm_rec.utils.config import ensure_dirs, load_config  # noqa: E402


def main() -> None:
    """本地推理入口：给定 user_id，输出 Top-K 推荐结果。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/local.yaml")
    parser.add_argument("--user_id", type=int, required=True)
    parser.add_argument("--top_k", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs()
    processed_dir = ROOT / "data" / "processed"
    semantic_dir = ROOT / "data" / "semantic_id"
    movies = pd.read_csv(processed_dir / "movies.csv")
    train = pd.read_csv(processed_dir / "train_interactions.csv")
    sequences = {int(x["user_id"]): x for x in read_jsonl(processed_dir / "user_sequences.jsonl")}
    sid_map_path = semantic_dir / "movie_sid_map.json"
    sid_map = json.loads(sid_map_path.read_text(encoding="utf-8")) if sid_map_path.exists() else {}

    if args.user_id not in sequences:
        raise ValueError(f"user_id={args.user_id} not found in processed sequences.")
    # 当前本地版使用训练历史作为输入；完整 LLM 推理可替换为“历史 SID -> 生成 SID -> 映射电影”。
    history = [int(x) for x in sequences[args.user_id]["train"]]
    top_k = args.top_k or int(cfg["inference"]["top_k"])

    recommender = CooccurrenceRecommender(train)
    ranked = recommender.rank(history, top_k=top_k)
    movie_by_id = movies.set_index("movie_id").to_dict(orient="index")
    rows = []
    for rank, movie_id in enumerate(ranked, start=1):
        # 推荐结果同时带上 title/genres/SID，方便面试时展示可解释链路。
        info = movie_by_id.get(movie_id, {})
        rows.append(
            {
                "rank": rank,
                "movie_id": int(movie_id),
                "title": info.get("title", ""),
                "genres": info.get("genres", ""),
                "semantic_id": sid_map.get(str(movie_id), ""),
            }
        )

    out = {
        "user_id": args.user_id,
        "history_movie_ids": history,
        "recommendations": rows,
        "backend": "local-cooccurrence-fallback",
    }
    out_path = ROOT / "outputs" / "recommendations.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Saved recommendations -> {out_path}")


if __name__ == "__main__":
    main()
