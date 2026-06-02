from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semanticid_llm_rec.data.preprocess import (  # noqa: E402
    load_downloaded_movielens,
    load_local_movielens,
    save_processed,
    split_user_sequences,
)
from semanticid_llm_rec.utils.config import ensure_dirs, load_config, resolve_project_path  # noqa: E402
from semanticid_llm_rec.utils.seed import set_seed  # noqa: E402


def main() -> None:
    """预处理入口：原始 MovieLens -> 统一表结构 + 用户序列切分。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/local.yaml")
    parser.add_argument("--download", action="store_true", help="Download MovieLens-1M if local pickle data is absent.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs()
    set_seed(int(cfg["project"]["seed"]))

    data_root = resolve_project_path(cfg["project"]["data_root"])
    processed_dir = ROOT / "data" / "processed"
    try:
        # 默认读取当前目录已有的 funrec-movielens-1m，不强制联网。
        users, movies, ratings = load_local_movielens(data_root)
        source = str(data_root)
    except FileNotFoundError:
        if not args.download:
            raise
        users, movies, ratings = load_downloaded_movielens(ROOT / "data" / "raw")
        source = str(ROOT / "data" / "raw" / "ml-1m")

    data_cfg = cfg["data"]
    # 按时间线做 train/valid/test，保证评估更接近真实推荐场景。
    sequences = split_user_sequences(
        ratings,
        min_interactions=int(data_cfg["min_interactions"]),
        max_history_len=int(data_cfg["max_history_len"]),
        max_users=data_cfg.get("max_users"),
    )
    save_processed(users, movies, ratings, sequences, processed_dir)
    print(
        f"Processed source={source} users={len(users)} movies={len(movies)} "
        f"ratings={len(ratings)} sequences={len(sequences)} -> {processed_dir}"
    )


if __name__ == "__main__":
    main()
