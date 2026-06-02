from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semanticid_llm_rec.data.preprocess import read_jsonl, write_jsonl  # noqa: E402
from semanticid_llm_rec.data.sft import build_sft_rows  # noqa: E402
from semanticid_llm_rec.utils.config import ensure_dirs, load_config  # noqa: E402


def main() -> None:
    """SFT 数据构造入口：用户历史 SID -> 下一个 SID。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/local.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs()
    processed_dir = ROOT / "data" / "processed"
    sid_path = ROOT / "data" / "semantic_id" / "movie_sid_map.json"
    if not sid_path.exists():
        raise FileNotFoundError("Run scripts/build_semantic_id.py before building SFT data.")

    sequences = read_jsonl(processed_dir / "user_sequences.jsonl")
    movie_sid_map = json.loads(sid_path.read_text(encoding="utf-8"))
    # 每个训练样本都长成：prompt=历史 SID 序列，response=目标电影 SID。
    rows = build_sft_rows(
        sequences,
        movie_sid_map,
        max_history_len=int(cfg["data"]["max_history_len"]),
        max_examples=cfg["data"].get("max_sft_examples"),
    )
    out_path = processed_dir / "sft_train.jsonl"
    write_jsonl(out_path, rows)
    print(f"Saved SFT rows={len(rows)} -> {out_path}")


if __name__ == "__main__":
    main()
