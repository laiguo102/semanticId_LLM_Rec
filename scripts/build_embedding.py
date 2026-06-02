from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semanticid_llm_rec.data.preprocess import build_item_texts  # noqa: E402
from semanticid_llm_rec.models.semantic_id import deterministic_hash_embeddings  # noqa: E402
from semanticid_llm_rec.utils.config import ensure_dirs, load_config  # noqa: E402


def encode_with_sentence_transformer(texts: list[str], model_name: str, batch_size: int) -> np.ndarray:
    """用语义模型把 title + genres 编码成 item embedding。"""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")


def main() -> None:
    """embedding 构建入口：电影文本 -> item_embedding.npy。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/local.yaml")
    parser.add_argument("--mode", choices=["local", "server"], default="local")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs()
    processed_dir = ROOT / "data" / "processed"
    out_path = processed_dir / "item_embedding.npy"
    ids_path = processed_dir / "item_ids.npy"
    if out_path.exists() and ids_path.exists() and not args.force:
        print(f"Embedding already exists: {out_path}")
        return

    movies_path = processed_dir / "movies.csv"
    if not movies_path.exists():
        raise FileNotFoundError("Run scripts/preprocess.py before building embeddings.")
    movies = pd.read_csv(movies_path)
    texts = build_item_texts(movies)
    emb_cfg = cfg["embedding"]
    allow_fallback = bool(emb_cfg.get("allow_hash_fallback", False)) and args.mode == "local"
    try:
        # server/local 都优先走真正的 SentenceTransformer embedding。
        embeddings = encode_with_sentence_transformer(
            texts,
            emb_cfg["model_name"],
            int(emb_cfg["batch_size"]),
        )
        backend = emb_cfg["model_name"]
    except Exception as exc:
        if not allow_fallback:
            raise RuntimeError(
                "SentenceTransformer embedding failed. On the server, install dependencies and "
                "ensure the model can be downloaded or is cached locally."
            ) from exc
        # 本地演示允许 hash fallback，确保没有网络或模型缓存时仍能跑完整流程。
        embeddings = deterministic_hash_embeddings(texts, dim=int(emb_cfg["dim"]))
        backend = "deterministic-hash-fallback"

    np.save(out_path, embeddings.astype("float32"))
    np.save(ids_path, movies["movie_id"].astype(int).to_numpy())
    print(f"Saved embeddings shape={embeddings.shape} backend={backend} -> {out_path}")


if __name__ == "__main__":
    main()
