from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semanticid_llm_rec.models.semantic_id import build_unique_sid_map, kmeans_residual_codes  # noqa: E402
from semanticid_llm_rec.utils.config import ensure_dirs, load_config  # noqa: E402


def encode_with_rqvae_checkpoint(embeddings: np.ndarray, cfg: dict, ckpt_path: Path) -> np.ndarray:
    """加载服务器训练好的 RQ-VAE checkpoint，把 embedding 编码成离散 code。"""
    try:
        import torch

        from semanticid_llm_rec.models.rqvae import RQVAE
    except Exception as exc:
        raise RuntimeError("torch is required to encode Semantic IDs from an RQ-VAE checkpoint.") from exc

    rq_cfg = cfg["rqvae"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RQVAE(
        input_dim=int(rq_cfg["input_dim"]),
        hidden_dim=int(rq_cfg["hidden_dim"]),
        latent_dim=int(rq_cfg["latent_dim"]),
        codebook_num=int(rq_cfg["codebook_num"]),
        codebook_size=int(rq_cfg["codebook_size"]),
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    batches = []
    with torch.no_grad():
        # 分批编码避免一次性把全部 item embedding 放进显存。
        for start in range(0, len(embeddings), 1024):
            batch = torch.from_numpy(embeddings[start : start + 1024].astype("float32")).to(device)
            batches.append(model.encode_codes(batch).cpu().numpy())
    return np.vstack(batches)


def main() -> None:
    """Semantic ID 构建入口：item embedding -> movie_id 与 SID 的双向映射。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/local.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs()
    processed_dir = ROOT / "data" / "processed"
    semantic_dir = ROOT / "data" / "semantic_id"
    out_path = semantic_dir / "movie_sid_map.json"
    if out_path.exists() and not args.force:
        print(f"Semantic ID map already exists: {out_path}")
        return

    emb_path = processed_dir / "item_embedding.npy"
    ids_path = processed_dir / "item_ids.npy"
    if not emb_path.exists() or not ids_path.exists():
        raise FileNotFoundError("Run scripts/build_embedding.py before building Semantic IDs.")

    embeddings = np.load(emb_path)
    movie_ids = np.load(ids_path).astype(int).tolist()
    rq_cfg = cfg["rqvae"]
    ckpt_path = ROOT / "checkpoints" / "rqvae" / "model.pt"
    if ckpt_path.exists():
        # 有服务器产物时，使用真正的 RQ-VAE codebook 生成 SID。
        codes = encode_with_rqvae_checkpoint(embeddings, cfg, ckpt_path)
        backend = f"rqvae-checkpoint:{ckpt_path}"
    else:
        if not bool(rq_cfg.get("allow_kmeans_fallback", False)):
            raise FileNotFoundError(
                "RQ-VAE checkpoint not found. Run scripts/train_rqvae.py first, or enable "
                "rqvae.allow_kmeans_fallback for local smoke demos."
            )
        # 本地无 checkpoint 时用残差 KMeans 兜底，保证 demo 可以完成。
        codes = kmeans_residual_codes(
            embeddings,
            codebook_num=int(rq_cfg["codebook_num"]),
            codebook_size=int(rq_cfg["codebook_size"]),
            seed=int(cfg["project"]["seed"]),
        )
        backend = "kmeans-residual-fallback"
    sid_map = build_unique_sid_map(movie_ids, codes)
    # 生成反向表，推理时可以从模型输出的 SID 找回 movie_id。
    sid_movie_map = {sid: movie_id for movie_id, sid in sid_map.items()}

    semantic_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sid_map, ensure_ascii=False, indent=2), encoding="utf-8")
    (semantic_dir / "sid_movie_map.json").write_text(
        json.dumps(sid_movie_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (semantic_dir / "semantic_ids.jsonl").open("w", encoding="utf-8") as f:
        for movie_id, sid in sorted(sid_map.items(), key=lambda x: int(x[0])):
            f.write(json.dumps({"movie_id": int(movie_id), "semantic_id": sid}, ensure_ascii=False) + "\n")
    print(f"Saved {len(sid_map)} unique Semantic IDs backend={backend} -> {out_path}")


if __name__ == "__main__":
    main()