from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semanticid_llm_rec.models.rqvae import RQVAE  # noqa: E402
from semanticid_llm_rec.utils.config import ensure_dirs, load_config  # noqa: E402
from semanticid_llm_rec.utils.seed import set_seed  # noqa: E402


def main() -> None:
    """服务器训练入口：用 item embedding 训练 RQ-VAE，并保存 codebook/encoder。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/server.yaml")
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        raise RuntimeError("torch is required for RQ-VAE training. Run `uv sync` on the server.") from exc

    cfg = load_config(args.config)
    ensure_dirs()
    set_seed(int(cfg["project"]["seed"]))

    emb_path = ROOT / "data" / "processed" / "item_embedding.npy"
    if not emb_path.exists():
        raise FileNotFoundError("Run scripts/build_embedding.py before train_rqvae.py")

    embeddings = np.load(emb_path).astype("float32")
    rq_cfg = cfg["rqvae"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # RQ-VAE 学到的离散 code 会在 build_semantic_id.py 中转成 <sid*> token。
    model = RQVAE(
        input_dim=int(rq_cfg["input_dim"]),
        hidden_dim=int(rq_cfg["hidden_dim"]),
        latent_dim=int(rq_cfg["latent_dim"]),
        codebook_num=int(rq_cfg["codebook_num"]),
        codebook_size=int(rq_cfg["codebook_size"]),
    ).to(device)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(embeddings)),
        batch_size=int(rq_cfg["batch_size"]),
        shuffle=True,
    )
    optim = torch.optim.AdamW(model.parameters(), lr=float(rq_cfg["lr"]))
    model.train()
    for epoch in range(int(rq_cfg["epochs"])):
        total = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{rq_cfg['epochs']}", leave=False)

        for (batch,) in loader:
            batch = batch.to(device)
            # 目标是重构 item embedding，同时让 encoder 输出贴近选中的 codebook 向量。
            optim.zero_grad(set_to_none=True)
            out = model(batch)
            out["loss"].backward()
            optim.step()
            total += float(out["loss"].item()) * len(batch)
        print(f"epoch={epoch + 1} loss={total / len(embeddings):.6f}")

    ckpt_dir = ROOT / "checkpoints" / "rqvae"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "model.pt")
    (ckpt_dir / "config.json").write_text(json.dumps(rq_cfg, indent=2), encoding="utf-8")
    print(f"Saved RQ-VAE checkpoint -> {ckpt_dir}")


if __name__ == "__main__":
    main()
