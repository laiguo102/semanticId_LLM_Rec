from __future__ import annotations

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except Exception:  # pragma: no cover - exercised only on machines without torch
    torch = None
    nn = None
    F = None


if torch is not None:

    class RQVAE(nn.Module):
        """Residual Quantized VAE：把连续 item embedding 压缩成多层离散 code。

        面试讲法：
        1. encoder 把 384 维 item embedding 映射到 latent space；
        2. 多个 codebook 逐层量化残差，得到可离散化的 Semantic ID；
        3. decoder 尝试重构原 embedding，训练目标是重构误差 + commitment loss。
        """

        def __init__(
            self,
            input_dim: int = 384,
            hidden_dim: int = 256,
            latent_dim: int = 64,
            codebook_num: int = 3,
            codebook_size: int = 256,
        ):
            super().__init__()
            self.codebook_num = codebook_num
            self.codebook_size = codebook_size
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )
            self.codebooks = nn.Parameter(torch.randn(codebook_num, codebook_size, latent_dim) * 0.02)

        def quantize(self, z):
            """逐层残差量化：每一层选择离当前残差最近的 codebook 向量。"""
            residual = z
            quantized = torch.zeros_like(z)
            codes = []
            for level in range(self.codebook_num):
                book = self.codebooks[level]
                # cdist 计算 batch 内每个 latent 与当前 codebook 所有向量的距离。
                distances = torch.cdist(residual.unsqueeze(1), book.unsqueeze(0)).squeeze(1)
                idx = distances.argmin(dim=-1)
                chosen = book[idx]
                # 当前层解释掉一部分语义，剩下的 residual 交给下一层继续量化。
                quantized = quantized + chosen
                residual = residual - chosen
                codes.append(idx)
            return quantized, torch.stack(codes, dim=1)

        def forward(self, x):
            """训练时返回总 loss、重构结果和离散 code。"""
            z = self.encoder(x)
            z_q, codes = self.quantize(z)
            # Straight-through estimator：前向用量化向量，反向近似把梯度传回 encoder。
            z_st = z + (z_q - z).detach()
            recon = self.decoder(z_st)
            recon_loss = F.mse_loss(recon, x)
            commit_loss = F.mse_loss(z.detach(), z_q)
            return {"loss": recon_loss + 0.25 * commit_loss, "recon": recon, "codes": codes}

        @torch.no_grad()
        def encode_codes(self, x):
            """推理/导出 SID 时只需要 encoder + quantize，不需要 decoder。"""
            z = self.encoder(x)
            _, codes = self.quantize(z)
            return codes

else:

    class RQVAE:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required for RQ-VAE training. Install dependencies with uv sync.")
