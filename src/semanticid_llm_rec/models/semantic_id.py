from __future__ import annotations

import hashlib
import re
from collections import defaultdict

import numpy as np


SID_RE = re.compile(r"<sid([123c])_(\d+)>")


def sid_from_codes(codes: list[int] | tuple[int, ...], collision_id: int | None = None) -> str:
    """把多层量化 code 转成人类可读、LLM 可生成的 Semantic ID token 序列。"""
    tokens = [f"<sid{i + 1}_{int(code)}>" for i, code in enumerate(codes[:3])]
    if collision_id is not None and collision_id > 0:
        tokens.append(f"<sidc_{int(collision_id)}>")
    return " ".join(tokens)


def parse_sid(text: str) -> tuple[tuple[int, int, int], int]:
    """从模型输出文本中解析 Semantic ID，返回三级 code 和冲突后缀。"""
    parts = SID_RE.findall(text)
    values: dict[str, int] = {}
    for level, value in parts:
        values[level] = int(value)
    if not {"1", "2", "3"}.issubset(values):
        raise ValueError(f"Invalid Semantic ID: {text}")
    return (values["1"], values["2"], values["3"]), values.get("c", 0)


def build_unique_sid_map(movie_ids: list[int], codes: np.ndarray) -> dict[str, str]:
    """构建 movie_id -> Semantic ID 映射，并用 sidc 后缀解决 code 冲突。

    RQ-VAE/KMeans 可能把多个物品量化到同一个三元组。生成式推荐要求 SID 能唯一映射回物品，
    所以同桶内第 2 个及之后的电影会追加 <sidc_n>。
    """
    if codes.shape[0] != len(movie_ids):
        raise ValueError("codes row count must match movie_ids length")
    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for movie_id, row in zip(movie_ids, codes):
        buckets[tuple(int(x) for x in row[:3])].append(int(movie_id))

    sid_map: dict[str, str] = {}
    for code, ids in buckets.items():
        for collision_idx, movie_id in enumerate(sorted(ids)):
            sid_map[str(movie_id)] = sid_from_codes(code, collision_idx if collision_idx else None)
    return sid_map


def deterministic_hash_embeddings(texts: list[str], dim: int = 384) -> np.ndarray:
    """本地兜底 embedding：不依赖网络/模型下载，但同一文本每次结果稳定。

    它不追求语义质量，只用于 smoke/local 环境验证后续 SID、SFT、推理和评估链路。
    """
    rows = []
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little", signed=False)
        rng = np.random.default_rng(seed)
        vec = rng.normal(size=dim).astype("float32")
        vec /= max(float(np.linalg.norm(vec)), 1e-12)
        rows.append(vec)
    return np.vstack(rows)


def kmeans_residual_codes(
    embeddings: np.ndarray,
    codebook_num: int = 3,
    codebook_size: int = 256,
    seed: int = 2026,
) -> np.ndarray:
    """用残差 KMeans 模拟 RQ-VAE 的多级量化，便于没有 checkpoint 时本地演示。

    每一级 KMeans 拟合当前残差，下一层继续量化剩余信息，最终得到形如 [sid1, sid2, sid3] 的 code。
    """
    from sklearn.cluster import MiniBatchKMeans

    x = embeddings.astype("float32")
    residual = x.copy()
    all_codes = []
    for level in range(codebook_num):
        # 数据量小于 codebook_size 时自动缩小簇数，避免 KMeans 报错。
        n_clusters = min(int(codebook_size), len(residual))
        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=seed + level,
            batch_size=max(4096, min(8192, len(residual))),
            n_init=3,
            max_iter=100,
        )
        labels = km.fit_predict(residual)
        all_codes.append(labels.astype("int64"))
        residual = residual - km.cluster_centers_[labels]
    return np.vstack(all_codes).T
