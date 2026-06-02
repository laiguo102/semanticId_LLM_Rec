from __future__ import annotations

import json
import pickle
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd


MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def _read_pickle(path: Path):
    """读取当前数据目录里已经整理好的 pickle 文件。"""
    with path.open("rb") as f:
        return pickle.load(f)


def load_local_movielens(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """优先使用本地的 funrec-movielens-1m 数据，避免每次联网下载。"""
    users_path = data_root / "users.pkl"
    movies_path = data_root / "movies.pkl"
    ratings_path = data_root / "ratings.pkl"
    if not (users_path.exists() and movies_path.exists() and ratings_path.exists()):
        missing = [str(p) for p in [users_path, movies_path, ratings_path] if not p.exists()]
        raise FileNotFoundError(f"MovieLens pickle files are missing: {missing}")

    users = _read_pickle(users_path)
    movies = _read_pickle(movies_path)
    ratings = _read_pickle(ratings_path)
    return normalize_users(users), normalize_movies(movies), normalize_ratings(ratings)


def download_movielens_1m(raw_dir: Path) -> Path:
    """兜底下载官方 MovieLens-1M 数据；本项目默认不依赖这一步。"""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "ml-1m.zip"
    extract_dir = raw_dir / "ml-1m"
    if extract_dir.exists():
        return extract_dir
    urllib.request.urlretrieve(MOVIELENS_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    return extract_dir


def load_downloaded_movielens(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """把官方 dat 格式读取成和本地 pickle 一致的 DataFrame 结构。"""
    data_dir = download_movielens_1m(raw_dir)
    users = pd.read_csv(
        data_dir / "users.dat",
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        encoding="latin-1",
    )
    movies = pd.read_csv(
        data_dir / "movies.dat",
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )
    ratings = pd.read_csv(
        data_dir / "ratings.dat",
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    movies["description"] = ""
    return normalize_users(users), normalize_movies(movies), normalize_ratings(ratings)


def normalize_users(users: pd.DataFrame) -> pd.DataFrame:
    """统一用户表字段类型，后续 join / groupby 时不会因为类型不一致出错。"""
    users = users.copy()
    users["user_id"] = users["user_id"].astype(int)
    return users


def normalize_movies(movies: pd.DataFrame) -> pd.DataFrame:
    """清洗电影表，保证 title/genres/description 至少是可拼接的字符串。"""
    movies = movies.copy()
    movies["movie_id"] = movies["movie_id"].astype(int)
    movies["title"] = movies["title"].fillna("").astype(str)
    movies["genres"] = movies["genres"].fillna("").astype(str)
    if "description" not in movies.columns:
        movies["description"] = ""
    movies["description"] = movies["description"].fillna("").astype(str)
    return movies.drop_duplicates("movie_id").sort_values("movie_id").reset_index(drop=True)


def normalize_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """清洗评分表，并按用户时间线排序，便于后续做序列推荐切分。"""
    ratings = ratings.copy()
    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["movie_id"] = ratings["movie_id"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)
    ratings["timestamp"] = ratings["timestamp"].astype(int)
    return ratings.sort_values(["user_id", "timestamp", "movie_id"]).reset_index(drop=True)


def split_user_sequences(
    ratings: pd.DataFrame,
    min_interactions: int = 5,
    max_history_len: int = 50,
    max_users: int | None = None,
) -> list[dict]:
    """按时间切分用户序列：历史用于训练，倒数第二个做验证，最后一个做测试。

    这是推荐系统里常见的 leave-one-out 协议，能模拟“根据过去行为预测下一次点击/观看”。
    """
    counts = ratings.groupby("user_id")["movie_id"].size()
    keep_users = counts[counts >= min_interactions].index
    filtered = ratings[ratings["user_id"].isin(keep_users)].copy()
    if max_users is not None:
        selected = sorted(filtered["user_id"].unique())[:max_users]
        filtered = filtered[filtered["user_id"].isin(selected)]

    sequences: list[dict] = []
    for user_id, group in filtered.groupby("user_id", sort=True):
        # 每个用户内部必须严格按时间排序，否则验证/测试会泄露未来信息。
        items = group.sort_values(["timestamp", "movie_id"])["movie_id"].astype(int).tolist()
        if len(items) < min_interactions:
            continue
        train_items = items[:-2]
        valid_item = items[-2]
        test_item = items[-1]
        sequences.append(
            {
                "user_id": int(user_id),
                "train": train_items[-max_history_len:],
                "valid": int(valid_item),
                "test": int(test_item),
                "full": items[-max_history_len:],
            }
        )
    return sequences


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """写 JSONL：一行一个样本，适合大规模训练数据流式读取。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    """读取 JSONL 文件，空行会被忽略。"""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_processed(
    users: pd.DataFrame,
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    sequences: list[dict],
    processed_dir: Path,
) -> None:
    """保存预处理产物，供 embedding、SFT 构造、推理和评估复用。"""
    processed_dir.mkdir(parents=True, exist_ok=True)
    users.to_csv(processed_dir / "users.csv", index=False)
    movies.to_csv(processed_dir / "movies.csv", index=False)
    ratings.to_csv(processed_dir / "ratings.csv", index=False)
    write_jsonl(processed_dir / "user_sequences.jsonl", sequences)

    train_rows = []
    valid_rows = []
    test_rows = []
    for row in sequences:
        user_id = row["user_id"]
        # train_interactions.csv 保留 position，方便共现 baseline 知道用户历史顺序。
        for rank, movie_id in enumerate(row["train"]):
            train_rows.append({"user_id": user_id, "movie_id": movie_id, "position": rank})
        valid_rows.append({"user_id": user_id, "history": row["train"], "target": row["valid"]})
        test_history = row["train"] + [row["valid"]]
        valid_history = test_history[-50:]
        test_rows.append({"user_id": user_id, "history": valid_history, "target": row["test"]})

    pd.DataFrame(train_rows).to_csv(processed_dir / "train_interactions.csv", index=False)
    write_jsonl(processed_dir / "valid_samples.jsonl", valid_rows)
    write_jsonl(processed_dir / "test_samples.jsonl", test_rows)


def build_item_texts(movies: pd.DataFrame) -> list[str]:
    """把电影元信息转成文本，作为 SentenceTransformer 的 item embedding 输入。"""
    texts = []
    for row in movies.itertuples(index=False):
        genres = str(row.genres).replace("|", " ")
        title = str(row.title)
        texts.append(f"{title} {genres}".strip())
    return texts
