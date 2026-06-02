from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """返回项目根目录，脚本从任意工作目录启动时都能定位文件。"""
    return Path(__file__).resolve().parents[3]


def load_config(path: str | Path = "config/local.yaml") -> dict[str, Any]:
    """读取 yaml 配置；相对路径默认按项目根目录解析。"""
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = project_root() / cfg_path
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(cfg_path)
    return cfg


def resolve_project_path(path: str | Path) -> Path:
    """把配置里的路径转成绝对路径，方便访问项目外的数据目录。"""
    p = Path(path)
    if p.is_absolute():
        return p
    return (project_root() / p).resolve()


def ensure_dirs() -> None:
    """创建约定产物目录，避免脚本写文件时因为目录不存在失败。"""
    root = project_root()
    for rel in [
        "data/raw",
        "data/processed",
        "data/semantic_id",
        "checkpoints/rqvae",
        "checkpoints/qwen_lora",
        "outputs",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)
