from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semanticid_llm_rec.data.preprocess import read_jsonl  # noqa: E402
from semanticid_llm_rec.data.sft import format_history_prompt  # noqa: E402
from semanticid_llm_rec.models.heuristic import CooccurrenceRecommender  # noqa: E402
from semanticid_llm_rec.models.semantic_id import sid_from_codes  # noqa: E402
from semanticid_llm_rec.utils.config import ensure_dirs, load_config, resolve_project_path  # noqa: E402


SID_TOKEN_RE = re.compile(r"<sid([123c])_(\d+)>")


def resolve_model_path(path: str | Path) -> Path:
    """解析模型路径：支持绝对路径，也支持相对项目根目录的路径。"""
    p = Path(path)
    if p.is_absolute():
        return p
    return (ROOT / p).resolve()


def extract_first_sid(text: str) -> str | None:
    values: dict[str, int] = {}
    collision_id: int | None = None

    for level, value in SID_TOKEN_RE.findall(text):
        if level in {"1", "2", "3"} and level not in values:
            values[level] = int(value)
        elif level == "c" and collision_id is None:
            collision_id = int(value)

        if {"1", "2"}.issubset(values):
            codes = [values["1"], values["2"]]
            if "3" in values:
                codes.append(values["3"])

            return sid_from_codes(
                codes,
                collision_id=collision_id,
            )

    return None


def load_sid_tables(semantic_dir: Path) -> tuple[dict[str, str], dict[str, int]]:
    """读取 movie_id -> SID 和 SID -> movie_id 两张表。"""
    movie_sid_path = semantic_dir / "movie_sid_map.json"
    sid_movie_path = semantic_dir / "sid_movie_map.json"
    if not movie_sid_path.exists() or not sid_movie_path.exists():
        raise FileNotFoundError("Run scripts/build_semantic_id.py before LLM inference.")

    movie_sid_map = json.loads(movie_sid_path.read_text(encoding="utf-8"))
    raw_sid_movie_map = json.loads(sid_movie_path.read_text(encoding="utf-8"))
    sid_movie_map = {str(sid): int(movie_id) for sid, movie_id in raw_sid_movie_map.items()}
    return movie_sid_map, sid_movie_map


def build_user_prompt(sequence: dict[str, Any], movie_sid_map: dict[str, str], max_history_len: int) -> tuple[str, list[int]]:
    """把用户历史 movie_id 转成训练时一致的 SID prompt。"""
    history = [int(x) for x in sequence["train"]][-max_history_len:]
    history_sids = [movie_sid_map[str(movie_id)] for movie_id in history if str(movie_id) in movie_sid_map]
    if not history_sids:
        raise ValueError("The selected user has no history items with Semantic IDs.")
    return format_history_prompt(history_sids), history


def load_tokenizer_and_model(model_path: Path, base_model: str | None, device: str):
    """加载本地 SFT 模型或 LoRA adapter。

    判断规则：
    - 如果 model_path 下存在 adapter_config.json，认为它是 LoRA adapter；
    - 否则认为 model_path 是完整微调后的 CausalLM 模型目录。
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("LLM inference requires torch and transformers. Run `uv sync` first.") from exc

    adapter_config_path = model_path / "adapter_config.json"
    is_lora_adapter = adapter_config_path.exists()
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    device_map = "auto" if device == "cuda" else None

    if is_lora_adapter:
        try:
            from peft import PeftModel
        except Exception as exc:
            raise RuntimeError("Loading a LoRA adapter requires peft. Run `uv sync` first.") from exc

        adapter_cfg = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        resolved_base_model = base_model or adapter_cfg.get("base_model_name_or_path")
        if not resolved_base_model:
            raise ValueError("LoRA adapter detected, but no base model is provided.")

        # tokenizer 优先从 adapter 目录读取；如果训练时没保存 tokenizer，则回退到 base model。
        tokenizer_source = model_path if (model_path / "tokenizer_config.json").exists() else resolved_base_model
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            resolved_base_model,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_path)
        backend = f"qwen-lora-adapter:{model_path}"
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        backend = f"qwen-sft-full:{model_path}"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if device == "cpu":
        model = model.to("cpu")
    model.eval()
    return tokenizer, model, backend


def generate_sid_candidates(
    tokenizer,
    model,
    prompt: str,
    *,
    device: str,
    top_k: int,
    beam_size: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> list[dict[str, str | None]]:
    """调用 Qwen 生成若干 SID 候选，并保留原始文本方便排错。"""
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("LLM inference requires torch.") from exc

    inputs = tokenizer(prompt, return_tensors="pt")
    if device == "cuda":
        inputs = {key: value.to("cuda") for key, value in inputs.items()}

    input_len = int(inputs["input_ids"].shape[-1])
    num_return_sequences = max(1, top_k)
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        # 采样模式更容易得到多样候选，适合模型只会输出少量重复 SID 时调试。
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "num_return_sequences": num_return_sequences,
            }
        )
    else:
        # beam search 更稳定；为了返回 top_k 条候选，beam 数至少等于返回条数。
        generation_kwargs.update(
            {
                "do_sample": False,
                "num_beams": max(beam_size, num_return_sequences),
                "num_return_sequences": num_return_sequences,
            }
        )

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)

    generations: list[dict[str, str | None]] = []
    for output_ids in outputs:
        new_tokens = output_ids[input_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        generations.append({"text": text, "semantic_id": extract_first_sid(text)})
    return generations


def build_recommendation_rows(
    generations: list[dict[str, str | None]],
    sid_movie_map: dict[str, int],
    movies: pd.DataFrame,
    seen_ids: set[int],
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str | None]]]:
    """把生成的 SID 候选映射成电影推荐结果。"""
    movie_by_id = movies.set_index("movie_id").to_dict(orient="index")
    used_movie_ids: set[int] = set()
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, str | None]] = []

    for gen in generations:
        sid = gen["semantic_id"]
        if sid is None:
            rejected.append({**gen, "reason": "no-valid-sid"})
            continue
        movie_id = sid_movie_map.get(sid)
        if movie_id is None:
            rejected.append({**gen, "reason": "sid-not-found"})
            continue
        if movie_id in seen_ids:
            rejected.append({**gen, "reason": "seen-item"})
            continue
        if movie_id in used_movie_ids:
            rejected.append({**gen, "reason": "duplicate-item"})
            continue

        info = movie_by_id.get(movie_id, {})
        rows.append(
            {
                "rank": len(rows) + 1,
                "movie_id": int(movie_id),
                "title": info.get("title", ""),
                "genres": info.get("genres", ""),
                "semantic_id": sid,
                "source": "llm",
            }
        )
        used_movie_ids.add(movie_id)
        if len(rows) >= top_k:
            break

    return rows, rejected


def fill_with_local_fallback(
    rows: list[dict[str, Any]],
    history: list[int],
    movies: pd.DataFrame,
    train: pd.DataFrame,
    movie_sid_map: dict[str, str],
    top_k: int,
) -> list[dict[str, Any]]:
    """当 LLM 候选不足 top_k 时，用本地共现推荐补齐，保证演示输出完整。"""
    if len(rows) >= top_k:
        return rows

    movie_by_id = movies.set_index("movie_id").to_dict(orient="index")
    used = {int(row["movie_id"]) for row in rows}
    recommender = CooccurrenceRecommender(train)
    fallback_ranked = recommender.rank(history, top_k=top_k + len(history))
    for movie_id in fallback_ranked:
        if movie_id in used:
            continue
        info = movie_by_id.get(movie_id, {})
        rows.append(
            {
                "rank": len(rows) + 1,
                "movie_id": int(movie_id),
                "title": info.get("title", ""),
                "genres": info.get("genres", ""),
                "semantic_id": movie_sid_map.get(str(movie_id), ""),
                "source": "local-cooccurrence-fill",
            }
        )
        used.add(movie_id)
        if len(rows) >= top_k:
            break
    return rows


def main() -> None:
    """LLM 本地推理入口：用户历史 -> Qwen 生成 SID -> SID 映射电影 -> Top-K 推荐。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/local.yaml")
    parser.add_argument("--user_id", type=int, required=True)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--model_path", default="checkpoints/qwen_lora", help="完整 SFT 模型或 LoRA adapter 目录。")
    parser.add_argument("--base_model", default=None, help="LoRA adapter 对应的基座模型，例如 Qwen/Qwen2.5-1.5B-Instruct。")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--beam_size", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--sample", action="store_true", help="使用采样生成候选；默认使用 beam search。")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--no_fallback", action="store_true", help="LLM 生成不足 top_k 时不使用本地共现补齐。")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs()
    processed_dir = ROOT / "data" / "processed"
    semantic_dir = ROOT / "data" / "semantic_id"

    try:
        import torch
    except Exception as exc:
        raise RuntimeError("LLM inference requires torch. Run `uv sync` first.") from exc

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

    movies = pd.read_csv(processed_dir / "movies.csv")
    train = pd.read_csv(processed_dir / "train_interactions.csv")
    sequences = {int(x["user_id"]): x for x in read_jsonl(processed_dir / "user_sequences.jsonl")}
    movie_sid_map, sid_movie_map = load_sid_tables(semantic_dir)

    if args.user_id not in sequences:
        raise ValueError(f"user_id={args.user_id} not found in processed sequences.")

    prompt, history = build_user_prompt(
        sequences[args.user_id],
        movie_sid_map,
        max_history_len=int(cfg["data"]["max_history_len"]),
    )
    top_k = args.top_k or int(cfg["inference"]["top_k"])
    beam_size = args.beam_size or int(cfg["inference"]["beam_size"])

    model_path = resolve_model_path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model_path does not exist: {model_path}")

    # LoRA adapter 通常会在 adapter_config.json 记录训练时的基座模型；
    # 只有你显式传入 --base_model 时才覆盖它，避免 local.yaml 里的旧配置误覆盖 1.5B 产物。
    base_model = args.base_model
    tokenizer, model, backend = load_tokenizer_and_model(model_path, base_model, device)
    generations = generate_sid_candidates(
        tokenizer,
        model,
        prompt,
        device=device,
        top_k=top_k,
        beam_size=beam_size,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    rows, rejected = build_recommendation_rows(
        generations,
        sid_movie_map,
        movies,
        seen_ids=set(history),
        top_k=top_k,
    )
    if not args.no_fallback:
        rows = fill_with_local_fallback(rows, history, movies, train, movie_sid_map, top_k)

    out = {
        "user_id": args.user_id,
        "history_movie_ids": history,
        "prompt": prompt,
        "recommendations": rows,
        "raw_generations": generations,
        "rejected_generations": rejected,
        "backend": backend if args.no_fallback or len(rows) == len([r for r in rows if r["source"] == "llm"]) else f"{backend}+fallback",
        "model_path": str(model_path),
        "device": device,
    }
    out_path = ROOT / "outputs" / "recommendations_llm.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Saved LLM recommendations -> {out_path}")


if __name__ == "__main__":
    main()
