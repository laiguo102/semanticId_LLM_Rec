from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semanticid_llm_rec.data.preprocess import read_jsonl, write_jsonl  # noqa: E402
from semanticid_llm_rec.data.sft import format_history_prompt  # noqa: E402
from semanticid_llm_rec.utils.config import ensure_dirs, load_config  # noqa: E402


SID_TOKEN_RE = re.compile(r"<sid([123c])_(\d+)>")


def resolve_model_path(path: str | Path) -> Path:
    """解析模型目录路径，支持绝对路径和相对项目根目录的路径。"""
    p = Path(path)
    if p.is_absolute():
        return p
    return (ROOT / p).resolve()


def required_sid_levels(codebook_num: int) -> list[str]:
    """根据码本层数得到必须出现的 SID 层级。

    例如：
    - codebook_num=2 时，需要 <sid1_*> <sid2_*>
    - codebook_num=3 时，需要 <sid1_*> <sid2_*> <sid3_*>
    """
    if codebook_num < 1:
        raise ValueError("codebook_num must be greater than 0.")
    if codebook_num > 3:
        raise ValueError("This project currently supports at most 3 SID levels.")
    return [str(i) for i in range(1, codebook_num + 1)]


def strict_extract_first_sid(text: str, codebook_num: int) -> str | None:
    """从模型输出里抽取第一个完整 SID，所需层数由 codebook_num 决定。

    评估脚本要比演示脚本更严格：只要缺少当前码本要求的任意一级 SID，就认为这次生成不可解析，
    避免把不完整输出错误地映射成推荐命中。
    """
    required_levels = set(required_sid_levels(codebook_num))
    values: dict[str, int] = {}
    collision_id: int | None = None
    for level, value in SID_TOKEN_RE.findall(text):
        if level in required_levels and level not in values:
            values[level] = int(value)
        elif level == "c" and collision_id is None:
            collision_id = int(value)

        if required_levels.issubset(values):
            sid = " ".join(f"<sid{level}_{values[level]}>" for level in required_sid_levels(codebook_num))
            if collision_id is not None and collision_id > 0:
                sid += f" <sidc_{collision_id}>"
            return sid
    return None


def sid_levels(sid: str | None) -> dict[str, int]:
    """把 SID 拆成各层 code，用于统计分层准确率。"""
    if sid is None:
        return {}
    return {level: int(value) for level, value in SID_TOKEN_RE.findall(sid)}


def infer_codebook_num(movie_sid_map: dict[str, str]) -> int | None:
    """从已有 SID 映射表自动推断码本层数。

    这比单纯相信 config 更稳，因为你可能已经重新生成了 2 级码本产物，
    但本地配置文件还停留在 3 级默认值。
    """
    for sid in movie_sid_map.values():
        numeric_levels = [int(level) for level, _ in SID_TOKEN_RE.findall(str(sid)) if level != "c"]
        if numeric_levels:
            return max(numeric_levels)
    return None


def load_sid_tables() -> tuple[dict[str, str], dict[str, int]]:
    """读取 SID 映射表：movie_id -> SID 和 SID -> movie_id。"""
    semantic_dir = ROOT / "data" / "semantic_id"
    movie_sid_path = semantic_dir / "movie_sid_map.json"
    sid_movie_path = semantic_dir / "sid_movie_map.json"
    if not movie_sid_path.exists() or not sid_movie_path.exists():
        raise FileNotFoundError("Run scripts/build_semantic_id.py before LLM evaluation.")

    movie_sid_map = json.loads(movie_sid_path.read_text(encoding="utf-8"))
    sid_movie_map = json.loads(sid_movie_path.read_text(encoding="utf-8"))
    return movie_sid_map, {str(sid): int(movie_id) for sid, movie_id in sid_movie_map.items()}


def build_prompt(history: list[int], movie_sid_map: dict[str, str], max_history_len: int) -> tuple[str, list[str]]:
    """把评估样本中的历史 movie_id 转成和 SFT 训练一致的 SID prompt。"""
    trimmed_history = [int(x) for x in history][-max_history_len:]
    history_sids = [movie_sid_map[str(movie_id)] for movie_id in trimmed_history if str(movie_id) in movie_sid_map]
    if not history_sids:
        raise ValueError("A sample has no history item with Semantic ID.")
    return format_history_prompt(history_sids), history_sids


def load_tokenizer_and_model(model_path: Path, base_model: str | None, device: str):
    """加载完整 SFT 模型或 LoRA adapter。

    - model_path 下有 adapter_config.json：按 LoRA adapter 加载；
    - 否则：按完整 CausalLM 模型目录加载。
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("LLM evaluation requires torch and transformers. Run `uv sync` first.") from exc

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


def generate_once(
    tokenizer,
    model,
    prompt: str,
    *,
    device: str,
    codebook_num: int,
    beam_size: int,
    num_return_sequences: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> list[dict[str, str | None]]:
    """单条样本生成。

    这里刻意不把 num_return_sequences 和 beam_size 强绑定：
    - greedy/beam 模式下，HuggingFace 要求 num_return_sequences <= num_beams；
    - 如果不满足，直接报错提示你调小返回条数或调大 beam，而不是自动加大 beam 吃显存。
    """
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("LLM evaluation requires torch.") from exc

    if not do_sample and num_return_sequences > beam_size:
        raise ValueError(
            "num_return_sequences cannot be greater than beam_size when --sample is not enabled. "
            "Use --num_return_sequences 1 for low-memory evaluation, or explicitly increase --beam_size."
        )

    inputs = tokenizer(prompt, return_tensors="pt")
    if device == "cuda":
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
    input_len = int(inputs["input_ids"].shape[-1])

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "num_return_sequences": num_return_sequences,
    }
    if do_sample:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
    else:
        generation_kwargs.update(
            {
                "do_sample": False,
                "num_beams": beam_size,
            }
        )

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)

    rows: list[dict[str, str | None]] = []
    for output_ids in outputs:
        generated_ids = output_ids[input_len:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        rows.append({"text": text, "semantic_id": strict_extract_first_sid(text, codebook_num)})
    return rows


def update_metrics(summary: dict[str, float], detail: dict[str, Any], level_keys: list[str]) -> None:
    """把单样本结果累加到 summary 计数器。"""
    summary["evaluated"] += 1
    if detail["parse_ok"]:
        summary["parse_ok"] += 1
    if detail["in_catalog"]:
        summary["in_catalog"] += 1
    if detail["top1_sid_hit"]:
        summary["top1_sid_hit"] += 1
    if detail["top1_item_hit"]:
        summary["top1_item_hit"] += 1
    if detail["candidate_sid_hit"]:
        summary["candidate_sid_hit"] += 1
    if detail["candidate_item_hit"]:
        summary["candidate_item_hit"] += 1
    for level in level_keys + ["c"]:
        if detail["level_hit"].get(level):
            summary[f"sid{level}_hit"] += 1


def finalize_metrics(summary: dict[str, float], level_keys: list[str]) -> dict[str, float]:
    """把计数转换成比例指标。"""
    total = max(int(summary["evaluated"]), 1)
    metrics = {
        "users": int(summary["evaluated"]),
        "ParseRate": summary["parse_ok"] / total,
        "CatalogRate": summary["in_catalog"] / total,
        "Top1SIDAccuracy": summary["top1_sid_hit"] / total,
        "Top1ItemAccuracy": summary["top1_item_hit"] / total,
        "CandidateSIDHitRate": summary["candidate_sid_hit"] / total,
        "CandidateItemHitRate": summary["candidate_item_hit"] / total,
        "SIDCollisionAccuracy": summary["sidc_hit"] / total,
    }
    for level in level_keys:
        metrics[f"SID{level}Accuracy"] = summary[f"sid{level}_hit"] / total
    return metrics


def main() -> None:
    """LLM 准确率检测入口：不使用本地 fallback，只统计 LLM 自身生成是否正确。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/local.yaml")
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--model_path", default="checkpoints/qwen_lora", help="完整 SFT 模型或 LoRA adapter 目录。")
    parser.add_argument("--base_model", default=None, help="LoRA adapter 对应的基座模型，例如 Qwen/Qwen2.5-1.5B-Instruct。")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max_users", type=int, default=None, help="最多评估多少个用户；为空时读取 config.eval.max_users。")
    parser.add_argument("--beam_size", type=int, default=1, help="低显存默认 1，即 greedy。")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="每个用户返回多少个 LLM 候选，不会自动抬高 beam。")
    parser.add_argument("--sid_levels", type=int, default=None, help="SID 层数；默认从 SID 映射表推断。")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--sample", action="store_true", help="使用采样生成候选；默认 greedy/beam。")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--save_details", action="store_true", help="保存逐样本明细到 jsonl，方便人工排错。")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs()

    try:
        import torch
    except Exception as exc:
        raise RuntimeError("LLM evaluation requires torch. Run `uv sync` first.") from exc

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

    processed_dir = ROOT / "data" / "processed"
    samples = read_jsonl(processed_dir / f"{args.split}_samples.jsonl")
    max_users = args.max_users if args.max_users is not None else cfg["eval"].get("max_users")
    if max_users is not None:
        samples = samples[: int(max_users)]

    movie_sid_map, sid_movie_map = load_sid_tables()
    inferred_codebook_num = infer_codebook_num(movie_sid_map)
    codebook_num = int(args.sid_levels or inferred_codebook_num or cfg.get("rqvae", {}).get("codebook_num", 3))
    level_keys = required_sid_levels(codebook_num)
    model_path = resolve_model_path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model_path does not exist: {model_path}")

    tokenizer, model, backend = load_tokenizer_and_model(model_path, args.base_model, device)
    summary = {
        "evaluated": 0.0,
        "parse_ok": 0.0,
        "in_catalog": 0.0,
        "top1_sid_hit": 0.0,
        "top1_item_hit": 0.0,
        "candidate_sid_hit": 0.0,
        "candidate_item_hit": 0.0,
        "sidc_hit": 0.0,
    }
    for level in level_keys:
        summary[f"sid{level}_hit"] = 0.0
    details: list[dict[str, Any]] = []

    for idx, row in enumerate(samples, start=1):
        user_id = int(row["user_id"])
        history = [int(x) for x in row["history"]]
        target_movie_id = int(row["target"])
        target_sid = movie_sid_map.get(str(target_movie_id))
        if target_sid is None:
            continue

        prompt, history_sids = build_prompt(history, movie_sid_map, int(cfg["data"]["max_history_len"]))
        generations = generate_once(
            tokenizer,
            model,
            prompt,
            device=device,
            codebook_num=codebook_num,
            beam_size=int(args.beam_size),
            num_return_sequences=int(args.num_return_sequences),
            max_new_tokens=int(args.max_new_tokens),
            do_sample=bool(args.sample),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
        )

        predicted_sids = [gen["semantic_id"] for gen in generations]
        predicted_movie_ids = [sid_movie_map.get(sid) if sid is not None else None for sid in predicted_sids]
        top1_sid = predicted_sids[0] if predicted_sids else None
        top1_movie_id = predicted_movie_ids[0] if predicted_movie_ids else None

        target_levels = sid_levels(target_sid)
        predicted_levels = sid_levels(top1_sid)
        level_hit = {
            level: predicted_levels.get(level) == target_levels.get(level)
            for level in level_keys + ["c"]
        }

        detail = {
            "index": idx,
            "user_id": user_id,
            "history_movie_ids": history,
            "history_sids": history_sids,
            "target_movie_id": target_movie_id,
            "target_sid": target_sid,
            "generations": generations,
            "predicted_sids": predicted_sids,
            "predicted_movie_ids": predicted_movie_ids,
            "top1_sid": top1_sid,
            "top1_movie_id": top1_movie_id,
            "parse_ok": top1_sid is not None,
            "in_catalog": top1_movie_id is not None,
            "top1_sid_hit": top1_sid == target_sid,
            "top1_item_hit": top1_movie_id == target_movie_id,
            "candidate_sid_hit": target_sid in predicted_sids,
            "candidate_item_hit": target_movie_id in predicted_movie_ids,
            "level_hit": level_hit,
        }
        update_metrics(summary, detail, level_keys)
        details.append(detail)

        if idx % 20 == 0:
            metrics = finalize_metrics(summary, level_keys)
            print(
                f"evaluated={idx}/{len(samples)} "
                f"top1_item_acc={metrics['Top1ItemAccuracy']:.4f} "
                f"parse_rate={metrics['ParseRate']:.4f}"
            )

    metrics = finalize_metrics(summary, level_keys)
    out = {
        "split": args.split,
        "metrics": metrics,
        "backend": backend,
        "model_path": str(model_path),
        "device": device,
        "beam_size": int(args.beam_size),
        "num_return_sequences": int(args.num_return_sequences),
        "sid_levels": codebook_num,
        "sample": bool(args.sample),
        "note": "Pure LLM evaluation. No local fallback is used.",
    }
    out_path = ROOT / "outputs" / "metrics_llm.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.save_details:
        details_path = ROOT / "outputs" / "llm_eval_predictions.jsonl"
        write_jsonl(details_path, details)
        print(f"Saved LLM evaluation details -> {details_path}")

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Saved LLM metrics -> {out_path}")


if __name__ == "__main__":
    main()
