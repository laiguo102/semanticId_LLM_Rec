from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semanticid_llm_rec.utils.config import ensure_dirs, load_config  # noqa: E402
from semanticid_llm_rec.utils.seed import set_seed  # noqa: E402


def main() -> None:
    """服务器训练入口：用 SFT JSONL 微调 Qwen LoRA，让模型生成下一个 SID。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/server.yaml")
    args = parser.parse_args()

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except Exception as exc:
        raise RuntimeError(
            "Qwen LoRA SFT requires torch, datasets, transformers and peft. Run `uv sync` on the server."
        ) from exc

    cfg = load_config(args.config)
    ensure_dirs()
    set_seed(int(cfg["project"]["seed"]))

    train_path = ROOT / "data" / "processed" / "sft_train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError("Run scripts/build_sft_dataset.py before train_sft.py")

    sft_cfg = cfg["sft"]
    tokenizer = AutoTokenizer.from_pretrained(sft_cfg["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        sft_cfg["base_model"],
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    # LoRA 只训练少量低秩适配参数，显存和训练成本远低于全量微调。
    lora_cfg = LoraConfig(
        r=int(sft_cfg["lora_r"]),
        lora_alpha=int(sft_cfg["lora_alpha"]),
        lora_dropout=float(sft_cfg["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=str(train_path), split="train")

    def tokenize(row):
        """只对 response 部分计算 loss，prompt 部分用 -100 屏蔽。"""
        prompt = row["prompt"]
        response = row["response"] + tokenizer.eos_token
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + response_ids)[: int(sft_cfg["max_seq_len"])]
        labels = [-100] * len(prompt_ids) + response_ids
        labels = labels[: int(sft_cfg["max_seq_len"])]
        attention_mask = [1] * len(input_ids)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)

    def collate(rows):
        """动态 padding 到当前 batch 最大长度，减少无效计算。"""
        max_len = max(len(row["input_ids"]) for row in rows)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for row in rows:
            pad_len = max_len - len(row["input_ids"])
            batch["input_ids"].append(row["input_ids"] + [tokenizer.pad_token_id] * pad_len)
            batch["attention_mask"].append(row["attention_mask"] + [0] * pad_len)
            batch["labels"].append(row["labels"] + [-100] * pad_len)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}

    out_dir = ROOT / "checkpoints" / "qwen_lora"
    args_train = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=float(sft_cfg["epochs"]),
        per_device_train_batch_size=int(sft_cfg["batch_size"]),
        gradient_accumulation_steps=int(sft_cfg["gradient_accumulation_steps"]),
        learning_rate=float(sft_cfg["lr"]),
        logging_steps=20,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to=[],
    )
    trainer = Trainer(model=model, args=args_train, train_dataset=tokenized, data_collator=collate)
    trainer.train()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved Qwen LoRA adapter -> {out_dir}")


if __name__ == "__main__":
    main()
