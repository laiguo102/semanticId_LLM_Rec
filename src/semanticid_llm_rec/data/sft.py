from __future__ import annotations

from typing import Iterable


def format_history_prompt(history_sids: list[str]) -> str:
    """把用户历史 SID 拼成 LLM 输入，让模型学习“根据历史预测下一个 SID”。"""
    history = "\n".join(history_sids)
    return f"User History:\n\n{history}\n\nPredict Next Item:"


def build_sft_rows(
    sequences: Iterable[dict],
    movie_sid_map: dict[str, str],
    max_history_len: int = 50,
    max_examples: int | None = None,
) -> list[dict]:
    """构造监督微调样本：prompt 是历史 SID，response 是下一个 item 的 SID。

    同一个用户序列会滑窗生成多个训练样本，这样 Qwen LoRA 能学习序列转移规律。
    """
    rows: list[dict] = []
    for seq in sequences:
        train = [int(x) for x in seq["train"]]
        if len(train) < 2:
            continue
        for target_pos in range(1, len(train)):
            # 只保留目标点之前的历史，避免把答案泄露给模型。
            history = train[max(0, target_pos - max_history_len) : target_pos]
            target = train[target_pos]
            history_sids = [movie_sid_map[str(i)] for i in history if str(i) in movie_sid_map]
            if not history_sids or str(target) not in movie_sid_map:
                continue
            rows.append(
                {
                    "user_id": int(seq["user_id"]),
                    "prompt": format_history_prompt(history_sids),
                    "response": movie_sid_map[str(target)],
                    "history_movie_ids": history,
                    "target_movie_id": int(target),
                }
            )
            if max_examples is not None and len(rows) >= max_examples:
                return rows
    return rows
