from __future__ import annotations

import pandas as pd

from semanticid_llm_rec.data.preprocess import split_user_sequences


def test_split_user_sequences_last_two_are_valid_and_test():
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 1, 2, 2, 2],
            "movie_id": [10, 11, 12, 13, 14, 20, 21, 22],
            "rating": [5.0] * 8,
            "timestamp": [1, 2, 3, 4, 5, 1, 2, 3],
        }
    )
    rows = split_user_sequences(ratings, min_interactions=5, max_history_len=3)
    assert len(rows) == 1
    assert rows[0]["user_id"] == 1
    assert rows[0]["train"] == [10, 11, 12]
    assert rows[0]["valid"] == 13
    assert rows[0]["test"] == 14


def test_split_user_sequences_truncates_train_history():
    ratings = pd.DataFrame(
        {
            "user_id": [1] * 7,
            "movie_id": [10, 11, 12, 13, 14, 15, 16],
            "rating": [5.0] * 7,
            "timestamp": list(range(7)),
        }
    )
    rows = split_user_sequences(ratings, min_interactions=5, max_history_len=2)
    assert rows[0]["train"] == [13, 14]
    assert rows[0]["valid"] == 15
    assert rows[0]["test"] == 16
