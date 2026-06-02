from __future__ import annotations

import numpy as np

from semanticid_llm_rec.models.semantic_id import build_unique_sid_map, parse_sid, sid_from_codes


def test_sid_roundtrip_three_tokens():
    sid = sid_from_codes([17, 43, 88])
    assert sid == "<sid1_17> <sid2_43> <sid3_88>"
    assert parse_sid(sid) == ((17, 43, 88), 0)


def test_sid_collision_adds_collision_token():
    sid_map = build_unique_sid_map([2, 1], np.array([[7, 8, 9], [7, 8, 9]]))
    assert sid_map["1"] == "<sid1_7> <sid2_8> <sid3_9>"
    assert sid_map["2"] == "<sid1_7> <sid2_8> <sid3_9> <sidc_1>"
    assert parse_sid(sid_map["2"]) == ((7, 8, 9), 1)
