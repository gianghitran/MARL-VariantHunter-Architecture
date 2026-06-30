"""
benign_interface.py
===================
Bước 3 — interface hợp nhất cho Benign Generator (HYBRID).

generate_benign_subgraph(source, mimicry_level): trả về benign bundle cùng schema
với *_detection.json (rút gọn). Mặc định hybrid = 70% UNICORN-derived + 30% synthetic.

Module ĐỘC LẬP — không import vào / không sửa Attack_Agent/. Chỉ phục vụ phía detector.
"""

import os
import sys
import numpy as np
import pandas as pd

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from unicorn_subgraph_sampler import (
    load_unicorn_edges, sample_benign_subgraph, BENIGN_IDS,
    build_malicious_corpus,   # re-export: corpus malicious từ UNICORN attack (125..149)
)
from synthetic_templates import generate_synthetic_subgraph

SYNTHETIC_RATIO = 0.30   # tỉ lệ synthetic trong hybrid (70/30)

# ── Cache df UNICORN benign đã load (tránh đọc lại file lớn) ──
_DF_CACHE = {}
_PRELOAD_IDS = None


def _get_df(rng):
    global _PRELOAD_IDS
    if _PRELOAD_IDS is None:
        ids = list(BENIGN_IDS)
        rng.shuffle(ids)
        _PRELOAD_IDS = ids[:15]          # giới hạn 15 file nguồn
    fid = int(rng.choice(_PRELOAD_IDS))
    if fid not in _DF_CACHE:
        _DF_CACHE[fid] = load_unicorn_edges(fid)
    return _DF_CACHE[fid]


def generate_benign_subgraph(source="hybrid", mimicry_level=0.4, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    use_synth = (source == "synthetic") or (source == "hybrid" and rng.random() < SYNTHETIC_RATIO)
    if use_synth:
        return generate_synthetic_subgraph(rng, mimicry_level)
    # UNICORN-derived (có fallback synthetic nếu sample fail)
    for _ in range(10):
        df = _get_df(rng)
        if df is not None and len(df) >= 100:
            b = sample_benign_subgraph(df, rng)
            if b is not None:
                return b
    return generate_synthetic_subgraph(rng, mimicry_level)


def build_benign_corpus(n, source="hybrid", mimicry_level=0.4, seed=0):
    rng = np.random.default_rng(seed)
    return [generate_benign_subgraph(source, mimicry_level, rng) for _ in range(n)]


# ── Adapter: bundle -> edges_df cột môi trường (MP/MF/SO + action-code env) ──
NODE2ENV = {"process": "MP", "file": "MF", "socket": "SO"}
# Khớp Attack_Agent EDGE_TYPE_MAP để benign & malicious dùng CÙNG action-code trong env
EDGE2ENV = {"FR": "EX", "WF": "WR", "RF": "RD", "ST": "ST", "RCV": "RF", "EX": "EX", "IJ": "IJ", "LOAD": "RD"}
ENV_COLS = ["actorID", "actor_type", "objectID", "object", "action", "timestamp"]


def bundle_to_edges_df(bundle) -> pd.DataFrame:
    id2type = {n["id"]: n["type"] for n in bundle["nodes"]}
    rows = []
    for e in bundle["edges"]:
        s, d = e["source"], e["target"]
        rows.append([str(s), NODE2ENV[id2type[s]], str(d), NODE2ENV[id2type[d]],
                     EDGE2ENV.get(e["type"], e["type"]), str(e["event_order"])])
    return pd.DataFrame(rows, columns=ENV_COLS)


if __name__ == "__main__":
    from collections import Counter
    corpus = build_benign_corpus(20, source="hybrid", mimicry_level=0.4, seed=0)
    srcs = Counter(b["source"] for b in corpus)
    print("hybrid corpus sources:", dict(srcs))
    df = bundle_to_edges_df(corpus[0])
    print("edges_df sample (env cols):")
    print(df.head(4).to_string(index=False))
