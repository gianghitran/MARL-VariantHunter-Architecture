"""
synthetic_templates.py
=======================
Bước 2 — synthetic augment cho Benign Generator (hybrid).

Sinh các edge-structure benign với TOPOLOGY có kiểm soát, rồi đưa qua
`finalize_bundle` (dùng chung với UNICORN-derived) để chuẩn hóa BỀ MẶT
(node-type ratio + action histogram) -> chỉ TOPOLOGY khác nhau.

Mục đích chính: tạo HARD NEGATIVES (benign có độ sâu/cross-link gần malicious)
-> chồng lấp vùng đặc trưng cấu trúc -> ép detector học đa đặc trưng thay vì 1
ngưỡng depth thô. Không đụng Attack_Agent/.
"""

import numpy as np
from unicorn_subgraph_sampler import finalize_bundle, E_RANGE


def _easy_topology(rng, e_target):
    """Hoạt động cục bộ bình thường: vài process-hub + nhiều leaf (star), nông."""
    sel = []
    n_hubs = int(rng.integers(2, 4))
    hubs = [f"p{h}" for h in range(n_hubs)]
    for i in range(n_hubs - 1):                   # nối hub qua file dùng chung (depth nhỏ)
        mid = f"shared{i}"
        sel.append((hubs[i], mid))
        sel.append((mid, hubs[i + 1]))
    leaf = 0
    while len(sel) < e_target:
        h = hubs[int(rng.integers(0, n_hubs))]
        sel.append((h, f"l{leaf}"))               # hub -> leaf file (RF/WF sau chuẩn hóa)
        leaf += 1
    return sel[:e_target]


def _hard_topology(rng, e_target, mimicry):
    """
    Hard negative (vd backup pipeline): chuỗi sâu + cross-link -> depth & clustering
    tiến gần malicious, tạo CHỒNG LẤP. depth ~ 4 + mimicry*8 (tối đa ~12).
    """
    target_depth = int(4 + mimicry * 8)
    chain = [f"c{i}" for i in range(target_depth + 1)]
    sel = [(chain[i], chain[i + 1]) for i in range(target_depth)]
    leaf = 0
    while len(sel) < e_target:
        i = int(rng.integers(0, len(chain)))
        if rng.random() < 0.35 and i + 2 < len(chain):
            sel.append((chain[i], chain[i + 2]))  # cross-link -> tăng clustering
        else:
            sel.append((chain[i], f"b{leaf}"))    # nhánh
            leaf += 1
    return sel[:e_target]


def generate_synthetic_subgraph(rng, mimicry_level=0.4):
    e_target = int(np.clip(round(rng.normal(62, 6)), E_RANGE[0], E_RANGE[1]))
    if rng.random() < mimicry_level:
        sel = _hard_topology(rng, e_target, mimicry_level)
    else:
        sel = _easy_topology(rng, e_target)
    return finalize_bundle(sel, rng, source="synthetic")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    from collections import Counter
    for ml in (0.0, 1.0):
        b = generate_synthetic_subgraph(rng, mimicry_level=ml)
        print(f"mimicry={ml}: {b['stats']} "
              f"nt={dict(Counter(n['type'] for n in b['nodes']))} "
              f"et={dict(Counter(e['type'] for e in b['edges']))}")
