"""
unicorn_subgraph_sampler.py
===========================
Bước 1 của Benign Generator (hybrid) — nguồn benign CHÍNH: cắt subgraph từ benign
UNICORN thật rồi chuẩn hóa về CÙNG vocabulary/độ mịn với Attack Agent.

KHÔNG đụng tới Attack_Agent/ — module độc lập, chỉ phục vụ phía detector.

Quy trình (xem BENIGN_GENERATOR_DESIGN.md §4A):
  1. Cắt subgraph ~54–78 cạnh từ benign UNICORN (temporal window + largest WCC).
  2. Suy node-type theo CẤU TRÚC (actor→process, pure-object→file, leaf→socket),
     post-balance về tỉ lệ mục tiêu (process 0.29 / file 0.64 / socket 0.07).
  3. Gán action-type để KHỚP histogram malicious (RF/WF/FR/EX/ST/RCV) — nhãn action
     chỉ là surface-matching; tín hiệu phân biệt nằm ở CẤU TRÚC thật của UNICORN.

Đầu ra: bundle {'label':'benign','nodes':[{id,type,role}],'edges':[{source,target,type,event_order}]}
cùng schema rút gọn với *_detection.json của Attack Agent.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNICORN_DIR = os.path.join(BASE_DIR, "unicorn")

# Mục tiêu bề mặt — đo từ 27 graph malicious thật (BENIGN_GENERATOR_DESIGN §2)
NODE_TARGET = {"process": 0.29, "file": 0.64, "socket": 0.07}
EDGE_TARGET = {"RF": 0.38, "WF": 0.32, "FR": 0.15, "EX": 0.05, "ST": 0.05, "RCV": 0.05}
E_RANGE = (54, 78)
BENIGN_IDS = range(0, 125)     # 0..124 là benign trong UNICORN
MALICIOUS_IDS = range(125, 150) # 125..149 là attack/malicious trong UNICORN (25 graph)
COLS = ["actorID", "actor_type", "objectID", "object", "action", "timestamp"]


def load_unicorn_edges(i: int) -> pd.DataFrame:
    fpath = os.path.join(UNICORN_DIR, f"{i}.txt")
    if not os.path.exists(fpath):
        return None
    return pd.read_csv(fpath, sep="\t", names=COLS, dtype=str)


def sample_benign_subgraph(df: pd.DataFrame, rng: np.random.Generator):
    """Cắt 1 subgraph benign liên thông ~E_target cạnh từ df (1 graph UNICORN)."""
    return _sample_subgraph(df, rng, label="benign")


def sample_malicious_subgraph(df: pd.DataFrame, rng: np.random.Generator):
    """
    Cắt 1 subgraph MALICIOUS liên thông từ 1 graph UNICORN attack (125..149).
    Xử lý Y HỆT benign (BFS liên thông + suy type cấu trúc + gán action khớp
    histogram) -> BỀ MẶT giống benign, tín hiệu phân biệt nằm ở CẤU TRÚC thật
    của graph attack UNICORN (degree distribution / connectivity), KHÔNG ở nhãn.
    """
    return _sample_subgraph(df, rng, label="malicious")


def _sample_subgraph(df: pd.DataFrame, rng: np.random.Generator, label: str = "benign"):
    """Lõi cắt subgraph liên thông dùng chung cho cả benign lẫn malicious."""
    n = len(df)
    # recenter quanh ~62 (mean malicious) thay vì uniform[54,78] (mean 66) để khớp size
    e_target = int(np.clip(round(rng.normal(62, 6)), E_RANGE[0], E_RANGE[1]))
    if n < e_target + 5:
        return None

    # 1. Temporal window rộng rồi rút thành phần liên thông
    W = min(n, e_target * 6)
    start = int(rng.integers(0, max(1, n - W)))
    win = df.iloc[start:start + W]
    src = win["actorID"].to_numpy()
    dst = win["objectID"].to_numpy()

    Gm = nx.Graph()
    for s, d in zip(src, dst):
        Gm.add_edge(s, d)
    comps = list(nx.connected_components(Gm))
    if not comps:
        return None
    largest = max(comps, key=len)

    # 2. Thu cạnh bằng BFS từ seed trong WCC lớn nhất -> subgraph LIÊN THÔNG
    #    (tránh artifact n_wcc: phân mảnh do thu theo thứ tự thời gian).
    from collections import deque
    adj = {}
    for s, d in zip(src, dst):
        if s in largest and d in largest and s != d:
            adj.setdefault(s, []).append((s, d))
            adj.setdefault(d, []).append((s, d))
    if not adj:
        return None
    seed = max(adj, key=lambda x: len(adj[x]))
    q, seen, sel, used = deque([seed]), {seed}, [], set()
    while q and len(sel) < e_target:
        u = q.popleft()
        for (s, d) in adj.get(u, []):
            if len(sel) >= e_target:
                break
            sel.append((s, d))
            nxt = d if u == s else s
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    if len(sel) < 30:
        return None
    src_tag = "unicorn_derived" if label == "benign" else "unicorn_malicious"
    return finalize_bundle(sel, rng, source=src_tag, label=label)


def finalize_bundle(sel, rng, source="unicorn_derived", label="benign"):
    """
    Chuẩn hóa BỀ MẶT dùng chung cho cả UNICORN-derived lẫn synthetic:
    suy node-type theo cấu trúc + post-balance tỉ lệ, gán action khớp histogram.
    `sel` = list cạnh (src, dst) đã có CẤU TRÚC mong muốn (topology giữ nguyên).
    """
    # 3. Suy node-type theo cấu trúc
    nodes = list(dict.fromkeys([x for e in sel for x in e]))  # giữ thứ tự
    outdeg = {x: 0 for x in nodes}
    deg = {x: 0 for x in nodes}
    for s, d in sel:
        outdeg[s] += 1
        deg[s] += 1
        deg[d] += 1
    N = len(nodes)

    # process = node có cạnh đi ra; post-balance về tỉ lệ mục tiêu
    n_proc = max(1, round(NODE_TARGET["process"] * N))
    n_sock = max(0, round(NODE_TARGET["socket"] * N))
    # ứng viên process: out-degree cao nhất
    by_out = sorted(nodes, key=lambda x: (-outdeg[x], -deg[x]))
    proc_set = set(by_out[:n_proc])
    # socket: trong các node KHÔNG phải process, lấy leaf (degree thấp nhất)
    non_proc = [x for x in nodes if x not in proc_set]
    by_deg = sorted(non_proc, key=lambda x: deg[x])
    sock_set = set(by_deg[:n_sock])
    ntype = {}
    for x in nodes:
        if x in proc_set:
            ntype[x] = "process"
        elif x in sock_set:
            ntype[x] = "socket"
        else:
            ntype[x] = "file"

    # 4. Gán action theo COUNT mục tiêu -> marginal edge-type KHỚP TUYỆT ĐỐI malicious
    #    (nhãn action của UNICORN không decode được hash -> đây là surface-matching;
    #     tín hiệu phân biệt nằm ở CẤU TRÚC, không phải nhãn action).
    E = len(sel)
    types = list(EDGE_TARGET.keys())
    counts = {t: int(round(EDGE_TARGET[t] * E)) for t in types}
    counts["RF"] += E - sum(counts.values())          # bù lệch làm tròn vào nhóm trội
    counts["RF"] = max(0, counts["RF"])
    pool = [t for t in types for _ in range(counts[t])]
    pool = (pool + ["RF"] * E)[:E]                      # đảm bảo đúng E nhãn
    rng.shuffle(pool)
    # Ưu tiên đặt ST/RCV lên cạnh có socket (giữ chút mạch lạc), phần còn lại random
    sock_edges = [i for i, (s, d) in enumerate(sel) if ntype[s] == "socket" or ntype[d] == "socket"]
    st_rcv_pos = [i for i, t in enumerate(pool) if t in ("ST", "RCV")]
    for k, ei in enumerate(sock_edges[:len(st_rcv_pos)]):
        j = st_rcv_pos[k]
        pool[ei], pool[j] = pool[j], pool[ei]
    edges = [{"source": s, "target": d, "type": pool[i],
              "event_order": i, "is_bridge": False} for i, (s, d) in enumerate(sel)]

    # 5. Role tối giản (không dùng trong feature leakage, chỉ cho schema)
    hub = by_out[0] if by_out else None
    node_list = [{"id": x, "type": ntype[x],
                  "role": "hub" if x == hub else ("internal" if ntype[x] == "process" else
                          ("input_artifact" if outdeg[x] == 0 else "output_artifact"))}
                 for x in nodes]

    return {"label": label, "source": source,
            "nodes": node_list, "edges": edges,
            "stats": {"nodes": N, "edges": len(edges)}}


def build_benign_corpus(n_graphs: int = 120, seed: int = 0,
                        max_files: int = 15, per_file: int = 12):
    """Sinh corpus benign-derived từ nhiều file UNICORN benign (0..124)."""
    return _build_corpus(BENIGN_IDS, "benign", n_graphs, seed, max_files, per_file)


def build_malicious_corpus(n_graphs: int = 25, seed: int = 0,
                           max_files: int = 25, per_file: int = 6):
    """
    Sinh corpus malicious-derived từ file UNICORN attack (125..149).
    Cùng pipeline với benign -> CÙNG format node-feature (build_node_feature 30-dim),
    KHÔNG hash w2v. Tín hiệu phân biệt = CẤU TRÚC graph attack thật của UNICORN.
    """
    return _build_corpus(MALICIOUS_IDS, "malicious", n_graphs, seed, max_files, per_file)


def _build_corpus(file_ids_src, label, n_graphs, seed, max_files, per_file):
    rng = np.random.default_rng(seed)
    file_ids = list(file_ids_src)
    rng.shuffle(file_ids)
    sampler = sample_benign_subgraph if label == "benign" else sample_malicious_subgraph
    corpus = []
    for fid in file_ids[:max_files]:
        df = load_unicorn_edges(fid)
        if df is None or len(df) < 100:
            continue
        for _ in range(per_file):
            b = sampler(df, rng)
            if b is not None:
                corpus.append(b)
            if len(corpus) >= n_graphs:
                return corpus
    return corpus


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    df = load_unicorn_edges(0)
    print(f"UNICORN benign graph 0: {len(df)} edges")
    b = sample_benign_subgraph(df, rng)
    from collections import Counter
    print("sample benign subgraph:", b["stats"])
    print("  node-type:", dict(Counter(n["type"] for n in b["nodes"])))
    print("  edge-type:", dict(Counter(e["type"] for e in b["edges"])))
