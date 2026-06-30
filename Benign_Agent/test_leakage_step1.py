"""
test_leakage_step1.py
=====================
Cổng kiểm chứng §8 cho UNICORN-derived benign (thuần, chưa augment).

So 3 nhóm đặc trưng giữa benign-derived vs malicious (Attack Agent detection JSON):
  Test 1 — Surface-overlap: in histogram trung bình từng lớp (kiểm tra chồng lấp).
  Test 2 — Surface-only probe: LogReg CHỈ trên đặc trưng bề mặt -> AUC PHẢI ~0.5
           (cao = còn format-leakage, hướng lai chưa sạch).
  Test 3 — Behavioral probe: LogReg trên đặc trưng cấu trúc (topology) -> AUC PHẢI cao
           (tín hiệu hành vi tồn tại & học được).

Đặc trưng cấu trúc CHỈ dùng topology (depth/components/density/clustering/articulation),
KHÔNG dùng nhãn technique/is_bridge (để công bằng — benign không có nhãn đó).
"""

import os
import json
import glob
import numpy as np
import networkx as nx
from collections import Counter

from unicorn_subgraph_sampler import BASE_DIR
from benign_interface import build_benign_corpus

EDGE_TYPES = ["RF", "WF", "FR", "EX", "ST", "RCV"]
NODE_TYPES = ["process", "file", "socket"]


def _graph(bundle):
    G = nx.DiGraph()
    G.add_nodes_from(n["id"] for n in bundle["nodes"])
    for e in bundle["edges"]:
        G.add_edge(e["source"], e["target"])
    return G


def features(bundle):
    nodes, edges = bundle["nodes"], bundle["edges"]
    N, E = len(nodes), len(edges)
    nt = Counter(n["type"] for n in nodes)
    et = Counter(e["type"] for e in edges)
    G = _graph(bundle)
    degs = [d for _, d in G.degree()] or [0]

    # SURFACE = FORMAT/SCALE thuần (vocabulary + kích thước) — detector KHÔNG được
    # dùng để "ăn gian". Degree/density KHÔNG thuộc đây (chúng là topology=hành vi).
    surface = [N, E,
               nt["process"] / N, nt["file"] / N, nt["socket"] / N,
               *[et[t] / E for t in EDGE_TYPES]]

    # STRUCTURAL = topology = hành vi (cái detector NÊN học)
    C = nx.condensation(G)
    depth = nx.dag_longest_path_length(C)           # causal depth (an toàn với chu trình)
    wcc = nx.number_weakly_connected_components(G)
    dens = nx.density(G)
    UG = G.to_undirected()
    clust = nx.average_clustering(UG) if UG.number_of_nodes() > 2 else 0.0
    artic = len(list(nx.articulation_points(UG))) if UG.number_of_nodes() > 2 else 0
    structural = [depth, wcc, dens, clust, artic, depth / max(N, 1), float(np.mean(degs))]
    return surface, structural


def load_malicious():
    out = []
    for f in sorted(glob.glob(os.path.join(BASE_DIR, "Attack_Agent", "result_handoff", "*", "*_detection.json"))):
        d = json.load(open(f, encoding="utf-8"))
        out.append({"nodes": d["nodes"], "edges": d["edges"]})
    return out


def run_for(source):
    print(f"\n########## SOURCE = {source} ##########")
    benign = build_benign_corpus(120, source=source, mimicry_level=0.5, seed=0)
    mal = load_malicious()
    from collections import Counter
    print(f"benign={len(benign)} (sources={dict(Counter(b['source'] for b in benign))})  malicious={len(mal)}")
    _evaluate(benign, mal)


def _evaluate(benign, mal):
    Bs = [features(b) for b in benign]
    Ms = [features(m) for m in mal]
    surf = np.array([s for s, _ in Bs] + [s for s, _ in Ms], dtype=float)
    struct = np.array([t for _, t in Bs] + [t for _, t in Ms], dtype=float)
    y = np.array([0] * len(Bs) + [1] * len(Ms))

    # ── Test 1: surface overlap ──
    surf_names = ["n_nodes", "n_edges", "f_proc", "f_file", "f_sock",
                  *[f"f_{t}" for t in EDGE_TYPES]]
    bmean = surf[y == 0].mean(0)
    mmean = surf[y == 1].mean(0)
    print("\n=== Test 1: Surface means (benign vs malicious) — muốn CHỒNG LẤP ===")
    for nm, bv, mv in zip(surf_names, bmean, mmean):
        print(f"  {nm:10s} benign={bv:8.3f}  malicious={mv:8.3f}")

    struct_names = ["causal_depth", "n_wcc", "density", "clustering", "articulation", "depth/N", "mean_deg"]
    bsm = struct[y == 0].mean(0)
    msm = struct[y == 1].mean(0)
    print("\n=== Structural means (benign vs malicious) — muốn KHÁC BIỆT ===")
    for nm, bv, mv in zip(struct_names, bsm, msm):
        print(f"  {nm:13s} benign={bv:8.3f}  malicious={mv:8.3f}")

    # ── Test 2 & 3: probes (ROC-AUC, 5-fold) ──
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    def probe(X):
        Xs = StandardScaler().fit_transform(X)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        return cross_val_score(clf, Xs, y, cv=cv, scoring="roc_auc").mean()

    auc_surf = probe(surf)
    auc_struct = probe(struct)
    print("\n=== Test 2: Surface-only probe AUC (MUỐN ~0.5, cao=leakage) ===")
    print(f"  AUC_surface    = {auc_surf:.3f}")
    print("=== Test 3: Behavioral probe AUC (MUỐN >0.85) ===")
    print(f"  AUC_structural = {auc_struct:.3f}")

    print("\n=== KẾT LUẬN ===")
    leak = auc_surf > 0.70
    behav = auc_struct > 0.85
    print(f"  Surface leakage? {'CÓ (xấu)' if leak else 'không (tốt)'}  (AUC={auc_surf:.3f})")
    print(f"  Hành vi tách được? {'CÓ (tốt)' if behav else 'KHÔNG (cần augment)'}  (AUC={auc_struct:.3f})")
    verdict = "PASS — benign sạch (format không leak, hành vi tách được)" if (not leak and behav) else "CẦN ĐIỀU CHỈNH"
    print(f"  => {verdict}")


if __name__ == "__main__":
    run_for("unicorn")
    run_for("hybrid")
