"""
darpa_tc_sampler.py
===================
Nguồn pretrain THẬT từ DARPA TC E3 (CADETS/THEIA/TRACE) — thay UNICORN (whole-graph,
không tách được ở mức subgraph) và Attack Agent JSON (synthetic, gắn result_handoff).

Module ĐỘC LẬP phía detector — KHÔNG đụng Attack_Agent/.

Pipeline:
  1. parse_darpa_tc(): stream CDM JSON -> DataFrame [actorID, actor_type, objectID,
     object, action, timestamp] với type ∈ {process,file,socket} và action ∈
     {read,write,execute,clone,send,mmap,load,...} — KHỚP TRỰC TIẾP graph_utils
     NODE_TYPE_VOCAB / ACTION_TYPE_VOCAB (build_node_feature 30-dim, không hash w2v).
  2. load_groundtruth(): đọc danh sách UUID node độc hại (convention THREATRACE).
  3. sample_attack_subgraph(): BFS k-hop SEED TỪ node malicious -> subgraph chứa
     tín hiệu attack thật (khác UNICORN cắt random).
     sample_benign_subgraph(): BFS từ node benign, loại node trong groundtruth.
  4. build_darpa_corpus(): corpus bundle (schema chung với unicorn_subgraph_sampler)
     -> dùng bundle_to_edges_df của benign_interface.

Tải dữ liệu (thủ công, ~GB) — đặt vào data/darpa_tc/<dataset>/:
  - CADETS E3 JSON: ta1-cadets-e3-official-2.json[.N], ta1-cadets-e3-official.json[.N]
    (DARPA TC GitHub / Google Drive mirror).
  - groundtruth: cadets.txt (UUID độc hại) từ repo THREATRACE (folder groundtruth/).
"""

import os
import re
import json
import glob
import pickle
from collections import deque, defaultdict

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DARPA_DIR = os.path.join(BASE_DIR, "data", "darpa_tc")
COLS = ["actorID", "actor_type", "objectID", "object", "action", "timestamp"]

E_TARGET_MEAN, E_TARGET_STD = 62, 6
E_RANGE = (54, 78)

# ── CDM type -> schema hệ thống (process/file/socket) ─────────────────────────
# Khớp NODE_TYPE_VOCAB của graph_utils (process/file/socket/network/ip).
def _subject_type(_t):    # mọi Subject (PROCESS/THREAD/UNIT/BASIC_BLOCK) -> process
    return "process"

def _object_type(cdm_kind):
    if "NetFlow" in cdm_kind:
        return "socket"
    if "File" in cdm_kind:
        return "file"
    # Memory/SrcSink/Registry/Unknown -> file (NODE2ENV chỉ có process/file/socket)
    return "file"

# EVENT_* -> action name khớp ACTION_TYPE_VOCAB (read/write/execute/clone/send/mmap/load)
EVENT_ACTION = {
    "EVENT_WRITE": "write", "EVENT_WRITE_SOCKET_PARAMS": "write", "EVENT_SENDMSG": "send",
    "EVENT_READ": "read", "EVENT_READ_SOCKET_PARAMS": "read", "EVENT_RECVMSG": "read",
    "EVENT_RECVFROM": "read", "EVENT_SENDTO": "send", "EVENT_CONNECT": "send",
    "EVENT_EXECUTE": "execute", "EVENT_FORK": "clone", "EVENT_CLONE": "clone",
    "EVENT_MMAP": "mmap", "EVENT_MPROTECT": "mmap",
    "EVENT_LOADLIBRARY": "load", "EVENT_OPEN": "read", "EVENT_CLOSE": "read",
    "EVENT_MODIFY_FILE_ATTRIBUTES": "write", "EVENT_RENAME": "write",
    "EVENT_UNLINK": "write", "EVENT_CREATE_OBJECT": "write",
    "EVENT_ACCEPT": "send", "EVENT_BIND": "send",
}

_UUID_RE = re.compile(r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}")


def _find_cdm(datum: dict):
    """Trả về (cdm_kind_short, body) từ datum, robust với namespace cdm18/19/20."""
    for k, v in datum.items():
        short = k.rsplit(".", 1)[-1]   # 'com.bbn...cdm18.Event' -> 'Event'
        return short, v
    return None, None


def _extract_uuid(ref):
    """predicateObject/subject có thể là {'com...UUID': '<uuid>'} hoặc str trực tiếp."""
    if ref is None:
        return None
    if isinstance(ref, str):
        return ref
    if isinstance(ref, dict):
        for v in ref.values():
            if isinstance(v, str):
                return v
    return None


def parse_darpa_tc(dataset: str = "cadets", force: bool = False) -> pd.DataFrame:
    """
    Parse mọi file JSON trong data/darpa_tc/<dataset>/ -> DataFrame cạnh.
    Cache ra <dataset>_edges.pkl (parse 1 lần; file gốc lớn).
    """
    ddir = os.path.join(DARPA_DIR, dataset)
    cache = os.path.join(ddir, f"{dataset}_edges.pkl")
    if os.path.exists(cache) and not force:
        print(f"[DARPA] Load cache {cache}")
        return pd.read_pickle(cache)

    # Mọi file JSON (DARPA chia nhiều phần .json, .json.1, .json.2, ...)
    files = sorted(glob.glob(os.path.join(ddir, "*.json*")))
    files = [f for f in files if not f.endswith((".pkl", ".tar.gz", ".txt"))]
    if not files:
        raise FileNotFoundError(
            f"Không thấy JSON DARPA trong {ddir}. Hãy tải CADETS E3 JSON + đặt vào đó "
            f"(xem docstring darpa_tc_sampler.py)."
        )

    node_type = {}       # uuid -> process/file/socket
    rows = []
    n_lines = n_evt = 0
    for fp in files:
        print(f"[DARPA] Parsing {os.path.basename(fp)} ...")
        with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                n_lines += 1
                line = line.strip()
                if not line or '"datum"' not in line:
                    continue
                try:
                    datum = json.loads(line)["datum"]
                except Exception:
                    continue
                kind, body = _find_cdm(datum)
                if body is None:
                    continue
                if kind == "Subject":
                    node_type[body.get("uuid")] = _subject_type(body.get("type"))
                elif kind in ("FileObject", "NetFlowObject", "MemoryObject",
                              "SrcSinkObject", "RegistryKeyObject", "UnnamedPipeObject"):
                    node_type[body.get("uuid")] = _object_type(kind)
                elif kind == "Event":
                    etype = body.get("type", "")
                    action = EVENT_ACTION.get(etype)
                    if action is None:
                        continue
                    su = _extract_uuid(body.get("subject"))
                    ob = _extract_uuid(body.get("predicateObject"))
                    if not su or not ob:
                        continue
                    ts = body.get("timestampNanos", 0)
                    rows.append((su, ob, action, ts))
                    n_evt += 1

    # Build DataFrame; gán type (default process cho subject chưa thấy, file cho object)
    data = []
    for su, ob, action, ts in rows:
        st = node_type.get(su, "process")
        ot = node_type.get(ob, "file")
        data.append([su, st, ob, ot, action, str(ts)])
    df = pd.DataFrame(data, columns=COLS)
    print(f"[DARPA] {dataset}: lines={n_lines:,} events={n_evt:,} edges={len(df):,} "
          f"nodes={df[['actorID','objectID']].stack().nunique():,}")
    df.to_pickle(cache)
    print(f"[DARPA] Cached -> {cache}")
    return df


def load_groundtruth(dataset: str = "cadets") -> set:
    """Đọc UUID node độc hại từ data/darpa_tc/<dataset>/<dataset>.txt (convention THREATRACE)."""
    ddir = os.path.join(DARPA_DIR, dataset)
    candidates = [os.path.join(ddir, f"{dataset}.txt"),
                  os.path.join(ddir, "groundtruth.txt")]
    mal = set()
    for fp in candidates:
        if os.path.exists(fp):
            with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    for u in _UUID_RE.findall(line):
                        mal.add(u)
            break
    print(f"[DARPA] groundtruth malicious UUIDs: {len(mal)}")
    return mal


# ── Subgraph sampling (BFS liên thông, seed có kiểm soát) ─────────────────────
def _build_adj(df: pd.DataFrame):
    adj = defaultdict(list)   # node -> list of row-index
    for idx, (s, d) in enumerate(zip(df["actorID"].to_numpy(), df["objectID"].to_numpy())):
        adj[s].append(idx)
        adj[d].append(idx)
    return adj


def _bfs_subgraph(df, adj, seed, rng, e_target, mal_uuids=None, avoid_mal=False):
    """Thu ~e_target cạnh liên thông từ seed bằng BFS. Trả list row-index hoặc None."""
    src = df["actorID"].to_numpy()
    dst = df["objectID"].to_numpy()
    q, seen, sel, used = deque([seed]), {seed}, [], set()
    while q and len(sel) < e_target:
        u = q.popleft()
        nbr = adj.get(u, [])
        rng.shuffle(nbr)
        for ei in nbr:
            if len(sel) >= e_target:
                break
            if ei in used:
                continue
            s, d = src[ei], dst[ei]
            if avoid_mal and mal_uuids and (s in mal_uuids or d in mal_uuids):
                continue
            used.add(ei)
            sel.append(ei)
            nxt = d if u == s else s
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    if len(sel) < 30:
        return None
    return sel


def _bundle_from_rows(df, sel, label, source):
    """list row-index -> bundle schema chung (type/action THẬT của DARPA)."""
    src = df["actorID"].to_numpy(); dst = df["objectID"].to_numpy()
    st = df["actor_type"].to_numpy(); ot = df["object"].to_numpy()
    act = df["action"].to_numpy()
    ntype, outdeg, deg = {}, defaultdict(int), defaultdict(int)
    edges = []
    for order, ei in enumerate(sel):
        s, d = src[ei], dst[ei]
        ntype.setdefault(s, st[ei]); ntype.setdefault(d, ot[ei])
        outdeg[s] += 1; deg[s] += 1; deg[d] += 1
        edges.append({"source": s, "target": d, "type": act[ei],
                      "event_order": order, "is_bridge": False})
    hub = max(outdeg, key=outdeg.get) if outdeg else None
    nodes = [{"id": x, "type": ntype[x],
              "role": "hub" if x == hub else ("internal" if ntype[x] == "process" else
                      ("input_artifact" if outdeg[x] == 0 else "output_artifact"))}
             for x in ntype]
    return {"label": label, "source": source, "nodes": nodes, "edges": edges,
            "stats": {"nodes": len(nodes), "edges": len(edges)}}


def sample_attack_subgraph(df, adj, mal_nodes, rng, dataset="cadets"):
    """BFS seed từ 1 node malicious -> subgraph chứa tín hiệu attack."""
    if not mal_nodes:
        return None
    e_target = int(np.clip(round(rng.normal(E_TARGET_MEAN, E_TARGET_STD)), *E_RANGE))
    seed = mal_nodes[int(rng.integers(0, len(mal_nodes)))]
    sel = _bfs_subgraph(df, adj, seed, rng, e_target)
    if sel is None:
        return None
    return _bundle_from_rows(df, sel, "malicious", f"darpa_{dataset}")


def sample_benign_subgraph(df, adj, benign_nodes, mal_uuids, rng, dataset="cadets"):
    """BFS từ node benign, loại cạnh chạm node malicious -> subgraph benign sạch."""
    if not benign_nodes:
        return None
    e_target = int(np.clip(round(rng.normal(E_TARGET_MEAN, E_TARGET_STD)), *E_RANGE))
    seed = benign_nodes[int(rng.integers(0, len(benign_nodes)))]
    sel = _bfs_subgraph(df, adj, seed, rng, e_target, mal_uuids=mal_uuids, avoid_mal=True)
    if sel is None:
        return None
    return _bundle_from_rows(df, sel, "benign", f"darpa_{dataset}")


def build_darpa_corpus(n_benign=125, n_malicious=125, dataset="cadets", seed=0):
    """
    Corpus {benign, malicious} từ DARPA TC. Bundle schema giống unicorn_subgraph_sampler
    -> dùng bundle_to_edges_df (benign_interface). type/action THẬT -> build_node_feature.
    """
    rng = np.random.default_rng(seed)
    df = parse_darpa_tc(dataset)
    mal_uuids = load_groundtruth(dataset)
    adj = _build_adj(df)
    all_nodes = list(adj.keys())
    mal_nodes = [n for n in all_nodes if n in mal_uuids]
    benign_nodes = [n for n in all_nodes if n not in mal_uuids]
    print(f"[DARPA] nodes total={len(all_nodes):,} | malicious-in-graph={len(mal_nodes):,} "
          f"| benign={len(benign_nodes):,}")
    if not mal_nodes:
        raise RuntimeError("Không có node malicious nào của groundtruth xuất hiện trong graph "
                           "-> kiểm tra lại file groundtruth/UUID.")

    benign, malicious = [], []
    guard = 0
    while len(malicious) < n_malicious and guard < n_malicious * 50:
        b = sample_attack_subgraph(df, adj, mal_nodes, rng, dataset)
        if b: malicious.append(b)
        guard += 1
    guard = 0
    while len(benign) < n_benign and guard < n_benign * 50:
        b = sample_benign_subgraph(df, adj, benign_nodes, mal_uuids, rng, dataset)
        if b: benign.append(b)
        guard += 1
    print(f"[DARPA] corpus: benign={len(benign)} malicious={len(malicious)}")
    return benign, malicious


if __name__ == "__main__":
    import sys
    ds = sys.argv[1] if len(sys.argv) > 1 else "cadets"
    try:
        ben, mal = build_darpa_corpus(20, 20, dataset=ds, seed=0)
        from collections import Counter
        if mal:
            print("sample malicious:", mal[0]["stats"],
                  "node-type:", dict(Counter(n["type"] for n in mal[0]["nodes"])),
                  "edge-type:", dict(Counter(e["type"] for e in mal[0]["edges"])))
        if ben:
            print("sample benign   :", ben[0]["stats"],
                  "node-type:", dict(Counter(n["type"] for n in ben[0]["nodes"])))
    except FileNotFoundError as e:
        print("DATA CHƯA SẴN:", e)
