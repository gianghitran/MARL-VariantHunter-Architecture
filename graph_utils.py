"""
graph_utils.py
==============
Chuyen subgraph DataFrame -> cau truc GAT-compatible.

Feature vector cho moi node (20-dim, khop voi GAT in_channels=20):
  - 6 dim: one-hot node_type (MP/TP/MF/TF/SO/unknown)
  - 6 dim: action frequency vector (WR/RD/EX/ST/RF/other), normalized
  - 3 dim: structural features (degree, in-degree, out-degree), normalized
  - 5 dim: zero-padding cho tuong lai (extra semantic features)

Thay the cach dung W2V (vocab 60 hash IDs, khong chua action names) de tranh
truong hop infer() tra ve np.zeros(30) lam cho GAT output = 0.
"""

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary
# ─────────────────────────────────────────────────────────────────────────────

NODE_TYPE_VOCAB = {
    "MP": 0, "TP": 1, "process": 1, "thread": 1,       # Malicious/Trusted Process
    "MF": 2, "TF": 3, "file": 3, "module": 3,          # Malicious/Trusted File
    "SO": 4, "network": 4, "socket": 4, "ip": 4,       # Socket / Network
    "unknown": 5,                                      # Fallback
}

ACTION_TYPE_VOCAB = {
    # Write / create
    "WF": 0, "WR": 0, "write": 0,
    # Read / fetch
    "RF": 1, "RD": 1, "RCV": 1, "read": 1,
    # Execute / spawn
    "EX": 2, "execute": 2, "clone": 2,
    # Send / connect / mmap
    "ST": 3, "mmap": 3, "send": 3,
    # File-read / receive
    "FR": 4, "load": 4,
    # Other / fallback
}

N_NODE_TYPES   = len(NODE_TYPE_VOCAB)     # 6
N_ACTION_TYPES = 5                         # 5 nhom action
EMBED_DIM      = 30                        # Khop voi GAT in_channels=30 (checkpoint weight shape)


def _node_type_onehot(node_type_str: str) -> np.ndarray:
    """One-hot vector kich thuoc N_NODE_TYPES cho node_type (6-dim)."""
    vec = np.zeros(N_NODE_TYPES, dtype=np.float32)
    idx = NODE_TYPE_VOCAB.get(str(node_type_str).strip(), NODE_TYPE_VOCAB["unknown"])
    vec[idx] = 1.0
    return vec


def _action_freq_vector(action_list: list) -> np.ndarray:
    """
    Tan suat cac action type cho mot node (5-dim), normalized.
    Nhom action theo 5 loai chinh: Write, Read, Execute, Send, FileRead.
    """
    vec = np.zeros(N_ACTION_TYPES, dtype=np.float32)
    for a in action_list:
        idx = ACTION_TYPE_VOCAB.get(str(a).strip(), -1)
        if 0 <= idx < N_ACTION_TYPES:
            vec[idx] += 1.0
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


def build_node_feature(node_type: str, action_list: list,
                       degree: int = 0, in_deg: int = 0, out_deg: int = 0,
                       max_degree: int = 1) -> np.ndarray:
    """
    Tao feature vector EMBED_DIM-dim (30-dim) cho mot node:
      [0:6]   node_type one-hot           (6-dim)
      [6:11]  action frequency            (5-dim)
      [11:14] structural (deg/in/out) norm (3-dim)
      [14:30] zero-pad (cho mo rong sau)   (16-dim)

    Tong = 30-dim, khop voi GAT in_channels=30.
    """
    feat = np.zeros(EMBED_DIM, dtype=np.float32)

    # Node type one-hot [0:6]
    feat[0:N_NODE_TYPES] = _node_type_onehot(node_type)

    # Action frequency [6:11]
    feat[N_NODE_TYPES:N_NODE_TYPES + N_ACTION_TYPES] = _action_freq_vector(action_list)

    # Structural features (normalized by max_degree) [11:14]
    md = max(max_degree, 1)
    feat[11] = degree  / md
    feat[12] = in_deg  / md
    feat[13] = out_deg / md

    # [14:30] zero-pad
# --- Legacy shim: infer() duoc giu lai de khong bi loi ImportError tu environment.py ---
# Nhung gio no dung tuple (action_list, node_type) hoac chi action_list
def infer(document, w2vmodel, encoder):
    """
    Su dung w2vmodel de embedding cho tap action hashes, ket hop Positional Encoding.
    """
    if isinstance(document, tuple) and len(document) == 2:
        action_list, node_type = document
    else:
        action_list = document
        node_type = "unknown"
        
    if not w2vmodel:
        return np.zeros(30, dtype=np.float32)
        
    vecs = []
    for i, action in enumerate(action_list):
        action_str = str(action).strip()
        if action_str in w2vmodel.wv:
            # Lấy vector tu Word2Vec [30]
            v = w2vmodel.wv[action_str].copy()
            if encoder is not None:
                try:
                    import torch
                    pos = torch.tensor([i], dtype=torch.long)
                    pos_emb = encoder(pos).squeeze(0).detach().numpy()
                    v += pos_emb
                except Exception:
                    pass
            vecs.append(v)
            
    if len(vecs) == 0:
        return np.zeros(30, dtype=np.float32)
        
    return np.mean(vecs, axis=0)


def prepare_graph(df):
    """
    Chuyen pandas.DataFrame (subgraph) sang cau truc du lieu cho GAT.

    Moi node duoc bieu dien bang vector 20-dim (build_node_feature):
      - node_type one-hot (6-dim)
      - action frequency (5-dim)
      - structural degree (3-dim)
      - zero-pad (6-dim)

    Returns:
        phrases    : list of tuples (action_list, node_type)
        feat_labels: list[int]          - dummy labels (-1)
        edge_index : list[[src], [dst]] - edge list dang index
        node_names : list[str]          - ten node theo thu tu
    """
    nodes      = {}   # node_id -> list of actions
    node_types = {}   # node_id -> type string
    in_degrees  = {}  # node_id -> in-degree count
    out_degrees = {}  # node_id -> out-degree count

    for row in df.itertuples():
        actor_id  = row.actorID
        object_id = row.objectID
        action    = row.action

        # Actor node
        nodes.setdefault(actor_id, []).append(action)
        if actor_id not in node_types:
            node_types[actor_id] = getattr(row, "actor_type", "unknown")
        out_degrees[actor_id] = out_degrees.get(actor_id, 0) + 1
        in_degrees.setdefault(actor_id, 0)

        # Object node
        nodes.setdefault(object_id, []).append(action)
        if object_id not in node_types:
            node_types[object_id] = getattr(row, "object_type", "unknown") if hasattr(row, "object_type") else getattr(row, "object", "unknown")
        in_degrees[object_id] = in_degrees.get(object_id, 0) + 1
        out_degrees.setdefault(object_id, 0)

    node_names = list(nodes.keys())
    max_deg = max(
        max((out_degrees.get(n, 0) + in_degrees.get(n, 0)) for n in node_names) if node_names else 0,
        1
    )

    # Build feature phrases (danh sach action, node_type) cho tung node
    features   = [(nodes[n], node_types.get(n, "unknown")) for n in node_names]
    feat_labels = [-1] * len(node_names)

    # Edge index: dung index trong node_names
    name_to_idx = {n: i for i, n in enumerate(node_names)}
    edge_index  = [[], []]
    for row in df.itertuples():
        src_idx = name_to_idx[row.actorID]
        dst_idx = name_to_idx[row.objectID]
        edge_index[0].append(src_idx)
        edge_index[1].append(dst_idx)

    return features, feat_labels, edge_index, node_names
