import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Node type encoding: chuyen node_type sang one-hot-like vector
# De GAT phan biet duoc Process / File / Socket / Unknown
# ─────────────────────────────────────────────────────────────────────────────

NODE_TYPE_VOCAB = {
    "MP": 0,       # Malicious Process
    "TP": 1,       # Trusted Process
    "MF": 2,       # Malicious File
    "TF": 3,       # Trusted File
    "SO": 4,       # Socket / IP
    "unknown": 5,  # Fallback
}

ACTION_TYPE_VOCAB = {
    "WR": 0, "write": 0,
    "RD": 1, "read": 1,
    "EX": 2, "execute": 2,
    "ST": 3, "mmap": 3,
    "RF": 4, "clone": 4,
    "IJ": 5,
}

N_NODE_TYPES  = len(NODE_TYPE_VOCAB)    # 6
N_ACTION_TYPES = len(ACTION_TYPE_VOCAB) // 2  # 6 unique actions
EMBED_DIM      = 30


def _node_type_onehot(node_type_str: str) -> np.ndarray:
    """One-hot vector kich thuoc N_NODE_TYPES cho node_type."""
    vec = np.zeros(N_NODE_TYPES, dtype=np.float32)
    idx = NODE_TYPE_VOCAB.get(str(node_type_str).strip(), NODE_TYPE_VOCAB["unknown"])
    vec[idx] = 1.0
    return vec


def _action_freq_vector(action_list: list) -> np.ndarray:
    """Tan suat cac action type cho mot node (normalized)."""
    vec = np.zeros(N_ACTION_TYPES, dtype=np.float32)
    for a in action_list:
        idx = ACTION_TYPE_VOCAB.get(str(a).lower(), -1)
        if idx >= 0:
            vec[idx] += 1.0
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


def infer(document, w2vmodel, encoder):
    """
    Tinh embedding cho mot node dua tren tap actions cua no.
    Ket qua la vector EMBED_DIM-dim (30-dim de phu hop GAT in_channels=30).
    """
    if w2vmodel is None:
        return np.random.rand(EMBED_DIM).astype(np.float32)
    word_embeddings = [w2vmodel.wv[word] for word in document if word in w2vmodel.wv]
    if not word_embeddings:
        return np.zeros(EMBED_DIM, dtype=np.float32)
    output_embedding = torch.tensor(np.array(word_embeddings), dtype=torch.float)
    if len(document) < 100000:
        output_embedding = encoder.embed(output_embedding)
    output_embedding = output_embedding.detach().cpu().numpy()
    return np.mean(output_embedding, axis=0)


def prepare_graph(df):
    """
    Chuyen pandas.DataFrame (subgraph) sang cau truc du lieu cho GAT.

    Moi node duoc bieu dien bang vector 30-dim:
      - W2V embedding cua cac action no thuc hien
      (Tuong lai: co the ket hop voi node_type one-hot neu tang EMBED_DIM)

    Returns:
        phrases    : list of list[str]  - tap action labels cho moi node
        feat_labels: list[int]          - dummy labels (-1)
        edge_index : list[[src], [dst]] - edge list dang index
        node_names : list[str]          - ten node theo thu tu
    """
    nodes      = {}   # node_id -> list of actions
    node_types = {}   # node_id -> type string (MP/MF/SO...)

    for row in df.itertuples():
        actor_id  = row.actorID
        object_id = row.objectID
        action    = row.action

        # Actor node
        nodes.setdefault(actor_id, []).append(action)
        if actor_id not in node_types:
            node_types[actor_id] = getattr(row, "actor_type", "unknown")

        # Object node
        nodes.setdefault(object_id, []).append(action)
        if object_id not in node_types:
            node_types[object_id] = getattr(row, "object", "unknown")

    node_names = list(nodes.keys())
    features   = [nodes[n] for n in node_names]
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
