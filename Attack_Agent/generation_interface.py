"""
generation_interface.py
========================
Wrapper cho Attack Agent pipeline (TAGAPT_generated_samples).

Gọi run_pipeline() như một hàm Python thay vì CLI,
trả về DataFrame-compatible dict để môi trường MARL sử dụng trực tiếp.

Chức năng chính:
  - generate_apt_subgraph(): Gọi pipeline và trả về dict gồm:
      'edges_df': pandas.DataFrame với các cột khớp với current_graph_df
      'nx_graph': networkx.DiGraph (full graph)
      'reward':   dict reward từ RewardCalculator (để tính PPO reward)
      'stats':    dict thống kê nodes/edges
  - NODE_TYPE_MAP: ánh xạ từ GAT node type sang actor_type/object của environment.
"""

import os
import sys
import random
import pandas as pd

# Thêm thư mục Attack_Agent vào sys.path để import các module của pipeline
_ATTACK_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _ATTACK_AGENT_DIR not in sys.path:
    sys.path.insert(0, _ATTACK_AGENT_DIR)

# Import pipeline core
try:
    from pipeline_runner import run_pipeline
    _PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"[GenerationInterface] WARNING: Cannot import run_pipeline: {e}")
    _PIPELINE_AVAILABLE = False

# Ánh xạ node_type của Attack Agent → actor_type/object của environment DataFrame
NODE_TYPE_MAP = {
    "process": "MP",   # Malicious Process
    "file":    "MF",   # Malicious File
    "socket":  "SO",   # Socket/IP
}

# Edge type map Attack Agent → environment action codes
EDGE_TYPE_MAP = {
    "FR":   "EX",  # Fork → Execute
    "WF":   "WR",  # Write File → WR
    "RF":   "RD",  # Read File  → RD
    "ST":   "ST",  # SendTo     → ST
    "RCV":  "RF",  # ReceiveFrom → RF
    "IJ":   "IJ",  # Inject     → IJ
    "EX":   "EX",  # Execute    → EX
    "LOAD": "RD",  # Load       → RD (map về Read)
}

# List các APT được hỗ trợ (mở rộng về sau)
SUPPORTED_APTS = [
    "APT28", "APT29", "APT32", "APT33", "APT38",
    "APT41", "Lazarus", "FIN7", "Carbanak"
]

SUPPORTED_VARIANTS = ["aligned", "evolved", "composite"]


def _nx_graph_to_dataframe(nx_graph) -> pd.DataFrame:
    """
    Chuyển đổi networkx.DiGraph từ Attack Agent pipeline
    sang pandas.DataFrame với các cột tương thích với environment.

    Columns: ['actorID', 'actor_type', 'objectID', 'object', 'action', 'timestamp']
    """
    rows = []
    for u, v, edge_data in nx_graph.edges(data=True):
        u_data = nx_graph.nodes[u]
        v_data = nx_graph.nodes[v]

        # Actor (source node)
        u_label    = u_data.get("label", str(u))
        u_ntype    = u_data.get("node_type", "process")
        u_type_str = NODE_TYPE_MAP.get(u_ntype, u_ntype)

        # Object (target node)
        v_label    = v_data.get("label", str(v))
        v_ntype    = v_data.get("node_type", "file")
        v_type_str = NODE_TYPE_MAP.get(v_ntype, v_ntype)

        # Edge action
        raw_action = edge_data.get("type", edge_data.get("edge_type", "WR"))
        action_code = EDGE_TYPE_MAP.get(raw_action.upper(), raw_action)

        # Timestamp: dùng event_order nếu có, nếu không dùng index
        timestamp = str(edge_data.get("event_order", edge_data.get("timestamp", "0")))

        rows.append({
            "actorID":    u_label,
            "actor_type": u_type_str,
            "objectID":   v_label,
            "object":     v_type_str,
            "action":     action_code,
            "timestamp":  timestamp,
        })

    if not rows:
        return pd.DataFrame(
            columns=["actorID", "actor_type", "objectID", "object", "action", "timestamp"]
        )

    return pd.DataFrame(rows)


def _generate_fallback_subgraph(apt_name: str = "APT28") -> dict:
    """
    Fallback khi pipeline không khả dụng (import error, thiếu dữ liệu).
    Sinh một subgraph APT đơn giản nhưng đúng định dạng để hệ thống không bị crash.
    """
    import random
    apt_prefix = apt_name.replace("APT", "")

    # Một số entity chuẩn APT
    processes = [f"MP-bash_{apt_prefix}", f"MP-python_{apt_prefix}", f"MP-curl_{apt_prefix}"]
    files     = [f"MF-payload_{apt_prefix}.sh", f"MF-config_{apt_prefix}.cfg"]
    sockets   = [f"SO-192.168.{random.randint(1,254)}.{random.randint(1,254)}"]
    actions   = ["EX", "WR", "RD", "ST", "RF"]

    rows = [
        # Lateral movement chain
        [processes[0], "MP", files[0],    "MF", "WR", "1"],
        [files[0],     "MF", processes[1],"MP", "EX", "2"],
        [processes[1], "MP", sockets[0],  "SO", "ST", "3"],
        [sockets[0],   "SO", processes[1],"MP", "RF", "4"],
        [processes[1], "MP", files[1],    "MF", "WR", "5"],
    ]

    df = pd.DataFrame(rows, columns=["actorID", "actor_type", "objectID", "object", "action", "timestamp"])
    print(f"[GenerationInterface] Using fallback subgraph for {apt_name} ({len(df)} edges).")
    return {
        "edges_df":  df,
        "nx_graph":  None,
        "reward":    {"total": 0.3, "coverage": 0.3, "validity": 0.5},
        "stats":     {"nodes": len(set(df["actorID"].tolist() + df["objectID"].tolist())), "edges": len(df)},
        "apt_name":  apt_name,
        "variant":   "fallback",
        "is_fallback": True,
    }


def generate_apt_subgraph(
    apt_name: str = None,
    variant_type: str = None,
    max_stages: int = 5,
    output_dir: str = None,
    cti_path: str = None,
    similarity_threshold: float = 0.8,
) -> dict:
    """
    Sinh một APT provenance subgraph bằng Attack Agent pipeline.

    Args:
        apt_name:            Tên APT group (vd: "APT28"). None = random.
        variant_type:        "aligned" | "evolved" | "composite". None = random.
        max_stages:          Số stage tối đa trong kill-chain.
        output_dir:          Thư mục lưu output của pipeline. None = temp dir.
        cti_path:            Path đến MITRE ATT&CK JSON. None = dùng default.
        similarity_threshold: Ngưỡng SCF/GNN cho pruning và composite.

    Returns:
        dict với các key:
            'edges_df':    pd.DataFrame với cột khớp current_graph_df
            'nx_graph':    networkx.DiGraph (full graph, có thể None nếu fallback)
            'reward':      dict reward từ RewardCalculator
            'stats':       dict {'nodes': int, 'edges': int}
            'apt_name':    str
            'variant':     str
            'is_fallback': bool
    """
    # Random APT/variant nếu không chỉ định → tăng diversity mỗi episode
    if apt_name is None:
        apt_name = random.choice(SUPPORTED_APTS)
    if variant_type is None:
        variant_type = random.choice(SUPPORTED_VARIANTS)

    # Default cti_path: relative từ thư mục Attack_Agent
    if cti_path is None:
        cti_path = os.path.join(_ATTACK_AGENT_DIR, "macro_planner", "data", "enterprise-attack.json")

    # Default output_dir
    if output_dir is None:
        output_dir = os.path.join(_ATTACK_AGENT_DIR, "result_handoff", apt_name)
    os.makedirs(output_dir, exist_ok=True)

    # Nếu pipeline không available, dùng fallback
    if not _PIPELINE_AVAILABLE:
        return _generate_fallback_subgraph(apt_name)

    # Đổi cwd tạm thời về Attack_Agent để các import relative hoạt động
    original_cwd = os.getcwd()
    try:
        os.chdir(_ATTACK_AGENT_DIR)
        print(f"[GenerationInterface] Running pipeline: APT={apt_name}, variant={variant_type}, stages<={max_stages}")
        result = run_pipeline(
            apt_name=apt_name,
            variant_type=variant_type,
            max_stages=max_stages,
            output_dir=output_dir,
            cti_path=cti_path,
            similarity_threshold=similarity_threshold,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[GenerationInterface] Pipeline error: {e}. Using fallback.")
        os.chdir(original_cwd)
        return _generate_fallback_subgraph(apt_name)
    finally:
        os.chdir(original_cwd)

    nx_graph = result.get("graph")
    reward   = result.get("reward", {})
    stats    = result.get("stats", {})

    # Chuyển đổi networkx graph → DataFrame
    if nx_graph is not None and nx_graph.number_of_edges() > 0:
        edges_df = _nx_graph_to_dataframe(nx_graph)
    else:
        print("[GenerationInterface] Empty graph from pipeline, using fallback.")
        return _generate_fallback_subgraph(apt_name)

    print(f"[GenerationInterface] Generated subgraph: {len(edges_df)} edges | "
          f"Reward total={reward.get('total', 0):.4f}")

    return {
        "edges_df":    edges_df,
        "nx_graph":    nx_graph,
        "reward":      reward,
        "stats":       {
            "nodes": stats.get("nodes", edges_df[["actorID", "objectID"]].values.ravel("K").tolist().__len__()),
            "edges": len(edges_df),
        },
        "apt_name":    apt_name,
        "variant":     variant_type,
        "is_fallback": False,
    }
