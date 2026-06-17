# graph_stitcher/action_masking.py
"""
OS-Level Action Space Masking for Provenance Graphs.

Enforces strict semantic rules that prevent invalid edges in OS provenance:
  - Process→Process: FR (Fork/Clone) only
  - Process→File:    RF (Read), WF (Write), or EX (Execute) — EX only for executables
  - Process→Socket:  ST (Send/Connect) only
  - File→Process:    RF (read/loaded by process) or LOAD
  - Socket→Process:  RCV (Receive) only

This module is used both at generation time (in _create_dummy_cluster) and
as a post-hoc validator (sanitize_graph) for MaskGAF-generated graphs.
"""

# ═══════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════

# File extensions that CANNOT be executed
NON_EXECUTABLE_EXTS = frozenset({
    ".txt", ".log", ".xml", ".csv", ".conf", ".cfg", ".ini",
    ".doc", ".docx", ".xls", ".xlsx", ".pdf", ".html", ".htm",
    ".json", ".yaml", ".yml", ".md", ".rst", ".dat", ".tmp",
    ".key", ".pem", ".crt", ".db", ".sqlite", ".bak",
    ".tar", ".gz", ".zip", ".enc", ".crontab",
})

# File extensions that CAN be executed
EXECUTABLE_EXTS = frozenset({
    ".sh", ".py", ".pl", ".rb", ".elf", ".bin", ".so",
    ".php", ".js", ".jar", ".class", ".service", ".bash",
    ".exe", ".dll", ".bat", ".ps1", ".cmd",
})

# ═══════════════════════════════════════════════════════════════════
# CORE RULES
# ═══════════════════════════════════════════════════════════════════

def _get_file_ext(label: str) -> str:
    """Extract file extension from a node label."""
    if not label:
        return ""
    # Handle labels like "payload_abc.elf" → ".elf"
    if "." in label:
        return "." + label.rsplit(".", 1)[-1].lower()
    return ""


def _is_executable(label: str) -> bool:
    """Check if a file label represents an executable file."""
    ext = _get_file_ext(label)
    if ext in EXECUTABLE_EXTS:
        return True
    if ext in NON_EXECUTABLE_EXTS:
        return False
    # Unknown extension: conservative — not executable
    return False


def get_valid_edges(src_type: str, tgt_type: str, tgt_label: str = "") -> set:
    """
    Return the set of valid edge types for a given (src_type, tgt_type) pair.

    Args:
        src_type: "process", "file", or "socket"
        tgt_type: "process", "file", or "socket"
        tgt_label: label of the target node (used for EX validation)

    Returns:
        Set of valid edge type strings, e.g. {"RF", "WF"}
    """
    src = src_type.lower()
    tgt = tgt_type.lower()

    if src == "process" and tgt == "process":
        # Rule B: Process→Process = Fork only
        return {"FR"}

    if src == "process" and tgt == "file":
        # Rule A: Process→File = RF or WF
        valid = {"RF", "WF"}
        # Rule C: EX only if file is executable
        if _is_executable(tgt_label):
            valid.add("EX")
        return valid

    if src == "process" and tgt == "socket":
        # Process→Socket = Send/Connect
        return {"ST"}

    if src == "file" and tgt == "process":
        # File→Process = rare reverse edge (loaded/triggered)
        return {"RF", "LOAD"}

    if src == "socket" and tgt == "process":
        # Socket→Process = Receive
        return {"RCV"}

    # Any other combo: no valid edges
    return set()


def validate_edge(src_type: str, tgt_type: str, tgt_label: str,
                  edge_type: str) -> bool:
    """
    Check if a specific edge is valid under OS provenance rules.

    Returns True if valid, False if violation.
    """
    valid = get_valid_edges(src_type, tgt_type, tgt_label)
    return edge_type in valid


def pick_valid_edge(src_type: str, tgt_type: str, tgt_label: str,
                    preferred: list = None) -> str:
    """
    Pick a valid edge type for the given node pair.
    Respects preference order if provided.

    Args:
        preferred: ordered list of preferred edge types to try first

    Returns:
        A valid edge type string, or "" if no valid edge exists
    """
    import random
    valid = get_valid_edges(src_type, tgt_type, tgt_label)
    if not valid:
        return ""

    if preferred:
        for pref in preferred:
            if pref in valid:
                return pref

    return random.choice(list(valid))


def sanitize_graph(G) -> tuple:
    """
    Post-hoc validation: scan all edges in a NetworkX DiGraph and
    fix any violations.

    Returns:
        (G, violations_list) where violations_list contains details
        of all edges that were corrected.
    """
    violations = []

    for u, v, data in list(G.edges(data=True)):
        edge_type = data.get("edge_type", "")
        src_type = G.nodes[u].get("node_type", "process")
        tgt_type = G.nodes[v].get("node_type", "process")
        tgt_label = G.nodes[v].get("label", "")

        if not validate_edge(src_type, tgt_type, tgt_label, edge_type):
            valid = get_valid_edges(src_type, tgt_type, tgt_label)
            if valid:
                import random
                new_type = random.choice(list(valid))
                violations.append({
                    "edge": (u, v),
                    "old_type": edge_type,
                    "new_type": new_type,
                    "reason": f"{src_type}→{tgt_type} cannot use '{edge_type}'"
                })
                G[u][v]["edge_type"] = new_type
            else:
                # No valid edge exists for this pair — remove edge
                violations.append({
                    "edge": (u, v),
                    "old_type": edge_type,
                    "new_type": "REMOVED",
                    "reason": f"No valid edge for {src_type}→{tgt_type}"
                })
                G.remove_edge(u, v)

    return G, violations
