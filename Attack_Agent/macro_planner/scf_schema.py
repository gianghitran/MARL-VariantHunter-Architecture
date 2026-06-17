"""Security Context Feature helpers for THREATIFY-style planning.

The existing GNN model consumes the original 25-dimensional numeric vector.
This module adds a richer, explainable context layer used for pruning, bridge
selection, reporting, and macro-to-micro semantic checks.
"""

TACTIC_ORDER = [
    "reconnaissance", "resource-development", "initial-access", "execution",
    "persistence", "privilege-escalation", "defense-evasion",
    "credential-access", "discovery", "lateral-movement", "collection",
    "command-and-control", "exfiltration", "impact",
]

TACTIC_CAPABILITIES = {
    "reconnaissance": {"network", "discovery"},
    "resource-development": {"file", "payload"},
    "initial-access": {"file", "execution", "network"},
    "execution": {"process", "execution", "file"},
    "persistence": {"persistence", "file", "process"},
    "privilege-escalation": {"privilege", "execution", "process"},
    "defense-evasion": {"file", "stealth"},
    "credential-access": {"credential", "file"},
    "discovery": {"discovery", "file", "network"},
    "lateral-movement": {"network", "execution", "credential"},
    "collection": {"collection", "file"},
    "command-and-control": {"network", "c2"},
    "exfiltration": {"network", "exfiltration", "file"},
    "impact": {"impact", "file"},
}

BRIDGE_BY_TRANSITION = {
    ("initial-access", "execution"): "file",
    ("execution", "persistence"): "file",
    ("execution", "privilege-escalation"): "file",
    ("credential-access", "exfiltration"): "socket",
    ("collection", "exfiltration"): "socket",
    ("discovery", "command-and-control"): "socket",
    ("lateral-movement", "command-and-control"): "socket",
}


def normalize_platform(raw: str) -> str:
    value = (raw or "").lower().replace(" ", "-")
    if "windows" in value:
        return "windows"
    if "linux" in value:
        return "linux"
    if "macos" in value or "mac-os" in value:
        return "macos"
    if "network" in value:
        return "network"
    return value


def tactic_capabilities(tactics) -> set:
    caps = set()
    for tactic in tactics or []:
        caps.update(TACTIC_CAPABILITIES.get(tactic, set()))
    return caps


def bridge_hint(src_tactic: str, tgt_tactic: str, src_caps=None, tgt_caps=None) -> str:
    key = (src_tactic, tgt_tactic)
    if key in BRIDGE_BY_TRANSITION:
        return BRIDGE_BY_TRANSITION[key]

    caps = set(src_caps or []) | set(tgt_caps or [])
    if caps & {"network", "c2", "exfiltration"}:
        return "socket"
    return "file"


def context_similarity(ctx_a: dict, ctx_b: dict) -> float:
    """Jaccard-style similarity over explainable context sets."""
    if not ctx_a or not ctx_b:
        return 0.0

    sets = []
    for key in ("tactics", "platforms", "capabilities", "data_sources"):
        sets.append((set(ctx_a.get(key, [])), set(ctx_b.get(key, []))))

    scores = []
    for left, right in sets:
        if not left and not right:
            continue
        union = left | right
        scores.append(len(left & right) / len(union) if union else 0.0)

    priv_a = float(ctx_a.get("privilege_score", 0.0))
    priv_b = float(ctx_b.get("privilege_score", 0.0))
    scores.append(1.0 - min(abs(priv_a - priv_b), 1.0))

    impact_a = 1.0 if ctx_a.get("impact") else 0.0
    impact_b = 1.0 if ctx_b.get("impact") else 0.0
    scores.append(1.0 if impact_a == impact_b else 0.0)

    return sum(scores) / len(scores) if scores else 0.0
