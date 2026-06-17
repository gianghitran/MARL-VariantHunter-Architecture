"""Export generated provenance graphs for downstream Detection Agents."""

import json
import os
from pathlib import Path


SCHEMA_VERSION = "threatify-tagapt-detection-v1"


def _json_safe(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _node_role(node_id: str, attrs: dict) -> str:
    if attrs.get("is_bridge"):
        return "bridge"
    if "__p0_" in node_id or attrs.get("is_hub"):
        return "hub"
    if "__fin" in node_id:
        return "input_artifact"
    if "__fh" in node_id:
        return "output_artifact"
    if "__exec" in node_id:
        return "executable_artifact"
    return "internal"


def export_detection_bundle(graph, kill_chain: list, output_dir: str,
                            output_name: str, apt_name: str,
                            variant_type: str, planner_meta: dict,
                            reward: dict, quality_valid: bool) -> dict:
    """Write a JSON bundle with full metadata needed by Detection Agents."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    nodes = []
    for node_id, attrs in graph.nodes(data=True):
        nodes.append({
            "id": node_id,
            "label": attrs.get("label", node_id),
            "type": attrs.get("node_type", "process"),
            "stage": attrs.get("stage", ""),
            "stage_index": attrs.get("stage_index"),
            "technique_id": attrs.get("technique_id"),
            "technique_name": attrs.get("technique_name"),
            "tactic": attrs.get("tactic"),
            "role": _node_role(node_id, attrs),
            "is_bridge": bool(attrs.get("is_bridge", False)),
        })

    edges = []
    for event_order, (src, tgt, attrs) in enumerate(graph.edges(data=True)):
        edge_type = attrs.get("edge_type", "")
        edges.append({
            "id": f"e{event_order:06d}",
            "source": src,
            "target": tgt,
            "type": edge_type,
            "event": edge_type,
            "event_order": event_order,
            "timestamp": event_order,
            "stage_index": attrs.get("stage_index"),
            "technique_id": attrs.get("technique_id"),
            "technique_name": attrs.get("technique_name"),
            "tactic": attrs.get("tactic"),
            "is_bridge": bool(attrs.get("bridge", False)),
            "src_tactic": attrs.get("src_tactic"),
            "tgt_tactic": attrs.get("tgt_tactic"),
            "src_technique_id": attrs.get("src_technique_id"),
            "tgt_technique_id": attrs.get("tgt_technique_id"),
            "bridge_hint": attrs.get("bridge_hint"),
            "transition_score": attrs.get("transition_score"),
        })

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "graph_id": output_name,
        "apt": apt_name,
        "variant_type": variant_type,
        "label": "malicious",
        "task": "apt_provenance_detection",
        "quality_valid": quality_valid,
        "stats": {
            "nodes": len(nodes),
            "edges": len(edges),
            "clusters": len(kill_chain),
        },
        "kill_chain": [
            {
                "stage_index": i,
                "technique_id": stage.get("technique_id"),
                "technique_name": stage.get("technique_name"),
                "tactic": stage.get("tactic"),
                "scf_context": stage.get("scf_context", {}),
                "next_transition": stage.get("next_transition"),
            }
            for i, stage in enumerate(kill_chain)
        ],
        "planner_meta": planner_meta,
        "reward": reward,
        "cluster_reports": graph.graph.get("cluster_reports", []),
        "node_schema": {
            "id": "stable node id",
            "label": "process/file/socket name",
            "type": "process | file | socket",
            "stage_index": "integer stage order, null for bridge nodes",
            "technique_id": "MITRE ATT&CK technique id for generated stage",
            "tactic": "MITRE ATT&CK tactic",
            "role": "hub | bridge | input_artifact | output_artifact | executable_artifact | internal",
        },
        "edge_schema": {
            "source": "source node id",
            "target": "target node id",
            "type": "FR | RF | WF | EX | ST | RCV | IJ | LOAD",
            "event_order": "synthetic temporal order",
            "timestamp": "same as event_order for now",
            "is_bridge": "true when edge connects two technique clusters",
        },
        "nodes": nodes,
        "edges": edges,
    }

    json_path = os.path.join(output_dir, f"{output_name}_detection.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(bundle), f, indent=2, ensure_ascii=False)

    return {"detection_json": json_path}
