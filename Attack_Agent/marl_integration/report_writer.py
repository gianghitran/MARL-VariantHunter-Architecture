"""Write explainable pipeline reports for generated provenance graphs."""

import json
import os
from pathlib import Path


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


def _stage_summary(stage: dict) -> dict:
    return {
        "technique_id": stage.get("technique_id"),
        "technique_name": stage.get("technique_name"),
        "tactic": stage.get("tactic"),
        "context": stage.get("scf_context", {}),
        "next_transition": stage.get("next_transition"),
    }


def write_pipeline_report(output_dir: str, output_name: str, apt_name: str,
                          variant_type: str, kill_chain: list,
                          planner_meta: dict, stats: dict, reward: dict,
                          quality_valid: bool, dot_path: str) -> dict:
    """Persist machine-readable and human-readable reports."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cluster_reports = []
    graph = stats.get("graph")
    if graph is not None:
        cluster_reports = graph.graph.get("cluster_reports", [])

    report = {
        "apt": apt_name,
        "variant_type": variant_type,
        "dot_path": dot_path,
        "quality_valid": quality_valid,
        "stats": {k: v for k, v in stats.items() if k != "graph"},
        "reward": reward,
        "planner_meta": planner_meta,
        "kill_chain": [_stage_summary(stage) for stage in kill_chain],
        "cluster_reports": cluster_reports,
    }

    json_path = os.path.join(output_dir, f"{output_name}_report.json")
    md_path = os.path.join(output_dir, f"{output_name}_report.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(report), f, indent=2, ensure_ascii=False)

    lines = [
        f"# {apt_name} {variant_type} provenance report",
        "",
        f"- DOT: `{dot_path}`",
        f"- Quality valid: `{quality_valid}`",
        f"- Nodes: `{report['stats'].get('nodes')}`",
        f"- Edges: `{report['stats'].get('edges')}`",
        f"- Clusters: `{report['stats'].get('clusters')}`",
        f"- Total reward: `{reward.get('total', 0.0):.4f}`",
        f"- Transition quality: `{reward.get('transition_quality', 0.0):.4f}`",
        f"- Cluster validity: `{reward.get('cluster_validity', 0.0):.4f}`",
        "",
        "## Kill Chain",
    ]

    for idx, stage in enumerate(kill_chain):
        trans = stage.get("next_transition", {})
        score = trans.get("score")
        bridge = trans.get("bridge_hint", "")
        suffix = ""
        if score is not None:
            suffix = f" -> score={score:.4f}, bridge={bridge}"
        lines.append(
            f"{idx + 1}. `{stage.get('technique_id')}` "
            f"{stage.get('technique_name')} ({stage.get('tactic')}){suffix}"
        )

    replacements = planner_meta.get("composite_replacements", [])
    removed = planner_meta.get("pruning", {}).get("removed", [])
    if replacements:
        lines.extend(["", "## Composite Replacements"])
        for item in replacements:
            lines.append(
                f"- `{item.get('replaced')}` -> "
                f"`{item.get('replacement_sequence')}` "
                f"from `{item.get('source_apt')}` "
                f"(similarity={item.get('similarity', 0.0):.4f})"
            )
    if removed:
        lines.extend(["", "## Pruned Techniques"])
        for item in removed:
            lines.append(
                f"- `{item.get('technique_id')}`: {item.get('reason')} "
                f"(incoming={item.get('incoming_score', 0.0):.4f}, "
                f"bypass={item.get('bypass_score', 0.0):.4f})"
            )

    invalid_clusters = [r for r in cluster_reports if not r.get("valid")]
    if invalid_clusters:
        lines.extend(["", "## Cluster Issues"])
        for report_item in invalid_clusters:
            lines.append(
                f"- `{report_item.get('technique_id')}` "
                f"{report_item.get('issues', [])}"
            )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return {"json": json_path, "markdown": md_path}
