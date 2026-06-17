"""Technique-aware semantic checks for generated OS provenance clusters."""

TACTIC_REQUIREMENTS = {
    "reconnaissance": {"any_edges": {"ST", "RF", "WF"}},
    "initial-access": {"any_edges": {"RF", "WF", "EX", "FR"}},
    "execution": {"any_edges": {"FR", "EX", "WF"}},
    "persistence": {"any_edges": {"WF", "RF", "FR"}},
    "privilege-escalation": {"any_edges": {"EX", "FR", "IJ"}},
    "defense-evasion": {"any_edges": {"WF", "RF", "EX"}},
    "credential-access": {"any_edges": {"RF", "WF"}},
    "discovery": {"any_edges": {"RF", "WF", "ST"}},
    "lateral-movement": {"any_edges": {"ST", "RF", "EX", "WF"}},
    "collection": {"any_edges": {"RF", "WF"}},
    "command-and-control": {"any_edges": {"ST", "RCV", "RF"}},
    "exfiltration": {"any_edges": {"ST", "WF", "RF", "RCV"}},
}


class ClusterSemanticValidator:
    """Validate that a cluster expresses the macro technique/tactic intent."""

    def validate(self, graph, stage: dict) -> dict:
        tactic = stage.get("tactic", "unknown")
        technique_id = stage.get("technique_id", "unknown")
        req = TACTIC_REQUIREMENTS.get(tactic, {})
        edge_types = {
            data.get("edge_type", "")
            for _, _, data in graph.edges(data=True)
        }
        node_types = {
            data.get("node_type", "")
            for _, data in graph.nodes(data=True)
        }

        issues = []
        required_any = req.get("any_edges", set())
        if required_any and not (edge_types & required_any):
            issues.append(
                f"{technique_id}/{tactic}: missing expected edge among {sorted(required_any)}"
            )

        if tactic in {"command-and-control", "exfiltration", "lateral-movement"}:
            if "socket" not in node_types and not (edge_types & {"ST", "RCV"}):
                issues.append(f"{technique_id}/{tactic}: expected network/socket activity")

        if tactic in {"persistence", "credential-access", "collection"}:
            if "file" not in node_types:
                issues.append(f"{technique_id}/{tactic}: expected file artifact")

        return {
            "technique_id": technique_id,
            "tactic": tactic,
            "valid": len(issues) == 0,
            "issues": issues,
            "edge_types": sorted(edge_types),
            "node_types": sorted(node_types),
        }
