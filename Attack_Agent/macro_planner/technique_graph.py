# macro_planner/technique_graph.py
"""
Xây dựng NetworkX graph của techniques từ MITRE relationships.
Node = technique_id, Edge = "uses" hoặc "related-to" relationship
"""
import json
import pickle
import networkx as nx
from pathlib import Path


class TechniqueGraphBuilder:
    def __init__(self, cti_json_path="macro_planner/data/enterprise-attack.json"):
        with open(cti_json_path, encoding="utf-8") as f:
            self.data = json.load(f)
        self.id_to_tid = {}  # {stix_id: technique_external_id}
        self._build_id_map()

    def _build_id_map(self):
        """Map STIX ID (attack-pattern--uuid) sang external ID (T1059)."""
        for obj in self.data["objects"]:
            if obj.get("type") != "attack-pattern":
                continue
            stix_id = obj.get("id", "")
            for ref in obj.get("external_references", []):
                if ref.get("source_name") == "mitre-attack":
                    self.id_to_tid[stix_id] = ref.get("external_id", "")

    def build_graph(self) -> nx.DiGraph:
        cache_path = Path("macro_planner/data/technique_graph.pkl")
        if cache_path.exists():
            print("[TechGraph] Loading from cache...")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        G = nx.DiGraph()

        # Add nodes
        for obj in self.data["objects"]:
            if obj.get("type") == "attack-pattern" and not obj.get("revoked"):
                for ref in obj.get("external_references", []):
                    if ref.get("source_name") == "mitre-attack":
                        tid = ref.get("external_id", "")
                        if tid.startswith("T"):
                            phases = [p["phase_name"] for p in obj.get("kill_chain_phases", [])]
                            G.add_node(tid, name=obj.get("name", ""), tactics=phases)

        # Add edges từ relationships
        for obj in self.data["objects"]:
            if obj.get("type") != "relationship":
                continue
            rel_type = obj.get("relationship_type", "")
            if rel_type not in ["uses", "subtechnique-of", "related-to"]:
                continue
            src = self.id_to_tid.get(obj.get("source_ref", ""), "")
            tgt = self.id_to_tid.get(obj.get("target_ref", ""), "")
            if src and tgt and src.startswith("T") and tgt.startswith("T"):
                G.add_edge(src, tgt, rel_type=rel_type)

        print(f"[TechGraph] Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(G, f)
        return G
