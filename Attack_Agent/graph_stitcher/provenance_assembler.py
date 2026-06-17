# graph_stitcher/provenance_assembler.py
"""
Ghép N subgraph clusters thành 1 final provenance graph.

Flow:
  1. Mỗi cluster được đọc từ file .dot output của TAGAPT
  2. Rename nodes để tránh collision (prefix stage_name)
  3. Thêm bridge edges từ hub node của cluster[i] đến hub node cluster[i+1]
  4. Export .dot final
"""
import networkx as nx
import re
import os
import html
from pathlib import Path
from .cluster_semantic_validator import ClusterSemanticValidator


class ProvenanceAssembler:
    def __init__(self, output_dir="./result"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def load_dot_subgraph(self, dot_path: str, prefix: str) -> nx.DiGraph:
        """
        Load .dot file và rename tất cả nodes với prefix.
        TAGAPT output .dot format: node_id [label="..."] hoặc node_id -> node_id2
        Parse từng dòng: phân loại edge (có "->") hay node definition.
        """
        G = nx.DiGraph()
        with open(dot_path, encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith("digraph") or line in ["{", "}"]:
                continue

            # Phân loại dòng: edge (có "->") hay node definition
            if "->" in line:
                # EDGE LINE: `src -> tgt [label="..."]`
                edge_m = re.match(
                    r'^([\w]+)\s*->\s*([\w]+)\s*(?:\[([^\]]*)\])?',
                    line
                )
                if edge_m:
                    src, tgt = edge_m.group(1), edge_m.group(2)
                    attrs_str = edge_m.group(3) or ""
                    label_m = re.search(r'label="([^"]*)"', attrs_str)
                    elabel = label_m.group(1) if label_m else ""
                    G.add_edge(
                        f"{prefix}__{src}",
                        f"{prefix}__{tgt}",
                        edge_type=elabel
                    )
            else:
                # NODE LINE: `node_id [label="..." ...]`
                node_m = re.match(
                    r'^([\w]+)\s*\[([^\]]*)\]',
                    line
                )
                if node_m:
                    node_id = node_m.group(1)
                    attrs_str = node_m.group(2)
                    label_m = re.search(r'label="([^"]*)"', attrs_str)
                    label = label_m.group(1) if label_m else node_id
                    type_m = re.search(r'type=(?:"([^"]*)"|([\w]+))', attrs_str)
                    ntype = ""
                    if type_m:
                        ntype = type_m.group(1) or type_m.group(2) or ""
                    G.add_node(
                        f"{prefix}__{node_id}",
                        label=label,
                        stage=prefix,
                        node_type=ntype
                    )

        return G

    def find_hub_node(self, G: nx.DiGraph) -> str:
        """Tìm hub node (node có degree cao nhất) trong subgraph."""
        if not G.nodes():
            return None
        degrees = dict(G.degree())
        return max(degrees, key=degrees.get)

    def assemble(self, cluster_dot_files: list, kill_chain_stages: list,
                  output_name: str = "final_provenance") -> nx.DiGraph:
        """
        Ghép các cluster thành final graph.

        Args:
            cluster_dot_files: ["/path/to/cluster_0.dot", ...]
            kill_chain_stages: [{"technique_id": "T1595", "tactic": "recon", ...}, ...]
            output_name: tên file output không có extension

        Returns: NetworkX DiGraph của final provenance graph
        """
        print(f"[Assembler] Assembling {len(cluster_dot_files)} clusters...")

        if len(cluster_dot_files) != len(kill_chain_stages):
            raise ValueError("Number of clusters must match kill_chain_stages")

        # Validate tactic diversity trước khi assemble
        tactics_in_chain = [s["tactic"] for s in kill_chain_stages]
        unique_tactics = set(tactics_in_chain)
        if len(unique_tactics) < 2:
            print(f"  WARNING: Kill-chain has low tactic diversity: {tactics_in_chain}")
            print(f"  Consider re-running KillChainPlanner with enforce_diversity=True")

        merged = nx.DiGraph()
        hub_nodes = []
        cluster_reports = []
        semantic_validator = ClusterSemanticValidator()

        for i, (dot_file, stage) in enumerate(zip(cluster_dot_files, kill_chain_stages)):
            tactic_short = stage["tactic"][:4]
            prefix = f"stage{i}_{tactic_short}"
            print(f"  Loading cluster {i}: {dot_file} -> prefix={prefix}")

            if not os.path.exists(dot_file):
                print(f"  WARNING: {dot_file} not found, skipping")
                hub_nodes.append(None)
                continue

            sg = self.load_dot_subgraph(dot_file, prefix)
            self._annotate_stage_graph(sg, stage, i, prefix)
            hub = self.find_hub_node(sg)
            if hub:
                sg.nodes[hub]["is_hub"] = True
            hub_nodes.append(hub)
            cluster_report = semantic_validator.validate(sg, stage)
            cluster_reports.append(cluster_report)
            if not cluster_report["valid"]:
                print(f"  WARNING: cluster semantic issues: {cluster_report['issues']}")

            print(f"  Cluster {i}: {sg.number_of_nodes()} nodes, hub={hub}")
            merged = nx.compose(merged, sg)

        # ── Bridge Injection ──────────────────────────────────────
        # Instead of direct Process→Process LATERAL_MOVE, inject
        # intermediary File or Socket nodes between consecutive stages.
        NETWORK_TACTICS = {
            "lateral-movement", "command-and-control", "exfiltration",
            "late", "comm", "exfi"
        }
        import random

        for i in range(len(hub_nodes) - 1):
            src_hub = hub_nodes[i]
            tgt_hub = hub_nodes[i + 1]
            if not src_hub or not tgt_hub:
                continue

            src_tactic = kill_chain_stages[i]["tactic"]
            tgt_tactic = kill_chain_stages[i + 1]["tactic"]
            bridge_id_base = f"bridge_s{i}_s{i+1}"

            # Prefer macro planner bridge hints, then fall back to tactic context.
            transition = kill_chain_stages[i].get("next_transition", {})
            requested_bridge = transition.get("bridge_hint", "")
            use_socket = requested_bridge == "socket" or (
                not requested_bridge and (
                    src_tactic in NETWORK_TACTICS or
                    tgt_tactic in NETWORK_TACTICS or
                    src_tactic[:4] in NETWORK_TACTICS or
                    tgt_tactic[:4] in NETWORK_TACTICS
                )
            )

            if use_socket:
                # ── Socket bridge: Process→ST→Socket→RCV→Process ──
                port = random.choice(["443", "80", "22", "4444", "8443"])
                ip = f"192.168.{random.randint(1,254)}.{random.randint(1,254)}"
                bridge_label = f"{ip}:{port}"
                bridge_id = f"{bridge_id_base}__sock"
                bridge_type = "socket"

                merged.add_node(bridge_id,
                                label=bridge_label,
                                stage=f"bridge_{i}_{i+1}",
                                node_type="socket",
                                is_bridge=True,
                                src_technique_id=kill_chain_stages[i].get("technique_id"),
                                tgt_technique_id=kill_chain_stages[i + 1].get("technique_id"),
                                src_tactic=src_tactic,
                                tgt_tactic=tgt_tactic)

                # ProcessA ──ST──> Socket
                merged.add_edge(src_hub, bridge_id,
                                edge_type="ST", bridge=True,
                                src_tactic=src_tactic,
                                tgt_tactic=tgt_tactic,
                                src_technique_id=kill_chain_stages[i].get("technique_id"),
                                tgt_technique_id=kill_chain_stages[i + 1].get("technique_id"),
                                transition_score=transition.get("score"),
                                bridge_hint=requested_bridge or "socket")
                # Socket ──RCV──> ProcessB
                merged.add_edge(bridge_id, tgt_hub,
                                edge_type="RCV", bridge=True,
                                src_tactic=src_tactic,
                                tgt_tactic=tgt_tactic,
                                src_technique_id=kill_chain_stages[i].get("technique_id"),
                                tgt_technique_id=kill_chain_stages[i + 1].get("technique_id"),
                                transition_score=transition.get("score"),
                                bridge_hint=requested_bridge or "socket")

                print(f"  Bridge [Socket]: {src_hub} ->ST-> {bridge_id} ->RCV-> {tgt_hub}")
            else:
                # ── File bridge: Process→WF→File→RF→Process ──
                file_names = ["staging_payload", "dropped_config", "exfil_data",
                              "lateral_tool", "pivot_script", "bridge_artifact"]
                file_exts = [".elf", ".sh", ".bin", ".dat"]
                bridge_label = f"{random.choice(file_names)}_s{i}s{i+1}{random.choice(file_exts)}"
                bridge_id = f"{bridge_id_base}__file"
                bridge_type = "file"

                merged.add_node(bridge_id,
                                label=bridge_label,
                                stage=f"bridge_{i}_{i+1}",
                                node_type="file",
                                is_bridge=True,
                                src_technique_id=kill_chain_stages[i].get("technique_id"),
                                tgt_technique_id=kill_chain_stages[i + 1].get("technique_id"),
                                src_tactic=src_tactic,
                                tgt_tactic=tgt_tactic)

                # ProcessA ──WF──> File
                merged.add_edge(src_hub, bridge_id,
                                edge_type="WF", bridge=True,
                                src_tactic=src_tactic,
                                tgt_tactic=tgt_tactic,
                                src_technique_id=kill_chain_stages[i].get("technique_id"),
                                tgt_technique_id=kill_chain_stages[i + 1].get("technique_id"),
                                transition_score=transition.get("score"),
                                bridge_hint=requested_bridge or "file")
                # File ──RF──> ProcessB (process reads the dropped file)
                merged.add_edge(bridge_id, tgt_hub,
                                edge_type="RF", bridge=True,
                                src_tactic=src_tactic,
                                tgt_tactic=tgt_tactic,
                                src_technique_id=kill_chain_stages[i].get("technique_id"),
                                tgt_technique_id=kill_chain_stages[i + 1].get("technique_id"),
                                transition_score=transition.get("score"),
                                bridge_hint=requested_bridge or "file")

                print(f"  Bridge [File]: {src_hub} ->WF-> {bridge_id} ->RF-> {tgt_hub}")

        print(f"[Assembler] Final graph: {merged.number_of_nodes()} nodes, "
              f"{merged.number_of_edges()} edges")

        out_path = self.output_dir / f"{output_name}.dot"
        self._export_dot(merged, out_path)
        merged.graph["cluster_reports"] = cluster_reports

        return merged

    def _annotate_stage_graph(self, G: nx.DiGraph, stage: dict,
                              stage_index: int, prefix: str):
        """Attach macro-stage metadata to every generated cluster node/edge."""
        technique_id = stage.get("technique_id")
        technique_name = stage.get("technique_name")
        tactic = stage.get("tactic")
        for _, attrs in G.nodes(data=True):
            attrs.update({
                "stage": prefix,
                "stage_index": stage_index,
                "technique_id": technique_id,
                "technique_name": technique_name,
                "tactic": tactic,
                "is_bridge": False,
            })
        for _, _, attrs in G.edges(data=True):
            attrs.update({
                "stage": prefix,
                "stage_index": stage_index,
                "technique_id": technique_id,
                "technique_name": technique_name,
                "tactic": tactic,
            })

    def _export_dot(self, G: nx.DiGraph, out_path: Path):
        """Export NetworkX graph sang .dot format with node_type preserved."""
        lines = ["digraph {"]
        for node, attrs in G.nodes(data=True):
            label = attrs.get("label", node)
            stage = attrs.get("stage", "")
            ntype = attrs.get("node_type", "process")
            color = self._get_stage_color(stage)
            dot_attrs = dict(attrs)
            dot_attrs.update({
                "label": label,
                "type": ntype,
                "style": "filled",
                "fillcolor": color,
            })
            lines.append(f'  "{node}" [{self._format_dot_attrs(dot_attrs)}]')
        for u, v, attrs in G.edges(data=True):
            etype = attrs.get("edge_type", "")
            bridge = attrs.get("bridge", False)
            dot_attrs = dict(attrs)
            dot_attrs.update({
                "label": etype,
                "style": "dashed" if bridge else "solid",
            })
            lines.append(f'  "{u}" -> "{v}" [{self._format_dot_attrs(dot_attrs)}]')
        lines.append("}")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[Assembler] Exported to {out_path}")

    def _format_dot_attrs(self, attrs: dict) -> str:
        rendered = []
        for key, value in attrs.items():
            if value is None:
                continue
            safe_key = str(key).replace("-", "_")
            if isinstance(value, bool):
                rendered.append(f'{safe_key}="{str(value).lower()}"')
            elif isinstance(value, (int, float)):
                rendered.append(f'{safe_key}="{value}"')
            else:
                safe_value = html.escape(str(value), quote=True)
                rendered.append(f'{safe_key}="{safe_value}"')
        return " ".join(rendered)

    def _get_stage_color(self, stage: str) -> str:
        colors = {
            "reco": "#FFE4B5", "init": "#FFB6C1", "exec": "#98FB98",
            "pers": "#87CEEB", "priv": "#DDA0DD", "defe": "#F0E68C",
            "stea": "#F0E68C", "cred": "#FF7F7F", "disc": "#90EE90",
            "late": "#20B2AA", "coll": "#FFD700", "comm": "#FFA500",
            "exfi": "#FF6347", "brid": "#D3D3D3", "reso": "#E6E6FA",
        }
        for key, color in colors.items():
            if key in stage:
                return color
        return "#FFFFFF"
