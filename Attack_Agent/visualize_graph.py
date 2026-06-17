#!/usr/bin/env python
"""
visualize_graph.py
==================
Render file .dot output từ pipeline thành hình ảnh PNG/SVG/PDF.

Hỗ trợ 2 backend:
  1. Graphviz (mặc định, đẹp hơn)  — cần cài graphviz system binary
  2. Matplotlib + NetworkX (fallback) — luôn có sẵn

Usage:
    python visualize_graph.py result/APT28_evolved_provenance.dot
    python visualize_graph.py result/APT28_evolved_provenance.dot --format svg
    python visualize_graph.py result/APT28_evolved_provenance.dot --backend matplotlib
    python visualize_graph.py result/APT28_evolved_provenance.dot --open
"""
import argparse
import os
import re
import sys
from pathlib import Path
import math


# ═══════════════════════════════════════════════════════════════════
# STAGE COLOR PALETTE — phân biệt rõ ràng các stage bằng màu sắc
# ═══════════════════════════════════════════════════════════════════

STAGE_COLORS = {
    "reco": {"fill": "#FFF3E0", "border": "#E65100", "label": "Reconnaissance"},
    "reso": {"fill": "#E3F2FD", "border": "#1565C0", "label": "Resource Dev"},
    "init": {"fill": "#FCE4EC", "border": "#C62828", "label": "Initial Access"},
    "exec": {"fill": "#E8F5E9", "border": "#2E7D32", "label": "Execution"},
    "pers": {"fill": "#E0F7FA", "border": "#00695C", "label": "Persistence"},
    "priv": {"fill": "#F3E5F5", "border": "#6A1B9A", "label": "Privilege Escalation"},
    "defe": {"fill": "#FFFDE7", "border": "#F57F17", "label": "Defense Evasion"},
    "stea": {"fill": "#FFFDE7", "border": "#F57F17", "label": "Stealth"},
    "cred": {"fill": "#FFEBEE", "border": "#B71C1C", "label": "Credential Access"},
    "disc": {"fill": "#E8F5E9", "border": "#1B5E20", "label": "Discovery"},
    "late": {"fill": "#E0F2F1", "border": "#004D40", "label": "Lateral Movement"},
    "coll": {"fill": "#FFF8E1", "border": "#FF6F00", "label": "Collection"},
    "comm": {"fill": "#FBE9E7", "border": "#BF360C", "label": "Command & Control"},
    "exfi": {"fill": "#FFCCBC", "border": "#D84315", "label": "Exfiltration"},
    "brid": {"fill": "#ECEFF1", "border": "#546E7A", "label": "Bridge"},
}

NODE_SHAPE_MAP = {
    "process": {"shape": "box", "mpl_marker": "s"},
    "file":    {"shape": "note", "mpl_marker": "D"},
    "socket":  {"shape": "hexagon", "mpl_marker": "h"},
}


def parse_dot_file(dot_path):
    """Parse .dot file thành lists of nodes và edges."""
    nodes = {}  # id -> {label, stage, fillcolor, ...}
    edges = []  # [(src, dst, attrs)]

    with open(dot_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("digraph") or line in ("{", "}"):
                continue

            # Edge: "src" -> "dst" [label="..." style=...]
            edge_m = re.match(
                r'"([^"]+)"\s*->\s*"([^"]+)"\s*\[label="([^"]*)"(?:\s+style=(\w+))?\]',
                line)
            if edge_m:
                src, dst, elabel, style = edge_m.groups()
                edges.append((src, dst, {
                    "label": elabel,
                    "style": style or "solid",
                    "bridge": (style == "dashed"),
                }))
                continue

            # Node: "id" [label="..." type=... style=filled fillcolor="..."]
            node_m = re.match(
                r'"([^"]+)"\s*\[label="([^"]*)"(?:[^\]]*style=(\w+))?'
                r'(?:[^\]]*fillcolor="([^"]*)")?\]',
                line)
            if node_m:
                nid, label, style, fillcolor = node_m.groups()
                # Infer stage from node id
                stage = "unknown"
                stage_m = re.search(r'(stage\d+_\w+)', nid)
                if stage_m:
                    stage = stage_m.group(1)
                elif "bridge" in nid:
                    stage_m = re.search(r'(bridge_s\d+_s\d+)', nid)
                    if stage_m:
                        stage = stage_m.group(1)
                # Infer node type from id prefix
                ntype = "process"
                if "__f" in nid or "__exec_" in nid:
                    ntype = "file"
                elif "__sock_" in nid:
                    ntype = "socket"

                nodes[nid] = {
                    "label": label,
                    "stage": stage,
                    "type": ntype,
                    "fillcolor": fillcolor or "#FFFFFF",
                }

    return nodes, edges


def _get_stage_info(stage_key):
    """Get color info for a stage."""
    for key, info in STAGE_COLORS.items():
        if key in stage_key:
            return info
    return {"fill": "#F5F5F5", "border": "#616161", "label": stage_key}


# ═══════════════════════════════════════════════════════════════════
# BACKEND 1: Graphviz (đẹp hơn, layout tự động)
# ═══════════════════════════════════════════════════════════════════

def render_graphviz(dot_path, output_path, fmt="png"):
    """Dùng graphviz Python library để render."""
    import graphviz

    nodes, edges = parse_dot_file(dot_path)

    # Group nodes by stage
    stages = {}
    for nid, attrs in nodes.items():
        s = attrs["stage"]
        stages.setdefault(s, []).append(nid)

    dot = graphviz.Digraph(
        name="APT_Provenance",
        format=fmt,
        engine="dot",
    )
    dot.attr(
        rankdir="TB",
        bgcolor="#1a1a2e",
        fontname="Consolas",
        fontsize="11",
        fontcolor="white",
        label=f"\\n{Path(dot_path).stem}\\nNodes: {len(nodes)} | Edges: {len(edges)}",
        labelloc="t",
        pad="0.5",
        nodesep="0.4",
        ranksep="0.8",
    )
    dot.attr("node", fontname="Consolas", fontsize="9")
    dot.attr("edge", fontname="Consolas", fontsize="8")

    # Create subgraphs (clusters) for each stage
    for stage_key in sorted(stages.keys()):
        stage_nodes = stages[stage_key]
        info = _get_stage_info(stage_key)
        stage_num = re.search(r'stage(\d+)', stage_nodes[0])
        stage_label = f"Stage {stage_num.group(1)}" if stage_num else stage_key

        with dot.subgraph(name=f"cluster_{stage_key}") as sub:
            sub.attr(
                label=f" {stage_label}: {info['label']} ",
                style="rounded,filled",
                fillcolor=info["fill"] + "30",  # semi-transparent
                color=info["border"],
                penwidth="2",
                fontcolor="white",
                fontsize="10",
            )

            for nid in stage_nodes:
                nattrs = nodes[nid]
                ntype = nattrs["type"]
                shape_info = NODE_SHAPE_MAP.get(ntype, {"shape": "ellipse"})

                sub.node(nid,
                    label=nattrs["label"],
                    shape=shape_info["shape"],
                    style="filled,rounded",
                    fillcolor=info["fill"],
                    color=info["border"],
                    fontcolor="#212121",
                    penwidth="1.5",
                )

    # Add edges
    for src, dst, eattrs in edges:
        if eattrs["bridge"]:
            dot.edge(src, dst,
                label=f" {eattrs['label']} ",
                style="dashed",
                color="#FF5722",
                fontcolor="#FF8A65",
                penwidth="2.5",
                arrowsize="1.2",
            )
        else:
            # Color edge by verb type
            verb_colors = {
                "WF": "#42A5F5", "RF": "#66BB6A", "EX": "#EF5350",
                "FR": "#AB47BC", "ST": "#FFA726", "IJ": "#EC407A",
            }
            ecolor = verb_colors.get(eattrs["label"], "#90A4AE")
            dot.edge(src, dst,
                label=f" {eattrs['label']} ",
                color=ecolor,
                fontcolor=ecolor,
                penwidth="1.5",
            )

    dot.render(output_path, cleanup=True)
    final_path = f"{output_path}.{fmt}"
    print(f"[Visualize] Graphviz render -> {final_path}")
    return final_path


# ═══════════════════════════════════════════════════════════════════
# BACKEND 2: Matplotlib + NetworkX (fallback)
# ═══════════════════════════════════════════════════════════════════

def render_matplotlib(dot_path, output_path, fmt="png"):
    """Dùng matplotlib + networkx để render với layout đẹp hơn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx

    nodes, edges = parse_dot_file(dot_path)

    G = nx.DiGraph()
    for nid, attrs in nodes.items():
        G.add_node(nid, **attrs)
    for src, dst, eattrs in edges:
        G.add_edge(src, dst, **eattrs)

    # 1. Custom horizontal hierarchical layout
    stage_groups = {}
    for nid in G.nodes():
        s = G.nodes[nid].get("stage", "")
        stage_groups.setdefault(s, []).append(nid)

    def parse_stage_order(stage_name):
        import re
        m = re.match(r'stage(\d+)', stage_name)
        if m: return float(m.group(1))
        m = re.match(r'bridge_(\d+)_(\d+)', stage_name)
        if m: return (float(m.group(1)) + float(m.group(2))) / 2.0
        return 999.0

    sorted_stages = sorted(stage_groups.keys(), key=parse_stage_order)
    
    pos = {}
    for stage_key in sorted_stages:
        stage_nodes = stage_groups[stage_key]
        hub = [n for n in stage_nodes if "__p0" in n]
        bridges = [n for n in stage_nodes if n.startswith("bridge")]
        others = [n for n in stage_nodes if n not in hub and n not in bridges]
        
        # Sort others so that files and sockets are grouped visually if needed
        all_ordered = hub + bridges + others
        
        x_val = parse_stage_order(stage_key) * 6.0
        y_step = 1.5
        
        for i, nid in enumerate(all_ordered):
            if i == 0 and (nid in hub or nid in bridges):
                pos[nid] = (x_val, 0)
            else:
                sign = 1 if i % 2 != 0 else -1
                offset = sign * ((i + 1) // 2) * y_step
                pos[nid] = (x_val, offset)

    # Calculate graph dimensions to scale figure appropriately
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    if not xs: xs = [0]
    if not ys: ys = [0]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    
    fig_w = max(20, w * 1.5)
    fig_h = max(12, h * 1.2)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), facecolor="#1e1e2f")
    ax.set_facecolor("#1e1e2f")

    # Draw edges first so they are under nodes
    bridge_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("bridge")]
    normal_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get("bridge")]

    nx.draw_networkx_edges(G, pos, edgelist=normal_edges,
                           edge_color="#78909C", width=1.5,
                           arrows=True, arrowsize=15, ax=ax,
                           node_size=1)
    nx.draw_networkx_edges(G, pos, edgelist=bridge_edges,
                           edge_color="#FF7043", width=2.5, style="dashed",
                           arrows=True, arrowsize=18, ax=ax,
                           node_size=1)

    # Draw edge labels manually to avoid NetworkX curved arrow bug
    edge_labels = {(u, v): d.get("label", "") for u, v, d in G.edges(data=True)}
    for (u, v), label in edge_labels.items():
        x = (pos[u][0] + pos[v][0]) / 2
        y = (pos[u][1] + pos[v][1]) / 2
        ax.text(x, y, label, size=8, color="#FFFFFF", family="monospace",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#2c2c44", edgecolor="none", alpha=0.8),
                horizontalalignment='center', verticalalignment='center')

    # Draw nodes as text boxes
    legend_handles = []
    seen_labels = set()
    for stage_key in sorted_stages:
        stage_nodes = stage_groups[stage_key]
        info = _get_stage_info(stage_key)
        
        labels = {nid: G.nodes[nid].get("label", nid) for nid in stage_nodes}
        nx.draw_networkx_labels(
            G, pos, labels=labels, ax=ax,
            font_size=9, font_family="monospace", font_color="#111111", font_weight="bold",
            bbox=dict(
                boxstyle="round4,pad=0.7" if "brid" not in stage_key else "square,pad=0.6",
                facecolor=info["fill"],
                edgecolor=info["border"],
                linewidth=2.0,
                alpha=1.0
            )
        )
        
        if info["label"] not in seen_labels:
            legend_handles.append(
                mpatches.Patch(facecolor=info["fill"], edgecolor=info["border"],
                               label=f"{info['label']}", linewidth=2.0))
            seen_labels.add(info["label"])

    # Title & legend
    ax.set_title(f"{Path(dot_path).stem}\nNodes: {G.number_of_nodes()} | "
                 f"Edges: {G.number_of_edges()}",
                 fontsize=22, color="white", fontfamily="monospace", fontweight="bold", pad=30)
    
    # Place legend
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.02, 0.98),
              fontsize=11, facecolor="#2a2a4e", edgecolor="#555", labelcolor="white",
              framealpha=0.9, title="Stages", title_fontsize=13)

    ax.axis("off")
    plt.tight_layout()

    final_path = f"{output_path}.{fmt}"
    plt.savefig(final_path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Visualize] Matplotlib render -> {final_path}")
    return final_path


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def visualize(dot_path, output=None, fmt="png", backend="graphviz", open_file=False):
    """
    Visualize file .dot thành hình ảnh.

    Args:
        dot_path: Đường dẫn file .dot
        output: Đường dẫn output (không có extension). Mặc định = tên file .dot
        fmt: "png", "svg", "pdf"
        backend: "graphviz" hoặc "matplotlib"
        open_file: Mở file sau khi render
    """
    if not os.path.exists(dot_path):
        print(f"ERROR: File not found: {dot_path}")
        return None

    if output is None:
        output = str(Path(dot_path).with_suffix(""))

    print(f"[Visualize] Input:   {dot_path}")
    print(f"[Visualize] Backend: {backend}")
    print(f"[Visualize] Format:  {fmt}")

    if backend == "graphviz":
        try:
            result = render_graphviz(dot_path, output, fmt)
        except Exception as e:
            print(f"[Visualize] Graphviz failed: {e}")
            print("[Visualize] Falling back to matplotlib...")
            result = render_matplotlib(dot_path, output, fmt)
    else:
        result = render_matplotlib(dot_path, output, fmt)

    if open_file and result and os.path.exists(result):
        os.startfile(result)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize APT provenance graph")
    parser.add_argument("dot_file", help="Path to .dot file")
    parser.add_argument("--output", "-o", default=None,
                        help="Output path (without extension)")
    parser.add_argument("--format", "-f", default="png",
                        choices=["png", "svg", "pdf"])
    parser.add_argument("--backend", "-b", default="graphviz",
                        choices=["graphviz", "matplotlib"])
    parser.add_argument("--open", action="store_true",
                        help="Open file after rendering")
    args = parser.parse_args()

    visualize(args.dot_file, args.output, args.format, args.backend, args.open)
