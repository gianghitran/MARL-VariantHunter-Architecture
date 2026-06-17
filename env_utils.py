import os
import networkx as nx
import matplotlib.pyplot as plt

def calculate_scf_impact(graph_df):
    cti_risk_scores = {
        'execute': 1.0,
        'mmap': 0.8,
        'write': 0.6,
        'read': 0.2,
        'clone': 0.9,
    }
    impact = 0.0
    for action in graph_df['action']:
        impact += cti_risk_scores.get(str(action).lower(), 0.1)
    return impact / max(len(graph_df), 1)

def visualize_current_graph(graph_df, run_dir):
    import time
    vis_dir = os.path.join(run_dir, "graph_visualize") if run_dir else "graph_visualize"
    os.makedirs(vis_dir, exist_ok=True)
    filename = f"graph_step_{int(time.time())}"
    save_path = os.path.join(vis_dir, filename)

    try:
        import graphviz
        dot = graphviz.Digraph(comment='Provenance Graph', format='pdf', engine='neato')
        dot.attr(
            rankdir="LR", size="9", fixedsize="false", splines="true",
            nodesep="0.3", ranksep="0", fontsize="10", overlap="scalexy", fontname="sans-serif"
        )
        dot.attr('node', fontname="sans-serif")
        dot.attr('edge', fontname="sans-serif")
        
        added_nodes = set()
        def get_shape(label):
            if 'MP' in label or 'TP' in label: return 'rectangle'
            elif 'SO' in label: return 'diamond'
            return 'ellipse'

        for idx, row in graph_df.iterrows():
            src = str(row['actorID'])
            dst = str(row['objectID'])
            action = str(row['action'])
            
            src_label = src.split('-')[0] + '-' + src.split('-')[1][:15] if '-' in src else src
            dst_label = dst.split('-')[0] + '-' + dst.split('-')[1][:15] if '-' in dst else dst
            
            if src not in added_nodes:
                dot.node(src, f"{len(added_nodes)}_{src_label}", shape=get_shape(src_label))
                added_nodes.add(src)
            if dst not in added_nodes:
                dot.node(dst, f"{len(added_nodes)}_{dst_label}", shape=get_shape(dst_label))
                added_nodes.add(dst)
            
            edge_label = f"{idx}: {action}"
            dot.edge(src, dst, label=edge_label, fontcolor='black', fontsize='10')
        
        dot.render(save_path, cleanup=True)
        print(f"[Visualization] Saved Graphviz chart to {save_path}.pdf")
        return
    except Exception as e:
        print(f"[Visualization] Graphviz error/missing: {e}. Falling back to matplotlib...")
        
    try:
        G = nx.DiGraph()
        for _, row in graph_df.iterrows():
            src = str(row['actorID'])
            dst = str(row['objectID'])
            action = str(row['action'])
            src_label = src.split('-')[0] + '-' + src.split('-')[1][:15] if '-' in src else src
            dst_label = dst.split('-')[0] + '-' + dst.split('-')[1][:15] if '-' in dst else dst
            G.add_node(src, label=src_label)
            G.add_node(dst, label=dst_label)
            G.add_edge(src, dst, label=action)
            
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, k=1.0)
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightgreen', edgecolors='black')
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=15, alpha=0.8)
        node_labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
        plt.title(f"Dynamic Provenance Graph | Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path + ".pdf", dpi=200, bbox_inches='tight')
        plt.close()
        print(f"[Visualization] Saved matplotlib graph to {save_path}.pdf")
    except Exception as fallback_err:
        print(f"[Visualization] Matplotlib Fallback Error: {fallback_err}")

def save_current_graph_to_txt(graph_df, run_dir):
    import time
    out_dir = os.path.join(run_dir, "generated_graphs") if run_dir else "generated_graphs"
    os.makedirs(out_dir, exist_ok=True)
    filename = f"graph_step_{int(time.time())}.txt"
    save_path = os.path.join(out_dir, filename)

    nodes_list = []
    node_to_idx = {}
    for _, row in graph_df.iterrows():
        if row['actorID'] not in node_to_idx:
            node_to_idx[row['actorID']] = len(nodes_list)
            nodes_list.append(row['actorID'])
        if row['objectID'] not in node_to_idx:
            node_to_idx[row['objectID']] = len(nodes_list)
            nodes_list.append(row['objectID'])
    
    edges_list = []
    for _, row in graph_df.iterrows():
        src_idx = node_to_idx[row['actorID']]
        dst_idx = node_to_idx[row['objectID']]
        edges_list.append(f"{src_idx} {dst_idx} {row['action']}")

    with open(save_path, "w") as f:
        f.write(f"{len(nodes_list)}\n")
        for node in nodes_list:
            f.write(f"{node}\n")
        f.write(f"{len(edges_list)}\n")
        for edge in edges_list:
            f.write(f"{edge}\n")
    
    print(f"[Generation Agent] Saved current graph txt to {save_path}")
