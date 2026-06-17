# marl_integration/reward_calculator.py
"""Reward signals for the THREATIFY -> TAGAPT pipeline."""

import networkx as nx
import numpy as np

from graph_stitcher.action_masking import validate_edge


REQUIRED_TACTICS = [
    "initial-access", "execution", "persistence",
    "discovery", "command-and-control", "exfiltration"
]


class RewardCalculator:
    def __init__(self, w_coverage=0.25, w_connect=0.2, w_valid=0.25,
                 w_novelty=0.05, w_chain_depth=0.1, w_transition=0.1,
                 w_cluster=0.05):
        self.weights = {
            "coverage": w_coverage,
            "connectivity": w_connect,
            "validity": w_valid,
            "novelty": w_novelty,
            "chain_depth": w_chain_depth,
            "transition_quality": w_transition,
            "cluster_validity": w_cluster,
        }

    def compute(self, graph: nx.DiGraph, kill_chain: list,
                training_graphs: list = None) -> dict:
        """Return scalar and decomposed rewards for macro and micro agents."""
        coverage = self._kill_chain_coverage(kill_chain)
        connectivity = self._graph_connectivity(graph)
        validity = self._os_validity(graph)
        novelty = self._novelty_score(graph, training_graphs or [])
        chain_depth = self._chain_depth(graph)
        transition_quality = self._transition_quality(kill_chain)
        cluster_validity = self._cluster_validity(graph)

        total = (
            self.weights["coverage"] * coverage
            + self.weights["connectivity"] * connectivity
            + self.weights["validity"] * validity
            + self.weights["novelty"] * novelty
            + self.weights["chain_depth"] * chain_depth
            + self.weights["transition_quality"] * transition_quality
            + self.weights["cluster_validity"] * cluster_validity
        )

        macro_reward = (
            0.45 * coverage
            + 0.35 * transition_quality
            + 0.15 * novelty
            + 0.05 * cluster_validity
        )
        micro_reward = (
            0.35 * validity
            + 0.25 * connectivity
            + 0.2 * chain_depth
            + 0.2 * cluster_validity
        )

        return {
            "total": float(total),
            "coverage": float(coverage),
            "connectivity": float(connectivity),
            "validity": float(validity),
            "novelty": float(novelty),
            "chain_depth": float(chain_depth),
            "transition_quality": float(transition_quality),
            "cluster_validity": float(cluster_validity),
            "macro_reward": float(macro_reward),
            "micro_reward": float(micro_reward),
        }

    def _kill_chain_coverage(self, kill_chain: list) -> float:
        covered = {stage["tactic"] for stage in kill_chain}
        return len(covered & set(REQUIRED_TACTICS)) / len(REQUIRED_TACTICS)

    def _graph_connectivity(self, graph: nx.DiGraph) -> float:
        if graph.number_of_nodes() == 0:
            return 0.0
        connected = sum(1 for n in graph.nodes() if graph.degree(n) > 0)
        return connected / graph.number_of_nodes()

    def _os_validity(self, graph: nx.DiGraph) -> float:
        """Ratio of edges accepted by the OS action mask."""
        if graph.number_of_edges() == 0:
            return 1.0
        valid = 0
        for u, v, attrs in graph.edges(data=True):
            edge_type = attrs.get("edge_type", "")
            src_type = graph.nodes[u].get("node_type", "process")
            tgt_type = graph.nodes[v].get("node_type", "process")
            tgt_label = graph.nodes[v].get("label", "")
            if validate_edge(src_type, tgt_type, tgt_label, edge_type):
                valid += 1
        return valid / graph.number_of_edges()

    def _transition_quality(self, kill_chain: list) -> float:
        scores = [
            float(stage["next_transition"]["score"])
            for stage in kill_chain
            if stage.get("next_transition", {}).get("score") is not None
        ]
        if not scores:
            return 0.5
        return float(np.mean(scores))

    def _cluster_validity(self, graph: nx.DiGraph) -> float:
        reports = graph.graph.get("cluster_reports", [])
        if not reports:
            return 0.5
        return sum(1 for report in reports if report.get("valid")) / len(reports)

    def _novelty_score(self, graph: nx.DiGraph, training_graphs: list) -> float:
        if not training_graphs:
            return 0.5
        node_labels = {d.get("label", "") for _, d in graph.nodes(data=True)}
        similarities = []
        for tg in training_graphs[:10]:
            train_labels = {d.get("label", "") for _, d in tg.nodes(data=True)}
            union = len(node_labels | train_labels)
            inter = len(node_labels & train_labels)
            similarities.append(inter / union if union else 0.0)
        return 1.0 - float(np.mean(similarities))

    def _chain_depth(self, graph: nx.DiGraph) -> float:
        if graph.number_of_nodes() == 0:
            return 0.0

        sources = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        if not sources:
            return 0.0

        import random as _rng
        if len(sources) > 50:
            sources = _rng.sample(sources, 50)

        depths = []
        for src in sources:
            visited = {src}
            queue = [(src, 0)]
            max_depth = 0
            while queue:
                node, depth = queue.pop(0)
                max_depth = max(max_depth, depth)
                for succ in graph.successors(node):
                    if succ not in visited:
                        visited.add(succ)
                        queue.append((succ, depth + 1))
            depths.append(max_depth)

        avg_depth = float(np.mean(depths)) if depths else 0.0
        return min(avg_depth / 6.0, 1.0)
