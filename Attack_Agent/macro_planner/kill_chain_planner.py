# macro_planner/kill_chain_planner.py
"""
Orchestrator: APT name → ordered kill-chain [T1595, T1566, T1059, ...]
Hỗ trợ 3 kiểu variant: aligned, evolved, composite.
"""
import json
import random
import numpy as np
from .scf_extractor import SCFExtractor
from .technique_graph import TechniqueGraphBuilder
from .gnn_link_predictor import GNNLinkPredictor
from .variant_pruner import VariantPruner
from .composite_generator import CompositeGenerator

# Thứ tự chuẩn của MITRE kill-chain phases
KILL_CHAIN_ORDER = [
    "reconnaissance", "resource-development", "initial-access", "execution",
    "persistence", "privilege-escalation", "defense-evasion", "credential-access",
    "discovery", "lateral-movement", "collection", "command-and-control", "exfiltration"
]


class KillChainPlanner:
    def __init__(self, cti_path="macro_planner/data/enterprise-attack.json"):
        self.scf = SCFExtractor(cti_path)
        self.scf.extract_all()
        graph_builder = TechniqueGraphBuilder(cti_path)
        self.tech_graph = graph_builder.build_graph()
        self.gnn = GNNLinkPredictor(self.scf.scf_cache, self.tech_graph)
        self.apt_techniques = self._load_apt_techniques(cti_path)
        self.last_variant_meta = {}

    def _load_apt_techniques(self, cti_path) -> dict:
        """Load mapping {apt_name: [technique_ids]}."""
        with open(cti_path, encoding="utf-8") as f:
            data = json.load(f)

        apt_names = {}
        for obj in data["objects"]:
            if obj.get("type") == "intrusion-set":
                apt_names[obj["id"]] = obj.get("name", "")

        tid_map = {}
        for obj in data["objects"]:
            if obj.get("type") == "attack-pattern":
                for ref in obj.get("external_references", []):
                    if ref.get("source_name") == "mitre-attack":
                        tid_map[obj["id"]] = ref.get("external_id", "")

        result = {}
        for obj in data["objects"]:
            if obj.get("type") != "relationship":
                continue
            if obj.get("relationship_type") != "uses":
                continue
            src = obj.get("source_ref", "")
            tgt = obj.get("target_ref", "")
            apt = apt_names.get(src)
            tid = tid_map.get(tgt)
            if apt and tid:
                result.setdefault(apt, []).append(tid)
        return result

    def plan_kill_chain(self, apt_name: str, variant_type: str = "aligned",
                        min_stages: int = 4, max_stages: int = 8,
                        similarity_threshold: float = 0.8) -> list:
        """
        Sinh kill-chain cho một APT.
        Returns: [{"technique_id": ..., "tactic": ...,
                   "technique_name": ..., "scf": np.array}, ...]
        """
        print(f"[Planner] Planning kill-chain for {apt_name} ({variant_type})...")

        base_techniques = self.apt_techniques.get(apt_name, [])
        if not base_techniques:
            print(f"[Planner] APT {apt_name} not found, using default starter")
            base_techniques = ["T1595", "T1566"]

        # Sort theo kill-chain order
        def get_tactic_order(tid):
            node = self.tech_graph.nodes.get(tid, {})
            tactics = node.get("tactics", [])
            for i, t in enumerate(KILL_CHAIN_ORDER):
                if t in tactics:
                    return i
            return 99

        sorted_techs = sorted(base_techniques, key=get_tactic_order)

        if variant_type == "aligned":
            chain = self._build_aligned(sorted_techs, min_stages, max_stages)
        elif variant_type == "evolved":
            chain = self._build_evolved(sorted_techs, min_stages, max_stages)
        else:  # composite
            chain = self._build_composite(
                sorted_techs, min_stages, max_stages,
                apt_name=apt_name, threshold=similarity_threshold
            )

        # Enforce tactic diversity — đảm bảo kill-chain cover các tactic thiết yếu
        chain = self._enforce_tactic_diversity(chain, min_stages)
        pruner = VariantPruner(self.scf, self.gnn, threshold=similarity_threshold)
        chain, prune_meta = pruner.prune(chain[:max_stages], min_stages=min_stages)
        self.last_variant_meta.setdefault("variant_type", variant_type)
        self.last_variant_meta.setdefault("composite_replacements", [])
        self.last_variant_meta["pruning"] = prune_meta
        self.last_variant_meta["threshold"] = similarity_threshold

        result = []
        for tid in chain:
            node = self.tech_graph.nodes.get(tid, {})
            tactics = node.get("tactics", [])
            context = self.scf.extract_context(tid)
            result.append({
                "technique_id": tid,
                "tactic": tactics[0] if tactics else "unknown",
                "technique_name": node.get("name", tid),
                "scf": self.scf.extract_scf(tid),
                "scf_context": context,
            })

        transitions = self.last_variant_meta.get("pruning", {}).get("transitions", [])
        for i, transition in enumerate(transitions):
            if i < len(result) - 1:
                result[i]["next_transition"] = transition

        print(f"[Planner] Kill-chain: {[r['technique_id'] for r in result]}")
        return result

    def _build_aligned(self, sorted_techs, min_s, max_s):
        """Prune và add links, giữ nguyên technique pool gốc."""
        chain = sorted_techs[:max_s]
        while len(chain) < min_s and chain:
            preds = self.gnn.predict_next_techniques(chain[-1], top_k=3)
            new = [p[0] for p in preds if p[0] not in chain]
            if not new:
                break
            chain.append(new[0])
        return chain

    def _enforce_tactic_diversity(self, chain: list, min_stages: int) -> list:
        """
        Đảm bảo kill-chain:
        - Mỗi tactic xuất hiện TỐI ĐA 1 lần
        - Cover đủ ESSENTIAL_TACTICS
        - Sắp xếp đúng thứ tự KILL_CHAIN_ORDER
        """
        ESSENTIAL_TACTICS = [
            "initial-access", "execution", "persistence",
            "discovery", "command-and-control", "exfiltration"
        ]

        # Bước 1: Deduplicate — giữ 1 technique per tactic
        # Ưu tiên giữ technique xuất hiện đầu tiên trong chain (đã sorted)
        seen_tactics = {}
        for tid in chain:
            tactics = self.tech_graph.nodes.get(tid, {}).get("tactics", [])
            primary_tactic = tactics[0] if tactics else "unknown"
            if primary_tactic not in seen_tactics:
                seen_tactics[primary_tactic] = tid

        deduplicated = list(seen_tactics.values())

        # Bước 2: Thêm techniques cho các essential tactics còn thiếu
        covered = set(seen_tactics.keys())
        missing = [t for t in ESSENTIAL_TACTICS if t not in covered]

        for tactic in missing:
            if len(deduplicated) >= 10:
                break
            # Tìm technique thuộc tactic này
            candidates = []
            for tid, node_data in self.tech_graph.nodes(data=True):
                node_tactics = node_data.get("tactics", [])
                if tactic in node_tactics and tid not in deduplicated:
                    # Score = SCF similarity với technique cuối trong chain
                    anchor = deduplicated[-1] if deduplicated else "T1566"
                    sim = self.scf.similarity(anchor, tid)
                    candidates.append((tid, sim))
            if candidates:
                best_tid = max(candidates, key=lambda x: x[1])[0]
                deduplicated.append(best_tid)
                seen_tactics[tactic] = best_tid

        # Bước 3: Sort theo KILL_CHAIN_ORDER
        def get_order(tid):
            tactics = self.tech_graph.nodes.get(tid, {}).get("tactics", [])
            for i, t in enumerate(KILL_CHAIN_ORDER):
                if t in tactics:
                    return i
            return 99

        deduplicated.sort(key=get_order)

        # Bước 4: Đảm bảo đủ min_stages (dùng GNN nếu thiếu)
        while len(deduplicated) < min_stages:
            preds = self.gnn.predict_next_techniques(deduplicated[-1], top_k=5)
            added = False
            for pred_tid, _, _ in preds:
                pred_tactics = self.tech_graph.nodes.get(pred_tid, {}).get("tactics", [])
                primary = pred_tactics[0] if pred_tactics else "unknown"
                if pred_tid not in deduplicated and primary not in seen_tactics:
                    deduplicated.append(pred_tid)
                    seen_tactics[primary] = pred_tid
                    added = True
                    break
            if not added:
                break

        deduplicated.sort(key=get_order)
        return deduplicated

    def _build_evolved(self, sorted_techs, min_s, max_s):
        """Thay thế 1-2 techniques bằng similar techniques từ same tactic."""
        chain = list(sorted_techs[:max_s])
        all_tids = list(self.scf.scf_cache.keys())
        all_vecs = np.array([self.scf.scf_cache[t] for t in all_tids])

        # Simple nearest-neighbor by cosine similarity (no sklearn dependency)
        replace_idx = random.sample(range(len(chain)), min(2, len(chain)))
        for ri in replace_idx:
            vec = self.scf.extract_scf(chain[ri]).reshape(1, -1)
            # Cosine similarity
            norms = np.linalg.norm(all_vecs, axis=1, keepdims=True).clip(min=1e-8)
            vec_norm = np.linalg.norm(vec).clip(min=1e-8)
            sims = (all_vecs @ vec.T).squeeze() / (norms.squeeze() * vec_norm)
            # Sort descending
            top_indices = np.argsort(-sims)
            # Tactic constraint: chỉ replace nếu candidate thuộc cùng tactic
            orig_tactics = set(self.tech_graph.nodes.get(chain[ri], {}).get("tactics", []))
            for ii in top_indices:
                candidate = all_tids[ii]
                cand_tactics = set(self.tech_graph.nodes.get(candidate, {}).get("tactics", []))
                if (candidate not in chain and candidate != chain[ri]
                        and orig_tactics & cand_tactics):
                    chain[ri] = candidate
                    break

        # Fill if too short
        while len(chain) < min_s and chain:
            preds = self.gnn.predict_next_techniques(chain[-1], top_k=3)
            new = [p[0] for p in preds if p[0] not in chain]
            if not new:
                break
            chain.append(new[0])
        return chain

    def _build_composite(self, sorted_techs, min_s, max_s,
                         apt_name: str = "", threshold: float = 0.8):
        """Lấy 1 subsequence từ APT khác và merge vào."""
        generator = CompositeGenerator(self.tech_graph, self.scf, self.apt_techniques)
        merged, replacements = generator.generate(
            sorted_techs[:max_s],
            apt_name=apt_name,
            threshold=threshold,
            max_replacements=2,
            max_stages=max_s,
        )
        self.last_variant_meta = {
            "variant_type": "composite",
            "composite_replacements": replacements,
        }

        def _sort_key(tid):
            node = self.tech_graph.nodes.get(tid, {})
            tactics = node.get("tactics", [])
            for i, t in enumerate(KILL_CHAIN_ORDER):
                if t in tactics:
                    return i
            return 99

        merged.sort(key=_sort_key)
        return merged[:max_s]
