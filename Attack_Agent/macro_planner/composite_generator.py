"""Composite variant generation by substituting techniques with sub-sequences."""

import numpy as np


class CompositeGenerator:
    """Find same-tactic sub-sequences from other APTs and merge them."""

    def __init__(self, tech_graph, scf_extractor, apt_techniques: dict):
        self.tech_graph = tech_graph
        self.scf = scf_extractor
        self.apt_techniques = apt_techniques

    def _primary_tactic(self, tid: str) -> str:
        tactics = self.tech_graph.nodes.get(tid, {}).get("tactics", [])
        return tactics[0] if tactics else "unknown"

    def _sequence_similarity(self, tid: str, seq: list) -> float:
        if not seq:
            return 0.0
        target_vec = self.scf.extract_scf(tid)
        seq_vecs = [self.scf.extract_scf(s) for s in seq]
        seq_vec = np.mean(seq_vecs, axis=0)
        norm = np.linalg.norm(target_vec) * np.linalg.norm(seq_vec)
        vec_sim = float(np.dot(target_vec, seq_vec) / norm) if norm > 0 else 0.0
        ctx_sim = max(self.scf.hybrid_similarity(tid, s) for s in seq)
        return 0.65 * vec_sim + 0.35 * ctx_sim

    def _candidate_sequences(self, target_tid: str, source_apt: str,
                             max_len: int = 3) -> list:
        target_tactic = self._primary_tactic(target_tid)
        source_chain = [
            tid for tid in self.apt_techniques.get(source_apt, [])
            if tid in self.tech_graph
        ]
        candidates = []
        for start in range(len(source_chain)):
            for end in range(start + 2, min(start + max_len, len(source_chain)) + 1):
                seq = source_chain[start:end]
                seq_tactics = {self._primary_tactic(tid) for tid in seq}
                if target_tactic in seq_tactics:
                    sim = self._sequence_similarity(target_tid, seq)
                    candidates.append((seq, sim))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    def generate(self, base_chain: list, apt_name: str, threshold: float = 0.8,
                 max_replacements: int = 2, max_stages: int = 8) -> tuple:
        chain = list(base_chain)
        replacements = []
        other_apts = [
            apt for apt in self.apt_techniques
            if apt != apt_name and self.apt_techniques.get(apt)
        ]

        for idx, tid in list(enumerate(base_chain)):
            if len(replacements) >= max_replacements:
                break
            best = None
            for other in other_apts:
                candidates = self._candidate_sequences(tid, other)
                if candidates and (best is None or candidates[0][1] > best[2]):
                    best = (other, candidates[0][0], candidates[0][1])

            if not best or best[2] < threshold:
                continue

            source_apt, seq, score = best
            current_pos = chain.index(tid) if tid in chain else -1
            if current_pos < 0:
                continue

            projected_len = len(chain) - 1 + len(seq)
            if projected_len > max_stages:
                seq = seq[:max(1, max_stages - len(chain) + 1)]
            if not seq:
                continue

            chain = chain[:current_pos] + seq + chain[current_pos + 1:]
            replacements.append({
                "replaced": tid,
                "replacement_sequence": seq,
                "source_apt": source_apt,
                "similarity": score,
            })

        return chain[:max_stages], replacements
