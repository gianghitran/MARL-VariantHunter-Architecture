"""Prune and annotate THREATIFY-style technique chains."""

from .scf_schema import bridge_hint


class VariantPruner:
    """Keep the most probable transitions using GNN and SCF similarity."""

    def __init__(self, scf_extractor, gnn=None, threshold: float = 0.8,
                 alpha: float = 0.55):
        self.scf = scf_extractor
        self.gnn = gnn
        self.threshold = threshold
        self.alpha = alpha

    def transition_score(self, src_tid: str, tgt_tid: str) -> dict:
        scf_score = self.scf.hybrid_similarity(src_tid, tgt_tid)
        gnn_score = None
        if self.gnn is not None:
            try:
                preds = self.gnn.predict_next_techniques(src_tid, top_k=50)
                for tid, prob, _ in preds:
                    if tid == tgt_tid:
                        gnn_score = float(prob)
                        break
            except Exception:
                gnn_score = None

        if gnn_score is None:
            score = scf_score
        else:
            score = self.alpha * gnn_score + (1.0 - self.alpha) * scf_score

        return {
            "source": src_tid,
            "target": tgt_tid,
            "score": score,
            "gnn_score": gnn_score,
            "scf_score": scf_score,
            "kept": score >= self.threshold,
        }

    def annotate(self, chain: list) -> tuple:
        transitions = []
        for i in range(len(chain) - 1):
            src = chain[i]
            tgt = chain[i + 1]
            trans = self.transition_score(src, tgt)
            src_ctx = self.scf.extract_context(src)
            tgt_ctx = self.scf.extract_context(tgt)
            trans["bridge_hint"] = bridge_hint(
                src_ctx.get("tactics", ["unknown"])[0] if src_ctx.get("tactics") else "unknown",
                tgt_ctx.get("tactics", ["unknown"])[0] if tgt_ctx.get("tactics") else "unknown",
                src_ctx.get("capabilities", []),
                tgt_ctx.get("capabilities", []),
            )
            transitions.append(trans)
        return chain, {"transitions": transitions, "removed": []}

    def prune(self, chain: list, min_stages: int = 4) -> tuple:
        """Remove weak middle techniques while preserving chain endpoints."""
        if len(chain) <= min_stages:
            return self.annotate(chain)

        kept = [chain[0]]
        removed = []
        for idx in range(1, len(chain) - 1):
            prev_tid = kept[-1]
            cur_tid = chain[idx]
            next_tid = chain[idx + 1]
            incoming = self.transition_score(prev_tid, cur_tid)
            bypass = self.transition_score(prev_tid, next_tid)
            can_remove = (
                len(chain) - len(removed) > min_stages
                and incoming["score"] < self.threshold
                and bypass["score"] >= incoming["score"]
            )
            if can_remove:
                removed.append({
                    "technique_id": cur_tid,
                    "reason": "weak transition and stronger/equal bypass",
                    "incoming_score": incoming["score"],
                    "bypass_score": bypass["score"],
                })
            else:
                kept.append(cur_tid)
        kept.append(chain[-1])

        kept, meta = self.annotate(kept)
        meta["removed"] = removed
        return kept, meta
