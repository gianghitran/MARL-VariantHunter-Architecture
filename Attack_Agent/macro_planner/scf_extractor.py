# macro_planner/scf_extractor.py
"""
Trích xuất SCF (Security Context Feature) vector từ MITRE ATT&CK enterprise-attack.json.
SCF vector encode: [tactic_onehot(14) | platform_onehot(9) | priv_score(1)
                   | impact_flag(1)] = 25 dims
"""
import json
import pickle
import numpy as np
from pathlib import Path
from .scf_schema import normalize_platform, tactic_capabilities, context_similarity

TACTICS = [
    "reconnaissance", "resource-development", "initial-access", "execution",
    "persistence", "privilege-escalation", "defense-evasion", "credential-access",
    "discovery", "lateral-movement", "collection", "command-and-control",
    "exfiltration", "impact"
]

PLATFORMS = [
    "windows", "linux", "macos", "network", "containers",
    "iaas", "saas", "office-365", "azure-ad"
]

PRIVS = ["user", "administrator", "system"]


class SCFExtractor:
    def __init__(self, cti_json_path="macro_planner/data/enterprise-attack.json"):
        print(f"[SCF] Loading CTI from {cti_json_path}...")
        with open(cti_json_path, encoding="utf-8") as f:
            self.data = json.load(f)
        self.techniques = {}   # {technique_id: object}
        self.scf_cache = {}    # {technique_id: np.array}
        self.context_cache = {} # {technique_id: rich explainable SCF dict}
        self._parse_techniques()

    def _parse_techniques(self):
        for obj in self.data["objects"]:
            if obj.get("type") != "attack-pattern":
                continue
            if obj.get("revoked", False) or obj.get("x_mitre_deprecated", False):
                continue
            ext_refs = obj.get("external_references", [])
            tid = None
            for ref in ext_refs:
                if ref.get("source_name") == "mitre-attack":
                    tid = ref.get("external_id", "")
            if tid:
                self.techniques[tid] = obj
        print(f"[SCF] Loaded {len(self.techniques)} techniques")

    def extract_scf(self, technique_id: str) -> np.ndarray:
        """Trả về SCF vector 25-dim cho một technique."""
        if technique_id in self.scf_cache:
            return self.scf_cache[technique_id]

        obj = self.techniques.get(technique_id)
        if obj is None:
            return np.zeros(25, dtype=np.float32)

        vec = np.zeros(25, dtype=np.float32)

        # [0:14] Tactic one-hot (technique có thể thuộc nhiều tactics)
        phases = obj.get("kill_chain_phases", [])
        for phase in phases:
            tactic = phase.get("phase_name", "").lower()
            if tactic in TACTICS:
                vec[TACTICS.index(tactic)] = 1.0

        # [14:23] Platform one-hot
        platforms = [p.lower() for p in obj.get("x_mitre_platforms", [])]
        for plat in platforms:
            for i, p in enumerate(PLATFORMS):
                if p in plat:
                    vec[14 + i] = 1.0

        # [23] Privileges: encode as max privilege level
        privs = [pr.lower() for pr in obj.get("x_mitre_permissions_required", [])]
        priv_score = 0.0
        if "system" in privs or "kernel" in privs:
            priv_score = 1.0
        elif "administrator" in privs or "root" in privs:
            priv_score = 0.5
        elif "user" in privs:
            priv_score = 0.2
        vec[23] = priv_score

        # [24] Impact flag
        impact_types = obj.get("x_mitre_impact_type", [])
        vec[24] = 1.0 if impact_types else 0.0

        self.scf_cache[technique_id] = vec
        return vec

    def extract_context(self, technique_id: str) -> dict:
        """Return richer explainable SCF context for THREATIFY-style logic."""
        if technique_id in self.context_cache:
            return self.context_cache[technique_id]

        obj = self.techniques.get(technique_id)
        if obj is None:
            ctx = {
                "technique_id": technique_id,
                "tactics": [],
                "platforms": [],
                "privilege_score": 0.0,
                "impact": False,
                "data_sources": [],
                "capabilities": [],
                "requires_network": False,
                "requires_execution": False,
                "produces_access": False,
            }
            self.context_cache[technique_id] = ctx
            return ctx

        tactics = [
            p.get("phase_name", "").lower()
            for p in obj.get("kill_chain_phases", [])
            if p.get("phase_name")
        ]
        platforms = sorted({
            normalize_platform(p)
            for p in obj.get("x_mitre_platforms", [])
            if p
        })
        data_sources = sorted({
            str(ds).lower()
            for ds in obj.get("x_mitre_data_sources", [])
            if ds
        })
        privs = [pr.lower() for pr in obj.get("x_mitre_permissions_required", [])]
        priv_score = 0.0
        if "system" in privs or "kernel" in privs:
            priv_score = 1.0
        elif "administrator" in privs or "root" in privs:
            priv_score = 0.5
        elif "user" in privs:
            priv_score = 0.2

        caps = tactic_capabilities(tactics)
        text_blob = " ".join([
            obj.get("name", ""),
            obj.get("description", ""),
            " ".join(data_sources),
        ]).lower()
        if any(word in text_blob for word in ("network", "remote", "c2", "command and control")):
            caps.add("network")
        if any(word in text_blob for word in ("credential", "password", "token", "hash")):
            caps.add("credential")
        if any(word in text_blob for word in ("file", "directory", "registry", "script")):
            caps.add("file")
        if any(word in text_blob for word in ("execute", "execution", "process", "command")):
            caps.add("execution")

        ctx = {
            "technique_id": technique_id,
            "name": obj.get("name", technique_id),
            "tactics": tactics,
            "platforms": platforms,
            "privilege_score": priv_score,
            "impact": bool(obj.get("x_mitre_impact_type", [])),
            "data_sources": data_sources,
            "capabilities": sorted(caps),
            "requires_network": "network" in caps,
            "requires_execution": "execution" in caps,
            "produces_access": bool(caps & {"credential", "persistence", "privilege"}),
        }
        self.context_cache[technique_id] = ctx
        return ctx

    def extract_all(self) -> dict:
        """Trích xuất SCF cho tất cả techniques, cache vào file."""
        cache_path = Path("macro_planner/data/technique_scf.pkl")
        if cache_path.exists():
            print("[SCF] Loading from cache...")
            with open(cache_path, "rb") as f:
                self.scf_cache = pickle.load(f)
            return self.scf_cache

        print("[SCF] Computing SCF for all techniques...")
        for tid in self.techniques:
            self.extract_scf(tid)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(self.scf_cache, f)
        print(f"[SCF] Saved {len(self.scf_cache)} SCF vectors to {cache_path}")
        return self.scf_cache

    def similarity(self, tid1: str, tid2: str) -> float:
        """Cosine similarity giữa 2 techniques."""
        v1, v2 = self.extract_scf(tid1), self.extract_scf(tid2)
        norm = (np.linalg.norm(v1) * np.linalg.norm(v2))
        return float(np.dot(v1, v2) / norm) if norm > 0 else 0.0

    def hybrid_similarity(self, tid1: str, tid2: str, alpha: float = 0.65) -> float:
        """Blend numeric SCF cosine with explainable context similarity."""
        vec_sim = self.similarity(tid1, tid2)
        ctx_sim = context_similarity(
            self.extract_context(tid1),
            self.extract_context(tid2),
        )
        return alpha * vec_sim + (1.0 - alpha) * ctx_sim
