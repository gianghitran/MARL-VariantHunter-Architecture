"""
experiments/datasets.py — xay dung corpus CO DINH (frozen) dung chung cho RQ1/RQ2/RQ3.

Hai nguon:
  1. DARPA TC E3 CADETS (benign + malicious) qua build_darpa_corpus + bundle_to_edges_df.
  2. APT variants sinh san trong Attack_Agent/result_handoff/<APT>/*_detection.json
     (aligned/evolved/composite) — cung schema bundle nen tai dung bundle_to_edges_df.

Corpus item = {"sample_id", "label" (0/1), "edges_df", "source", "apt", "variant"}.

Cache corpus xuong experiments/results/cache/ de tranh parse lai pkl 3.4GB moi RQ.
Manifest (chi metadata, khong co edges_df) ghi ra experiments/results/manifests/.
"""
import os
import sys
import json
import glob
import pickle

import numpy as np
import pandas as pd

# ── path setup ──
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "Benign_Agent")):
    if p not in sys.path:
        sys.path.insert(0, p)

from benign_interface import bundle_to_edges_df, build_benign_corpus   # noqa: E402
from darpa_tc_sampler import build_darpa_corpus                        # noqa: E402

_RESULTS_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
# EXP_RESULTS_DIR tach ket qua/manifest sang thu muc khac; CACHE luon dung goc
# de tai dung DARPA pkl (tranh parse lai ~3.4GB).
RESULTS_DIR   = os.environ.get("EXP_RESULTS_DIR") or _RESULTS_BASE
CACHE_DIR     = os.path.join(_RESULTS_BASE, "cache")
MANIFEST_DIR  = os.path.join(RESULTS_DIR, "manifests")
HANDOFF_DIR   = os.path.join(_REPO_ROOT, "Attack_Agent", "result_handoff")

# Seed CO DINH cho viec sampling DARPA -> moi RQ thay cung tap mau (so sanh duoc).
DARPA_SEED = 20240629
DARPA_DATASET = os.environ.get("DARPA_DATASET", "cadets")

for _d in (CACHE_DIR, MANIFEST_DIR):
    os.makedirs(_d, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# DARPA corpus (benign + malicious), cached
# ─────────────────────────────────────────────────────────────────────────────
def get_darpa_corpus(n_benign: int = 60, n_malicious: int = 60, seed: int = DARPA_SEED):
    """
    Tra ve list corpus items tu DARPA CADETS. Cache theo (dataset, n_ben, n_mal, seed).
    """
    cache_path = os.path.join(
        CACHE_DIR, f"darpa_{DARPA_DATASET}_{n_benign}_{n_malicious}_{seed}.pkl"
    )
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    ben_bundles, mal_bundles = build_darpa_corpus(
        n_benign=n_benign, n_malicious=n_malicious, dataset=DARPA_DATASET, seed=seed
    )
    corpus = []
    for i, b in enumerate(ben_bundles):
        corpus.append({
            "sample_id": f"darpa_ben_{i:03d}", "label": 0,
            "edges_df": bundle_to_edges_df(b), "source": "darpa", "apt": None, "variant": None,
        })
    for i, b in enumerate(mal_bundles):
        corpus.append({
            "sample_id": f"darpa_mal_{i:03d}", "label": 1,
            "edges_df": bundle_to_edges_df(b), "source": "darpa", "apt": None, "variant": None,
        })
    with open(cache_path, "wb") as f:
        pickle.dump(corpus, f)
    return corpus


# ─────────────────────────────────────────────────────────────────────────────
# APT variant corpus (malicious) tu result_handoff JSON
# ─────────────────────────────────────────────────────────────────────────────
def get_apt_variant_corpus(variants=None, apts=None):
    """
    Doc tat ca *_provenance_detection.json trong result_handoff.

    Args:
        variants : iterable {aligned,evolved,composite} de loc (None = tat ca).
        apts     : iterable ten APT de loc (None = tat ca).

    Returns: list corpus items (label=1).
    """
    variants = set(variants) if variants else None
    apts = set(apts) if apts else None

    corpus = []
    pattern = os.path.join(HANDOFF_DIR, "*", "*_provenance_detection.json")
    for fpath in sorted(glob.glob(pattern)):
        try:
            with open(fpath) as f:
                bundle = json.load(f)
        except Exception as e:
            print(f"[datasets] skip {fpath}: {e}")
            continue
        apt = bundle.get("apt")
        variant = bundle.get("variant_type")
        if apts and apt not in apts:
            continue
        if variants and variant not in variants:
            continue
        if not bundle.get("nodes") or not bundle.get("edges"):
            continue
        try:
            edges_df = bundle_to_edges_df(bundle)
        except Exception as e:
            print(f"[datasets] convert fail {apt}/{variant}: {e}")
            continue
        corpus.append({
            "sample_id": f"{apt}_{variant}", "label": 1,
            "edges_df": edges_df, "source": "apt_variant", "apt": apt, "variant": variant,
        })
    return corpus


# ─────────────────────────────────────────────────────────────────────────────
# Camouflage (evasion) — mo phong CHINH XAC co che cua environment.py:
#   detect target = APT subgraph + k benign edges,  k = len(apt) * ratio,
#   benign lay tu build_benign_corpus(source="hybrid", mimicry_level=0.4).
# Day la huong tan cong evasion cua he thong (benign mimicry / LOTL) -> tao ra
# cac mau ma baseline (B0) BO SOT, dung de do Recovery Rate cho RQ1/RQ2.
# ─────────────────────────────────────────────────────────────────────────────
_CAMO_POOL = None


def _camo_pool(seed: int = 99):
    """Pool benign-gen dung lam camouflage (cache 1 lan), khop environment.py."""
    global _CAMO_POOL
    if _CAMO_POOL is None:
        dfs = [bundle_to_edges_df(b)
               for b in build_benign_corpus(40, source="hybrid", mimicry_level=0.4, seed=seed)]
        _CAMO_POOL = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return _CAMO_POOL


def camouflage_corpus(mal_items, ratios=(0.0, 1.0, 2.0, 3.0), seed: int = 99):
    """
    Sinh cac phien ban camouflage cua moi mau malicious o nhieu muc ratio.

    ratio=0.0 -> ban goc (khong camo). ratio>0 -> them len(apt)*ratio canh benign.
    Tra ve list item moi (label=1) voi sample_id hau to '__camo{ratio}' va field
    'camo_ratio'. Dung chung pool nen reproducible theo seed.
    """
    pool = _camo_pool(seed)
    out = []
    for idx, item in enumerate(mal_items):
        apt = item["edges_df"]
        cols = list(apt.columns)
        for r in ratios:
            k = int(round(len(apt) * r))
            if k <= 0 or len(pool) == 0:
                cam = apt.copy()
            else:
                rng = np.random.RandomState((seed + idx * 131 + int(r * 1000)) % (2**31 - 1))
                camo = pool.sample(n=k, replace=(k > len(pool)), random_state=rng)
                # Dong cot pool khop cot apt (pool co cung 6 cot env)
                camo = camo[[c for c in cols if c in camo.columns]].copy()
                cam = pd.concat([apt, camo], ignore_index=True)
            out.append({
                **item,
                "sample_id": f"{item['sample_id']}__camo{r}",
                "edges_df": cam,
                "camo_ratio": r,
            })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Manifest (metadata only, de audit/reproduce)
# ─────────────────────────────────────────────────────────────────────────────
def save_manifest(name: str, corpus, extra: dict = None):
    """Ghi metadata cua corpus (KHONG kem edges_df) ra manifests/<name>.json."""
    meta = {
        "name": name,
        "darpa_dataset": DARPA_DATASET,
        "darpa_seed": DARPA_SEED,
        "n_total": len(corpus),
        "n_mal": sum(1 for c in corpus if c["label"] == 1),
        "n_ben": sum(1 for c in corpus if c["label"] == 0),
        "samples": [
            {"sample_id": c["sample_id"], "label": c["label"],
             "source": c["source"], "apt": c["apt"], "variant": c["variant"]}
            for c in corpus
        ],
    }
    if extra:
        meta.update(extra)
    path = os.path.join(MANIFEST_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Corpus builders cho tung RQ (dung chung -> so sanh duoc, ghi manifest)
# ─────────────────────────────────────────────────────────────────────────────
RQ1_CAMO_RATIOS = (0.0, 0.5, 1.0, 2.0)   # mix: vai mau B0 bat duoc, vai mau bo sot
RQ2_CAMO_RATIOS = (0.0, 1.0, 2.0)


def build_rq1_corpus(n_darpa_ben=30, n_darpa_mal=15, camo_ratios=RQ1_CAMO_RATIOS,
                     seed=DARPA_SEED):
    """
    RQ1/RQ3 test corpus:
      benign    = DARPA benign
      malicious = camouflage(APT variants[ALL] + DARPA malicious) o nhieu muc ratio
    """
    darpa = get_darpa_corpus(n_benign=n_darpa_ben, n_malicious=n_darpa_mal, seed=seed)
    ben = [c for c in darpa if c["label"] == 0]
    mal_darpa = [c for c in darpa if c["label"] == 1]
    av = get_apt_variant_corpus()
    mal = camouflage_corpus(av + mal_darpa, ratios=camo_ratios)
    corpus = ben + mal
    save_manifest("rq1_test", corpus, {"camo_ratios": list(camo_ratios),
                                       "n_darpa_ben": n_darpa_ben, "n_darpa_mal": n_darpa_mal})
    return corpus


# Ho APT giu rieng cho RQ1 hold-out (baseline darpa_ho KHONG train tren cac ho nay).
RQ1_HELDOUT_APTS = ("APT29", "APT33", "Lazarus")
# Ghi de qua EXP_RQ1_HO_CAMO (vd "0.0,3.0,5.0,7.0"): camo cao -> baseline bat dau
# bo sot (FN0>0) -> do duoc Recovery cua MARL. Mac dinh (0,3,5,7): camo nang ngoai
# vung ben cua baseline (gay >=3) de RQ1 co FN0>0 va Recovery co y nghia.
RQ1_HO_CAMO_RATIOS = tuple(
    float(x) for x in os.environ.get("EXP_RQ1_HO_CAMO", "0.0,3.0,5.0,7.0").split(",")
)


def build_rq1_heldout_corpus(heldout_apts=RQ1_HELDOUT_APTS, n_darpa_ben=30,
                             camo_ratios=RQ1_HO_CAMO_RATIOS):
    """
    RQ1 hold-out test corpus:
      benign    = DARPA benign
      malicious = camouflage(APT variants thuoc cac HO HELD-OUT) — baseline darpa_ho
                  CHUA TUNG thay cac ho nay khi pretrain -> accuracy realistic.
    """
    darpa = get_darpa_corpus(n_benign=n_darpa_ben, n_malicious=1)
    ben = [c for c in darpa if c["label"] == 0]
    av = get_apt_variant_corpus(apts=heldout_apts)
    mal = camouflage_corpus(av, ratios=camo_ratios)
    corpus = ben + mal
    save_manifest("rq1_heldout_test", corpus,
                  {"heldout_apts": list(heldout_apts), "camo_ratios": list(camo_ratios)})
    return corpus


def build_rq2_corpus(n_darpa_ben=30, held_variants=("evolved", "composite"),
                     camo_ratios=RQ2_CAMO_RATIOS, seed=DARPA_SEED):
    """
    RQ2 test corpus (Leave-Variant-Out):
      benign    = DARPA benign
      malicious = camouflage(APT variants thuoc held_variants = UNSEEN khi train)
    Co-evolution chi train tren 'aligned' (qua EXP_ALLOWED_VARIANTS) -> evolved/composite
    la chua thay.
    """
    darpa = get_darpa_corpus(n_benign=n_darpa_ben, n_malicious=1, seed=seed)
    ben = [c for c in darpa if c["label"] == 0]
    av = get_apt_variant_corpus(variants=held_variants)
    mal = camouflage_corpus(av, ratios=camo_ratios)
    corpus = ben + mal
    save_manifest("rq2_test", corpus, {"held_variants": list(held_variants),
                                       "camo_ratios": list(camo_ratios)})
    return corpus


if __name__ == "__main__":
    # Smoke: in ra so luong mau co the build.
    av = get_apt_variant_corpus()
    print(f"APT variants found: {len(av)}")
    by_var = {}
    for c in av:
        by_var.setdefault(c["variant"], 0)
        by_var[c["variant"]] += 1
    print("  by variant:", by_var)
    print("  apts:", sorted({c['apt'] for c in av}))
