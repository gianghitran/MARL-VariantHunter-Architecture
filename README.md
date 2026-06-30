# MARL-VariantHunter

**Multi-Agent Reinforcement Learning for adaptive APT-variant generation and detection on provenance graphs.**

A PPO coordinator drives a closed adversarial loop: a **Generation Agent** synthesizes new APT attack variants, and a **Detection Agent** (GAT + MLP with EWC continual learning) co-evolves to catch them — without forgetting prior knowledge.

---
## Author
- Gia-Nghi Tran
- Da-Vit Nguyen

## Overal

Working end-to-end and benchmarked:

- **Detection backbone migrated to GAT** (Graph Attention Network); `GCN`/`SAGEConv` retained as legacy.
- **Continual learning stabilized** — joint EWC (GAT+MLP) anchored on a DARPA reference set (125 benign + 125 malicious), λ tuned to **500** (lower λ collapses; the anchor prevents catastrophic forgetting).
- **Reproducible experiment suite** under `experiments/` answering three research questions, with multi-seed aggregation (mean ± std) and a parallel orchestrator.
- **Latest results** (3 seeds, 50 episodes, λ=500) — see below.

### Latest results

| RQ | Question | Result |
|----|----------|--------|
| **RQ1** | Does MARL recover attacks the baseline misses? | On held-out APT families under heavy camouflage (r≥3), baseline recall **0.667** (FN₀=12/36) → MARL recall **0.778±0.104**, Recovery **0.333±0.312**, ΔF1 **+0.071**, Retention **1.000** (no forgetting). |
| **RQ2** | Does the Generation Agent help against unseen variants? | On unseen `evolved`+`composite` variants, F1 **0.963 → 0.976** (EWC-only) / **0.975** (full); most of the gain comes from EWC continual learning, Generation's contribution is positive but seed-dependent. |
| **RQ3** | Does the classifier head architecture matter? | `mlp` **0.980±0.010** (most stable, only head to reach Nash) > `deep` **0.831±0.160** > `linear` **0.447±0.374**. |

> Single-seed and high-camouflage RQ1 variants are kept in separate result folders (e.g. `experiments/results_rq1_hicamo/`) so runs never overwrite each other.

---

## Architecture

<img width="524" alt="Architecture" src="https://github.com/user-attachments/assets/3ffef9fd-3291-497f-9627-eff046f4a81f" />


**Episode loop:** encode current graph state → coordinator picks an action → run the agent → collect reward → PPO update → offline EWC update of the Detection Agent from the replay buffer → save artifacts to `runs/<timestamp>/` (or `EXP_RUN_DIR`).

### Detection Agent

`GAT(in→64→20)` mean-pooled to a 20-d latent → `DetectionMLP(20→32→2)` → softmax; malicious if `P(malicious) ≥ 0.5`. Hard/false-negative samples enter a prioritized replay buffer; after each episode an EWC batch update adapts the classifier while the Fisher anchor protects DARPA knowledge. The classifier head is swappable (`mlp` / `linear` / `deep`) via `EXP_CLASSIFIER`.

---

## Datasets

- **DARPA TC E3 CADETS** — 12,858 ground-truth malicious UUIDs; balanced BFS k-hop subgraphs (~62 edges each), sampled with a fixed seed so every experiment shares one frozen test set.
- **APT variants** — 27 graphs across 9 families (APT28/29/32/33/38/41, Lazarus, FIN7, Carbanak) × {`aligned`, `evolved`, `composite`}.
- **Camouflage** — each malicious graph is mixed with `r × |edges|` benign edges (benign mimicry / living-off-the-land); higher `r` is the main driver of baseline misses.

---

## Setup & run

```bash
# environment (conda)
conda activate nmuggle_APT          # Python 3.10, PyTorch + PyTorch Geometric

# 1. pretrain the baseline detector  → trained_weights/darpa/{gat,mlp}.pth
python Benign_Agent/pretrain_on_generated.py
#    held-out baseline (RQ1)         → trained_weights/darpa_ho/{gat,mlp}.pth
EXP_PRETRAIN_HELDOUT_APTS="APT29,APT33,Lazarus" \
  EXP_PRETRAIN_OUTDIR="trained_weights/darpa_ho" \
  python Benign_Agent/pretrain_on_generated.py

# 2. single MARL co-evolution run
python main.py

# 3. full benchmark (5 configs × 3 seeds in parallel, then eval + consolidate)
bash experiments/run_multiseed_e50.sh
cat experiments/results/summary.md
```

### Key environment knobs

| Variable | Effect |
|----------|--------|
| `EXP_SEED`, `EXP_NUM_EPISODES`, `EXP_RUN_DIR` | seed / episode budget / output dir |
| `EXP_EWC_LAMBDA` | EWC strength (default 30; benchmarks use 500) |
| `EXP_CLASSIFIER` | detection head: `mlp` \| `linear` \| `deep` |
| `EXP_DISABLE_GENERATION`, `EXP_DISABLE_EWC` | ablations |
| `EXP_PRETRAIN_DIR`, `EXP_ALLOWED_APTS`, `EXP_ALLOWED_VARIANTS` | held-out / leave-variant-out splits |
| `EXP_RQ1_HO_CAMO`, `EXP_RESULTS_DIR` | RQ1 camouflage ratios / isolated result folder |

---

## Repository layout

```
main.py                  PPO coordinator — main co-evolution loop + EWC integration
environment.py           MARL env — agents, replay buffer, joint-EWC continual learning
networks.py              GAT backbone, DetectionMLP (mlp/linear/deep), CoordinatorNetwork
Attack_Agent/            APT variant generation + handoff bundles
Benign_Agent/            benign-graph generation + detector pretraining
experiments/             RQ1/RQ2/RQ3 scripts, datasets, orchestrators, consolidate.py
trained_weights/         darpa/ and darpa_ho/ baselines (GAT + MLP)
runs/<timestamp>/        per-run artifacts (checkpoints, logs, generated graphs)
data/darpa_tc/           DARPA CADETS source + ground truth
```

See [`experiments/README.md`](experiments/README.md) for the benchmark details and result schema.
