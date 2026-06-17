# THREATIFY -> TAGAPT Pipeline

This project follows the second architecture direction:

1. Macro Planner: THREATIFY-style APT variant generation at MITRE ATT&CK
   technique level.
2. Micro Expander: TAGAPT/MaskGAF-style OS provenance subgraph generation for
   each technique stage.
3. Graph Stitcher: merge stage clusters into one provenance graph using
   OS-valid file/socket bridge nodes.
4. Quality and reward: validate topology/OS semantics and compute macro/micro
   reward signals.
5. Detection handoff: export a normalized JSON bundle for the downstream
   Detection Agent.

## Main Entry Point

```sh
python pipeline_runner.py --apt APT28 --variant evolved --stages 5 --output ./result
```

Optional MaskGAF checkpoint mode:

```sh
python pipeline_runner.py --apt APT28 --variant evolved --stages 5 \
  --checkpoint ./save_pretrain/exp_ASG_CTI_epoch3_1gpu/checkpoint2 \
  --data ./data_preprocessed/CTI \
  --output ./result
```

If no checkpoint/data is provided, `pipeline_runner.py` falls back to
constraint-aware dummy OS provenance clusters. This is useful for validating the
macro-to-micro pipeline without running the full generative model.

## Detection Agent Output Contract

Each run emits both visualization and machine-readable files:

- `<APT>_<variant>_provenance.dot`: Graphviz visualization/interchange graph.
- `<APT>_<variant>_provenance_detection.json`: primary Detection Agent input.
- `<APT>_<variant>_provenance_report.json`: generation quality/reward report.
- `<APT>_<variant>_provenance_report.md`: human-readable report.
- `<APT>_quality_report.txt`: compact validation summary.

Use `*_provenance_detection.json` as the default Detection Agent input. It
contains node/edge schemas, synthetic temporal order, MITRE technique/tactic
metadata, bridge metadata, reward, and validation reports.

See `DETECTION_AGENT_INTERFACE.md` for the full handoff schema.

## Core Modules

- `macro_planner/`: THREATIFY-inspired SCF extraction, technique graph building,
  GNN link prediction, and aligned/evolved/composite kill-chain planning.
- `technique_to_entity_map/`: maps MITRE techniques to OS entity constraints
  used by the Micro Expander.
- `train_CTI.py`, `model_CTI.py`, `MaskGAF.py`, `dataloader.py`: TAGAPT/MaskGAF
  model and generation code.
- `trans_gendata_CTI.py`: includes conversion from MaskGAF raw output to DOT
  clusters.
- `graph_stitcher/`: action masking and final provenance graph assembly.
- `graph_stitcher/detection_exporter.py`: exports normalized graph bundles for
  Detection Agent ingestion.
- `edge_validator.py`: stricter type-level and instance-level edge validation
  helpers.
- `marl_integration/`: reward calculation for coverage, connectivity, validity,
  novelty, and chain depth.
- `visualize_graph.py`: render final DOT outputs.

## Preserved Data

- `macro_planner/data/enterprise-attack.json`: MITRE CTI knowledge base.
- `macro_planner/data/*.pkl`, `link_pred_model.pt`: cached SCF/technique graph
  and trained link predictor artifacts.
- `data_preprocessed/`: TAGAPT model data arrays.
- `save_pretrain/`: MaskGAF pretrained checkpoints.
