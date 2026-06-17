# Detection Agent Interface

This file documents the contract between the Generation side
(`THREATIFY -> TAGAPT`) and the downstream Detection Agent.

## Generated Files

Each pipeline run writes these files under the selected `--output` directory:

- `<APT>_<variant>_provenance.dot`
  Graphviz visualization file. It now includes detection metadata, but should be
  treated as a visualization/interchange format.
- `<APT>_<variant>_provenance_detection.json`
  Primary machine-readable input for the Detection Agent.
- `<APT>_<variant>_provenance_report.json`
  Generation report with reward, pruning, transition, and cluster validation.
- `<APT>_<variant>_provenance_report.md`
  Human-readable report.
- `<APT>_quality_report.txt`
  Short quality summary.

## Recommended Input

The Detection Agent should consume:

```text
*_provenance_detection.json
```

The DOT file is useful for debugging and visualization, but the JSON bundle is
more stable because it preserves temporal order and macro/micro metadata.

## JSON Schema

Top-level fields:

- `schema_version`: currently `threatify-tagapt-detection-v1`.
- `graph_id`: stable graph/run id.
- `apt`: APT profile used for generation.
- `variant_type`: `aligned`, `evolved`, or `composite`.
- `label`: currently `malicious`.
- `quality_valid`: boolean result from OS/semantic validation.
- `stats`: node/edge/cluster counts.
- `kill_chain`: ordered MITRE ATT&CK stages.
- `planner_meta`: pruning, composite replacement, and transition metadata.
- `reward`: Generation-side reward signals.
- `cluster_reports`: semantic validation result for each generated cluster.
- `nodes`: provenance entities.
- `edges`: temporal provenance events.

Node fields:

- `id`: stable node id.
- `label`: process/file/socket display name.
- `type`: `process`, `file`, or `socket`.
- `stage_index`: macro stage index; `null` for bridge nodes.
- `technique_id`: MITRE ATT&CK technique id.
- `technique_name`: technique name.
- `tactic`: MITRE tactic.
- `role`: `hub`, `bridge`, `input_artifact`, `output_artifact`,
  `executable_artifact`, or `internal`.
- `is_bridge`: true for injected inter-stage bridge nodes.

Edge fields:

- `id`: stable event id.
- `source`: source node id.
- `target`: target node id.
- `type`: OS event type. Current values include `FR`, `RF`, `WF`, `EX`, `ST`,
  `RCV`, `IJ`, and `LOAD`.
- `event_order`: synthetic temporal order.
- `timestamp`: same value as `event_order` in the current prototype.
- `stage_index`: macro stage index; may be null for bridge edges.
- `technique_id`: technique id for intra-stage edges.
- `tactic`: tactic for intra-stage edges.
- `is_bridge`: true for inter-stage bridge edges.
- `src_technique_id`, `tgt_technique_id`: bridge source/target technique ids.
- `src_tactic`, `tgt_tactic`: bridge source/target tactics.
- `bridge_hint`: `file` or `socket`.
- `transition_score`: macro transition score from SCF/GNN logic.

## Minimal Detector Loader

```python
import json
import networkx as nx

with open("APT28_composite_provenance_detection.json", encoding="utf-8") as f:
    bundle = json.load(f)

G = nx.DiGraph()
for node in bundle["nodes"]:
    G.add_node(node["id"], **node)

for edge in bundle["edges"]:
    G.add_edge(edge["source"], edge["target"], **edge)

label = bundle["label"]
```

## Current Prototype Boundaries

- Output is malicious-only generation data. Benign samples must come from the
  Detection Agent side or another benign provenance source.
- `timestamp` is synthetic. If the detector requires wall-clock timestamps,
  map `event_order` to a time scale during ingestion.
- The Generation side has internal reward validation, but it is not yet a full
  RL policy trained from Detection Agent feedback.
- The closed loop should call this pipeline to generate hard samples, run the
  Detection Agent, then feed detection confidence/FN results back into the MARL
  reward.

## Feedback Back To Generation

After inference, the Detection Agent should return a JSON object like:

```json
{
  "graph_id": "APT28_composite_provenance",
  "predicted_label": "benign",
  "malicious_score": 0.21,
  "confidence": 0.79,
  "is_false_negative": true,
  "detector_name": "SLOT-extension",
  "notes": "low score despite malicious generated label"
}
```

Required fields are:

- `graph_id`
- `predicted_label`
- `malicious_score`

Optional but recommended:

- `confidence`
- `is_false_negative`
- `detector_name`
- `notes`

Generation-side helper:

```python
from marl_integration.detection_feedback import compute_closed_loop_reward

closed_loop = compute_closed_loop_reward(feedback, generation_reward)
```

## Example Run

```powershell
E:\miniconda3\envs\pentest\python.exe .\pipeline_runner.py `
  --apt APT28 `
  --variant composite `
  --stages 6 `
  --threshold 0.8 `
  --output .\result_handoff
```
