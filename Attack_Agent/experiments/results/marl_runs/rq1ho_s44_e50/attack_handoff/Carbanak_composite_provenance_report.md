# Carbanak composite provenance report

- DOT: `experiments/results/marl_runs/rq1ho_s44_e50/attack_handoff/Carbanak_composite_provenance.dot`
- Quality valid: `True`
- Nodes: `47`
- Edges: `62`
- Clusters: `4`
- Total reward: `0.7418`
- Transition quality: `0.4599`
- Cluster validity: `0.7500`

## Kill Chain
1. `T1588.002` Tool (resource-development) -> score=0.1680, bridge=socket
2. `T1091` Replication Through Removable Media (lateral-movement) -> score=0.5170, bridge=socket
3. `T1072` Software Deployment Tools (execution) -> score=0.6947, bridge=socket
4. `T1201` Password Policy Discovery (discovery)

## Composite Replacements
- `T1588.002` -> `['T1588.002']` from `WIRTE` (similarity=1.0000)
- `T1078` -> `['T1685.005']` from `Play` (similarity=0.9771)

## Pruned Techniques
- `T1543.003`: weak transition and stronger/equal bypass (incoming=0.3258, bypass=0.6947)

## Cluster Issues
- `T1091` ['T1091/lateral-movement: expected network/socket activity']
