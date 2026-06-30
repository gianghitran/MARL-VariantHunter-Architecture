# Lazarus aligned provenance report

- DOT: `experiments/results/marl_runs/rq3deep_s42_e50/attack_handoff/Lazarus_aligned_provenance.dot`
- Quality valid: `True`
- Nodes: `48`
- Edges: `65`
- Clusters: `4`
- Total reward: `0.7978`
- Transition quality: `0.5496`
- Cluster validity: `1.0000`

## Kill Chain
1. `T1595` Active Scanning (reconnaissance) -> score=0.3623, bridge=socket
2. `T1053` Scheduled Task/Job (execution) -> score=0.6877, bridge=file
3. `T1037` Boot or Logon Initialization Scripts (persistence) -> score=0.5987, bridge=socket
4. `T1046` Network Service Discovery (discovery)

## Pruned Techniques
- `T1566`: weak transition and stronger/equal bypass (incoming=0.1750, bypass=0.3623)
