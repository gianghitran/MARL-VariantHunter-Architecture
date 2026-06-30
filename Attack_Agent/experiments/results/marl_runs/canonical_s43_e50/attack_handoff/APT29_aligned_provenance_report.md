# APT29 aligned provenance report

- DOT: `experiments/results/marl_runs/canonical_s43_e50/attack_handoff/APT29_aligned_provenance.dot`
- Quality valid: `True`
- Nodes: `38`
- Edges: `51`
- Clusters: `4`
- Total reward: `0.7776`
- Transition quality: `0.6931`
- Cluster validity: `1.0000`

## Kill Chain
1. `T1595.002` Vulnerability Scanning (reconnaissance) -> score=0.4559, bridge=socket
2. `T1133` External Remote Services (persistence) -> score=0.8446, bridge=socket
3. `T1078.003` Local Accounts (stealth) -> score=0.7787, bridge=socket
4. `T1053` Scheduled Task/Job (execution)

## Pruned Techniques
- `T1588.002`: weak transition and stronger/equal bypass (incoming=0.2100, bypass=0.4559)
