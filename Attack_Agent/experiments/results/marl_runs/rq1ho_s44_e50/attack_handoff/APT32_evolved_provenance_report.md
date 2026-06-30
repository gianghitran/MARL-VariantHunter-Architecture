# APT32 evolved provenance report

- DOT: `experiments/results/marl_runs/rq1ho_s44_e50/attack_handoff/APT32_evolved_provenance.dot`
- Quality valid: `True`
- Nodes: `39`
- Edges: `53`
- Clusters: `4`
- Total reward: `0.7736`
- Transition quality: `0.6946`
- Cluster validity: `1.0000`

## Kill Chain
1. `T1598.003` Spearphishing Link (reconnaissance) -> score=0.4604, bridge=socket
2. `T1133` External Remote Services (persistence) -> score=0.8446, bridge=socket
3. `T1078.003` Local Accounts (stealth) -> score=0.7787, bridge=socket
4. `T1053` Scheduled Task/Job (execution)

## Pruned Techniques
- `T1583.001`: weak transition and stronger/equal bypass (incoming=0.2217, bypass=0.4604)
