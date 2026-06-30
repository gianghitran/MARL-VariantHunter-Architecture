# APT33 aligned provenance report

- DOT: `experiments/results/marl_runs/rq3linear_s44_e50/attack_handoff/APT33_aligned_provenance.dot`
- Quality valid: `True`
- Nodes: `42`
- Edges: `57`
- Clusters: `4`
- Total reward: `0.7242`
- Transition quality: `0.7145`
- Cluster validity: `1.0000`

## Kill Chain
1. `T1588.002` Tool (resource-development) -> score=0.4615, bridge=socket
2. `T1078` Valid Accounts (stealth) -> score=0.9034, bridge=socket
3. `T1078.003` Local Accounts (stealth) -> score=0.7787, bridge=socket
4. `T1053` Scheduled Task/Job (execution)

## Pruned Techniques
- `T1566.001`: weak transition and stronger/equal bypass (incoming=0.1680, bypass=0.4615)
