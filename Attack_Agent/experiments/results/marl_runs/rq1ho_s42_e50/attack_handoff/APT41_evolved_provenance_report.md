# APT41 evolved provenance report

- DOT: `experiments/results/marl_runs/rq1ho_s42_e50/attack_handoff/APT41_evolved_provenance.dot`
- Quality valid: `True`
- Nodes: `47`
- Edges: `62`
- Clusters: `4`
- Total reward: `0.6755`
- Transition quality: `0.5288`
- Cluster validity: `1.0000`

## Kill Chain
1. `T1595.003` Wordlist Scanning (reconnaissance) -> score=0.2217, bridge=socket
2. `T1588.002` Tool (resource-development) -> score=0.4615, bridge=socket
3. `T1078` Valid Accounts (stealth) -> score=0.9034, bridge=socket
4. `T1078.003` Local Accounts (stealth)

## Pruned Techniques
- `T1078.001`: weak transition and stronger/equal bypass (incoming=0.1575, bypass=0.4615)
