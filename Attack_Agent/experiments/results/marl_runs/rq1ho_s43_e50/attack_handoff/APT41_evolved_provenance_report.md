# APT41 evolved provenance report

- DOT: `experiments/results/marl_runs/rq1ho_s43_e50/attack_handoff/APT41_evolved_provenance.dot`
- Quality valid: `False`
- Nodes: `45`
- Edges: `61`
- Clusters: `4`
- Total reward: `0.7040`
- Transition quality: `0.7898`
- Cluster validity: `1.0000`

## Kill Chain
1. `T1595.003` Wordlist Scanning (reconnaissance) -> score=0.4660, bridge=socket
2. `T1078` Valid Accounts (stealth) -> score=1.0000, bridge=socket
3. `T1078.001` Default Accounts (stealth) -> score=0.9034, bridge=socket
4. `T1078.003` Local Accounts (stealth)

## Pruned Techniques
- `T1584.001`: weak transition and stronger/equal bypass (incoming=0.2333, bypass=0.4660)
