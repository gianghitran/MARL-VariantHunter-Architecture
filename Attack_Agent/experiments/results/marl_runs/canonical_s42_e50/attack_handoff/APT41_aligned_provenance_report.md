# APT41 aligned provenance report

- DOT: `experiments/results/marl_runs/canonical_s42_e50/attack_handoff/APT41_aligned_provenance.dot`
- Quality valid: `False`
- Nodes: `48`
- Edges: `63`
- Clusters: `4`
- Total reward: `0.6901`
- Transition quality: `0.7898`
- Cluster validity: `1.0000`

## Kill Chain
1. `T1595.003` Wordlist Scanning (reconnaissance) -> score=0.4660, bridge=socket
2. `T1078` Valid Accounts (stealth) -> score=1.0000, bridge=socket
3. `T1078.001` Default Accounts (stealth) -> score=0.9034, bridge=socket
4. `T1078.003` Local Accounts (stealth)

## Pruned Techniques
- `T1588.002`: weak transition and stronger/equal bypass (incoming=0.2217, bypass=0.4660)
