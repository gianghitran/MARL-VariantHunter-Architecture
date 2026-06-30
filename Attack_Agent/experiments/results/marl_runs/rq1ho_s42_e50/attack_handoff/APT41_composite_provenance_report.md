# APT41 composite provenance report

- DOT: `experiments/results/marl_runs/rq1ho_s42_e50/attack_handoff/APT41_composite_provenance.dot`
- Quality valid: `False`
- Nodes: `41`
- Edges: `56`
- Clusters: `4`
- Total reward: `0.7040`
- Transition quality: `0.7898`
- Cluster validity: `1.0000`

## Kill Chain
1. `T1598.002` Spearphishing Attachment (reconnaissance) -> score=0.4660, bridge=socket
2. `T1078` Valid Accounts (stealth) -> score=1.0000, bridge=socket
3. `T1078.001` Default Accounts (stealth) -> score=0.9034, bridge=socket
4. `T1078.003` Local Accounts (stealth)

## Composite Replacements
- `T1595.003` -> `['T1598.002']` from `Star Blizzard` (similarity=1.0000)
- `T1596.005` -> `['T1589.001']` from `APT28` (similarity=1.0000)

## Pruned Techniques
- `T1588.002`: weak transition and stronger/equal bypass (incoming=0.2217, bypass=0.4660)
