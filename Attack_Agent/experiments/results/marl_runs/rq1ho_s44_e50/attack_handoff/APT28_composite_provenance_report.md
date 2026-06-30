# APT28 composite provenance report

- DOT: `experiments/results/marl_runs/rq1ho_s44_e50/attack_handoff/APT28_composite_provenance.dot`
- Quality valid: `True`
- Nodes: `57`
- Edges: `77`
- Clusters: `5`
- Total reward: `0.8183`
- Transition quality: `0.6831`
- Cluster validity: `1.0000`

## Kill Chain
1. `T1598.002` Spearphishing Attachment (reconnaissance) -> score=0.4649, bridge=socket
2. `T1133` External Remote Services (persistence) -> score=0.8446, bridge=socket
3. `T1078.003` Local Accounts (stealth) -> score=0.7787, bridge=socket
4. `T1053` Scheduled Task/Job (execution) -> score=0.6443, bridge=socket
5. `T1046` Network Service Discovery (discovery)

## Composite Replacements
- `T1589.001` -> `['T1598.002']` from `Star Blizzard` (similarity=1.0000)
- `T1591` -> `['T1591']` from `Volt Typhoon` (similarity=1.0000)
