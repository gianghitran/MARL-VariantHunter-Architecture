# APT33 evolved provenance report

- DOT: `D:\notebook_UITNam3\Nam3_ki2\dacn\TAGAPT\TAGAPT_APTDetection\Attack_Agent\result_handoff\APT33/APT33_evolved_provenance.dot`
- Quality valid: `True`
- Nodes: `43`
- Edges: `58`
- Clusters: `4`
- Total reward: `0.7571`
- Transition quality: `0.5436`
- Cluster validity: `1.0000`

## Kill Chain
1. `T1588.002` Tool (resource-development) -> score=0.1750, bridge=socket
2. `T1189` Drive-by Compromise (initial-access) -> score=0.6771, bridge=socket
3. `T1078.003` Local Accounts (stealth) -> score=0.7787, bridge=socket
4. `T1053` Scheduled Task/Job (execution)

## Pruned Techniques
- `T1078.001`: weak transition and stronger/equal bypass (incoming=0.6266, bypass=0.6771)
