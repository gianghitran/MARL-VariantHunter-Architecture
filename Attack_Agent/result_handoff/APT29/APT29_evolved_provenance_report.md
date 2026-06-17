# APT29 evolved provenance report

- DOT: `D:\notebook_UITNam3\Nam3_ki2\dacn\TAGAPT\TAGAPT_APTDetection\Attack_Agent\result_handoff\APT29/APT29_evolved_provenance.dot`
- Quality valid: `True`
- Nodes: `41`
- Edges: `56`
- Clusters: `4`
- Total reward: `0.7679`
- Transition quality: `0.6961`
- Cluster validity: `1.0000`

## Kill Chain
1. `T1589.002` Email Addresses (reconnaissance) -> score=0.4649, bridge=socket
2. `T1133` External Remote Services (persistence) -> score=0.8446, bridge=socket
3. `T1078.003` Local Accounts (stealth) -> score=0.7787, bridge=socket
4. `T1053` Scheduled Task/Job (execution)

## Pruned Techniques
- `T1608.001`: weak transition and stronger/equal bypass (incoming=0.2333, bypass=0.4649)
