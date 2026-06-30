# FIN7 evolved provenance report

- DOT: `/home/tmkhiem/nmuggle/LLM-ProvenanceGraph_APTDetection/Attack_Agent/result_handoff/FIN7/FIN7_evolved_provenance.dot`
- Quality valid: `True`
- Nodes: `43`
- Edges: `58`
- Clusters: `4`
- Total reward: `0.7670`
- Transition quality: `0.6585`
- Cluster validity: `1.0000`

## Kill Chain
1. `T1591` Gather Victim Org Information (reconnaissance) -> score=0.3524, bridge=socket
2. `T1133` External Remote Services (persistence) -> score=0.8446, bridge=socket
3. `T1078.003` Local Accounts (stealth) -> score=0.7787, bridge=socket
4. `T1053` Scheduled Task/Job (execution)

## Pruned Techniques
- `T1584.001`: weak transition and stronger/equal bypass (incoming=0.2100, bypass=0.3524)
