# Lazarus evolved provenance report

- DOT: `/home/tmkhiem/nmuggle/LLM-ProvenanceGraph_APTDetection/Attack_Agent/result_handoff/Lazarus/Lazarus_evolved_provenance.dot`
- Quality valid: `True`
- Nodes: `44`
- Edges: `59`
- Clusters: `4`
- Total reward: `0.7879`
- Transition quality: `0.6012`
- Cluster validity: `1.0000`

## Kill Chain
1. `T1598.001` Spearphishing Service (reconnaissance) -> score=0.4572, bridge=socket
2. `T1053` Scheduled Task/Job (execution) -> score=0.6642, bridge=file
3. `T1098` Account Manipulation (persistence) -> score=0.6821, bridge=socket
4. `T1046` Network Service Discovery (discovery)

## Pruned Techniques
- `T1566.002`: weak transition and stronger/equal bypass (incoming=0.1680, bypass=0.4572)
