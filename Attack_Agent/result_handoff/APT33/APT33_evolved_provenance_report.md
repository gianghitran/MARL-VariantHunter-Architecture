# APT33 evolved provenance report

- DOT: `/home/tmkhiem/nmuggle/LLM-ProvenanceGraph_APTDetection/Attack_Agent/result_handoff/APT33/APT33_evolved_provenance.dot`
- Quality valid: `True`
- Nodes: `45`
- Edges: `61`
- Clusters: `4`
- Total reward: `0.7574`
- Transition quality: `0.4906`
- Cluster validity: `1.0000`

## Kill Chain
1. `T1588.002` Tool (resource-development) -> score=0.1680, bridge=socket
2. `T1566.001` Spearphishing Attachment (initial-access) -> score=0.6296, bridge=socket
3. `T1078.001` Default Accounts (stealth) -> score=0.6741, bridge=socket
4. `T1098.003` Additional Cloud Roles (persistence)

## Pruned Techniques
- `T1059.009`: weak transition and stronger/equal bypass (incoming=0.4553, bypass=0.6741)
