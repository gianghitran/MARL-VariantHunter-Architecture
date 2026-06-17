# Lazarus evolved provenance report

- DOT: `D:\notebook_UITNam3\Nam3_ki2\dacn\TAGAPT\TAGAPT_APTDetection\Attack_Agent\result_handoff\Lazarus/Lazarus_evolved_provenance.dot`
- Quality valid: `True`
- Nodes: `44`
- Edges: `59`
- Clusters: `4`
- Total reward: `0.8083`
- Transition quality: `0.5825`
- Cluster validity: `1.0000`

## Kill Chain
1. `T1589.002` Email Addresses (reconnaissance) -> score=0.4611, bridge=socket
2. `T1053` Scheduled Task/Job (execution) -> score=0.6877, bridge=file
3. `T1037` Boot or Logon Initialization Scripts (persistence) -> score=0.5987, bridge=socket
4. `T1046` Network Service Discovery (discovery)

## Pruned Techniques
- `T1195`: weak transition and stronger/equal bypass (incoming=0.1680, bypass=0.4611)
