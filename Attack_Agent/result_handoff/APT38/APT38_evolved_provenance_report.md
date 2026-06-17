# APT38 evolved provenance report

- DOT: `D:\notebook_UITNam3\Nam3_ki2\dacn\TAGAPT\TAGAPT_APTDetection\Attack_Agent\result_handoff\APT38/APT38_evolved_provenance.dot`
- Quality valid: `True`
- Nodes: `41`
- Edges: `56`
- Clusters: `4`
- Total reward: `0.7219`
- Transition quality: `0.6475`
- Cluster validity: `1.0000`

## Kill Chain
1. `T1588.002` Tool (resource-development) -> score=0.1750, bridge=file
2. `T1574.012` COR_PROFILER (stealth) -> score=0.9475, bridge=socket
3. `T1047` Windows Management Instrumentation (execution) -> score=0.8201, bridge=socket
4. `T1197` BITS Jobs (stealth)

## Pruned Techniques
- `T1195.002`: weak transition and stronger/equal bypass (incoming=0.1750, bypass=0.1750)
