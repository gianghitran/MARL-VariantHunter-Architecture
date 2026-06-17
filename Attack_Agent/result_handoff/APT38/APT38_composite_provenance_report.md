# APT38 composite provenance report

- DOT: `D:\notebook_UITNam3\Nam3_ki2\dacn\TAGAPT\TAGAPT_APTDetection\Attack_Agent\result_handoff\APT38/APT38_composite_provenance.dot`
- Quality valid: `True`
- Nodes: `61`
- Edges: `82`
- Clusters: `5`
- Total reward: `0.7986`
- Transition quality: `0.4863`
- Cluster validity: `1.0000`

## Kill Chain
1. `T1588.002` Tool (resource-development) -> score=0.1750, bridge=socket
2. `T1189` Drive-by Compromise (initial-access) -> score=0.4398, bridge=file
3. `T1059.003` Windows Command Shell (execution) -> score=0.8201, bridge=socket
4. `T1197` BITS Jobs (stealth) -> score=0.5104, bridge=socket
5. `T1615` Group Policy Discovery (discovery)

## Composite Replacements
- `T1588.002` -> `['T1588.002']` from `WIRTE` (similarity=1.0000)
- `T1583.001` -> `['T1583.006']` from `Medusa Group` (similarity=1.0000)
