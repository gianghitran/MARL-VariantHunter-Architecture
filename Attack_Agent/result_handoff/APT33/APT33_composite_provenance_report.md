# APT33 composite provenance report

- DOT: `D:\notebook_UITNam3\Nam3_ki2\dacn\TAGAPT\TAGAPT_APTDetection\Attack_Agent\result_handoff\APT33/APT33_composite_provenance.dot`
- Quality valid: `True`
- Nodes: `45`
- Edges: `61`
- Clusters: `4`
- Total reward: `0.7348`
- Transition quality: `0.7145`
- Cluster validity: `1.0000`

## Kill Chain
1. `T1588.002` Tool (resource-development) -> score=0.4615, bridge=socket
2. `T1078` Valid Accounts (stealth) -> score=0.9034, bridge=socket
3. `T1078.003` Local Accounts (stealth) -> score=0.7787, bridge=socket
4. `T1053` Scheduled Task/Job (execution)

## Composite Replacements
- `T1588.002` -> `['T1588.002']` from `WIRTE` (similarity=1.0000)
- `T1566.001` -> `['T1566.001']` from `Dragonfly` (similarity=1.0000)

## Pruned Techniques
- `T1566.001`: weak transition and stronger/equal bypass (incoming=0.1680, bypass=0.4615)
