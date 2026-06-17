# FIN7 composite provenance report

- DOT: `D:\notebook_UITNam3\Nam3_ki2\dacn\TAGAPT\TAGAPT_APTDetection\Attack_Agent\result_handoff\FIN7/FIN7_composite_provenance.dot`
- Quality valid: `True`
- Nodes: `42`
- Edges: `57`
- Clusters: `4`
- Total reward: `0.7778`
- Transition quality: `0.6946`
- Cluster validity: `1.0000`

## Kill Chain
1. `T1589.001` Credentials (reconnaissance) -> score=0.4604, bridge=socket
2. `T1133` External Remote Services (persistence) -> score=0.8446, bridge=socket
3. `T1078.003` Local Accounts (stealth) -> score=0.7787, bridge=socket
4. `T1053` Scheduled Task/Job (execution)

## Composite Replacements
- `T1591` -> `['T1589.001']` from `APT28` (similarity=1.0000)
- `T1591.004` -> `['T1589.001']` from `APT28` (similarity=1.0000)

## Pruned Techniques
- `T1608.005`: weak transition and stronger/equal bypass (incoming=0.2217, bypass=0.4604)
