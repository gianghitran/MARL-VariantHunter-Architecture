# Lazarus composite provenance report

- DOT: `D:\notebook_UITNam3\Nam3_ki2\dacn\TAGAPT\TAGAPT_APTDetection\Attack_Agent\result_handoff\Lazarus/Lazarus_composite_provenance.dot`
- Quality valid: `True`
- Nodes: `47`
- Edges: `62`
- Clusters: `4`
- Total reward: `0.8009`
- Transition quality: `0.5085`
- Cluster validity: `1.0000`

## Kill Chain
1. `T1589.001` Credentials (reconnaissance) -> score=0.1680, bridge=socket
2. `T1566.002` Spearphishing Link (initial-access) -> score=0.6755, bridge=file
3. `T1072` Software Deployment Tools (execution) -> score=0.6821, bridge=socket
4. `T1033` System Owner/User Discovery (discovery)

## Composite Replacements
- `T1595` -> `['T1589.001', 'T1591']` from `APT28` (similarity=0.9918)
- `T1566` -> `['T1566.002', 'T1684.001']` from `WIRTE` (similarity=0.9784)

## Pruned Techniques
- `T1505.003`: weak transition and stronger/equal bypass (incoming=0.6821, bypass=0.6821)
