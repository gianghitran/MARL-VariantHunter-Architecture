# APT38 aligned provenance report

- DOT: `experiments/results/marl_runs/rq1ho_s44_e50/attack_handoff/APT38_aligned_provenance.dot`
- Quality valid: `True`
- Nodes: `56`
- Edges: `75`
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
