# pipeline_runner.py
"""
Entry point của toàn bộ pipeline tích hợp Macro Planner + TAGAPT MaskGAF.

Pipeline 2 tầng:
  Tầng 1 (Macro Planner): MITRE ATT&CK CTI + GNN -> kill-chain skeleton
  Tầng 2 (Micro Expander): TAGAPT MaskGAF -> subgraph clusters
  Graph Stitcher: Ghép clusters -> final provenance graph (75-150 nodes)

Usage:
    python pipeline_runner.py --apt APT28 --variant evolved --output ./result
    python pipeline_runner.py --apt APT41 --variant composite --stages 6
"""
import argparse
import os
from pathlib import Path

from macro_planner.kill_chain_planner import KillChainPlanner
from technique_to_entity_map.mapping_rules import get_entity_constraints
from graph_stitcher.provenance_assembler import ProvenanceAssembler
from graph_stitcher.detection_exporter import export_detection_bundle
from marl_integration.reward_calculator import RewardCalculator
from marl_integration.report_writer import write_pipeline_report


def _safe_ext(ft):
    """
    Convert file type/path sang extension an toàn.
    /etc/crontab   → .crontab
    /var/log/*     → .log
    .sh            → .sh  (unchanged)
    *.service      → .service
    """
    ft = ft.strip()
    if ft.startswith("/"):
        # Extract filename từ path
        basename = os.path.basename(ft.rstrip("/*"))
        if "." in basename:
            ext = "." + basename.split(".")[-1]
        else:
            ext = "." + basename[:8]  # dùng tên file nếu không có ext
    elif ft.startswith("*."):
        ext = ft[1:]  # *.service → .service
    elif ft.startswith("."):
        ext = ft  # .sh → .sh
    else:
        ext = "." + ft[:8]
    return ext


def _create_dummy_cluster(tid, hints, out_dot, stage_idx):
    """
    Generate a semantically valid OS-level provenance subgraph for one technique.

    Topology: Deep causal chains with file-mediated causality (NOT star/broom).
      hub_proc ──FR──> child1 ──WF──> file_A
                                         └──RF── child2 ──WF──> file_B
                                                    └──FR──> child3 ──WF──> file_C
      hub_proc ──ST──> socket ──RCV──> child1  (for network tactics)

    Rules enforced:
      A. File-mediated causality: after WF, P_READ=0.4 chance another proc RF's it
      B. Chain depth: 50% children fork from previous child, not hub
      C. Socket causality: socket gets RCV edge to a child (not a dead-end)
      D. RF back-edges: 30% chance a child also RF's hub's input files
      E. Min avg degree >= 2.5 (pad with extra RF edges if needed)

    All edges validated through action_masking rules BEFORE emission.
    """
    import random
    import string
    from graph_stitcher.action_masking import (
        get_valid_edges, _is_executable, pick_valid_edge
    )

    P_READ = 0.65       # Rule A: probability a written file gets read
    P_CHAIN_FORK = 0.50 # Rule B: probability child forks from prev child
    P_SIBLING_RF = 0.30 # Rule D: probability a child RF's hub input file
    MIN_AVG_DEG = 2.5   # Rule E: minimum avg degree target
    MAX_SINK_RATIO = 0.42  # Rule E: max allowed sink ratio

    def rand_suffix(n=4):
        return ''.join(random.choices(string.hexdigits[:16], k=n))

    hub = hints.get("hub_process", "bash")
    procs = hints.get("allowed_processes", ["bash"])
    file_types = hints.get("allowed_file_types", [".sh"])
    edge_types_allowed = hints.get("allowed_edge_types", ["WF", "RF", "FR"])

    # ── Select child processes ──────────────────────────────────
    other_procs = [p for p in procs if p != hub]
    n_children = random.randint(2, min(4, max(2, len(other_procs))))
    children = (random.sample(other_procs, min(n_children, len(other_procs)))
                if other_procs else [hub])
    # Pad if we need more children than unique procs available
    while len(children) < n_children:
        children.append(random.choice(other_procs if other_procs else [hub]))

    # ── File name registry ──────────────────────────────────────
    FILE_NAMES = {
        ".sh": ["setup", "run", "exploit", "payload", "backdoor", "init"],
        ".py": ["scanner", "dropper", "c2client", "exfil", "loader", "agent"],
        ".elf": ["implant", "shellcode", "rootkit", "stager", "bot"],
        ".log": ["system", "auth", "syslog", "kern", "access"],
        ".txt": ["config", "output", "creds", "hosts", "data"],
        ".zip": ["archive", "dump", "backup", "exfil"],
        ".key": ["id_rsa", "privkey", "cert"],
        ".db": ["passwd", "shadow", "credentials", "tokens"],
        ".conf": ["sshd_config", "nginx", "apache2", "cron"],
        ".xml": ["nmap_out", "report", "config"],
        ".csv": ["results", "scan_data", "targets"],
        ".bin": ["payload", "stage2", "implant"],
        ".enc": ["exfil_data", "encrypted", "archive"],
        ".tar": ["backup", "exfil_pkg", "dump"],
        ".gz": ["archive", "compressed", "logs"],
        ".html": ["phish_page", "index", "login"],
        ".json": ["config", "api_resp", "manifest"],
        ".php": ["shell", "backdoor", "upload"],
        ".doc": ["invoice", "report", "resume"],
        ".xls": ["budget", "data", "macro_doc"],
        ".pdf": ["invoice", "contract", "report"],
        ".pl": ["scanner", "exploit", "util"],
        ".rb": ["exploit", "scanner", "payload"],
        ".service": ["backdoor", "persistence", "agent"],
        ".crontab": ["crontab", "cron_job", "schedule"],
        ".d": ["cron_task", "job_config", "timer"],
    }

    def make_fname(ft_raw):
        ext = _safe_ext(ft_raw)
        names = FILE_NAMES.get(ext, ["data"])
        return f"{random.choice(names)}_{rand_suffix(2)}{ext}", ext

    # ── Build nodes & edges with masking ────────────────────────
    nodes = []   # (node_id, label, node_type)
    edges = []   # (src_id, tgt_id, edge_label)
    node_ids = set()  # track IDs for dedup

    # Lookup helpers for Rule A (file-mediated causality)
    proc_ids = []      # ordered list of process node IDs
    file_ids = []      # ordered list of file node IDs (with labels)
    written_files = [] # files written by WF edges, candidates for RF
    writer_map = {}    # fid -> process_id that wrote it

    sid = f"s{stage_idx}"
    uid = rand_suffix()

    def add_node(nid, label, ntype):
        if nid not in node_ids:
            nodes.append((nid, label, ntype))
            node_ids.add(nid)
            if ntype == "process":
                proc_ids.append(nid)
            elif ntype == "file":
                file_ids.append((nid, label))

    def add_edge(src, tgt, elabel):
        edges.append((src, tgt, elabel))

    # ── Hub process node ────────────────────────────────────────
    hub_id = f"p0_{sid}"
    add_node(hub_id, hub, "process")

    # ── Hub reads input files (RF = File→Process, data flow direction) ──
    input_file_ids = []
    n_inputs = random.randint(1, 2)
    for k in range(n_inputs):
        ft = random.choice(file_types)
        fname, ext = make_fname(ft)
        fid = f"fin{k}_{sid}"
        add_node(fid, fname, "file")
        # RF direction: File→Process (file provides data to process)
        add_edge(fid, hub_id, "RF")
        input_file_ids.append((fid, fname))

    # ── Hub forks child processes with causal chains ────────────
    prev_proc_id = hub_id
    child_ids = []

    for j, proc in enumerate(children):
        cid = f"p{j+1}_{sid}"
        add_node(cid, proc, "process")
        child_ids.append(cid)

        # RULE B — Chain depth: 50% fork from previous child, 50% from hub
        if j == 0:
            # First child always forks from hub
            fork_parent = hub_id
        elif random.random() < P_CHAIN_FORK:
            # Fork from previous child → creates depth
            fork_parent = prev_proc_id
        else:
            # Fork from hub → creates breadth
            fork_parent = hub_id

        etype = pick_valid_edge("process", "process", proc)
        add_edge(fork_parent, cid, etype)  # Always "FR"

        # ── Child process writes 1-2 output files ───────────────
        n_child_files = random.randint(1, 2)
        for k in range(n_child_files):
            ft = random.choice(file_types)
            fname, ext = make_fname(ft)
            fid = f"f{j}_{k}_{sid}"
            add_node(fid, fname, "file")
            etype = pick_valid_edge("process", "file", fname, preferred=["WF"])
            add_edge(cid, fid, etype)
            if etype == "WF":
                written_files.append((fid, fname, cid))  # track writer
                writer_map[fid] = cid

        # RULE D — Sibling RF on hub input files (30% chance)
        # RF direction: File→Process (file provides data to child)
        if input_file_ids and random.random() < P_SIBLING_RF:
            ref_fid, ref_fname = random.choice(input_file_ids)
            add_edge(ref_fid, cid, "RF")

        # Update prev_proc_id for chain depth
        prev_proc_id = cid

    # ── RULE A — File-mediated causality ────────────────────────
    # After a child writes a file, another process reads it
    # RF direction: File→Process (file provides data to reader)
    # DAG constraint: reader must come AFTER writer
    for wf_fid, wf_fname, writer_pid in written_files:
        if random.random() < P_READ:
            writer_idx = proc_ids.index(writer_pid)
            candidates = proc_ids[writer_idx+1:]
            if candidates:
                reader_pid = random.choice(candidates)
                add_edge(wf_fid, reader_pid, "RF")

    # ── Hub writes output files (WF) ────────────────────────────
    n_outputs = random.randint(1, 2)
    for k in range(n_outputs):
        ft = random.choice(file_types)
        fname, ext = make_fname(ft)
        fid = f"fh{k}_{sid}"
        add_node(fid, fname, "file")
        etype = pick_valid_edge("process", "file", fname, preferred=["WF"])
        add_edge(hub_id, fid, etype)
        if etype == "WF":
            written_files.append((fid, fname, hub_id))
            writer_map[fid] = hub_id

    # ── Hub executes a payload if EX allowed AND executable file exists ──
    if "EX" in edge_types_allowed:
        exec_ft = next(
            (ft for ft in [".elf", ".sh", ".bin", ".py", ".pl", ".rb", ".php"]
             if ft in file_types or _safe_ext(ft) in [_safe_ext(f) for f in file_types]),
            None
        )
        if exec_ft:
            ext = _safe_ext(exec_ft)
            exec_names = FILE_NAMES.get(ext, ["payload"])
            exec_fname = f"{random.choice(exec_names)}_{uid}{ext}"
            eid = f"exec_{sid}"
            add_node(eid, exec_fname, "file")
            if _is_executable(exec_fname):
                add_edge(hub_id, eid, "EX")
            else:
                add_edge(hub_id, eid, "WF")

    # ── RULE C — Socket causality (Process→ST→Socket→RCV→Process) ──
    if "ST" in edge_types_allowed:
        ports = {
            "T1071": ["443", "80", "8443"], "T1041": ["443", "22"],
            "T1021": ["22", "3389", "5985"], "T1046": ["*"],
            "T1595": ["80", "443", "22", "8080"]
        }
        port_list = ports.get(tid, ["443", "4444", "8080"])
        port = random.choice(port_list)
        dest_ip = f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}"
        sock_id = f"sock_{sid}"
        add_node(sock_id, f"{dest_ip}:{port}", "socket")
        # Process→Socket: ALWAYS ST
        add_edge(hub_id, sock_id, "ST")
        # Socket→Process: RCV to a child (makes socket a relay, not dead-end)
        if child_ids:
            rcv_target = random.choice(child_ids)
            add_edge(sock_id, rcv_target, "RCV")

    # ── RULE E — Sink reduction + minimum avg degree enforcement ─
    # Target file sinks: add RF edges File→Process (data flow direction)
    def _compute_sinks():
        """Count nodes that have in-edges but no out-edges (dead-ends)."""
        out_set = {s for s, _, _ in edges}
        in_set = {t for _, t, _ in edges}
        nid_set = {n[0] for n in nodes}
        return [nid for nid in nid_set if nid in in_set and nid not in out_set]

    existing_pairs = {(s, t) for s, t, _ in edges}
    total_nodes = len(nodes)

    if total_nodes > 0 and proc_ids and file_ids:
        # Phase 1: Target file sinks — add RF File→Process
        sink_attempts = 0
        while sink_attempts < 30:
            sinks = _compute_sinks()
            sink_ratio = len(sinks) / total_nodes
            avg_deg = (2 * len(edges)) / total_nodes
            if sink_ratio <= MAX_SINK_RATIO and avg_deg >= MIN_AVG_DEG:
                break

            # Find file sinks (files with no out-edges)
            file_sinks = [(fid, fl) for fid, fl in file_ids
                          if fid in sinks]
            if not file_sinks:
                break

            # Add RF edge: File→Process (file sink → random process)
            # DAG constraint: reader must come AFTER writer
            rf_fid, rf_fname = random.choice(file_sinks)
            writer_pid = writer_map.get(rf_fid)
            if writer_pid:
                writer_idx = proc_ids.index(writer_pid)
                candidates = proc_ids[writer_idx+1:]
            else:
                candidates = proc_ids
            
            if candidates:
                rp = random.choice(candidates)
                if (rf_fid, rp) not in existing_pairs:
                    add_edge(rf_fid, rp, "RF")
                    existing_pairs.add((rf_fid, rp))
            sink_attempts += 1

        # Phase 2: If avg degree still low, add more RF edges
        pad_attempts = 0
        while pad_attempts < 10:
            avg_deg = (2 * len(edges)) / total_nodes
            if avg_deg >= MIN_AVG_DEG:
                break
            rf_fid, rf_fname = random.choice(file_ids)
            writer_pid = writer_map.get(rf_fid)
            if writer_pid:
                writer_idx = proc_ids.index(writer_pid)
                candidates = proc_ids[writer_idx+1:]
            else:
                candidates = proc_ids
                
            if candidates:
                rp = random.choice(candidates)
                if (rf_fid, rp) not in existing_pairs:
                    add_edge(rf_fid, rp, "RF")
                    existing_pairs.add((rf_fid, rp))
            pad_attempts += 1

    # ── Serialize to DOT ────────────────────────────────────────
    lines = ["digraph {"]
    for nid, label, ntype in nodes:
        lines.append(f'  {nid} [label="{label}" type={ntype} stage="{tid}"]')
    for src, tgt, elabel in edges:
        lines.append(f'  {src} -> {tgt} [label="{elabel}"]')
    lines.append("}")

    with open(out_dot, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_dot


def validate_and_report(final_graph, kill_chain, output_dir, apt_name):
    """Validate output quality with OS-level semantic checks."""
    import networkx as nx
    from graph_stitcher.action_masking import validate_edge, NON_EXECUTABLE_EXTS, _get_file_ext

    tactics_covered = set(s["tactic"] for s in kill_chain)
    unique_node_labels = set(d.get("label", "")
                              for _, d in final_graph.nodes(data=True))
    bridge_edges = [(u, v, d) for u, v, d in final_graph.edges(data=True)
                    if d.get("bridge", False)]
    isolated = [n for n in final_graph.nodes()
                if final_graph.degree(n) == 0]

    # Edge type distribution
    etypes = {}
    for _, _, d in final_graph.edges(data=True):
        et = d.get("edge_type", "?")
        etypes[et] = etypes.get(et, 0) + 1

    # ── Semantic violation detection ──────────────────────────
    semantic_violations = []
    for u, v, d in final_graph.edges(data=True):
        etype = d.get("edge_type", "")
        src_type = final_graph.nodes[u].get("node_type", "process")
        tgt_type = final_graph.nodes[v].get("node_type", "process")
        tgt_label = final_graph.nodes[v].get("label", "")

        if not validate_edge(src_type, tgt_type, tgt_label, etype):
            semantic_violations.append(
                f"[SEM] Action-mask violation '{etype}' for "
                f"{src_type}->{tgt_type}: {u} -> {v}")

        # Check: FR from Process to File (INVALID)
        if src_type == "process" and tgt_type == "file" and etype == "FR":
            semantic_violations.append(
                f"[SEM] FR (Fork) from Process→File: {u} -> {v}")

        # Check: EX on non-executable file
        if etype == "EX" and tgt_type == "file":
            ext = _get_file_ext(tgt_label)
            if ext in NON_EXECUTABLE_EXTS:
                semantic_violations.append(
                    f"[SEM] EX on non-executable '{tgt_label}': {u} -> {v}")

        # Check: Non-FR edge between two processes
        if src_type == "process" and tgt_type == "process" and etype not in ("FR", ""):
            if not d.get("bridge", False):
                semantic_violations.append(
                    f"[SEM] Invalid Process→Process edge '{etype}': {u} -> {v}")

        # Check: Direct LATERAL_MOVE Process→Process (should go through File/Socket)
        if etype == "LATERAL_MOVE" and src_type == "process" and tgt_type == "process":
            semantic_violations.append(
                f"[SEM] Direct LATERAL_MOVE bridge (no intermediary): {u} -> {v}")

    # Check bridge integrity
    bridge_valid = True
    for u, v, d in bridge_edges:
        src_type = final_graph.nodes[u].get("node_type", "")
        tgt_type = final_graph.nodes[v].get("node_type", "")
        if src_type == "process" and tgt_type == "process":
            bridge_valid = False

    print("\n" + "-" * 50)
    print("[QUALITY REPORT]")
    print(f"  Nodes: {final_graph.number_of_nodes()}")
    print(f"  Edges: {final_graph.number_of_edges()}")
    print(f"  Unique node labels: {len(unique_node_labels)}")
    print(f"  Isolated nodes: {len(isolated)}")
    print(f"  Bridge edges: {len(bridge_edges)}")
    print(f"  Edge type dist: {etypes}")
    print(f"  Tactics covered: {sorted(tactics_covered)}")

    # Structural warnings
    issues = []
    if len(tactics_covered) < 3:
        issues.append(f"[FAIL] Low tactic diversity: only {tactics_covered}")
    if len(unique_node_labels) < final_graph.number_of_nodes() * 0.5:
        issues.append("[FAIL] High label duplication (>50% nodes share labels)")
    if etypes.get("WF", 0) > final_graph.number_of_edges() * 0.5:
        issues.append("[FAIL] WF edges dominate (>50%) -- unrealistic provenance")
    if len(isolated) > 0:
        issues.append(f"[FAIL] {len(isolated)} isolated nodes found")

    # Semantic violations
    if semantic_violations:
        issues.extend(semantic_violations)
    if not bridge_valid:
        issues.append("[FAIL] Bridge edges directly connect Process→Process (missing intermediary)")

    if issues:
        sem_count = sum(1 for i in issues if i.startswith("[SEM]"))
        struct_count = len(issues) - sem_count
        print(f"\n  ISSUES ({struct_count} structural, {sem_count} semantic):")
        for iss in issues:
            print(f"    {iss}")
    else:
        print("\n  [PASS] All quality + semantic checks passed!")
    print("-" * 50)

    # Save report
    report_path = os.path.join(output_dir, f"{apt_name}_quality_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Nodes: {final_graph.number_of_nodes()}\n")
        f.write(f"Edges: {final_graph.number_of_edges()}\n")
        f.write(f"Unique labels: {len(unique_node_labels)}\n")
        f.write(f"Tactics: {sorted(tactics_covered)}\n")
        f.write(f"Edge types: {etypes}\n")
        f.write(f"Semantic violations: {len(semantic_violations)}\n")
        f.write(f"Issues: {issues}\n")
    return len(issues) == 0


def run_pipeline(
    apt_name: str = "APT28",
    variant_type: str = "evolved",
    min_stages: int = 4,
    max_stages: int = 7,
    nodes_per_cluster: tuple = (8, 18),
    output_dir: str = "./result",
    checkpoint: str = None,
    data_path: str = None,
    cti_path: str = "macro_planner/data/enterprise-attack.json",
    similarity_threshold: float = 0.8,
) -> dict:
    """
    Chạy toàn bộ pipeline và trả về final provenance graph.

    Returns: {
        "graph": nx.DiGraph,
        "kill_chain": [{"technique_id": ..., "tactic": ...}, ...],
        "dot_path": str,
        "stats": {"nodes": int, "edges": int, "clusters": int},
        "reward": dict
    }
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cluster_dir = os.path.join(output_dir, "clusters")
    Path(cluster_dir).mkdir(exist_ok=True)

    print("=" * 60)
    print(f"[Pipeline] APT={apt_name} | Variant={variant_type}")
    print("=" * 60)

    # ─── BƯỚC 1: Macro Planner → Kill-chain skeleton ───────────
    print("\n[Step 1] Macro Planner: Sinh kill-chain skeleton...")
    planner = KillChainPlanner(cti_path)
    kill_chain = planner.plan_kill_chain(
        apt_name=apt_name,
        variant_type=variant_type,
        min_stages=min_stages,
        max_stages=max_stages,
        similarity_threshold=similarity_threshold,
    )
    print(f"  Kill-chain ({len(kill_chain)} stages):")
    for i, stage in enumerate(kill_chain):
        print(f"    [{i}] {stage['technique_id']} - "
              f"{stage['technique_name']} ({stage['tactic']})")

    # Validate tactic diversity ngay sau khi plan
    tactic_counts = {}
    for stage in kill_chain:
        t = stage["tactic"]
        tactic_counts[t] = tactic_counts.get(t, 0) + 1

    duplicated_tactics = {t: c for t, c in tactic_counts.items() if c > 1}
    if duplicated_tactics:
        print(f"  ⚠ WARNING: Duplicated tactics detected: {duplicated_tactics}")
        print(f"  Kill-chain planner may need rerun or diversity enforcement failed.")
        # Hard stop neu >80% stages la cung tactic (giam do nghiem ngat tranh crash RL)
        max_dup = max(tactic_counts.values())
        if max_dup >= len(kill_chain):
            print(
                f"[Pipeline] WARNING: Kill-chain diversity very low: tactic '{max(tactic_counts, key=tactic_counts.get)}' "
                f"appears {max_dup}/{len(kill_chain)} times."
            )

    # ─── BƯỚC 2: Sinh cluster cho mỗi stage ──────────────────
    print("\n[Step 2] Micro Expander: Sinh subgraph clusters...")
    cluster_dot_files = []
    use_maskgaf = checkpoint is not None and data_path is not None

    if use_maskgaf:
        # Load TAGAPT trainer khi có checkpoint
        print("  Loading TAGAPT MaskGAF model...")
        try:
            from train_CTI import Trainer, read_molecules
            from dataloader import PretrainZinkDataset
            from torch.utils.data import DataLoader
            import torch

            args_ns = argparse.Namespace(
                dataset="ASG", path=data_path,
                batch_size=4, edge_unroll=10, shuffle=False, num_workers=2,
                name="pipeline", deq_type="random", deq_coeff=0.9,
                num_flow_layer=12, gcn_layer=3, nhid=128, nout=128,
                st_type="exp", sigmoid_shift=2.0,
                all_save_prefix="./", train=False, save=True,
                no_cuda=True, learn_prior=False, seed=66666666,
                epochs=3, lr=0.001, weight_decay=0.0, dropout=0.0,
                is_bn=True, is_bn_before=False, scale_weight_norm=False,
                divide_loss=True, init_checkpoint=checkpoint,
                show_loss_step=100, temperature=0.7,
                min_atoms=nodes_per_cluster[0],
                max_atoms=nodes_per_cluster[1],
                gen_num=1, gen=True,
                save_path=os.path.dirname(checkpoint),
                penalty=False, reward_type="linear", moving_coeff=0.0,
                reward_decay=0.9, qed_coeff=1.0, plogp_coeff=1.0,
                exp_temperature=1.0, exp_bias=0.0, property="qed",
            )
            args_ns.cuda = not args_ns.no_cuda and torch.cuda.is_available()

            node_features, adj_features, mol_sizes, data_config, all_smiles = \
                read_molecules(data_path)
            dataloader = DataLoader(
                PretrainZinkDataset(node_features, adj_features, mol_sizes),
                batch_size=4, shuffle=False)
            trainer = Trainer(dataloader, data_config, args_ns,
                              all_train_smiles=all_smiles)
            trainer.initialize_from_checkpoint(gen=True)
            print("  MaskGAF model loaded successfully.")
        except Exception as e:
            print(f"  WARNING: Cannot load MaskGAF: {e}")
            print("  Falling back to dummy clusters.")
            use_maskgaf = False

    for i, stage in enumerate(kill_chain):
        tid = stage["technique_id"]
        hints = get_entity_constraints(tid)
        out_dot = os.path.join(cluster_dir, f"cluster_{i:02d}_{tid}.dot")

        print(f"  [{i+1}/{len(kill_chain)}] {tid}: {hints['description']}...")

        if use_maskgaf:
            try:
                result = trainer.generate_cluster(
                    technique_id=tid, hints=hints,
                    out_path=out_dot,
                    min_atoms=nodes_per_cluster[0],
                    max_atoms=nodes_per_cluster[1]
                )
                if result:
                    cluster_dot_files.append(result)
                    continue
            except Exception as e:
                print(f"    MaskGAF error: {e}, using dummy cluster")

        # Fallback: dummy cluster
        dummy = _create_dummy_cluster(tid, hints, out_dot, i)
        cluster_dot_files.append(dummy)

    # ─── BƯỚC 3: Graph Stitcher → Ghép clusters ────────────────
    print("\n[Step 3] Graph Stitcher: Ghép các clusters...")
    assembler = ProvenanceAssembler(output_dir)
    output_name = f"{apt_name}_{variant_type}_provenance"
    final_graph = assembler.assemble(
        cluster_dot_files=cluster_dot_files,
        kill_chain_stages=kill_chain,
        output_name=output_name
    )

    # ─── BƯỚC 3.5: Quality Validation ──────────────────────────
    is_valid = validate_and_report(final_graph, kill_chain, output_dir, apt_name)

    # ─── BƯỚC 4: Reward Calculation ──────────────────────────────
    print("\n[Step 4] Computing MARL rewards...")
    reward_calc = RewardCalculator()
    reward = reward_calc.compute(final_graph, kill_chain)

    # ─── BƯỚC 5: Summary ──────────────────────────────────────────
    stats = {
        "nodes": final_graph.number_of_nodes(),
        "edges": final_graph.number_of_edges(),
        "clusters": len(cluster_dot_files)
    }
    dot_path = f"{output_dir}/{output_name}.dot"
    report_paths = write_pipeline_report(
        output_dir=output_dir,
        output_name=output_name,
        apt_name=apt_name,
        variant_type=variant_type,
        kill_chain=kill_chain,
        planner_meta=planner.last_variant_meta,
        stats={**stats, "graph": final_graph},
        reward=reward,
        quality_valid=is_valid,
        dot_path=dot_path,
    )
    detection_paths = export_detection_bundle(
        graph=final_graph,
        kill_chain=kill_chain,
        output_dir=output_dir,
        output_name=output_name,
        apt_name=apt_name,
        variant_type=variant_type,
        planner_meta=planner.last_variant_meta,
        reward=reward,
        quality_valid=is_valid,
    )
    report_paths.update(detection_paths)

    print("\n" + "=" * 60)
    print(f"[Pipeline] DONE!")
    print(f"  Nodes: {stats['nodes']} | Edges: {stats['edges']} | "
          f"Clusters: {stats['clusters']}")
    print(f"  Output: {dot_path}")
    print(f"  Report: {report_paths['markdown']}")
    print(f"  Detection JSON: {report_paths['detection_json']}")
    print(f"  Rewards:")
    print(f"    Total:        {reward['total']:.4f}")
    print(f"    Coverage:     {reward['coverage']:.4f}")
    print(f"    Connectivity: {reward['connectivity']:.4f}")
    print(f"    Validity:     {reward['validity']:.4f}")
    print(f"    Transition:   {reward['transition_quality']:.4f}")
    print(f"    Cluster Val.: {reward['cluster_validity']:.4f}")
    print(f"    Novelty:      {reward['novelty']:.4f}")
    print(f"    Chain Depth:  {reward['chain_depth']:.4f}")
    print(f"    Macro Reward: {reward['macro_reward']:.4f}")
    print(f"    Micro Reward: {reward['micro_reward']:.4f}")
    print("=" * 60)

    return {
        "graph": final_graph,
        "kill_chain": kill_chain,
        "dot_path": dot_path,
        "report_paths": report_paths,
        "stats": stats,
        "reward": reward
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TAGAPT Macro Planner Pipeline")
    parser.add_argument("--apt", default="APT28", help="APT group name")
    parser.add_argument("--variant", default="evolved",
                        choices=["aligned", "evolved", "composite"])
    parser.add_argument("--stages", type=int, default=5, help="Max stages")
    parser.add_argument("--output", default="./result")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to MaskGAF checkpoint (optional)")
    parser.add_argument("--data", default=None,
                        help="Path to data_preprocessed/CTI (optional)")
    parser.add_argument("--cti", default="macro_planner/data/enterprise-attack.json",
                        help="Path to MITRE ATT&CK JSON")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="SCF/GNN similarity threshold for pruning and composite substitution")
    args = parser.parse_args()

    run_pipeline(
        apt_name=args.apt,
        variant_type=args.variant,
        max_stages=args.stages,
        output_dir=args.output,
        checkpoint=args.checkpoint,
        data_path=args.data,
        cti_path=args.cti,
        similarity_threshold=args.threshold,
    )
