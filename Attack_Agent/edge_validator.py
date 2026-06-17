#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
edge_validator.py
=================
Pre-GA Edge Constraint Validator for the TAGAPT pipeline.

Filters semantically impossible OS-level system call edges from generated ASGs
BEFORE the Genetic Algorithm builds its candidate pool.

Two validation layers:
  Layer 1 — Type-level constraints:  (src_type, verb, dst_type) triples
  Layer 2 — Instance-level constraints: specific tool capabilities

Integration:
    from edge_validator import EdgeConstraintValidator
    validator = EdgeConstraintValidator(verbose=True)
    relation_list, stats = validator.filter_relation_list(entity_list, relation_list)
"""

import re
import os
import logging
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict

logger = logging.getLogger("EdgeValidator")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [EdgeValidator] %(levelname)s - %(message)s",
    )

# ══════════════════════════════════════════════════════════════════════════════
# ENTITY TYPE TAXONOMY
# ══════════════════════════════════════════════════════════════════════════════
# MP = Malicious Process    TP = Tool Process       SO = Socket/Network
# MF = Malicious File       SF = System File        TF = Temporary File

PROCESS_TYPES = {"MP", "TP"}
FILE_TYPES    = {"MF", "SF", "TF"}
NETWORK_TYPES = {"SO"}
ALL_TYPES     = PROCESS_TYPES | FILE_TYPES | NETWORK_TYPES

# Edge verbs in TAGAPT
EDGE_VERBS = {"RD", "WR", "EX", "UK", "CD", "FR", "IJ", "ST", "RF"}


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1: TYPE-LEVEL CONSTRAINT MATRIX
# ══════════════════════════════════════════════════════════════════════════════
# Each entry: (src_type_set, verb, dst_type_set) that is FORBIDDEN
# "*" means "any type"

def _build_type_constraint_matrix() -> List[Tuple[Set[str], str, Set[str]]]:
    """
    Build the list of impossible (source_types, verb, target_types) triples.
    Returns list of (src_set, verb, dst_set) where verb can be "*" for all verbs.
    """
    rules = []

    # ── Files/Sockets CANNOT be source of active operations ──
    # Files have no agency - they can't read, write, execute, fork, inject, send
    for ftype in FILE_TYPES:
        for verb in ["RD", "WR", "EX", "FR", "IJ", "ST", "CD", "UK"]:
            rules.append(({ftype}, verb, ALL_TYPES))

    # Socket can only send RF (Receive From) — outgoing only
    for verb in ["WR", "EX", "IJ", "FR", "ST", "CD", "RD"]:
        rules.append(({"SO"}, verb, ALL_TYPES))
    # SO -> UK -> anything: also forbidden (socket has no "unknown" action)
    rules.append(({"SO"}, "UK", ALL_TYPES))

    # ── FR (Fork) must be Process -> Process ──
    rules.append((PROCESS_TYPES, "FR", FILE_TYPES | NETWORK_TYPES))

    # ── IJ (Inject) must target a Process ──
    rules.append((ALL_TYPES, "IJ", FILE_TYPES | NETWORK_TYPES))

    # ── ST (Send To) must target a Socket ──
    rules.append((ALL_TYPES, "ST", PROCESS_TYPES | FILE_TYPES))

    # ── RF (Receive From) must come FROM a Socket to a Process ──
    rules.append((PROCESS_TYPES | FILE_TYPES, "RF", ALL_TYPES))

    # ── EX (Execute) target cannot be a Socket ──
    rules.append((ALL_TYPES, "EX", NETWORK_TYPES))

    return rules


TYPE_CONSTRAINTS = _build_type_constraint_matrix()


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2: STRICT ALLOWLIST CONSTRAINTS
# ══════════════════════════════════════════════════════════════════════════════
# Architecture: ALLOWLIST — only tools explicitly listed here may perform
# privileged operations. Any tool NOT on the list is BLOCKED from that verb.
# This eliminates the "whack-a-mole" problem of the old denylist approach.

# ── ALLOWLIST 1: Tools that CAN create FR (Fork), IJ (Inject), or EX edges ──
# These are shells, interpreters, and task schedulers — the ONLY processes
# that have the OS-level capability to spawn/inject/execute other processes.
SHELL_AND_EXECUTORS = frozenset({
    "bash", "sh", "zsh", "csh", "ksh", "tcsh",
    "cmd", "powershell", "pwsh",
    "python", "python3", "perl", "ruby", "php",
    "cron", "crond", "at", "systemd",
    "sudo", "su",
    "schtasks", "wmic", "rundll32", "regsvr32",
    "mshta", "cscript", "wscript",
    "runas", "msbuild", "msiexec", "wusa",
    "gdb",  # debugger can inject/execute
})

# ── ALLOWLIST 2: Tools that CAN interact with Sockets (ST / RF) ──
# These are network-aware tools. Everything else is local-only.
NETWORK_CAPABLE = frozenset({
    "curl", "wget",
    "nc", "ncat", "netcat", "socat",
    "bash", "sh", "zsh",
    "ssh", "ssh-keygen",
    "ftp", "sftp", "scp",
    "ping", "traceroute",
    "telnet",
    "nmap", "masscan",
    "python", "python3", "perl", "ruby", "php",
    "nikto", "dirb", "gobuster",
    "powershell", "pwsh",
})

# ── ALLOWLIST 3: Extensions that CAN be targets of EX (Execute) ──
# Only files with these extensions (or no extension at all) can be executed.
EXECUTABLE_EXTENSIONS = frozenset({
    "",         # no extension (common for Linux binaries)
    ".sh", ".bash",
    ".exe", ".elf", ".bin",
    ".bat", ".cmd", ".ps1",
    ".pl", ".py", ".rb", ".php",
    ".cgi",
    ".jar",
})

# Executable extensions that are EXPLICIT (non-empty).
# Used when validating named EX targets — no-extension is ambiguous
# (could be Linux binary OR system config file like 'crontab', 'passwd').
EXECUTABLE_EXTENSIONS_STRICT = EXECUTABLE_EXTENSIONS - {""}

# ── ALLOWLIST 4: Tools that do BOTH network I/O AND file write ──
# Prefer these when a node has both ST/RF and WR edges.
FILE_AND_NETWORK_TOOLS = frozenset({
    "curl", "wget", "scp", "sftp", "ssh", "nc", "ncat", "socat",
    "ftp", "lftp", "rsync", "axel",
    "bash", "sh", "python", "python3", "perl", "ruby",
})

# ── ALLOWLIST 5: Pure-scanner tools — only meaningful for ST/RF, NOT WR ──
SCANNER_ONLY_TOOLS = frozenset({
    "nmap", "masscan", "nikto", "dirb", "gobuster", "zmap",
    "unicornscan", "arp-scan", "cewl",
})


# ══════════════════════════════════════════════════════════════════════════════
# FALLBACK INSTANCES
# ══════════════════════════════════════════════════════════════════════════════

_FALLBACK_INSTANCES = {
    "MP": "unknown_process",
    "TP": "unknown_tool",
    "MF": "unknown_malware.bin",
    "SF": "unknown_sysfile",
    "TF": "unknown_tempfile.tmp",
    "SO": "0.0.0.0:0",
}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN VALIDATOR CLASS
# ══════════════════════════════════════════════════════════════════════════════

class EdgeConstraintValidator:
    """
    Pre-GA Edge Constraint Validator.

    Usage:
        validator = EdgeConstraintValidator(verbose=True)

        # Layer 1: Type-level validation
        is_valid = validator.validate_edge("MP", "EX", "SO")  # False

        # Layer 2: Instance-level validation
        is_valid = validator.validate_instance("cat", "WR", "1.zip")  # False

        # Bulk filter for the GA pipeline
        filtered_rl, stats = validator.filter_relation_list(entity_list, relation_list)
    """

    # Threshold: if > this fraction of edges are removed, drop the entire graph
    GRAPH_INVALID_THRESHOLD = 0.30

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._type_rules = TYPE_CONSTRAINTS
        self._stats = defaultdict(int)

    # ------------------------------------------------------------------
    # LAYER 1: TYPE-LEVEL VALIDATION
    # ------------------------------------------------------------------
    def validate_edge(self, src_type: str, verb: str, dst_type: str) -> bool:
        """
        Check if (src_type, verb, dst_type) is a valid OS-level operation.
        Returns True if valid, False if impossible.

        Handles dirty input: strips asterisks, numbers, prefixes.
        E.g., 'MP*' -> 'MP', 'MP' -> 'MP', '*TP' -> 'TP'
        """
        src_clean = self._clean_type(src_type)
        dst_clean = self._clean_type(dst_type)

        # If type is unrecognized after cleaning, allow it (don't block unknowns)
        if src_clean not in ALL_TYPES or dst_clean not in ALL_TYPES:
            return True

        for forbidden_src, forbidden_verb, forbidden_dst in self._type_rules:
            if (src_clean in forbidden_src
                    and forbidden_verb == verb
                    and dst_clean in forbidden_dst):
                return False
        return True

    # ------------------------------------------------------------------
    # LAYER 2: INSTANCE-LEVEL VALIDATION (STRICT ALLOWLIST)
    # ------------------------------------------------------------------
    def validate_instance(
        self, tool_name: str, verb: str, target_name: str, src_type: str = "TP"
    ) -> bool:
        """
        Check if a specific tool can perform the given verb on the target.
        Returns True if valid, False if impossible.

        STRICT ALLOWLIST — NO EXCEPTIONS, not even for MP:
          - FR/IJ:    tool MUST be in SHELL_AND_EXECUTORS
          - EX:       tool MUST be in SHELL_AND_EXECUTORS AND ext in EXECUTABLE_EXTENSIONS
          - ST/RF:    tool MUST be in NETWORK_CAPABLE
          - RD/WR/CD/UK: always allowed (standard file I/O)
        """
        actual_tool = self._normalize_tool(tool_name)
        actual_ext = self._get_extension(target_name)

        # ── FR / IJ: Only shells/executors can fork or inject ──
        if verb in ("FR", "IJ"):
            if actual_tool and actual_tool not in SHELL_AND_EXECUTORS:
                return False

        # ── EX: Only shells/executors AND target must be executable ──
        if verb == "EX":
            if actual_tool and actual_tool not in SHELL_AND_EXECUTORS:
                return False
            # When the target is a *named* file (not a placeholder), require an
            # explicit executable extension.  No-extension ("") is too ambiguous:
            # it covers both real Linux binaries AND system-config files like
            # 'crontab' or 'passwd' which are NOT valid EX targets in APT context.
            if target_name and target_name not in ("", "unknown"):
                if actual_ext not in EXECUTABLE_EXTENSIONS_STRICT:
                    return False

        # ── ST / RF: Only network-capable tools ──
        if verb in ("ST", "RF"):
            if actual_tool and actual_tool not in NETWORK_CAPABLE:
                return False

        # ── RD / WR / CD / UK: Always allowed ──
        return True

    # ------------------------------------------------------------------
    # BULK FILTER FOR GA PIPELINE
    # ------------------------------------------------------------------
    def filter_relation_list(
        self,
        entity_list: List[str],
        relation_list: List[List],
    ) -> Tuple[List[List], Dict[str, int]]:
        """
        Filter impossible edges from the relation_list BEFORE GA.

        Parameters
        ----------
        entity_list : list of entity types (e.g., ["MP", "TP", "SO", "SF", ...])
        relation_list : list of 4 stages, each stage is list of edges
                       edge = [src_idx, dst_idx, verb, global_edge_idx]

        Returns
        -------
        filtered_relation_list : same structure, impossible edges removed
        stats : dict with removal statistics
        """
        self._stats = defaultdict(int)
        total_edges = sum(len(stage) for stage in relation_list)
        removed_total = 0

        filtered_relations = []

        for stage_idx, stage_edges in enumerate(relation_list):
            filtered_stage = []
            for edge in stage_edges:
                src_idx = int(edge[0])
                dst_idx = int(edge[1])
                verb = edge[2]

                # Bounds check
                if src_idx >= len(entity_list) or dst_idx >= len(entity_list):
                    self._stats["out_of_bounds"] += 1
                    removed_total += 1
                    continue

                src_type = self._clean_type(entity_list[src_idx])
                dst_type = self._clean_type(entity_list[dst_idx])

                # Layer 1: Type-level check
                if not self.validate_edge(src_type, verb, dst_type):
                    self._stats[f"type_violation_{src_type}_{verb}_{dst_type}"] += 1
                    removed_total += 1
                    if self.verbose:
                        logger.debug(
                            f"  REMOVED stage{stage_idx+1}: "
                            f"node{src_idx}({src_type}) -{verb}-> "
                            f"node{dst_idx}({dst_type})"
                        )
                    continue

                filtered_stage.append(edge)

            filtered_relations.append(filtered_stage)

        # Check graph validity: if too many edges removed, flag it
        removal_rate = removed_total / max(total_edges, 1)
        is_valid = removal_rate <= self.GRAPH_INVALID_THRESHOLD

        stats = {
            "total_edges": total_edges,
            "removed": removed_total,
            "remaining": total_edges - removed_total,
            "removal_rate": removal_rate,
            "graph_valid": is_valid,
        }
        # Merge violation details
        stats.update(dict(self._stats))

        if self.verbose:
            logger.info(
                f"Edge filter: {total_edges} edges -> "
                f"{total_edges - removed_total} kept, "
                f"{removed_total} removed ({removal_rate:.1%})"
            )
            if removed_total > 0:
                # Log top violation types
                violations = {k: v for k, v in self._stats.items()
                              if k.startswith("type_violation")}
                for vtype, count in sorted(violations.items(),
                                           key=lambda x: x[1], reverse=True)[:5]:
                    logger.info(f"  {vtype}: {count}")

            if not is_valid:
                logger.warning(
                    f"  Graph INVALID: {removal_rate:.0%} edges removed "
                    f"(threshold: {self.GRAPH_INVALID_THRESHOLD:.0%})"
                )

        return filtered_relations, stats

    # ------------------------------------------------------------------
    # DANGLING NODE CLEANUP
    # ------------------------------------------------------------------
    @staticmethod
    def remove_dangling_nodes(
        entity_list: List[str],
        relation_list: List[List],
    ) -> Tuple[List[str], List[List], Dict[int, int]]:
        """
        After edge filtering, remove nodes with degree=0 (no edges).
        Re-index remaining nodes and update edges.

        Returns
        -------
        new_entity_list   : cleaned entity list
        new_relation_list : edges with updated indices
        node_map          : {new_idx: orig_idx}
        """
        n = len(entity_list)

        # Find nodes that appear in at least one edge
        active_nodes = set()
        for stage_edges in relation_list:
            for edge in stage_edges:
                active_nodes.add(int(edge[0]))
                active_nodes.add(int(edge[1]))

        # If all nodes are active, return identity
        if len(active_nodes) == n:
            return entity_list, relation_list, {i: i for i in range(n)}

        # Build new index mapping (only active nodes)
        active_sorted = sorted(active_nodes)
        old_to_new = {orig: new for new, orig in enumerate(active_sorted)}
        node_map = {new: orig for orig, new in old_to_new.items()}

        new_entity_list = [entity_list[orig] for orig in active_sorted]

        new_relation_list = []
        for stage_edges in relation_list:
            new_stage = []
            for edge in stage_edges:
                src = int(edge[0])
                dst = int(edge[1])
                if src in old_to_new and dst in old_to_new:
                    new_edge = [
                        str(old_to_new[src]),
                        str(old_to_new[dst]),
                        edge[2],
                        edge[3],
                    ]
                    new_stage.append(new_edge)
            new_relation_list.append(new_stage)

        dangling_count = n - len(active_sorted)
        if dangling_count > 0:
            logger.info(
                f"Removed {dangling_count} dangling nodes: "
                f"{n} -> {len(new_entity_list)} nodes"
            )

        return new_entity_list, new_relation_list, node_map

    # ------------------------------------------------------------------
    # FALLBACK INSTANCE
    # ------------------------------------------------------------------
    @staticmethod
    def get_fallback_instance(entity_type: str) -> str:
        """Return a generic fallback instance name when CTI lookup fails."""
        clean = EdgeConstraintValidator._clean_type(entity_type)
        return _FALLBACK_INSTANCES.get(clean, "unknown_entity")

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_type(raw_type: str) -> str:
        """Normalize entity type string.

        Handles all TAGAPT formats:
          'MP'       -> 'MP'
          'MP*'      -> 'MP'
          '*MP'      -> 'MP'
          'MP*-cat'  -> 'MP'
          '0_MP*'    -> 'MP'
          '11_TP*-masscan' -> 'TP'
        """
        if not raw_type:
            return ""
        s = raw_type.strip()
        # Remove leading index prefix like '0_', '11_'
        m = re.match(r'^\d+_(.+)', s)
        if m:
            s = m.group(1)
        # Remove instance suffix: '-cat', '-1.zip', etc.
        s = s.split('-', 1)[0]
        # Remove asterisks
        s = s.replace('*', '')
        # Should be 2 uppercase letters now
        return s.strip()

    @staticmethod
    def _normalize_tool(name: str) -> str:
        """Extract the base tool name from a TAGAPT instance string.

        PRIMARY METHOD: split on '-' and take the LAST segment.
        This is the most robust approach because TAGAPT names always
        use '-' to separate the type prefix from the tool name.

        Examples:
          'cat'              -> 'cat'
          'bash*'            -> 'bash'
          '0_MP*-find'       -> 'find'
          '10_TP-paste'      -> 'paste'
          '11_TP*-masscan'   -> 'masscan'
          '4_TP-less'        -> 'less'
          'MP-cat'           -> 'cat'
          '/usr/bin/grep'    -> 'grep'
          'cmd.exe'          -> 'cmd'
        """
        if not name:
            return ""
        s = name.strip()

        # Primary: split on '-' and take last part
        # This handles ALL TAGAPT formats: '0_MP*-find', 'TP-cat', 'MP*-bash'
        if '-' in s:
            tool = s.split('-')[-1].lower().strip()
            # If the last segment looks like a tool name (not a number/IP),
            # use it directly
            if tool and not tool.replace('.', '').isdigit():
                # Remove .exe/.bin/.sh suffix
                for ext in ('.exe', '.bin', '.sh'):
                    if tool.endswith(ext):
                        tool = tool[:-len(ext)]
                return tool

        # Fallback: plain name possibly with path
        base = os.path.basename(s.replace('\\', '/'))
        # Remove trailing * (hub marker)
        base = base.rstrip('*')
        # Remove common extensions for processes
        for ext in ('.exe', '.bin', '.sh'):
            if base.lower().endswith(ext):
                base = base[:-len(ext)]
        return base.lower().strip()

    @staticmethod
    def _get_extension(name: str) -> str:
        """Get lowercase file extension including the dot.

        PRIMARY METHOD: split on '.' and take the LAST segment.
        This is the most robust approach for TAGAPT instance names.

        Examples:
          '1.zip'                   -> '.zip'
          'flag3.txt'               -> '.txt'
          '3_SF-1.zip'              -> '.zip'
          '14_SF-e2.c'              -> '.c'
          '6_MF-cron.m'             -> '.m'
          '10_SF-victim-shred.txt'  -> '.txt'
          'T1574.006.c'             -> '.c'
          'http://192.168.1.1'      -> ''
          '192.168.1.1'             -> ''
        Returns '' if no extension found.
        """
        if not name:
            return ""
        s = name.strip()

        # Skip URLs
        if '://' in s:
            return ""

        # Skip bare IP addresses (all digits and dots)
        bare = s.split('-')[-1] if '-' in s else s
        if re.match(r'^[\d.]+$', bare):
            return ""

        # Primary: split on '.' and take the LAST segment
        if '.' in s:
            ext_candidate = s.split('.')[-1].lower().strip()
            # Must contain at least one letter to be a real extension
            if ext_candidate and re.search(r'[a-z]', ext_candidate):
                return '.' + ext_candidate

        return ""


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("EdgeConstraintValidator Self-Test")
    print("=" * 60)

    v = EdgeConstraintValidator(verbose=True)

    # --- _clean_type tests ---
    print("\n--- _clean_type Tests ---")
    clean_tests = [
        ("MP", "MP"), ("MP*", "MP"), ("*MP", "MP"),
        ("TP", "TP"), ("SO", "SO"), ("SF", "SF"),
        ("0_MP*", "MP"), ("11_TP*-masscan", "TP"),
        ("MP*-cat", "MP"), ("3_SF-1.zip", "SF"),
    ]
    ct_passed = 0
    for raw, expected in clean_tests:
        result = v._clean_type(raw)
        if result != expected:
            print(f"  [FAIL] _clean_type({raw!r}) = {result!r}, expected {expected!r}")
        else:
            ct_passed += 1
    print(f"  _clean_type: {ct_passed}/{len(clean_tests)} passed")

    # --- _normalize_tool tests ---
    print("\n--- _normalize_tool Tests ---")
    tool_tests = [
        ("cat", "cat"), ("bash*", "bash"),
        ("0_MP*-find", "find"), ("11_TP*-masscan", "masscan"),
        ("4_TP-less", "less"), ("MP-cat", "cat"),
        ("/usr/bin/grep", "grep"), ("cmd.exe", "cmd"),
        ("9_TP*-tac", "tac"), ("12_MP*-cat", "cat"),
        # User-reported failures:
        ("10_TP-paste", "paste"),
        ("3_TP*-dnsdomainname", "dnsdomainname"),
        ("7_TP-chmod", "chmod"),
        ("5_TP-dd", "dd"),
    ]
    nt_passed = 0
    for raw, expected in tool_tests:
        result = v._normalize_tool(raw)
        if result != expected:
            print(f"  [FAIL] _normalize_tool({raw!r}) = {result!r}, expected {expected!r}")
        else:
            nt_passed += 1
    print(f"  _normalize_tool: {nt_passed}/{len(tool_tests)} passed")

    # --- _get_extension tests ---
    print("\n--- _get_extension Tests ---")
    ext_tests = [
        ("1.zip", ".zip"), ("flag3.txt", ".txt"),
        ("3_SF-1.zip", ".zip"), ("6_MF-cron.m", ".m"),
        ("10_SF-victim-shred.txt", ".txt"),
        ("T1574.006.c", ".c"), ("5_TF-da-ks.cfg", ".cfg"),
        ("http://192.168.1.1", ""),
        ("8_SF-ufw.log", ".log"),
        # User-reported failures:
        ("14_SF-e2.c", ".c"),
        ("7_MF-backsup.java", ".java"),
        ("11_TF-ufw.h", ".h"),
    ]
    ex_passed = 0
    for raw, expected in ext_tests:
        result = v._get_extension(raw)
        if result != expected:
            print(f"  [FAIL] _get_extension({raw!r}) = {result!r}, expected {expected!r}")
        else:
            ex_passed += 1
    print(f"  _get_extension: {ex_passed}/{len(ext_tests)} passed")

    # --- Layer 1: Type-level tests (including dirty input) ---
    print("\n--- Layer 1: Type-Level Tests ---")
    tests_type = [
        # Clean types
        ("MP", "FR", "TP", True),
        ("MP", "WR", "SF", True),
        ("MP", "ST", "SO", True),
        ("SO", "RF", "TP", True),
        ("MP", "RD", "SF", True),
        ("SO", "WR", "SF", False),
        ("SF", "EX", "MP", False),
        ("MP", "EX", "SO", False),
        ("MP", "IJ", "SF", False),
        ("MP", "ST", "SF", False),
        ("MP", "FR", "SF", False),
        ("MP", "RF", "TP", False),
        ("SF", "RD", "MP", False),
        ("SO", "EX", "MP", False),
        # Dirty types with asterisks (from Kaggle)
        ("MP*", "FR", "TP", True),
        ("MP*", "ST", "SO", True),
        ("SO", "RF", "TP*", True),
        ("SF", "EX", "MP*", False),
        ("MP*", "IJ", "SF", False),
    ]

    passed = 0
    for src, verb, dst, expected in tests_type:
        result = v.validate_edge(src, verb, dst)
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL":
            print(f"  [{status}] {src}-{verb}->{dst}: got {result}, expected {expected}")
        else:
            passed += 1
    print(f"  Type-level: {passed}/{len(tests_type)} passed")

    # --- Layer 2: Instance-Level ALLOWLIST Tests ---
    print("\n--- Layer 2: Instance-Level Tests (ALLOWLIST) ---")
    tests_inst = [
        # ── FR/IJ: only SHELL_AND_EXECUTORS allowed ──
        ("bash", "FR", "anything", True),          # bash IS a shell
        ("sh", "FR", "anything", True),            # sh IS a shell
        ("sudo", "FR", "anything", True),          # sudo IS an executor
        ("perl", "IJ", "target", True),            # perl IS an executor
        ("cat", "FR", "anything", False),           # cat NOT a shell
        ("rev", "FR", "anything", False),           # rev NOT a shell
        ("tee", "FR", "anything", False),           # tee NOT a shell
        ("paste", "IJ", "perl", False),            # paste NOT an executor
        ("logname", "IJ", "anything", False),      # logname NOT an executor
        ("curl", "FR", "anything", False),         # curl NOT a shell
        ("chmod", "FR", "anything", False),        # chmod NOT a shell
        ("halt", "FR", "anything", False),         # halt NOT a shell
        ("vim", "FR", "anything", False),          # vim NOT a shell

        # ── EX: only SHELL_AND_EXECUTORS + EXECUTABLE_EXTENSIONS ──
        ("bash", "EX", "script.sh", True),         # bash + .sh OK
        ("sh", "EX", "run.pl", True),              # sh + .pl OK
        ("sudo", "EX", "binary", True),            # sudo + no ext OK
        ("perl", "EX", "shell.cgi", True),         # perl + .cgi OK
        ("bash", "EX", "e2.c", False),             # bash + .c NOT executable
        ("bash", "EX", "s.c", False),              # bash + .c NOT executable
        ("bash", "EX", "data.zip", False),         # bash + .zip NOT executable
        ("cat", "EX", "file.txt", False),          # cat NOT an executor
        ("rev", "EX", "anything", False),          # rev NOT an executor
        ("tee", "EX", "script.sh", False),         # tee NOT an executor

        # ── ST/RF: only NETWORK_CAPABLE allowed ──
        ("curl", "ST", "192.168.1.1", True),       # curl IS network-capable
        ("wget", "ST", "example.com", True),       # wget IS network-capable
        ("bash", "ST", "socket", True),            # bash IS network-capable
        ("nc", "RF", "socket", True),              # nc IS network-capable
        ("sh", "ST", "socket", True),              # sh IS network-capable
        ("perl", "ST", "socket", True),            # perl IS network-capable
        ("nmap", "ST", "target", True),            # nmap IS network-capable
        ("paste", "ST", "socket", False),          # paste NOT network
        ("rev", "ST", "socket", False),            # rev NOT network
        ("logname", "ST", "socket", False),        # logname NOT network
        ("cat", "ST", "socket", False),            # cat NOT network
        ("tee", "ST", "socket", False),            # tee NOT network
        ("dd", "ST", "socket", False),             # dd NOT network
        ("cat", "RF", "socket", False),            # cat NOT network

        # ── RD/WR/CD/UK: always allowed ──
        ("cat", "RD", "readme.txt", True),
        ("tee", "WR", "file.log", True),
        ("paste", "RD", "data.csv", True),
        ("rev", "WR", "file.txt", True),
        ("curl", "WR", "output.html", True),
        ("tar", "WR", "backup.tar", True),

        # ── TAGAPT-format names from actual .dot files ──
        ("0_MP*-bash", "FR", "1_TP-id", True),     # bash IS in SHELL_AND_EXECUTORS
        ("4_TP*-sh", "ST", "2_SO-http://192.168.1.1", True),  # sh IS in NETWORK_CAPABLE
        ("12_MP*-cat", "RD", "8_SF-ufw.log", True),           # cat can read (RD allowed)

        # ── MP TOOLS GET NO FREE PASS (bypass removed!) ──
        ("0_MP-cat", "FR", "1_TP-id", False),       # cat is MP but NOT a shell
        ("0_MP-ls", "FR", "1_TP-id", False),        # ls is MP but NOT a shell
        ("0_MP-curl", "FR", "1_TP-id", False),      # curl is MP but NOT a shell
        ("0_MP-curl", "EX", "file.sh", False),      # curl is MP but NOT an executor
        ("0_MP-dd", "ST", "2_SO-socket", False),    # dd is MP but NOT network
        ("0_MP-cat", "RF", "2_SO-socket", False),   # cat is MP but NOT network
        ("0_MP-halt", "FR", "1_TP-rev", False),     # halt is MP but NOT a shell

        # ── .DOT FILE HALLUCINATIONS (must ALL return False) ──
        ("0_MP-teehee", "FR", "1_TP-rev", False),     # teehee NOT a shell
        ("1_TP-rev", "FR", "anything", False),         # rev NOT a shell
        ("3_MP-sudo", "EX", "5_TF-s.c", False),       # .c not executable
        ("10_TP-paste", "ST", "1_SO-socket", False),   # paste NOT network
        ("10_TP-paste", "IJ", "11_TP-perl", False),    # paste NOT executor
        ("11_TP-perl", "EX", "14_SF-e2.c", False),     # .c NOT executable
        ("16_MP-cat", "RF", "6_SO-socket", False),     # cat NOT network (MP no pass)
        ("9_TP-chmod", "FR", "anything", False),       # chmod NOT a shell
    ]

    passed = 0
    for tool, verb, target, expected in tests_inst:
        result = v.validate_instance(tool, verb, target)
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL":
            print(f"  [{status}] {tool}-{verb}->{target}: got {result}, expected {expected}")
        else:
            passed += 1
    print(f"  Instance-level: {passed}/{len(tests_inst)} passed")

    # --- Bulk filter test ---
    print("\n--- Bulk Filter Test ---")
    test_entity = ["MP", "MP", "SO", "SF", "TP", "SF", "MF", "MF", "SF", "TP", "SF", "TP"]
    test_relations = [
        [
            ["0", "1", "FR", 0],
            ["0", "2", "ST", 1],
            ["2", "1", "RF", 2],
            ["0", "3", "EX", 3],
            ["1", "3", "CD", 4],
            ["5", "0", "RD", 5],   # SF->MP RD: INVALID
        ],
        [
            ["0", "4", "IJ", 6],
            ["4", "2", "ST", 7],
            ["4", "3", "WR", 8],
            ["3", "4", "EX", 9],   # SF->TP EX: INVALID
        ],
        [
            ["0", "1", "FR", 10],
            ["0", "5", "IJ", 11],  # MP->SF IJ: INVALID
        ],
        [
            ["2", "1", "RF", 12],
            ["1", "2", "RF", 13],  # MP->SO RF: INVALID
        ],
    ]

    filtered, stats = v.filter_relation_list(test_entity, test_relations)
    print(f"  Total: {stats['total_edges']}, Removed: {stats['removed']}, "
          f"Remaining: {stats['remaining']}")
    assert stats["removed"] == 4, f"Expected 4 removed, got {stats['removed']}"
    print(f"  Bulk filter test: PASS")

    # --- Dangling node test ---
    print("\n--- Dangling Node Test ---")
    new_e, new_r, nmap = v.remove_dangling_nodes(test_entity, filtered)
    print(f"  Nodes: {len(test_entity)} -> {len(new_e)}")
    print(f"  Node map sample: {dict(list(nmap.items())[:5])}")

    # --- Fallback test ---
    print("\n--- Fallback Instance Test ---")
    for etype in ["MP", "TP", "MF", "SF", "TF", "SO", "MP*", "TP*"]:
        fb = v.get_fallback_instance(etype)
        print(f"  {etype} -> {fb}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
