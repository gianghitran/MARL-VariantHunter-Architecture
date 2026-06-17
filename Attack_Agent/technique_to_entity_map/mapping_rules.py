# technique_to_entity_map/mapping_rules.py
"""
Mapping từ MITRE technique sang OS entity constraints.
Mỗi entry định nghĩa:
  - allowed_processes: loại process được phép xuất hiện
  - allowed_file_types: loại file được phép
  - allowed_edge_types: loại edge được phép
  - hub_process: process trung tâm của cluster
  - description: mô tả ngắn
"""

TECHNIQUE_ENTITY_MAP = {
    # ── RECONNAISSANCE ────────────────────────────────────────
    "T1595": {  # Active Scanning
        "allowed_processes": ["nmap", "masscan", "zmap", "netcat", "ping", "traceroute"],
        "allowed_file_types": [".log", ".txt", ".xml"],
        "allowed_edge_types": ["ST", "RF", "WF"],
        "hub_process": "nmap",
        "description": "Network scanning"
    },
    "T1592": {  # Gather Victim Host Information
        "allowed_processes": ["curl", "wget", "python3", "bash"],
        "allowed_file_types": [".html", ".json", ".txt"],
        "allowed_edge_types": ["ST", "RF", "WF"],
        "hub_process": "curl",
        "description": "Host info gathering"
    },

    # ── INITIAL ACCESS ────────────────────────────────────────
    "T1566": {  # Phishing
        "allowed_processes": ["thunderbird", "mutt", "python3", "bash"],
        "allowed_file_types": [".pdf", ".doc", ".xls", ".zip", ".elf"],
        "allowed_edge_types": ["RF", "WF", "EX", "FR"],
        "hub_process": "python3",
        "description": "Phishing delivery"
    },
    "T1190": {  # Exploit Public-Facing Application
        "allowed_processes": ["apache2", "nginx", "python3", "php", "node"],
        "allowed_file_types": [".php", ".py", ".sh", ".elf"],
        "allowed_edge_types": ["ST", "RF", "WF", "EX", "FR"],
        "hub_process": "nginx",
        "description": "Web exploit"
    },

    # ── EXECUTION ─────────────────────────────────────────────
    "T1059": {  # Command and Scripting Interpreter
        "allowed_processes": ["bash", "sh", "python3", "python", "perl", "ruby"],
        "allowed_file_types": [".sh", ".py", ".pl", ".rb", ".elf", ".bin"],
        "allowed_edge_types": ["FR", "EX", "RF", "WF", "IJ"],
        "hub_process": "bash",
        "description": "Script execution"
    },
    "T1059.001": {  # PowerShell (mapped to bash equivalent on Linux)
        "allowed_processes": ["bash", "python3", "ruby"],
        "allowed_file_types": [".sh", ".py", ".elf"],
        "allowed_edge_types": ["FR", "EX", "RF", "WF"],
        "hub_process": "bash",
        "description": "Scripting interpreter"
    },

    # ── PERSISTENCE ───────────────────────────────────────────
    "T1053": {  # Scheduled Task/Job
        "allowed_processes": ["cron", "at", "systemd", "bash"],
        "allowed_file_types": ["/etc/cron.d/*", "/etc/crontab", ".sh", ".service"],
        "allowed_edge_types": ["RF", "WF", "FR"],
        "hub_process": "cron",
        "description": "Scheduled persistence"
    },
    "T1547": {  # Boot/Logon Autostart
        "allowed_processes": ["systemd", "bash", "python3"],
        "allowed_file_types": [".service", ".sh", ".elf", "/etc/init.d/*"],
        "allowed_edge_types": ["RF", "WF", "EX"],
        "hub_process": "systemd",
        "description": "Autostart persistence"
    },

    # ── PRIVILEGE ESCALATION ──────────────────────────────────
    "T1068": {  # Exploitation for Privilege Escalation
        "allowed_processes": ["bash", "python3", "sudo", "su"],
        "allowed_file_types": [".elf", ".bin", ".sh"],
        "allowed_edge_types": ["EX", "FR", "IJ"],
        "hub_process": "sudo",
        "description": "Privilege escalation exploit"
    },

    # ── DEFENSE EVASION ───────────────────────────────────────
    "T1070": {  # Indicator Removal
        "allowed_processes": ["bash", "rm", "shred", "python3"],
        "allowed_file_types": [".log", ".sh", "/var/log/*"],
        "allowed_edge_types": ["RF", "WF", "EX"],
        "hub_process": "bash",
        "description": "Log/artifact removal"
    },

    # ── CREDENTIAL ACCESS ─────────────────────────────────────
    "T1003": {  # OS Credential Dumping
        "allowed_processes": ["bash", "python3", "mimikatz", "sudo"],
        "allowed_file_types": ["/etc/passwd", "/etc/shadow", ".db", ".txt"],
        "allowed_edge_types": ["RF", "WF", "EX", "FR"],
        "hub_process": "python3",
        "description": "Credential dumping"
    },

    # ── DISCOVERY ─────────────────────────────────────────────
    "T1083": {  # File and Directory Discovery
        "allowed_processes": ["ls", "find", "bash", "python3"],
        "allowed_file_types": [".txt", ".conf", ".db", ".log"],
        "allowed_edge_types": ["RF", "WF"],
        "hub_process": "find",
        "description": "File/dir discovery"
    },
    "T1046": {  # Network Service Discovery
        "allowed_processes": ["nmap", "netstat", "ss", "bash"],
        "allowed_file_types": [".txt", ".log", ".xml"],
        "allowed_edge_types": ["ST", "RF", "WF"],
        "hub_process": "nmap",
        "description": "Network service scan"
    },

    # ── LATERAL MOVEMENT ──────────────────────────────────────
    "T1021": {  # Remote Services
        "allowed_processes": ["ssh", "scp", "rsync", "bash"],
        "allowed_file_types": [".sh", ".elf", ".key"],
        "allowed_edge_types": ["ST", "RF", "WF", "EX"],
        "hub_process": "ssh",
        "description": "Remote service access"
    },

    # ── COLLECTION ───────────────────────────────────────────
    "T1005": {  # Data from Local System
        "allowed_processes": ["bash", "python3", "cat", "find"],
        "allowed_file_types": [".doc", ".pdf", ".txt", ".db", ".csv", ".zip"],
        "allowed_edge_types": ["RF", "WF"],
        "hub_process": "python3",
        "description": "Local data collection"
    },

    # ── COMMAND AND CONTROL ───────────────────────────────────
    "T1071": {  # Application Layer Protocol
        "allowed_processes": ["curl", "wget", "python3", "nc", "bash"],
        "allowed_file_types": [".sh", ".bin", ".py"],
        "allowed_edge_types": ["ST", "RF", "WF", "FR"],
        "hub_process": "curl",
        "description": "C2 communication"
    },

    # ── EXFILTRATION ──────────────────────────────────────────
    "T1041": {  # Exfiltration Over C2 Channel
        "allowed_processes": ["curl", "wget", "python3", "nc", "ssh"],
        "allowed_file_types": [".zip", ".tar", ".gz", ".enc"],
        "allowed_edge_types": ["ST", "RF", "WF", "FR"],
        "hub_process": "curl",
        "description": "Data exfiltration"
    },
}


def get_entity_constraints(technique_id: str) -> dict:
    """Trả về constraints cho technique_id. Fallback về generic nếu không tìm thấy."""
    if technique_id in TECHNIQUE_ENTITY_MAP:
        return TECHNIQUE_ENTITY_MAP[technique_id]

    # Thử parent technique (T1059.001 → T1059)
    parent = technique_id.split(".")[0]
    if parent in TECHNIQUE_ENTITY_MAP:
        return TECHNIQUE_ENTITY_MAP[parent]

    # Generic fallback
    return {
        "allowed_processes": ["bash", "python3", "sh"],
        "allowed_file_types": [".sh", ".py", ".txt", ".elf"],
        "allowed_edge_types": ["RF", "WF", "EX", "FR"],
        "hub_process": "bash",
        "description": f"Generic ({technique_id})"
    }
