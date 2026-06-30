"""
experiments/run_marl.py — wrapper CO CACHE quanh closed-loop MARL (main.py).

Moi dieu kien thi nghiem duoc dinh danh boi `tag` + dict env-var. Neu best_mlp.pth
cho tag do da ton tai -> tai dung (KHONG chay lai). Nho vay 1 canonical run phuc vu
RQ1-B1 / RQ2-C2 / RQ3-A0 ma chi chay MARL mot lan.

Chay main.py qua subprocess de cach ly hoan toan (main.py set seed toan cuc, redirect
stdout, sua trong so in-place — khong nen goi nhieu lan trong cung process).
"""
import os
import re
import sys
import subprocess

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARL_RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "marl_runs")
os.makedirs(MARL_RUNS_DIR, exist_ok=True)


def parse_run_log(run_dir: str) -> dict:
    """
    Doc run.log -> {f1_curve, episodes_to_nash, stop_reason, last_f1}.
      - f1_curve: list F1 theo episode (parse dong 'Global F1 : x').
      - episodes_to_nash: so episode khi [STOP-CONVERGED] (None neu khong hoi tu).
    """
    log_path = os.path.join(run_dir, "run.log")
    f1_curve, episodes_to_nash, stop_reason = [], None, None
    if not os.path.exists(log_path):
        return {"f1_curve": [], "episodes_to_nash": None, "stop_reason": None, "last_f1": None}

    f1_re = re.compile(r"Global F1\s*:\s*([0-9.]+)")
    nash_re = re.compile(r"\[STOP-CONVERGED\].*sau\s+(\d+)\s+episodes")
    with open(log_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = f1_re.search(line)
            if m:
                try:
                    f1_curve.append(float(m.group(1)))
                except ValueError:
                    pass
            m = nash_re.search(line)
            if m:
                episodes_to_nash = int(m.group(1))
                stop_reason = "CONVERGED"
            elif "[STOP-PLATEAU]" in line:
                stop_reason = stop_reason or "PLATEAU"
            elif "[STOP-FAILURE]" in line:
                stop_reason = stop_reason or "FAILURE"
    return {
        "f1_curve": f1_curve,
        "episodes_to_nash": episodes_to_nash,
        "stop_reason": stop_reason,
        "last_f1": f1_curve[-1] if f1_curve else None,
    }


def run_marl(tag: str, env_overrides: dict = None, seed: int = 42,
             num_episodes: int = None, force: bool = False) -> dict:
    """
    Chay (hoac tai dung) mot MARL run.

    Args:
        tag           : dinh danh dieu kien (vd 'canonical', 'rq2_c1', 'rq3_linear').
        env_overrides : dict env-var bo sung (EXP_CLASSIFIER, EXP_DISABLE_GENERATION,
                        EXP_DISABLE_EWC, EXP_ALLOWED_VARIANTS, ...).
        seed          : EXP_SEED.
        num_episodes  : EXP_NUM_EPISODES (None = mac dinh 150 cua main.py).
        force         : True -> chay lai du da co checkpoint.

    Returns: {tag, run_dir, checkpoint, reused (bool), **parse_run_log(...)}
    """
    run_dir = os.path.join(MARL_RUNS_DIR, tag)

    def _resolve_ckpt():
        # Tra ve (mlp_path, gat_path). Uu tien best (co cai thien); neu khong co thi
        # final (head+GAT cuoi cung, dung kien truc). Eval PHAI dung dung cap GAT+head
        # vi co-evolution train GAT chung voi head (freeze_gat_cl=False).
        best_mlp = os.path.join(run_dir, "best_mlp.pth")
        best_gat = os.path.join(run_dir, "best_gat.pth")
        final_mlp = os.path.join(run_dir, "final_mlp.pth")
        final_gat = os.path.join(run_dir, "final_gat.pth")
        if os.path.exists(best_mlp):
            return best_mlp, (best_gat if os.path.exists(best_gat) else None)
        if os.path.exists(final_mlp):
            return final_mlp, (final_gat if os.path.exists(final_gat) else None)
        return None, None

    existing, existing_gat = _resolve_ckpt()
    if existing is not None and not force:
        print(f"[run_marl] tag='{tag}' -> REUSE existing {existing}")
        return {"tag": tag, "run_dir": run_dir, "checkpoint": existing,
                "gat_checkpoint": existing_gat, "reused": True, **parse_run_log(run_dir)}

    os.makedirs(run_dir, exist_ok=True)
    env = os.environ.copy()
    env["EXP_RUN_DIR"] = run_dir
    env["EXP_SEED"] = str(seed)
    if num_episodes is not None:
        env["EXP_NUM_EPISODES"] = str(num_episodes)
    for k, v in (env_overrides or {}).items():
        env[k] = str(v)

    print(f"[run_marl] tag='{tag}' -> RUN main.py "
          f"(seed={seed}, episodes={num_episodes or 'default'}, "
          f"overrides={env_overrides or {}})")
    proc = subprocess.run(
        [sys.executable, os.path.join(_REPO_ROOT, "main.py")],
        cwd=_REPO_ROOT, env=env,
    )
    if proc.returncode != 0:
        print(f"[run_marl] WARNING: main.py exited with code {proc.returncode}")

    checkpoint, gat_checkpoint = _resolve_ckpt()
    if checkpoint is None:
        print(f"[run_marl] WARNING: no mlp checkpoint for tag='{tag}'. "
              f"Falling back to pretrained in load_detector.")
    elif checkpoint.endswith("final_mlp.pth"):
        print(f"[run_marl] tag='{tag}': no improvement over baseline -> using final_mlp.pth "
              f"(detector cuoi cung, dung kien truc).")

    return {"tag": tag, "run_dir": run_dir, "checkpoint": checkpoint,
            "gat_checkpoint": gat_checkpoint, "reused": False, **parse_run_log(run_dir)}


# ─────────────────────────────────────────────────────────────────────────────
# Standard runs (tag scheme dung CHUNG -> cache chia se giua RQ1/RQ2/RQ3)
# Tat ca deu EXP_ALLOWED_VARIANTS=aligned (Leave-Variant-Out): co-evolution chi train
# tren variant 'aligned'; evolved/composite la unseen.
# ─────────────────────────────────────────────────────────────────────────────
def _ep_tag(episodes):
    return str(episodes) if episodes is not None else "def"


def canonical_run(seed=42, episodes=None, force=False):
    """Run #1: gen ON, EWC ON, clf=mlp. Phuc vu RQ1-B1 / RQ2-C2 / RQ3-A0."""
    return run_marl(
        tag=f"canonical_s{seed}_e{_ep_tag(episodes)}",
        env_overrides={"EXP_ALLOWED_VARIANTS": "aligned"},
        seed=seed, num_episodes=episodes, force=force,
    )


def c1_run(seed=42, episodes=None, force=False):
    """Run #2 (RQ2-C1): gen OFF, EWC ON -> co lap dong gop cua EWC."""
    return run_marl(
        tag=f"rq2c1_s{seed}_e{_ep_tag(episodes)}",
        env_overrides={"EXP_ALLOWED_VARIANTS": "aligned", "EXP_DISABLE_GENERATION": "1"},
        seed=seed, num_episodes=episodes, force=force,
    )


def rq1_heldout_run(seed=42, episodes=None, force=False,
                    pretrain_dir="trained_weights/darpa_ho",
                    train_apts=("APT28", "APT32", "APT38", "APT41", "FIN7", "Carbanak")):
    """
    RQ1 hold-out B1: co-evolution xuat phat tu baseline darpa_ho (da loai ho test),
    Generation chi sinh cac ho TRAIN (EXP_ALLOWED_APTS) -> ho test van UNSEEN xuyen suot.
    """
    abs_pre = pretrain_dir if os.path.isabs(pretrain_dir) else os.path.join(_REPO_ROOT, pretrain_dir)
    return run_marl(
        tag=f"rq1ho_s{seed}_e{_ep_tag(episodes)}",
        env_overrides={
            "EXP_PRETRAIN_DIR": abs_pre,
            "EXP_ALLOWED_APTS": ",".join(train_apts),
        },
        seed=seed, num_episodes=episodes, force=force,
    )


def classifier_run(name="mlp", seed=42, episodes=None, force=False):
    """RQ3: A0=mlp(==canonical), A1=linear (run #3), A3=deep (run #4)."""
    name = name.lower()
    if name == "mlp":
        return canonical_run(seed=seed, episodes=episodes, force=force)
    return run_marl(
        tag=f"rq3{name}_s{seed}_e{_ep_tag(episodes)}",
        env_overrides={"EXP_ALLOWED_VARIANTS": "aligned", "EXP_CLASSIFIER": name},
        seed=seed, num_episodes=episodes, force=force,
    )
