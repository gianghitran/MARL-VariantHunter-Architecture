"""
main.py
=======
Diem khoi chay cho MARL-VariantHunter (PPO Coordinator Training Loop).

Kien truc 3-agent cooperative:
  Agent 0 -> Generation Agent (RL-based Attack Agent) -- action = 0
  Agent 1 -> Pruning Agent (Graph Coarsening)          -- action = 1
  Agent 2 -> Detection Agent (GATs + MLP + IsolationForest) -- action = 2

PPO Coordinator dieu phoi 3 agents:
  - input : graph state representation (128-dim)
  - output: action in {0, 1, 2}

Reward closed-loop:
  - action 0: base_reward (RewardCalculator) + evasion_bonus (1 - confidence) + cl_evasion_bonus - penalty
  - action 1: compression_ratio x (1 - loss_causal)
  - action 2: precision + recall - 2*FN + scf_impact

Co-evolution Stopping (ImprovedStoppingCriterion):
  Thu tu uu tien:
    C. Failure/Degradation  -> dung som + canh bao
    A. Nash Equilibrium     -> Global_F1 >= 0.75 AND std(evasion)<0.04 AND FN_rate<0.20 (12 ep)
    B. Reward Plateau       -> std(reward)<0.10 AND no improvement (15 ep)
    D. Hard cap             -> episode >= num_episodes (150)

Continual Learning:
  - EWC + Prioritized Replay Buffer (Detection MLP)
  - Checkpoint sau moi episode neu dat ky luc moi
"""

import os
import sys
import datetime
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import statistics

from networks import CoordinatorNetwork
from environment import ProvenanceGraphEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)


# =============================================================================
# Improved Stopping Criterion  (thay the AdversarialConvergenceMonitor cu)
# =============================================================================

class ImprovedStoppingCriterion:
    """
    Co-evolution stopping criterion voi 4 tang uu tien:

    C. FAILURE / DEGRADATION (dung som, canh bao):
       - FN_rate > failure_fn_rate trong patience_failure episodes lien tiep
         -> Attack thong tri, Detection khong hoc duoc
       - Global_F1 < 0.05 trong 20 episodes (sau episode 20)
         -> Detection collapse hoan toan

    A. NASH EQUILIBRIUM (dung thanh cong - tieu chi chinh):
       - avg(Global_F1) >= f1_thresh     trong patience_nash episodes lien tiep
       - std(evasion_rate) < evasion_delta  trong cung window
       - avg(FN_rate) < fn_rate_thresh      trong cung window
       Global_F1 do tu evaluate_detection_mlp() (sklearn) -- nam trong [0,1]

    B. REWARD PLATEAU (dung trung tinh):
       - std(total_reward) < reward_delta     trong patience_plateau episodes
       - best_combined_score khong cai thien trong patience_plateau episodes

    D. HARD CAP:
       - episode >= num_episodes (xu ly o vong lap ngoai, khong qua day)
    """

    def __init__(
        self,
        patience_nash:    int   = 12,
        patience_plateau: int   = 15,
        patience_failure: int   = 15,
        f1_thresh:        float = 0.75,
        evasion_delta:    float = 0.04,
        fn_rate_thresh:   float = 0.20,
        failure_fn_rate:  float = 0.80,
        reward_delta:     float = 0.10,
    ):
        self.patience_nash    = patience_nash
        self.patience_plateau = patience_plateau
        self.patience_failure = patience_failure
        self.f1_thresh        = f1_thresh
        self.evasion_delta    = evasion_delta
        self.fn_rate_thresh   = fn_rate_thresh
        self.failure_fn_rate  = failure_fn_rate
        self.reward_delta     = reward_delta

        # Buffers voi maxlen = window lon nhat
        _win = max(patience_nash, patience_plateau, patience_failure)
        self.f1_buf      = deque(maxlen=_win)
        self.evasion_buf = deque(maxlen=_win)
        self.fn_buf      = deque(maxlen=_win)
        self.reward_buf  = deque(maxlen=_win)

        # Bests
        self.best_f1           = 0.0
        self.best_evasion_rate = 0.0

        # Counters cho Failure detection
        self._fn_high_count   = 0   # so ep lien tiep FN_rate cao
        self._collapse_count  = 0   # so ep lien tiep F1 rat thap

        # Counter cho Plateau detection
        self._no_improve_count = 0
        self._best_combined    = -float("inf")

        self.episode = 0

    def update(
        self,
        global_f1:    float,
        evasion_rate: float,
        fn_rate:      float,
        total_reward: float,
        combined_score: float,
    ) -> tuple:
        """
        Cap nhat sau moi episode.

        Args:
            global_f1:      F1 tu evaluate_detection_mlp() [0, 1]
            evasion_rate:   FN / det_calls [0, 1]
            fn_rate:        FN / (FN + TP) thuc su [0, 1]
            total_reward:   tong reward episode
            combined_score: avg_det + evasion_rate (dung cho best checkpoint)

        Returns:
            (stop_reason, stop_msg)
            stop_reason: "CONTINUE" | "CONVERGED" | "PLATEAU" | "FAILURE"
        """
        import statistics

        self.episode += 1
        safe_f1 = global_f1 if global_f1 is not None else 0.0

        # -- Update bests --
        if safe_f1 > self.best_f1:
            self.best_f1 = safe_f1
        if evasion_rate > self.best_evasion_rate:
            self.best_evasion_rate = evasion_rate

        # -- Append buffers --
        self.f1_buf.append(safe_f1)
        self.evasion_buf.append(evasion_rate)
        self.fn_buf.append(fn_rate)
        self.reward_buf.append(total_reward)

        # -- Plateau: track no-improvement --
        if combined_score > self._best_combined:
            self._best_combined    = combined_score
            self._no_improve_count = 0
        else:
            self._no_improve_count += 1

        # ─────────────────────────────────────────────────────────
        # C. FAILURE / DEGRADATION  (uu tien cao nhat)
        # ─────────────────────────────────────────────────────────
        # C-a: Attack thong tri lien tiep
        if fn_rate > self.failure_fn_rate:
            self._fn_high_count += 1
        else:
            self._fn_high_count = 0

        if self._fn_high_count >= self.patience_failure:
            return (
                "FAILURE",
                f"Attack dominated: FN_rate={fn_rate:.3f} > {self.failure_fn_rate} "
                f"for {self._fn_high_count} eps",
            )

        # C-b: Detection collapse
        if self.episode > 20 and safe_f1 < 0.05:
            self._collapse_count += 1
        else:
            self._collapse_count = 0

        if self._collapse_count >= 20:
            return (
                "FAILURE",
                f"Detection collapse: Global_F1={safe_f1:.4f} < 0.05 for 20 eps",
            )

        # ─────────────────────────────────────────────────────────
        # A. NASH EQUILIBRIUM
        # ─────────────────────────────────────────────────────────
        if len(self.f1_buf) >= self.patience_nash:
            window_f1  = list(self.f1_buf)[-self.patience_nash:]
            window_eva = list(self.evasion_buf)[-self.patience_nash:]
            window_fn  = list(self.fn_buf)[-self.patience_nash:]

            avg_f1  = statistics.mean(window_f1)
            std_eva = statistics.stdev(window_eva) if len(window_eva) > 1 else float("inf")
            avg_fn  = statistics.mean(window_fn)

            if (
                avg_f1  >= self.f1_thresh
                and std_eva < self.evasion_delta
                and avg_fn  < self.fn_rate_thresh
            ):
                return (
                    "CONVERGED",
                    f"Nash Equilibrium: avg_F1={avg_f1:.3f}>={self.f1_thresh} | "
                    f"EvaStd={std_eva:.3f}<{self.evasion_delta} | "
                    f"avg_FN={avg_fn:.3f}<{self.fn_rate_thresh} "
                    f"(window={self.patience_nash} eps)",
                )

        # ─────────────────────────────────────────────────────────
        # B. REWARD PLATEAU
        # ─────────────────────────────────────────────────────────
        if len(self.reward_buf) >= self.patience_plateau:
            window_rew = list(self.reward_buf)[-self.patience_plateau:]
            rew_std = statistics.stdev(window_rew) if len(window_rew) > 1 else float("inf")

            if (
                rew_std < self.reward_delta
                and self._no_improve_count >= self.patience_plateau
            ):
                return (
                    "PLATEAU",
                    f"Reward plateau: std={rew_std:.4f}<{self.reward_delta} | "
                    f"no improvement for {self._no_improve_count} eps",
                )

        return ("CONTINUE", "")

    def status_str(self, ep: int) -> str:
        """Chuoi trang thai ngan gon de in trong log."""
        f1_now  = self.f1_buf[-1]  if self.f1_buf  else 0.0
        eva_now = self.evasion_buf[-1] if self.evasion_buf else 0.0
        fn_now  = self.fn_buf[-1]  if self.fn_buf  else 0.0
        # std/avg tren cua so Nash — cho thay con cach Nash bao xa (gate chinh).
        win_eva = list(self.evasion_buf)[-self.patience_nash:]
        win_f1  = list(self.f1_buf)[-self.patience_nash:]
        eva_std = statistics.stdev(win_eva) if len(win_eva) > 1 else float("inf")
        avg_f1  = statistics.mean(win_f1) if win_f1 else 0.0
        return (
            f"avgF1={avg_f1:.3f}(>={self.f1_thresh}) | "
            f"EvaStd={eva_std:.3f}(<{self.evasion_delta}) | "
            f"FN_rate={fn_now:.3f}(<{self.fn_rate_thresh}) | "
            f"NoImprove={self._no_improve_count}/{self.patience_plateau} "
            f"[Nash {len(self.f1_buf)}/{self.patience_nash}]"
        )


# =============================================================================
# Main PPO Training Loop
# =============================================================================

class _TeeStream:
    """Ghi dong thoi ra console va file log."""
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()

    def isatty(self):
        return False


def run_marl_ppo():
    print("=" * 60)
    print("  MARL-VariantHunter: Initializing PPO Coordinator")
    print("=" * 60)

    base_dir  = os.path.dirname(os.path.abspath(__file__))
    gat_path  = os.path.join(base_dir, "trained_weights", "darpa", "gat.pth")
    w2v_path  = os.path.join(base_dir, "trained_weights", "darpa", "w2v.model")

    # ── EXP_SEED: dat seed toan cuc cho reproducibility (mac dinh: khong dat) ──
    _exp_seed = os.environ.get("EXP_SEED")
    if _exp_seed is not None:
        import random as _random
        import numpy as _np
        s = int(_exp_seed)
        _random.seed(s); _np.random.seed(s)
        torch.manual_seed(s); torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[Main] EXP_SEED set -> {s} (deterministic)")

    # ── Tao run_dir: EXP_RUN_DIR ghi de (cho experiment harness), neu khong dung timestamp ──
    _exp_run_dir = os.environ.get("EXP_RUN_DIR")
    if _exp_run_dir:
        run_dir = _exp_run_dir
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir   = os.path.join(base_dir, "runs", timestamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[Main] Run artifacts -> {run_dir}")

    # ── Redirect stdout/stderr vao run.log (giu nguyen console output) ─
    _log_file   = open(os.path.join(run_dir, "run.log"), "w", encoding="utf-8", buffering=1)
    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr
    sys.stdout   = _TeeStream(_orig_stdout, _log_file)
    sys.stderr   = _TeeStream(_orig_stderr, _log_file)

    # ── Khoi tao Environment va Coordinator ───────────────────────
    env         = ProvenanceGraphEnv(gcn_model_path=gat_path, w2v_model_path=w2v_path, run_dir=run_dir)
    coordinator = CoordinatorNetwork(state_dim=128, num_actions=3).to(device)

    # Load checkpoint neu co
    ckpt_path = os.path.join(base_dir, "ppo_coordinator.pth")
    if os.path.exists(ckpt_path):
        try:
            coordinator.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"[Main] Loaded PPO Coordinator checkpoint from {ckpt_path}")
        except Exception as e:
            print(f"[Main] Could not load checkpoint ({e}). Starting fresh.")

    optimizer = optim.Adam(coordinator.parameters(), lr=1e-3)

    # ── PPO Hyperparameters ────────────────────────────────────────
    gamma            = 0.99
    clip_epsilon     = 0.2
    entropy_coeff    = 0.01
    value_coeff      = 0.5
    num_episodes     = 150   # Hard cap (tang tu 100 -> 150 theo proposal)
    # EXP_NUM_EPISODES: ghi de hard-cap episodes (smoke run cho experiment harness)
    if os.environ.get("EXP_NUM_EPISODES"):
        num_episodes = int(os.environ["EXP_NUM_EPISODES"])
        print(f"[Main] EXP_NUM_EPISODES override -> {num_episodes} episodes")
    ppo_epochs       = 4      # Epochs moi episode
    max_steps        = 15     # Buoc moi episode
    min_det_per_ep   = 2      # Bat buoc goi Detection it nhat 2 lan/episode

    ACTION_NAMES = {0: "Generate", 1: "Prune   ", 2: "Detect  "}

    # ── Improved Stopping Criterion ─────────────────────────────────
    stopping = ImprovedStoppingCriterion(
        patience_nash    = 12,
        patience_plateau = 15,
        patience_failure = 15,
        f1_thresh        = 0.75,
        evasion_delta    = 0.04,
        fn_rate_thresh   = 0.20,
        failure_fn_rate  = 0.80,
        reward_delta     = 0.10,
    )

    # ── Best checkpoint tracking ────────────────────────────────────
    best_combined_score = -float("inf")

    # ── Training loop ──────────────────────────────────────────────
    for episode in range(num_episodes):
        import time as _time
        ep_start = _time.time()

        state = env.reset()
        done  = False
        step  = 0

        log_probs    = []
        values       = []
        rewards      = []
        states       = []
        actions_buf  = []
        fn_count     = 0      # Dem so FN trong episode
        tp_count     = 0      # Dem so TP trong episode (cho FN_rate)
        det_calls    = 0      # Dem so lan goi Detection
        fn_rate_win  = getattr(env, "last_fn_rate_win", 0.0)  # FN_rate windowed (StopMonitor)

        episode_confidence_scores = []
        episode_det_scores        = []

        print(f"\n{'-'*60}")
        print(f"[Episode {episode+1:03d}/{num_episodes}] START")
        print(f"{'-'*60}")

        # ── Trajectory Collection ──────────────────────────────────
        while not done and step < max_steps:
            step += 1
            state = state.to(device)
            logits, value = coordinator(state)

            probs  = F.softmax(logits, dim=-1)
            m      = Categorical(probs)

            # Nguyen tac buoc Detection:
            # 1. Neu dang o step giua (step == max_steps//2) va chua co Det nao -> Force Detect
            # 2. Neu sap het episode (steps_left < deficit) -> Force Detect
            steps_left = max_steps - step
            det_deficit = min_det_per_ep - det_calls
            mid_step   = max_steps // 2

            force_detect = (
                (det_calls == 0 and step == mid_step) or            # Chua co Det nao den giua ep
                (det_deficit > 0 and steps_left < det_deficit)       # Sap het, thieu Det
            )

            # Fix 4: Drift -> trigger Generation. Neu buoc Detection truoc do phat hien
            # drift (Wasserstein > tau), buoc nay uu tien sinh variant moi (tru khi
            # dang phai force Detect). Hien thuc hoa "Drift -> Generation" cua Gap 1.
            force_generate = getattr(env, "drift_triggered", False) and not force_detect
            # EXP_DISABLE_GENERATION (RQ2 C0/C1): khong bao gio ep Generation (Action 0 da
            # la no-op trong env, day chi tranh log DRIFT-TRIGGER vo nghia).
            if os.environ.get("EXP_DISABLE_GENERATION"):
                force_generate = False

            if force_detect:
                action = torch.tensor(2, device=device)   # Force Detection
            elif force_generate:
                action = torch.tensor(0, device=device)   # Drift-triggered Generation
                print(f"  [DRIFT-TRIGGER] drift={getattr(env,'last_drift',0.0):.4f} "
                      f"> tau={env.drift_tau:.3f} -> force Generation")
                env.drift_triggered = False
            else:
                action = m.sample()


            next_state, reward, done, info = env.step(action.item())

            conf = info.get("confidence_score", info.get("mlp_attack_prob", 0.0))
            episode_confidence_scores.append(conf)

            # Thu thap det_score neu la buoc Detection
            if action.item() == 2:
                det_calls += 1
                fn_count  += info.get("FN", 0)
                tp_count  += info.get("TP", 0)
                # FN_rate WINDOWED (1 - recall tren cua so 20 buoc) — tin hieu on dinh
                # cho StopMonitor (per-episode fn_rate voi 1-2 call qua nhieu cho Nash).
                fn_rate_win = info.get("fn_rate_win", fn_rate_win)
                det_score  = info.get("precision", 0.0) + info.get("recall", 0.0) \
                             - 2.0 * info.get("FN", 0) + info.get("scf_impact", 0.0)
                episode_det_scores.append(det_score)

            log_probs.append(m.log_prob(torch.tensor(action.item(), device=device)))
            values.append(value)
            rewards.append(reward)
            states.append(state)
            actions_buf.append(action)

            # ── Step-level log ─────────────────────────────────────
            action_name = ACTION_NAMES.get(action.item(), "?")
            print(
                f"  [Ep {episode+1:03d} | Step {step:02d}/{max_steps}] "
                f"Action={action_name} | "
                f"Reward={reward:+.4f} | "
                f"Conf={conf:.3f} | "
                f"Drift={info.get('drift', 0.0):.3f} | "
                f"Compress={info.get('compression_ratio', 0.0):.0%} | "
                f"DetCalls={det_calls} | "
                f"GraphEdges={len(env.current_graph_df)}"
            )

            state = next_state

        # ── Compute Returns & Advantages ──────────────────────────
        returns = []
        R       = 0.0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)

        returns    = torch.tensor(returns, dtype=torch.float32).to(device)
        values_t   = torch.cat(values).squeeze(-1)
        advantages = returns - values_t.detach()

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages - advantages.mean()

        old_log_probs  = torch.stack(log_probs).detach()
        states_tensor  = torch.stack(states).detach()
        actions_tensor = torch.stack(actions_buf).detach()

        # ── PPO Update ─────────────────────────────────────────────
        for _ in range(ppo_epochs):
            logits_new, values_new = coordinator(states_tensor)
            probs_new  = F.softmax(logits_new, dim=-1)
            m_new      = Categorical(probs_new)
            new_log_probs = m_new.log_prob(actions_tensor)

            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1        = ratio * advantages
            surr2        = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
            policy_loss  = -torch.min(surr1, surr2).mean()
            value_loss   = F.mse_loss(values_new.squeeze(-1), returns)
            entropy_loss = m_new.entropy().mean()

            loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(coordinator.parameters(), max_norm=0.5)
            optimizer.step()

        # ── EWC Continual Learning -- Offline Batch Update ─────────
        # EXP_DISABLE_EWC (RQ2 C0): bo qua cap nhat continual learning -> detector dung yen
        # o trong so pretrained (chi danh gia, khong hoc). Van goi evaluate de lay global_f1.
        if os.environ.get("EXP_DISABLE_EWC"):
            global_f1 = env.evaluate_detection_mlp(run_dir=run_dir)
        else:
            global_f1 = env.train_detection_agent_ewc(run_dir=run_dir, batch_size=8)
        if global_f1 is None:
            global_f1 = env.best_det_score

        # ── Episode Metrics ────────────────────────────────────────
        avg_conf   = sum(episode_confidence_scores) / len(episode_confidence_scores) \
                     if episode_confidence_scores else 0.0
        avg_det    = sum(episode_det_scores) / len(episode_det_scores) \
                     if episode_det_scores else 0.0
        # evasion_rate = ty le buoc Detection bi FN trong tong so lan Detection duoc goi
        evasion_rate = fn_count / det_calls if det_calls > 0 else 0.0
        # fn_rate = ty le FN thuc su (FN / (FN + TP))
        fn_rate  = fn_count / (fn_count + tp_count) if (fn_count + tp_count) > 0 else 0.0
        total_rew  = sum(rewards)

        # ── Best Checkpoint (luu vao run_dir) ─────────────────────
        combined_score = avg_det + evasion_rate
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_ckpt_path = os.path.join(run_dir, "ppo_coordinator_best.pth")
            torch.save(coordinator.state_dict(), best_ckpt_path)
            print(f"[Checkpoint] New best combined score={combined_score:.4f} -> {best_ckpt_path}")

        # ── MAB stats ──────────────────────────────────────────────
        mab_str = " | ".join([
            f"Arm{k}(u={env.mab.alpha[k]/(env.mab.alpha[k]+env.mab.beta[k]):.2f})"
            for k in range(env.mab.n_arms)
        ])

        # ── Closed-loop summary ────────────────────────────────────
        cl = getattr(env, "last_closed_loop_reward", {})

        # ── Update Stopping Criteria ───────────────────────────────
        # Nash dung evasion/FN WINDOWED (cua so 20 buoc detection) thay vi per-episode:
        # per-episode (1-2 call) chi nhan {0,0.5,1.0} -> std qua nhieu -> Nash khong bao
        # gio trigger. Windowed -> std co nghia, Nash do duoc hoi tu that.
        stop_reason, stop_msg = stopping.update(global_f1, fn_rate_win, fn_rate_win, total_rew, combined_score)

        ep_elapsed = _time.time() - ep_start

        # ── Episode Summary Log ────────────────────────────────────
        print(f"\n{'='*60}")
        print(
            f"[Episode {episode+1:03d}/{num_episodes}] DONE "
            f"({ep_elapsed:.1f}s | {step} steps)"
        )
        print(f"  Total Reward   : {total_rew:.4f}")
        print(f"  Avg Confidence : {avg_conf:.4f}")
        print(f"  Actions        : {[ACTION_NAMES.get(a.item(),a.item()) for a in actions_buf]}")
        print(f"  Det Calls/FN   : {det_calls} calls | FN={fn_count} TP={tp_count} | "
              f"EvasionRate={evasion_rate:.3f} | FN_rate={fn_rate:.3f}")
        print(f"  Avg Det Score  : {avg_det:.4f}  (precision+recall-2*FN+scf)")
        print(f"  Global F1      : {global_f1:.4f}" if global_f1 is not None else "  Global F1      : N/A")
        print(f"  Graph Size     : {len(env.current_graph_df)} edges")
        print(f"  Best MLP Score : {env.best_det_score:.4f}")
        print(f"  Replay Buffer  : {len(env.replay_buffer)} samples")
        print(f"  MAB Stats      : {mab_str}")
        if cl:
            print(
                f"  Closed-loop    : total={cl.get('closed_loop_total', 0):.4f} | "
                f"evasion={cl.get('detector_evasion', 0):.4f} | "
                f"fn_bonus={cl.get('hard_sample_bonus', 0):.4f}"
            )
        print(f"  StopMonitor    : {stopping.status_str(episode+1)}")
        print(f"  StopReason     : {stop_reason}" + (f" | {stop_msg}" if stop_msg else ""))
        print("=" * 60)

        # ── Kiem tra dieu kien dung ────────────────────────────────
        if stop_reason == "CONVERGED":
            print(f"\n[STOP-CONVERGED] He thong dat Nash Equilibrium sau {episode+1} episodes!")
            print(f"[STOP-CONVERGED] {stop_msg}")
            print(f"[STOP-CONVERGED] Best Global F1    = {stopping.best_f1:.4f}")
            print(f"[STOP-CONVERGED] Best Evasion Rate = {stopping.best_evasion_rate:.4f}")
            break
        elif stop_reason == "PLATEAU":
            print(f"\n[STOP-PLATEAU] Reward plateau sau {episode+1} episodes.")
            print(f"[STOP-PLATEAU] {stop_msg}")
            print(f"[STOP-PLATEAU] Best Global F1 = {stopping.best_f1:.4f}")
            break
        elif stop_reason == "FAILURE":
            print(f"\n[STOP-FAILURE] He thong suy thoai sau {episode+1} episodes!")
            print(f"[STOP-FAILURE] {stop_msg}")
            print("[STOP-FAILURE] Nen kiem tra lai hyperparameters hoac du lieu.")
            break


    # ── Save detection head CUOI CUNG (luon luon) ─────────────────
    # best_mlp.pth chi luu khi Global F1 cai thien; neu run khong cai thien (vd RQ3
    # classifier moi train tu dau / collapse) thi van can head dung kien truc de danh
    # gia -> final_mlp.pth bao dam luon co artifact.
    final_mlp_path = os.path.join(run_dir, "final_mlp.pth")
    torch.save(env.detection_mlp.state_dict(), final_mlp_path)
    # Luu CA GAT cuoi cung (co-evolution train GAT chung head) de eval dung cap GAT+head.
    torch.save(env.gat.state_dict(), os.path.join(run_dir, "final_gat.pth"))
    print(f"[Main] Final Detector (GAT+MLP) saved -> {run_dir}")

    # ── Save PPO Coordinator final (vao run_dir) ──────────────────
    save_path = os.path.join(run_dir, "ppo_coordinator_final.pth")
    torch.save(coordinator.state_dict(), save_path)
    print(f"\n[Main] Training complete. PPO Coordinator saved -> {save_path}")
    print(f"[Main] Best checkpoint -> {os.path.join(run_dir, 'ppo_coordinator_best.pth')}")
    print(f"[Main] Best combined score = {best_combined_score:.4f}")
    print(f"[Main] All artifacts (log, checkpoints, graphs, CSV) -> {run_dir}")

    # ── Restore stdout/stderr va dong log file ─────────────────────
    sys.stdout = _orig_stdout
    sys.stderr = _orig_stderr
    _log_file.close()
    print(f"[Main] Log saved -> {os.path.join(run_dir, 'run.log')}")


if __name__ == "__main__":
    run_marl_ppo()
