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

Co-evolution Convergence (Nash Equilibrium Proxy):
  - Convergence khi ca 2 agent dat dinh:
      1. Det convergence: avg_det_score >= det_converge_thresh trong patience episodes
      2. Attack convergence: evasion_rate on dinh (|evasion_rate - prev| < 0.05) trong patience episodes
  - Training chay toi da num_episodes episodes neu convergence chua dat duoc

Continual Learning:
  - EWC + Prioritized Replay Buffer (Detection MLP)
  - Checkpoint sau moi episode neu dat ky luc moi
"""

import os
import datetime
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque

from networks import CoordinatorNetwork
from environment import ProvenanceGraphEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)


# =============================================================================
# Adversarial Convergence Monitor
# =============================================================================

class AdversarialConvergenceMonitor:
    """
    Giam sat hoi tu cua he thong MARL Adversarial.

    Co-evolution convergence dat khi DONG THOI thoa:
    1. DETECTION CONVERGENCE:
       - avg_det_score (precision + recall - lambda*FN) on dinh >= det_thresh
         trong 'patience' episodes lien tiep
       Ngan nghia: Detection da hoc tot, FN -> 0

    2. ATTACK EQUILIBRIUM:
       - evasion_rate (ty le FN tren tong cuoc tan cong) on dinh
         (bien dong < evasion_delta trong 'patience' episodes)
       Ngan nghia: Attack Agent khong the cai thien them duoc

    3. REWARD PLATEAU:
       - Total reward dao dong it (std < reward_delta) trong 'patience' episodes
       Ngan nghia: PPO Policy da on dinh
    """

    def __init__(
        self,
        patience: int = 10,
        det_thresh: float = 0.7,
        evasion_delta: float = 0.05,
        reward_delta: float = 0.1,
    ):
        self.patience      = patience
        self.det_thresh    = det_thresh
        self.evasion_delta = evasion_delta
        self.reward_delta  = reward_delta

        self.det_scores    = deque(maxlen=patience)
        self.evasion_rates = deque(maxlen=patience)
        self.total_rewards = deque(maxlen=patience)

        self.best_det_score    = 0.0
        self.best_evasion_rate = 0.0
        self.converged         = False
        self.converge_reason   = ""

    def update(self, det_score: float, evasion_rate: float, total_reward: float):
        """Cap nhat sau moi episode va kiem tra convergence."""
        self.det_scores.append(det_score)
        self.evasion_rates.append(evasion_rate)
        self.total_rewards.append(total_reward)

        if det_score > self.best_det_score:
            self.best_det_score = det_score
        if evasion_rate > self.best_evasion_rate:
            self.best_evasion_rate = evasion_rate

        if len(self.det_scores) < self.patience:
            return False  # Chua du du lieu de danh gia

        # --- Dieu kien 1: Detection Convergence ---
        avg_det = sum(self.det_scores) / len(self.det_scores)
        det_converged = avg_det >= self.det_thresh

        # --- Dieu kien 2: Attack Equilibrium ---
        import statistics
        if len(self.evasion_rates) >= 2:
            evasion_std = statistics.stdev(self.evasion_rates)
            attack_equilibrium = evasion_std < self.evasion_delta
        else:
            attack_equilibrium = False

        # --- Dieu kien 3: Reward Plateau ---
        if len(self.total_rewards) >= 2:
            reward_std = statistics.stdev(self.total_rewards)
            reward_plateau = reward_std < self.reward_delta
        else:
            reward_plateau = False

        if det_converged and attack_equilibrium:
            self.converged = True
            self.converge_reason = (
                f"Nash Equilibrium: Det={avg_det:.3f}>={self.det_thresh} | "
                f"EvasionStd={evasion_std:.3f}<{self.evasion_delta}"
            )
            return True

        if reward_plateau and det_converged:
            self.converged = True
            self.converge_reason = (
                f"Reward Plateau: Det={avg_det:.3f} | RewardStd={reward_std:.3f}<{self.reward_delta}"
            )
            return True

        return False

    def status_str(self) -> str:
        """Chuoi trang thai de in ra log moi episode."""
        if len(self.det_scores) == 0:
            return "Chua co du lieu"
        avg_det  = sum(self.det_scores) / len(self.det_scores)
        avg_eva  = sum(self.evasion_rates) / len(self.evasion_rates)
        buf_len  = len(self.det_scores)
        return (
            f"Det={avg_det:.3f} (best={self.best_det_score:.3f}) | "
            f"EvasionRate={avg_eva:.3f} (best={self.best_evasion_rate:.3f}) | "
            f"Window={buf_len}/{self.patience}"
        )


# =============================================================================
# Main PPO Training Loop
# =============================================================================

def run_marl_ppo():
    print("=" * 60)
    print("  MARL-VariantHunter: Initializing PPO Coordinator")
    print("=" * 60)

    base_dir  = os.path.dirname(os.path.abspath(__file__))
    gat_path  = os.path.join(base_dir, "trained_weights", "unicorn", "unicorn0.pth")
    w2v_path  = os.path.join(base_dir, "trained_weights", "unicorn", "unicorn.model")

    # Tao run_dir theo timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = os.path.join(base_dir, "runs", timestamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[Main] Run artifacts -> {run_dir}")

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
    num_episodes     = 100    # Toi da 100 episodes (se dung truoc neu convergence)
    ppo_epochs       = 4      # Epochs moi episode
    max_steps        = 15     # Buoc moi episode

    # ── Adversarial Convergence Monitor ─────────────────────────────────
    # Det score threshold = 0.7 (precision+recall dat 70%),
    # Evasion delta = 0.05 (evasion rate on dinh trong +-5%),
    # Patience = 8 episodes (phai on dinh lien tiep moi xem la hoi tu)
    conv_monitor = AdversarialConvergenceMonitor(
        patience      = 8,
        det_thresh    = 0.7,
        evasion_delta = 0.05,
        reward_delta  = 0.15,
    )

    # ── Best checkpoint tracking ────────────────────────────────────
    best_combined_score = -float("inf")

    # ── Training loop ──────────────────────────────────────────────
    for episode in range(num_episodes):
        state = env.reset()
        done  = False
        step  = 0

        log_probs    = []
        values       = []
        rewards      = []
        states       = []
        actions_buf  = []
        fn_count     = 0      # Dem so FN trong episode
        det_calls    = 0      # Dem so lan goi Detection

        episode_confidence_scores = []
        episode_det_scores        = []

        # ── Trajectory Collection ──────────────────────────────────
        while not done and step < max_steps:
            step += 1
            state = state.to(device)
            logits, value = coordinator(state)

            probs  = F.softmax(logits, dim=-1)
            m      = Categorical(probs)
            action = m.sample()

            next_state, reward, done, info = env.step(action.item())

            conf = info.get("confidence_score", info.get("mlp_attack_prob", 0.0))
            episode_confidence_scores.append(conf)

            # Thu thap det_score neu la buoc Detection
            if action.item() == 2:
                det_calls += 1
                fn_count  += info.get("FN", 0)
                det_score  = info.get("precision", 0.0) + info.get("recall", 0.0) \
                             - 2.0 * info.get("FN", 0) + info.get("scf_impact", 0.0)
                episode_det_scores.append(det_score)

            log_probs.append(m.log_prob(action))
            values.append(value)
            rewards.append(reward)
            states.append(state)
            actions_buf.append(action)

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
        env.train_detection_agent_ewc(run_dir=run_dir, batch_size=8)

        # ── Episode Metrics ────────────────────────────────────────
        avg_conf   = sum(episode_confidence_scores) / len(episode_confidence_scores) \
                     if episode_confidence_scores else 0.0
        avg_det    = sum(episode_det_scores) / len(episode_det_scores) \
                     if episode_det_scores else 0.0
        # evasion_rate = ty le buoc Detection bi FN trong tong so lan Detection duoc goi
        evasion_rate = fn_count / det_calls if det_calls > 0 else 0.0
        total_rew  = sum(rewards)

        # ── Best Checkpoint ────────────────────────────────────────
        # Combined = det_score + evasion_rate (ca 2 agent tot = he thong co-evolve manh)
        combined_score = avg_det + evasion_rate
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_ckpt_path = os.path.join(base_dir, "ppo_coordinator_best.pth")
            torch.save(coordinator.state_dict(), best_ckpt_path)
            print(f"[Checkpoint] New best combined score={combined_score:.4f} -> {best_ckpt_path}")

        # ── MAB stats ──────────────────────────────────────────────
        mab_str = " | ".join([
            f"Arm{k}(u={env.mab.alpha[k]/(env.mab.alpha[k]+env.mab.beta[k]):.2f})"
            for k in range(env.mab.n_arms)
        ])

        # ── Closed-loop summary ────────────────────────────────────
        cl = getattr(env, "last_closed_loop_reward", {})

        # ── Convergence Monitor update ─────────────────────────────
        converged = conv_monitor.update(avg_det, evasion_rate, total_rew)

        # ── Episode Log ────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"Episode {episode+1}/{num_episodes}")
        print(f"  Total Reward   : {total_rew:.4f}")
        print(f"  Avg Confidence : {avg_conf:.4f}")
        print(f"  Actions        : {[a.item() for a in actions_buf]}")
        print(f"  Det Calls/FN   : {det_calls} calls | FN={fn_count} | EvasionRate={evasion_rate:.2f}")
        print(f"  Avg Det Score  : {avg_det:.4f}  (precision+recall-2*FN+scf)")
        print(f"  Graph Size     : {len(env.current_graph_df)} edges")
        print(f"  Best Score     : {env.best_det_score:.4f}")
        print(f"  Replay Buffer  : {len(env.replay_buffer)} samples")
        print(f"  MAB Stats      : {mab_str}")
        if cl:
            print(
                f"  Closed-loop    : total={cl.get('closed_loop_total', 0):.4f} | "
                f"evasion={cl.get('detector_evasion', 0):.4f} | "
                f"fn_bonus={cl.get('hard_sample_bonus', 0):.4f}"
            )
        print(f"  Convergence    : {conv_monitor.status_str()}")
        print("=" * 60)

        # ── Kiem tra convergence ───────────────────────────────────
        if converged:
            print(f"\n[Convergence] He thong dat Nash Equilibrium sau {episode+1} episodes!")
            print(f"[Convergence] Ly do: {conv_monitor.converge_reason}")
            print(f"[Convergence] Best Det Score = {conv_monitor.best_det_score:.4f}")
            print(f"[Convergence] Best Evasion Rate = {conv_monitor.best_evasion_rate:.4f}")
            break

    # ── Save PPO Coordinator (final) ──────────────────────────────
    save_path = os.path.join(base_dir, "ppo_coordinator.pth")
    torch.save(coordinator.state_dict(), save_path)
    print(f"\n[Main] Training complete. PPO Coordinator saved -> {save_path}")
    print(f"[Main] Best checkpoint -> {os.path.join(base_dir, 'ppo_coordinator_best.pth')}")
    print(f"[Main] Best combined score = {best_combined_score:.4f}")


if __name__ == "__main__":
    run_marl_ppo()
