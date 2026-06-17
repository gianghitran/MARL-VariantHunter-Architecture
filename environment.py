"""
environment.py
==============
Môi trường RL (ProvenanceGraphEnv) cho MARL-VariantHunter.

Pipeline 3-agent:
  Action 0 → Generation Agent (Attack Agent) — sinh APT provenance subgraph
  Action 1 → Pruning Agent — nén đồ thị giữ quan hệ nhân quả
  Action 2 → Detection Agent (GATs-based) — phát hiện + tạo feedback

Feedback Closed-loop:
  Detection Agent → Confidence Score → PPO Reward cho Coordinator
  Detection Agent → FN/Gradient info → Prioritized Replay Buffer
  Coordinator     → episode reward signal → Attack Agent evasion pressure

Tất cả LLM code đã bị gỡ bỏ.
"""

import os
import json
import zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from gensim.models import Word2Vec
from sklearn.ensemble import IsolationForest

from networks import GAT, DetectionMLP, PositionalEncoder
import sys

from env_utils import calculate_scf_impact, visualize_current_graph, save_current_graph_to_txt
from graph_utils import prepare_graph, infer

# Thêm đường dẫn để import PruningAgent
base_dir = os.path.dirname(os.path.abspath(__file__))
pruning_dir = os.path.join(base_dir, "Pruning_Agent")
if pruning_dir not in sys.path:
    sys.path.append(pruning_dir)
try:
    from pruning_agent import PruningAgent
except ImportError:
    print("Warning: Could not import PruningAgent. Fallback to basic pruning.")
    PruningAgent = None

# Import Attack Agent generation interface
attack_agent_dir = os.path.join(base_dir, "Attack_Agent")
if attack_agent_dir not in sys.path:
    sys.path.insert(0, attack_agent_dir)
try:
    from generation_interface import generate_apt_subgraph
    _ATTACK_AGENT_AVAILABLE = True
    print("[Env] Attack Agent (generation_interface) loaded successfully.")
except ImportError as e:
    print(f"[Env] WARNING: Cannot import Attack Agent: {e}. Using dummy fallback.")
    _ATTACK_AGENT_AVAILABLE = False

# Import marl_integration.detection_feedback (từ Attack_Agent)
try:
    from marl_integration.detection_feedback import compute_closed_loop_reward
    _CLOSED_LOOP_AVAILABLE = True
except ImportError:
    _CLOSED_LOOP_AVAILABLE = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Thompson Sampling MAB — chọn sampling rate cho Detection Agent
# ─────────────────────────────────────────────────────────────────────────────

class ThompsonSamplingMAB:
    """
    Multi-Armed Bandit dùng Thompson Sampling để chọn tỷ lệ lấy mẫu cạnh tối ưu
    cho Detection Agent trên streaming provenance data.

    3 Arms tương ứng 3 mức sampling:
        Arm 0: 60%  — tiết kiệm tài nguyên
        Arm 1: 80%  — cân bằng (default)
        Arm 2: 95%  — toàn diện nhất
    """

    SAMPLING_RATES = [0.60, 0.80, 0.95]

    def __init__(self):
        self.n_arms   = len(self.SAMPLING_RATES)
        self.alpha    = np.ones(self.n_arms, dtype=np.float64)   # successes
        self.beta     = np.ones(self.n_arms, dtype=np.float64)   # failures
        self.last_arm = 1   # Arm 1 (80%) mặc định

    def select_arm(self) -> int:
        samples       = np.random.beta(self.alpha, self.beta)
        self.last_arm = int(np.argmax(samples))
        return self.last_arm

    def select_indices(self, total_edges: int) -> np.ndarray:
        arm  = self.select_arm()
        rate = self.SAMPLING_RATES[arm]
        num  = max(1, int(total_edges * rate))
        print(f"[MAB] Arm {arm} selected (rate={rate:.0%}, θ~Beta({self.alpha[arm]:.1f},{self.beta[arm]:.1f}))")
        return np.random.choice(total_edges, num, replace=False)

    def update(self, arm: int, reward_positive: bool):
        if reward_positive:
            self.alpha[arm] += 1.0
        else:
            self.beta[arm] += 1.0
        print(f"[MAB] Updated Arm {arm}: α={self.alpha[arm]:.1f}, β={self.beta[arm]:.1f}")

    def get_stats(self) -> dict:
        stats = {}
        for k in range(self.n_arms):
            stats[f"arm{k}_rate"]  = self.SAMPLING_RATES[k]
            stats[f"arm{k}_alpha"] = float(self.alpha[k])
            stats[f"arm{k}_beta"]  = float(self.beta[k])
            stats[f"arm{k}_mean"]  = float(self.alpha[k] / (self.alpha[k] + self.beta[k]))
        return stats


# ─────────────────────────────────────────────────────────────────────────────
# Prioritized Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────

class PrioritizedReplayBuffer:
    """
    Buffer ưu tiên lưu trữ các subgraph gây ra False Negative (FN=1).

    Priority Score = boundary_dist = 1 − |prob − 0.5| × 2
    (Mẫu càng gần ranh giới 0.5 → thông tin càng cao → ưu tiên cao hơn)

    Dùng để thực hiện Offline Batch Update cho DetectionMLP qua EWC.
    """

    def __init__(self, maxlen: int = 500):
        self.maxlen   = maxlen
        self.buffer   = []   # list of (−priority, counter, sample_dict)
        self._counter = 0

    def add(self, sample: dict, priority: float):
        entry = (-priority, self._counter, sample)
        self._counter += 1
        self.buffer.append(entry)
        if len(self.buffer) > self.maxlen:
            self.buffer.sort(key=lambda x: x[0])
            self.buffer = self.buffer[: self.maxlen]

    def sample(self, batch_size: int) -> list:
        if not self.buffer:
            return []
        priorities = np.array([-e[0] for e in self.buffer], dtype=np.float64)
        priorities = priorities / (priorities.sum() + 1e-9)
        n = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), size=n, replace=False, p=priorities)
        return [self.buffer[i][2] for i in indices]

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────────────
# Môi trường MARL chính
# ─────────────────────────────────────────────────────────────────────────────

class ProvenanceGraphEnv:
    """
    Môi trường RL ba agent:
      - Action 0: Generation Agent (Attack Agent RL-based)
      - Action 1: Pruning Agent (GraphSAGE + Metis-style coarsening)
      - Action 2: Detection Agent (GATs + IsolationForest + MLP)

    Feedback closed-loop:
      Detection Confidence Score → PPO Reward cho Action 0 (Attack Agent pressure)
    """

    def __init__(self, gcn_model_path: str, w2v_model_path: str, run_dir: str = None):
        self.state_dim          = 128
        self.lambda_fn_penalty  = 2.0
        self.run_dir            = run_dir

        # ── Load pre-trained GAT (Detection Agent backbone) ──────────
        self.gat = GAT(in_channels=30, out_channels=20, hidden_dim=64, heads=8, dropout=0.3).to(device)
        # Load pre-trained GAT weights (partial, strict=False)
        if os.path.exists(gcn_model_path):
            try:
                state = torch.load(gcn_model_path, map_location=device)
                self.gat.load_state_dict(state, strict=False)
                print(f"[Env] Loaded partial GAT weights from {gcn_model_path} (strict=False).")
            except Exception as e:
                print(f"[Env] Could not load GAT weights: {e}")
        self.gat.eval()

        # ── Load pre-trained Detection MLP ───────────────────────────
        self.detection_mlp = DetectionMLP(input_dim=20, hidden_dim=32).to(device)

        # Thu tu uu tien load:
        # 1. best_mlp.pth (global best tu tat ca cac lan chay truoc)
        # 2. trained_weights/unicorn/mlp.pth (pre-trained goc)
        # 3. Random init
        global_best_mlp  = os.path.join(base_dir, "best_mlp.pth")
        original_mlp_path = os.path.join(base_dir, "trained_weights", "unicorn", "mlp.pth")

        if os.path.exists(global_best_mlp):
            try:
                self.detection_mlp.load_state_dict(
                    torch.load(global_best_mlp, map_location=device), strict=False
                )
                print(f"[Env] Loaded global best DetectionMLP from {global_best_mlp}")
            except Exception as e:
                print(f"[Env] Could not load best_mlp.pth ({e}). Trying original weights.")
                if os.path.exists(original_mlp_path):
                    self.detection_mlp.load_state_dict(
                        torch.load(original_mlp_path, map_location=device), strict=False
                    )
                    print(f"[Env] Loaded original MLP weights from {original_mlp_path}")
        elif os.path.exists(original_mlp_path):
            self.detection_mlp.load_state_dict(
                torch.load(original_mlp_path, map_location=device), strict=False
            )
            print(f"[Env] Loaded original MLP weights from {original_mlp_path}")
        else:
            print("[Env] No pre-trained DetectionMLP found. Starting from random init.")

        self.detection_mlp.eval()

        self.mab           = ThompsonSamplingMAB()
        self.replay_buffer = PrioritizedReplayBuffer(maxlen=500)

        self.best_det_score       = 0.0
        self.det_optimizer        = torch.optim.Adam(self.detection_mlp.parameters(), lr=1e-4)
        self._ewc_initialized     = False

        self.encoder = PositionalEncoder(30)
        try:
            self.w2vmodel = Word2Vec.load(w2v_model_path)
        except Exception:
            print("[Env] W2V model not found. Using random embeddings.")
            self.w2vmodel = None

        # Kiểm tra dữ liệu base graph
        base_graph_path = "unicorn/0.txt"
        if os.path.exists("unicorn.zip"):
            with zipfile.ZipFile("unicorn.zip", "r") as zip_ref:
                zip_ref.extractall("unicorn")

        if self.w2vmodel is None or not os.path.exists(base_graph_path):
            raise FileNotFoundError(
                "CRITICAL ERROR: Không tìm thấy W2V model hoặc 'unicorn/0.txt'. "
                "Vui lòng cung cấp dữ liệu thật để train."
            )

        # Trang thai tracking
        self.current_graph_df    = None
        self.last_mlp_prob       = 0.0
        self.last_precision      = 0.0
        self.last_recall         = 0.0
        self.last_FN             = 0
        self.last_scf_impact     = 0.0
        self.last_gen_reward     = {}    # reward dict tu Attack Agent pipeline
        self.last_confidence_score = 0.0
        self.accumulated_evasion_bonus = 0.0
        self.failed_evasion_penalty = 0.0

        # Sliding window de tinh Precision/Recall tich luy co y nghia thong ke
        # (cua so 20 buoc Detection gan nhat)
        from collections import deque
        self._window_TP = deque(maxlen=20)
        self._window_FP = deque(maxlen=20)
        self._window_FN = deque(maxlen=20)
        self._total_det_steps = 0   # Tong so lan Detection duoc goi

        self.reset()

    # ─────────────────────────────────────────────────────────────────
    # Reset
    # ─────────────────────────────────────────────────────────────────

    def reset(self):
        """Load đồ thị base từ dataset thật, trả về initial state."""
        base_graph_path = "2.txt"
        if os.path.exists(base_graph_path):
            try:
                with open(base_graph_path, "r") as f:
                    lines = [l.strip() for l in f if l.strip()]
                num_nodes  = int(lines[0])
                nodes      = lines[1 : num_nodes + 1]
                num_edges  = int(lines[num_nodes + 1])
                edges      = lines[num_nodes + 2 : num_nodes + 2 + num_edges]
                parsed     = []
                for edge in edges:
                    parts = edge.split()
                    if len(parts) >= 3:
                        src_idx, dst_idx, action = int(parts[0]), int(parts[1]), parts[2]
                        parsed.append([
                            nodes[src_idx], "unknown",
                            nodes[dst_idx], "unknown",
                            action, "0",
                        ])
                df = pd.DataFrame(
                    parsed,
                    columns=["actorID", "actor_type", "objectID", "object", "action", "timestamp"],
                )
                self.current_graph_df = df.copy()
            except Exception as e:
                print(f"[Env] Error loading base graph: {e}")
                self._load_dummy_graph()
        else:
            self._load_dummy_graph()

        return self._get_graph_state_representation()

    def _load_dummy_graph(self):
        self.current_graph_df = pd.DataFrame(
            [["111", "MP", "222", "MF", "WR", "10000"]],
            columns=["actorID", "actor_type", "objectID", "object", "action", "timestamp"],
        )

    # ─────────────────────────────────────────────────────────────────
    # State Representation
    # ─────────────────────────────────────────────────────────────────

    def _get_graph_state_representation(self) -> torch.Tensor:
        """Chuyển đồ thị hiện tại thành vector state 128-dim."""
        num_nodes = len(
            pd.unique(self.current_graph_df[["actorID", "objectID"]].values.ravel("K"))
        )
        num_edges = len(self.current_graph_df)
        state     = np.zeros(self.state_dim, dtype=np.float32)
        state[0]  = num_nodes
        state[1]  = num_edges
        # Thêm density, avg degree và confidence score vào state
        state[2]  = num_edges / (num_nodes + 1e-9)   # edge density
        state[3]  = self.last_confidence_score        # Detection confidence
        state[4]  = float(self.last_FN)               # FN flag
        return torch.tensor(state, dtype=torch.float32).to(device)

    # ─────────────────────────────────────────────────────────────────
    # Action 0: Generation Agent (RL Attack Agent)
    # ─────────────────────────────────────────────────────────────────

    def _generation_agent(self) -> float:
        """
        Gọi Attack Agent pipeline để sinh APT provenance subgraph mới.
        Append subgraph vào current_graph_df để mô phỏng sự tiến hóa.

        Reward:
          - Base: generation_reward['total'] từ RewardCalculator
          - Evasion bonus: (1 − confidence_score) * 0.5
            → Nếu Detection Agent không phát hiện được đồ thị vừa sinh (confidence thấp)
              → Attack Agent nhận thêm reward để khuyến khích tính evasive.

        Returns:
            float: reward cho PPO Coordinator
        """
        if _ATTACK_AGENT_AVAILABLE:
            gen_output = generate_apt_subgraph(max_stages=5)
        else:
            # Fallback khi Attack Agent không available
            gen_output = _generate_dummy_fallback_subgraph()

        new_edges_df   = gen_output.get("edges_df", pd.DataFrame())
        generation_rew = gen_output.get("reward", {})
        self.last_gen_reward = generation_rew

        if new_edges_df.empty:
            print("[Generation Agent] Empty subgraph generated.")
            return -0.5

        # Đảm bảo cột khớp
        for col in self.current_graph_df.columns:
            if col not in new_edges_df.columns:
                new_edges_df[col] = "unknown"

        # Append subgraph mới vào đồ thị hiện tại
        prev_size = len(self.current_graph_df)
        self.current_graph_df = pd.concat(
            [self.current_graph_df, new_edges_df[self.current_graph_df.columns]],
            ignore_index=True,
        )
        print(
            f"[Generation Agent] Appended {len(new_edges_df)} edges "
            f"({prev_size} → {len(self.current_graph_df)} total edges). "
            f"APT={gen_output.get('apt_name','?')} | variant={gen_output.get('variant','?')} | "
            f"is_fallback={gen_output.get('is_fallback', False)}"
        )

        visualize_current_graph(self.current_graph_df, self.run_dir)
        save_current_graph_to_txt(self.current_graph_df, self.run_dir)

        # Evasion pressure: nếu detection confidence thấp → Attack Agent đang evasive
        evasion_bonus = (1.0 - self.last_confidence_score) * 0.5 + self.accumulated_evasion_bonus
        base_reward   = float(generation_rew.get("total", 0.3))
        reward        = base_reward + evasion_bonus - self.failed_evasion_penalty

        print(
            f"[Generation Agent] Reward: base={base_reward:.4f} + "
            f"evasion_bonus={evasion_bonus:.4f} - penalty={self.failed_evasion_penalty:.4f} = {reward:.4f}"
        )

        # Reset accumulators cho các lượt chạy tiếp theo
        self.accumulated_evasion_bonus = 0.0
        self.failed_evasion_penalty = 0.0

        return reward

    # ─────────────────────────────────────────────────────────────────
    # Action 1: Pruning Agent
    # ─────────────────────────────────────────────────────────────────

    def _pruning_agent(self) -> float:
        """
        Nen do thi subgraph bang PruningAgent, giu nguyen quan he nhan qua.

        Input : self.current_graph_df  (subgraph tich luy tu Generation Agent)
        Output: self.current_graph_df  (da duoc nen, chi giu lai canh quan trong)
        Reward: compression_ratio * (1 - loss_causal)
        """
        if len(self.current_graph_df) <= 50:
            return 0.0

        original_edges = len(self.current_graph_df)

        if PruningAgent is not None:
            nodes_list, node_to_idx, node_types = [], {}, []
            for _, row in self.current_graph_df.iterrows():
                if row["actorID"] not in node_to_idx:
                    node_to_idx[row["actorID"]] = len(nodes_list)
                    nodes_list.append(row["actorID"])
                    node_types.append(row["actor_type"])   # MP / MF / SO
                if row["objectID"] not in node_to_idx:
                    node_to_idx[row["objectID"]] = len(nodes_list)
                    nodes_list.append(row["objectID"])
                    node_types.append(row["object"])       # MP / MF / SO

            relation_list = [[], [], [], []]
            original_rows = self.current_graph_df.reset_index(drop=True)
            for idx in range(len(original_rows)):
                row     = original_rows.iloc[idx]
                src_idx = node_to_idx[row["actorID"]]
                dst_idx = node_to_idx[row["objectID"]]
                relation_list[0].append([str(src_idx), str(dst_idx), str(row["action"]), idx])

            target_nodes = max(50, int(len(nodes_list) * 0.7))
            agent        = PruningAgent(target_nodes=target_nodes, verbose=False)
            try:
                pruned_entity, pruned_relations, node_map = agent.prune(
                    node_types, relation_list, stage=1
                )
                new_idx_to_name = {ni: nodes_list[oi] for ni, oi in node_map.items()}
                new_idx_to_type = {ni: node_types[oi] for ni, oi in node_map.items()}

                new_edges = []
                for new_edge in pruned_relations[0]:
                    ns, nd, ac, orig_row_idx = new_edge
                    ns, nd = int(ns), int(nd)
                    if ns not in new_idx_to_name or nd not in new_idx_to_name:
                        continue
                    sn = new_idx_to_name[ns]; st = new_idx_to_type[ns]
                    dn = new_idx_to_name[nd]; dt = new_idx_to_type[nd]
                    try:
                        ts = original_rows.iloc[int(orig_row_idx)]["timestamp"]
                    except (IndexError, ValueError):
                        ts = "0"
                    new_edges.append([sn, st, dn, dt, ac, ts])

                if new_edges:
                    self.current_graph_df = pd.DataFrame(
                        new_edges, columns=self.current_graph_df.columns
                    )
                    ratio       = 1.0 - len(new_edges) / original_edges
                    loss_causal = max(0.0, ratio - 0.5)
                    reward      = ratio * (1.0 - loss_causal) + 0.1
                    print(
                        f"[Pruning Agent] Compressed {ratio:.1%} "
                        f"({original_edges} -> {len(new_edges)} edges) | "
                        f"loss_causal={loss_causal:.2f} | reward={reward:.4f}"
                    )
                    return float(reward)

                print("[Pruning Agent] No edges after pruning. Keeping original.")
                return -0.1

            except Exception as e:
                print(f"[Pruning Agent] Error: {e}. Falling back to simple trim.")

        # Fallback
        if len(self.current_graph_df) > 1000:
            keep_n = int(len(self.current_graph_df) * 0.7)
            self.current_graph_df = self.current_graph_df.iloc[-keep_n:].reset_index(drop=True)
            print(f"[Pruning Agent] Fallback trim: kept last {keep_n} edges.")
        return 0.1

    # ─────────────────────────────────────────────────────────────────
    # Action 2: Detection Agent (GATs + Confidence Score + Gradient)
    # ─────────────────────────────────────────────────────────────────

    def _detection_agent(self) -> float:
        """
        Detection Agent dựa trên GATs + MLP + IsolationForest.

        Quy trình:
          1. Thompson Sampling MAB chọn sampling rate
          2. GAT forward_with_attention() → node embeddings + attention weights
          3. Mean-pool → graph_latent → MLP → confidence score
          4. Gradient: backward qua confidence loss để đo độ nhạy
          5. IsolationForest trên node embeddings → anomaly ratio
          6. Tính precision/recall/FN, cập nhật MAB, lưu hard sample
          7. Trả về reward + ghi nhận closed-loop feedback

        Returns:
            float: reward = (precision + recall − λ·FN) + scf_impact
        """
        try:
            total_edges = len(self.current_graph_df)
            if total_edges == 0:
                return 0.0

            # 1. MAB chọn subgraph
            selected_indices = self.mab.select_indices(total_edges)
            chosen_arm       = self.mab.last_arm
            mab_subgraph     = self.current_graph_df.iloc[selected_indices].copy()

            phrases, labels, edges, _ = prepare_graph(mab_subgraph)
            if len(phrases) == 0:
                return 0.0

            nodes        = [infer(x, self.w2vmodel, self.encoder) for x in phrases]
            x_tensor     = torch.tensor(np.array(nodes), dtype=torch.float32).to(device)
            edge_index   = torch.tensor(edges, dtype=torch.long).to(device)

            # 2. GAT forward với attention weights
            self.gat.eval()
            self.detection_mlp.eval()

            # Cần gradient để tính Gradient-weighted Confidence Score
            x_tensor.requires_grad_(True)

            with torch.enable_grad():
                # forward_with_attention trả về node_embeddings + alpha (attention per edge)
                node_embeddings, attention_weights = self.gat.forward_with_attention(
                    x_tensor, edge_index
                )

                # 3. Mean-pool → graph_latent → MLP
                graph_latent = node_embeddings.mean(dim=0).unsqueeze(0)   # [1, 20]
                mlp_probs    = self.detection_mlp(graph_latent)           # [1, 2]
                mlp_attack_prob = mlp_probs[0][1]                         # scalar tensor

                # 4. Tính Gradient w.r.t. node input features
                #    Gradient này đo mức độ ảnh hưởng của từng node đến quyết định malicious
                malicious_score = mlp_attack_prob
                malicious_score.backward(retain_graph=False)

            # Confidence Score (scalar) — dùng làm PPO Reward signal
            confidence_score       = mlp_attack_prob.item()
            self.last_confidence_score = confidence_score

            # Gradient w.r.t. input: [N, 30] — tóm tắt thành L2 norm mỗi node
            if x_tensor.grad is not None:
                grad_norms  = x_tensor.grad.detach().norm(dim=1)          # [N]
                # Top-k critical nodes theo gradient
                top_k       = min(5, len(grad_norms))
                topk_vals, topk_idx = torch.topk(grad_norms, top_k)
                print(
                    f"[Detection GAT] Top-{top_k} critical nodes (grad norm): "
                    f"{topk_vals.cpu().numpy().round(4).tolist()}"
                )
            else:
                grad_norms = None

            # 5. IsolationForest trên node embeddings
            latent_features = node_embeddings.detach()
            predicted_attack = 0
            anomaly_ratio    = 0.0

            if confidence_score > 0.5:
                latent_np = latent_features.cpu().numpy()
                clf = IsolationForest(random_state=0)
                if len(latent_np) > 1:
                    clf.fit(latent_np)
                    iforest_preds = clf.predict(latent_np)
                    anomaly_ratio = float((iforest_preds == -1).mean())
                if anomaly_ratio > 0.3:
                    predicted_attack = 1

            # 6. Ground Truth & Prediction
            # Tat ca do thi trong moi truong nay deu den tu Attack Agent -> luon la APT (GT=1)
            # Ngoai ra: neu do thi lon (>50 canh) thi chac chan la APT phuc tap
            is_attack_gt = 1  # Attack Agent luon sinh do thi tan cong

            pred_label = "Malicious" if predicted_attack else "Benign"
            gt_label   = "Malicious"  # Luon la Malicious

            print(
                f"[Detection GAT] Confidence={confidence_score:.4f} | "
                f"Anomaly={anomaly_ratio:.2%} | Pred={pred_label} | GT={gt_label}"
            )

            # Attention weights summary
            if attention_weights is not None:
                avg_attn = attention_weights.detach().mean().item()
                max_attn = attention_weights.detach().max().item()
                print(f"[Detection GAT] Attention: avg={avg_attn:.4f}, max={max_attn:.4f}")

            # Phat nang Attack Agent neu confidence qua cao (bi lo lieu)
            if confidence_score >= 0.95:
                self.failed_evasion_penalty += 2.0
                print(f"[Feedback] Bi phat manh (conf={confidence_score:.4f}) -> Phat Attack Agent {self.failed_evasion_penalty} o luot tiep theo.")

            # TP / FP / FN (tu moi buoc nay)
            TP_step = 1 if (predicted_attack == 1 and is_attack_gt == 1) else 0
            FP_step = 1 if (predicted_attack == 1 and is_attack_gt == 0) else 0
            FN_step = 1 if (predicted_attack == 0 and is_attack_gt == 1) else 0

            # Cap nhat sliding window
            self._window_TP.append(TP_step)
            self._window_FP.append(FP_step)
            self._window_FN.append(FN_step)
            self._total_det_steps += 1

            # Precision / Recall tu sliding window (co y nghia thong ke hon 1 mau)
            sum_TP = sum(self._window_TP)
            sum_FP = sum(self._window_FP)
            sum_FN = sum(self._window_FN)
            precision = sum_TP / (sum_TP + sum_FP + 1e-9)
            recall    = sum_TP / (sum_TP + sum_FN + 1e-9)

            # FN cua buoc nay (dung cho Replay Buffer va Evasion tracking)
            FN = FN_step

            # SCF Impact: trung binh co trong so, chuan hoa ve [0, 1]
            # Max possible avg = 1.0 (neu tat ca action la 'execute' = 1.0)
            scf_impact_raw = calculate_scf_impact(mab_subgraph)
            scf_impact     = min(scf_impact_raw, 1.0)  # Dam bao nam trong [0, 1]

            self.last_mlp_prob  = confidence_score
            self.last_precision = float(precision)
            self.last_recall    = float(recall)
            self.last_FN        = FN
            self.last_scf_impact = float(scf_impact)

            # MAB update
            mab_success = (TP == 1) or (anomaly_ratio > 0.3)
            self.mab.update(chosen_arm, reward_positive=mab_success)

            # 7. Prioritized Replay Buffer: lưu Hard Samples khi FN=1
            if FN == 1:
                boundary_dist = 1.0 - abs(confidence_score - 0.5) * 2.0
                hard_sample   = {
                    "latent":        graph_latent.detach().cpu(),
                    "label":         is_attack_gt,
                    "mlp_prob":      confidence_score,
                    "anomaly_ratio": anomaly_ratio,
                    "grad_norms":    grad_norms.cpu() if grad_norms is not None else None,
                    "attention_max": max_attn if attention_weights is not None else 0.0,
                }
                self.replay_buffer.add(hard_sample, priority=boundary_dist)
                print(
                    f"[Replay Buffer] Hard Sample added (FN=1, conf={confidence_score:.4f}, "
                    f"priority={boundary_dist:.4f}). Buffer size: {len(self.replay_buffer)}"
                )

                # Lưu hard sample subgraph vào file
                if getattr(self, "run_dir", None):
                    hard_path = os.path.join(self.run_dir, "hard_samples.txt")
                    with open(hard_path, "a", encoding="utf-8") as f:
                        f.write(
                            f"--- Hard Sample (FN=1) | Conf: {confidence_score:.4f} | "
                            f"Anomaly: {anomaly_ratio:.4f} ---\n"
                        )
                        for _, row in mab_subgraph.iterrows():
                            f.write(
                                f"{row['actorID']}\t{row['actor_type']}\t"
                                f"{row['objectID']}\t{row['object']}\t"
                                f"{row['action']}\t{row['timestamp']}\n"
                            )
                        f.write("\n")

                # Closed-loop feedback → compute_closed_loop_reward
                if _CLOSED_LOOP_AVAILABLE and self.last_gen_reward:
                    feedback = {
                        "graph_id":        "marl_env_graph",
                        "predicted_label": pred_label,
                        "malicious_score": confidence_score,
                        "confidence":      confidence_score,
                        "is_false_negative": bool(FN == 1),
                        "detector_name":   "GATs-SLOT-extension",
                    }
                    cl_reward = compute_closed_loop_reward(feedback, self.last_gen_reward)
                    print(
                        f"[Closed-Loop] closed_loop_total={cl_reward['closed_loop_total']:.4f} | "
                        f"evasion={cl_reward['detector_evasion']:.4f} | "
                        f"fn_bonus={cl_reward['hard_sample_bonus']:.4f}"
                    )
                    self.last_closed_loop_reward = cl_reward
                    self.accumulated_evasion_bonus += cl_reward.get("detector_evasion", 0.0)

            reward = (precision + recall - self.lambda_fn_penalty * FN) + scf_impact
            print(
                f"[Detection GAT] Precision={precision:.2f} | Recall={recall:.2f} | "
                f"SCF={scf_impact:.2f} | FN={FN} | Reward={reward:.4f}"
            )
            return float(reward)

        except Exception as e:
            print(f"[Detection Agent] Error: {e}")
            import traceback; traceback.print_exc()
            return -1.0

    # ─────────────────────────────────────────────────────────────────
    # Continual Learning: EWC + Prioritized Replay
    # ─────────────────────────────────────────────────────────────────

    def train_detection_agent_ewc(self, run_dir: str = None, batch_size: int = 8):
        """
        Offline Batch Update trên DetectionMLP:
          1. Sample batch từ PrioritizedReplayBuffer
          2. CrossEntropyLoss + EWC penalty
          3. Backward + optimizer step
          4. Lưu best_mlp.pth nếu metric tốt hơn
          5. Recompute Fisher (chuẩn bị task tiếp theo)
        """
        if len(self.replay_buffer) < 2:
            print(f"[EWC] Buffer quá nhỏ ({len(self.replay_buffer)} mẫu). Bỏ qua.")
            return

        batch = self.replay_buffer.sample(batch_size)
        if not batch:
            return

        x_list, y_list = [], []
        for s in batch:
            x_list.append(s["latent"])
            y_list.append(s["label"])

        x_batch = torch.cat(x_list, dim=0).to(device)
        y_batch = torch.tensor(y_list, dtype=torch.long).to(device)

        if not self._ewc_initialized and len(self.replay_buffer) >= 4:
            self.detection_mlp.compute_fisher([(x_batch, y_batch)], device, num_samples=len(batch))
            self._ewc_initialized = True

        self.detection_mlp.train()
        self.det_optimizer.zero_grad()

        outputs    = self.detection_mlp(x_batch)
        log_out    = torch.log(outputs + 1e-9)
        ce_loss    = F.nll_loss(log_out, y_batch)
        ewc_pen    = self.detection_mlp.ewc_loss()
        loss       = ce_loss + ewc_pen

        loss.backward()
        self.det_optimizer.step()
        self.detection_mlp.eval()

        with torch.no_grad():
            preds     = outputs.argmax(dim=1)
            correct   = (preds == y_batch).float().mean().item()
            det_score = correct

        ewc_val = ewc_pen.item() if hasattr(ewc_pen, "item") else float(ewc_pen)
        print(
            f"[EWC] Loss={loss.item():.4f} (CE={ce_loss.item():.4f} + EWC={ewc_val:.4f}) "
            f"| Acc={correct:.2%} | Buffer={len(self.replay_buffer)}"
        )

        if det_score > self.best_det_score:
            self.best_det_score = det_score
            # Luu vao run_dir (log theo timestamp)
            if run_dir is not None:
                os.makedirs(run_dir, exist_ok=True)
                run_path = os.path.join(run_dir, "best_mlp.pth")
                torch.save(self.detection_mlp.state_dict(), run_path)
                print(f"[EWC] Saved run checkpoint -> {run_path} (score={det_score:.4f})")
            # Luu vao thu muc goc (persistent best - dung cho lan chay tiep theo)
            base_dir = os.path.dirname(os.path.abspath(__file__))
            global_best_path = os.path.join(base_dir, "best_mlp.pth")
            torch.save(self.detection_mlp.state_dict(), global_best_path)
            print(f"[EWC] New global best DetectionMLP -> {global_best_path} (score={det_score:.4f})")

        if self._ewc_initialized:
            self.detection_mlp.compute_fisher(
                [(x_batch.detach(), y_batch.detach())], device, num_samples=len(batch)
            )

    # ─────────────────────────────────────────────────────────────────
    # Step
    # ─────────────────────────────────────────────────────────────────

    def step(self, action: int):
        """
        Thực thi một action của Coordinator.

        Actions:
          0 → Generation Agent (Attack Agent RL-based)
          1 → Pruning Agent
          2 → Detection Agent (GATs)

        Returns:
            (next_state, reward, done, info)
        """
        reward = 0.0
        done   = False

        if action == 0:
            # ── Generation Agent ──────────────────────────────
            reward = self._generation_agent()

        elif action == 1:
            # ── Pruning Agent ─────────────────────────────────
            reward = self._pruning_agent()

        elif action == 2:
            # ── Detection Agent ───────────────────────────────
            reward = self._detection_agent()
            # KHÔNG dừng episode ở đây để cho phép Generation Agent chạy tiếp tục trong cùng episode
            # done = True

        next_state = self._get_graph_state_representation()

        # Giới hạn kích thước đồ thị tối đa
        if len(self.current_graph_df) > 1000:
            done = True

        info = {
            "mlp_attack_prob":    self.last_mlp_prob,
            "confidence_score":   self.last_confidence_score,
            "precision":          self.last_precision,
            "recall":             self.last_recall,
            "FN":                 self.last_FN,
            "scf_impact":         getattr(self, "last_scf_impact", 0.0),
            "replay_buf_size":    len(self.replay_buffer),
            "mab_stats":          self.mab.get_stats(),
            "gen_reward_total":   float(self.last_gen_reward.get("total", 0.0)),
            "closed_loop_reward": getattr(self, "last_closed_loop_reward", {}),
        }

        return next_state, reward, done, info


# ─────────────────────────────────────────────────────────────────────────────
# Fallback helper (dùng khi Attack Agent không available)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_dummy_fallback_subgraph() -> dict:
    import random
    rows = [
        ["MP-bash", "MP", "MF-payload.sh", "MF", "WR", "1"],
        ["MF-payload.sh", "MF", "MP-python", "MP", "EX", "2"],
        ["MP-python", "MP", f"SO-192.168.{random.randint(1,254)}.{random.randint(1,254)}", "SO", "ST", "3"],
    ]
    df = pd.DataFrame(rows, columns=["actorID", "actor_type", "objectID", "object", "action", "timestamp"])
    return {"edges_df": df, "nx_graph": None, "reward": {"total": 0.2}, "stats": {"nodes": 4, "edges": 3}, "apt_name": "DummyAPT", "variant": "fallback", "is_fallback": True}
