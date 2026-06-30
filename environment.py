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

from networks import GAT, DetectionMLP, PositionalEncoder, make_classifier
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

# Import Benign Generator (module ĐỘC LẬP, không thuộc Attack Agent) — nguồn benign
# cùng mức với subgraph Attack Agent (thay UNICORN thô làm lớp benign).
benign_agent_dir = os.path.join(base_dir, "Benign_Agent")
if benign_agent_dir not in sys.path:
    sys.path.insert(0, benign_agent_dir)
try:
    from benign_interface import build_benign_corpus, bundle_to_edges_df
    _BENIGN_GEN_AVAILABLE = True
    print("[Env] Benign Generator loaded successfully.")
except ImportError as e:
    print(f"[Env] WARNING: Cannot import Benign Generator: {e}. Fallback to benign generator.")
    _BENIGN_GEN_AVAILABLE = False

# Nguon reference (eval/EWC-anchor/drift) = DARPA TC E3 (thay UNICORN). Dataset chon
# qua bien moi truong DARPA_DATASET (cadets|theia|trace). Fallback Benign Generator +
# Attack Agent JSON neu data DARPA chua tai -> he van chay.
DARPA_DATASET = os.environ.get("DARPA_DATASET", "cadets")

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
# Evasion Camouflage Bandit (Hướng 1 — env-level evasion action selection)
# ─────────────────────────────────────────────────────────────────────────────

class EvasionCamouflageBandit:
    """
    Attacker chọn MỨC CAMOUFLAGE: trộn benign-gen quanh APT subgraph (benign
    mimicry / LOTL) để pha loãng tín hiệu -> né detector. Đây là "evasion action"
    ở env-level (KHÔNG sửa Attack Agent; variant/stage đã chứng minh không né được).

    Epsilon-greedy trên reward định hình = evaded - cost*ratio: ưu tiên mức camo
    NHỎ NHẤT mà vẫn né (evasion hiệu quả, tránh degenerate max-camo). Khi detector
    học bắt được camo thấp (qua EWC) -> bandit leo lên camo cao -> arms race ->
    co-evolution khép kín.
    """
    RATIOS = [0.0, 0.5, 1.0, 2.0]

    def __init__(self, eps: float = 0.2, cost: float = 0.15):
        self.n        = len(self.RATIOS)
        self.q        = np.zeros(self.n)   # mean reward per arm
        self.cnt      = np.zeros(self.n)
        self.eps      = eps
        self.cost     = cost
        self.last_arm = 0

    def select(self) -> float:
        if np.random.rand() < self.eps or self.cnt.sum() == 0:
            self.last_arm = int(np.random.randint(self.n))
        else:
            self.last_arm = int(np.argmax(self.q))
        return self.RATIOS[self.last_arm]

    def update(self, arm: int, evaded: bool):
        ratio = self.RATIOS[arm]
        r     = (1.0 if evaded else 0.0) - self.cost * ratio
        self.cnt[arm] += 1
        self.q[arm]   += (r - self.q[arm]) / self.cnt[arm]
        print(f"[CamoBandit] arm={arm} (ratio={ratio}) evaded={evaded} r={r:+.2f} | "
              f"Q={np.round(self.q, 2).tolist()} cnt={self.cnt.astype(int).tolist()}")

    def get_stats(self) -> dict:
        return {f"camo{self.RATIOS[k]}_Q": float(self.q[k]) for k in range(self.n)}


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
        total = priorities.sum()
        if total > 0:
            priorities = priorities / total
        else:
            priorities = np.ones_like(priorities) / len(priorities)
            
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

        # EXP_PRETRAIN_DIR: thu muc weights khoi tao detector. Mac dinh trained_weights/darpa.
        # Dung cho RQ1 hold-out: co-evolution xuat phat tu baseline da loai ho APT test.
        _pretrain_dir = os.environ.get("EXP_PRETRAIN_DIR") or os.path.join(base_dir, "trained_weights", "darpa")
        if os.environ.get("EXP_PRETRAIN_DIR"):
            print(f"[Env] EXP_PRETRAIN_DIR -> loading detector weights from {_pretrain_dir}")

        # ── Load pre-trained GAT (Detection Agent backbone) ──────────
        self.gat = GAT(in_channels=30, out_channels=20, hidden_dim=64, heads=8, dropout=0.3).to(device)
        gat_model_path = os.path.join(_pretrain_dir, "gat.pth")
        if os.path.exists(gat_model_path):
            try:
                state = torch.load(gat_model_path, map_location=device)
                self.gat.load_state_dict(state)
                print(f"[Env] Loaded GAT weights from {gat_model_path}.")
            except Exception as e:
                print(f"[Env] Could not load GAT weights: {e}")
        else:
            print(f"[Env] WARNING: {gat_model_path} not found. Using random GAT weights.")
        self.gat.eval()

        # ── Load pre-trained Detection MLP ───────────────────────────
        # EXP_CLASSIFIER (RQ3): swap kien truc head {mlp|linear|deep}. Mac dinh "mlp"
        # khop checkpoint goc. Khi != mlp, mlp.pth khong load duoc (shape khac) ->
        # strict=False ben duoi se bo qua va head bat dau tu random init (dung y do RQ3).
        self._classifier_name = os.environ.get("EXP_CLASSIFIER", "mlp").lower()
        self.detection_mlp = make_classifier(self._classifier_name, input_dim=20).to(device)
        if self._classifier_name != "mlp":
            print(f"[Env] EXP_CLASSIFIER={self._classifier_name} -> classifier head {self.detection_mlp.hidden_dims}")

        # Thu tu uu tien load:
        # 1. trained_weights/darpa/mlp.pth (pre-trained goc, khong phu thuoc run khac)
        # 2. Random init
        # NOTE: global best_mlp.pth da KHONG con duoc doc de tranh lan lan giua cac lan chay.
        #       Best model cua tung run duoc luu trong runs/<timestamp>/best_mlp.pth.
        original_mlp_path = os.path.join(_pretrain_dir, "mlp.pth")

        if os.path.exists(original_mlp_path):
            try:
                self.detection_mlp.load_state_dict(
                    torch.load(original_mlp_path, map_location=device), strict=False
                )
                print(f"[Env] Loaded original DetectionMLP weights from {original_mlp_path}")
            except Exception as e:
                print(f"[Env] Could not load mlp.pth ({e}). Starting from random init.")
        else:
            print("[Env] No pre-trained DetectionMLP found. Starting from random init.")

        self.detection_mlp.eval()

        self.mab           = ThompsonSamplingMAB()
        self.replay_buffer = PrioritizedReplayBuffer(maxlen=500)

        self.best_det_score       = 0.0
        # Joint optimizer: GAT + MLP duoc train end-to-end.
        # lr_gat < lr_mlp: GAT (backbone) di cham hon MLP (head). Da NANG manh (Fix CL):
        # 5e-5/1e-4 truoc day qua nho -> boundary chi dich ~0.008 conf/update, khong
        # kip vuot 0.5 cho FN camo. Attention pooling can RESHAPE latent de TACH camo,
        # viec do can lr_gat du lon (test isolation hoc duoc o lr~1e-3). EWC anchor (1
        # lan) + benign-anchor 1:1 moi buoc van bao ve ky nang cu nen lr cao an toan.
        self.lr_gat               = 2e-4
        self.lr_mlp               = 3e-4
        self.det_optimizer        = torch.optim.Adam([
            {"params": self.gat.parameters(),           "lr": self.lr_gat},
            {"params": self.detection_mlp.parameters(), "lr": self.lr_mlp},
        ])
        self.grad_clip_norm       = 5.0   # noi tu 1.0 (Fix 3) sau khi nhan da dung
        # Continual-learning rebalance (Fix CL): truoc day model DONG BANG (TP/FP/TN
        # bat dong, chi FN tang) vi qua thien ve STABILITY. Cac nut van dieu chinh:
        #   - warmup_updates = 0: GAT (attention pooling = co che khang camo) train
        #     NGAY tu dau, khong dong bang -> co the hoc focus vao APT-core trong camo.
        #   - ewc_inner_steps = 8: chay NHIEU buoc gradient/episode thay vi 1 -> boundary
        #     di chuyen DU XA cho FN camo vuot 0.5 (1 buoc/episode qua cham vs toc do camo).
        #   - mal_loss_boost = 1.0 + benign-anchor 1:1: can bang. (v1 hoc duoc FN: TP 27->38;
        #     v2 benign-heavy 1.3:1 lam dong bang lai -> da revert ve v1 1:1.)
        self.warmup_updates       = 0     # Fix CL: KHONG dong bang GAT (attention = key)
        self.ewc_inner_steps      = 8     # Fix CL: K buoc gradient/EWC-call (truoc 1)
        self.mal_loss_boost       = 1.0   # CAN BANG (1:1 sampling + boost=1.0)
        # JOINT EWC (GAT+MLP): MO BANG GAT de backbone thich nghi voi bien the song
        # (giam FN — frozen GAT la nut that vi MLP khong tach duoc cai dac trung khong
        # tach). Fisher-anchor mo rong sang GAT giu tri thuc cu -> chong FP creep/quen
        # benign. Day la giai phap CL day du: plasticity backbone + stability co kiem soat.
        self.freeze_gat_cl        = False
        # he so EWC chung GAT+MLP — VUA DU: cho phep hoc FN nhung neo (tren reference DARPA)
        # du manh chong sup do (lambda=8 sup do; 30 on dinh). Cho phep ghi de qua EXP_EWC_LAMBDA
        # de chay lambda-sweep / tai lap thuc nghiem (default GIU 30.0 nhu da chot).
        self.ewc_lambda_joint     = float(os.environ.get("EXP_EWC_LAMBDA", 30.0))
        self._ewc_fisher          = {}    # name -> Fisher diagonal (GAT+MLP)
        self._ewc_star            = {}    # name -> theta* (anchor gan-pretrain)
        self._ewc_initialized     = False
        self._gat_update_count    = 0     # track so lan GAT duoc update (cadence Fisher + warm-up)

        # Fix 1: subgraph APT vua sinh — detect/score tren day, KHONG tren do thi tich luy
        self.last_generated_subgraph = None
        # Huong 1: env-level evasion qua camouflage (benign mimicry / LOTL)
        self.camo_bandit     = EvasionCamouflageBandit()
        self.last_camo_arm   = 0
        self.last_camo_ratio = 0.0
        self._camo_pool      = None
        # Fix 2: nguong bat buoc Pruning truoc Detection (kiem soat temporal explosion)
        self.prune_threshold      = 100
        self.last_compression_ratio = 0.0
        # Fix 4: drift detection (Wasserstein latent benign-ref vs stream)
        self.drift_tau            = 0.5   # nguong trigger Generation (tunable sau khi xem log)
        self.last_drift           = 0.0
        self.drift_triggered      = False
        self._benign_ref_emb      = None
        # Fix (bug info): TP cua buoc detect gan nhat (de main.py tinh fn_rate dung)
        self.last_TP              = 0
        self.last_det_latency     = 0.0

        self.encoder = PositionalEncoder(30)   # khop voi EMBED_DIM=30 trong graph_utils
        try:
            self.w2vmodel = Word2Vec.load(w2v_model_path)
        except Exception:
            print("[Env] W2V model not found. Using random embeddings.")
            self.w2vmodel = None

        # Khong con phu thuoc UNICORN/W2V: feature dung build_node_feature (30-dim),
        # base graph reset co dummy fallback, reference dung DARPA TC. W2V chi giu cho
        # tuong thich chu ky cu (None cung chay binh thuong).
        if self.w2vmodel is None:
            print("[Env] W2V khong dung (build_node_feature thay the) — bo qua.")

        # Trang thai tracking
        self.current_graph_df    = None
        self.last_mlp_prob       = 0.0
        self.last_precision      = 0.0
        self.last_recall         = 0.0
        self.last_FN             = 0
        self.last_fn_rate_win    = 0.0   # FN_rate tren cua so 20 buoc (cho StopMonitor)
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
        # EXP_DISABLE_GENERATION (RQ2 ablation C0/C1): tat hoan toan Generation Agent.
        # Action 0 tro thanh no-op (khong sinh subgraph moi, khong doi current_graph_df),
        # tra ve reward trung tinh. Detection van chay tren du lieu hien co.
        if os.environ.get("EXP_DISABLE_GENERATION"):
            self.last_gen_reward = {}
            return 0.0

        if _ATTACK_AGENT_AVAILABLE:
            # Ghi output sinh ra vao run_dir/attack_handoff CUA RUN NAY thay vi
            # Attack_Agent/result_handoff (vung committed bi ghi de moi run). Chi truyen
            # output_dir (tham so co san cua generation_interface) -> KHONG dung internals
            # Attack Agent. run_dir=None (test/standalone) -> fallback ve result_handoff.
            gen_out_dir = (
                os.path.join(self.run_dir, "attack_handoff")
                if getattr(self, "run_dir", None) else None
            )
            gen_output = generate_apt_subgraph(max_stages=5, output_dir=gen_out_dir)
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

        # Fix 1: subgraph APT vua sinh (tin hieu tan cong dam dac). Detection score TREN day.
        apt_sub = new_edges_df[self.current_graph_df.columns].copy()

        # Huong 1: env-level EVASION — chon muc camouflage (tron benign-gen quanh APT
        # -> benign mimicry/LOTL) de ne detector. Bandit hoc muc camo toi uu.
        ratio = self.camo_bandit.select()
        self.last_camo_arm   = self.camo_bandit.last_arm
        self.last_camo_ratio = ratio
        k = int(len(apt_sub) * ratio)
        pool = self._get_camouflage_pool()
        if k > 0 and len(pool) > 0:
            camo = pool.sample(n=k, replace=(k > len(pool)))[self.current_graph_df.columns]
            self.last_generated_subgraph = pd.concat([apt_sub, camo], ignore_index=True)
        else:
            self.last_generated_subgraph = apt_sub
        print(f"[Generation Agent] Camouflage ratio={ratio} -> detect target = "
              f"{len(apt_sub)} APT + {k} benign = {len(self.last_generated_subgraph)} edges")

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

    def _get_camouflage_pool(self):
        """Pool cạnh benign-gen dùng làm camouflage (cache 1 lần). Benign mimicry."""
        if self._camo_pool is not None:
            return self._camo_pool
        empty = self.current_graph_df.iloc[:0].copy() if self.current_graph_df is not None else pd.DataFrame()
        if not _BENIGN_GEN_AVAILABLE:
            self._camo_pool = empty
            return self._camo_pool
        try:
            dfs = [bundle_to_edges_df(b)
                   for b in build_benign_corpus(40, source="hybrid", mimicry_level=0.4, seed=99)]
            self._camo_pool = pd.concat(dfs, ignore_index=True) if dfs else empty
            print(f"[CamoBandit] Camouflage pool: {len(self._camo_pool)} benign-gen edges cached.")
        except Exception as ex:
            print(f"[CamoBandit] Camouflage pool error: {ex}")
            self._camo_pool = empty
        return self._camo_pool

    # ─────────────────────────────────────────────────────────────────
    # Action 1: Pruning Agent
    # ─────────────────────────────────────────────────────────────────

    def _pruning_agent(self, target_ratio: float = 0.7) -> float:
        """
        Nen do thi subgraph bang PruningAgent, giu nguyen quan he nhan qua.

        Args:
            target_ratio: ty le node MUON GIU lai (0.7 = giu 70% -> nen 30%).
                          Forced-prune-before-detection truyen 0.3 de nen manh (~70%).

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

            target_nodes = max(20, int(len(nodes_list) * target_ratio))
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

                if new_edges and len(new_edges) < original_edges:
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

                # PruningAgent khong nen duoc (no edges / khong giam) -> fall through
                # xuong recency-trim ben duoi de explosion luon duoc kiem soat.
                print("[Pruning Agent] PruningAgent khong giam duoc -> dung recency-trim.")

            except Exception as e:
                print(f"[Pruning Agent] Error: {e}. Falling back to recency trim.")

        # Fallback: recency-based summarization (giu N canh moi nhat theo thoi gian).
        # Crude nhung dam bao explosion luon duoc kiem soat va co so lieu compression
        # khi PruningAgent khong kha dung / khong nen duoc.
        keep_n = max(50, int(original_edges * target_ratio))
        if original_edges > keep_n:
            self.current_graph_df = self.current_graph_df.iloc[-keep_n:].reset_index(drop=True)
            ratio = 1.0 - keep_n / original_edges
            print(
                f"[Pruning Agent] Fallback recency-trim: {original_edges} -> {keep_n} edges "
                f"(compression={ratio:.1%})"
            )
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
            import time as _time
            _t0 = _time.time()

            # Fix 2: BAT BUOC Pruning truoc Detection — kiem soat temporal explosion
            # va tao du lieu do compression ratio (Gap 2).
            self._force_prune_before_detection()

            # Fix 1: Detect/score tren SUBGRAPH APT vua sinh (tin hieu dam dac),
            # KHONG tren do thi tich luy da bi pha loang boi base benign.
            detect_df = self.last_generated_subgraph
            if detect_df is None or len(detect_df) == 0:
                detect_df = self.current_graph_df   # fallback (vd: detect truoc khi co generate)

            total_edges = len(detect_df)
            if total_edges == 0:
                return 0.0

            # 1. MAB chọn subgraph (tren subgraph APT)
            selected_indices = self.mab.select_indices(total_edges)
            chosen_arm       = self.mab.last_arm
            mab_subgraph     = detect_df.iloc[selected_indices].copy()

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
                # 3. ATTENTION-WEIGHTED POOLING → graph_latent (chống camouflage):
                #    node APT đáng ngờ (attention cao) trội hơn benign padding.
                graph_latent, node_embeddings, attention_weights = self.gat.graph_latent(
                    x_tensor, edge_index, return_nodes=True
                )                                                         # [1, 20]
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

            # 5. IsolationForest: chi dung de tinh anomaly_ratio lam feature phu cho reward
            # KHONG dung IsolationForest de quyet dinh phan loai vi:
            # - IsolationForest(contamination=0.3).fit(X).predict(X) luon label dung 30% la outlier
            # - anomaly_ratio luon >= 0.3 -> iforest_boost luon = 1 -> predicted=1 luon -> 100% TP (gia)
            latent_features = node_embeddings.detach()
            anomaly_ratio   = 0.0

            if len(latent_features) > 1:
                latent_np     = latent_features.cpu().numpy()
                clf           = IsolationForest(random_state=0, contamination="auto")
                clf.fit(latent_np)
                # Dung decision_function (score) thay vi binary predict:
                # score < 0: outlier (cang am cang anomalous)
                # score > 0: inlier
                scores        = clf.decision_function(latent_np)
                # anomaly_ratio: ty le node co score < nguong -0.05 (coi la anomalous ro rang)
                anomaly_ratio = float((scores < -0.05).mean())

            # 5b. Fix 4: Concept Drift — Wasserstein giua phan phoi latent cua subgraph
            # hien tai vs phan phoi latent benign tham chieu (UNICORN). Drift cao =>
            # cau truc temporal-causal da doi (LOTL/fileless/AI-TTP) => can sinh variant.
            drift = self._compute_latent_drift(node_embeddings.detach())
            self.last_drift      = drift
            self.drift_triggered = drift > self.drift_tau
            print(
                f"[DRIFT] wasserstein={drift:.4f} (tau={self.drift_tau:.3f}) -> "
                f"trigger_generation={self.drift_triggered}"
            )

            # 6. Quyet dinh phan loai: CHI dua vao MLP confidence (thuc su tu model)
            # IsolationForest anomaly_ratio chi dung de augment reward, khong quyet dinh label
            predicted_attack = 1 if confidence_score >= 0.5 else 0

            # Huong 1: cap nhat Evasion Camouflage Bandit — ne thanh cong neu graph
            # (APT + camouflage) bi phan loai Benign (FN). Khep kin co-evolution:
            # attacker hoc muc camo ne duoc -> sinh FN -> EWC train -> detector bat
            # duoc -> attacker leo camo cao hon.
            self.camo_bandit.update(self.last_camo_arm, evaded=(predicted_attack == 0))

            # 7. Ground Truth
            # Graph = APT subgraph (+ camouflage benign) -> VAN la tan cong (GT=1):
            # day la benign-mimicry attack, detector phai phat hien APT trong camo.
            is_attack_gt = 1

            pred_label = "Malicious" if predicted_attack else "Benign"
            gt_label   = "Malicious"

            print(
                f"[Detection GAT] Conf={confidence_score:.4f} | "
                f"Pred={pred_label} (MLP-driven) | "
                f"Anomaly(aux)={anomaly_ratio:.2%} | GT={gt_label}"
            )

            if attention_weights is not None:
                avg_attn = attention_weights.detach().mean().item()
                max_attn = attention_weights.detach().max().item()
                print(f"[Detection GAT] Attention: avg={avg_attn:.4f}, max={max_attn:.4f}")

            # Phat Attack Agent neu bi phat hien qua ro rang
            if confidence_score >= 0.95:
                self.failed_evasion_penalty += 2.0
                print(
                    f"[Feedback] Bi phat manh (conf={confidence_score:.4f}) -> "
                    f"Attack Agent penalty={self.failed_evasion_penalty:.1f}"
                )

            # TP / FP / FN (dua tren prediction cuoi cung)
            TP_step = 1 if (predicted_attack == 1 and is_attack_gt == 1) else 0
            FP_step = 1 if (predicted_attack == 1 and is_attack_gt == 0) else 0  # luon 0 vi GT=1
            FN_step = 1 if (predicted_attack == 0 and is_attack_gt == 1) else 0
            TN_step = 1 if (predicted_attack == 0 and is_attack_gt == 0) else 0  # luon 0 vi GT=1

            # Cap nhat sliding window (TN cung duoc track trong window)
            self._window_TP.append(TP_step)
            self._window_FP.append(FP_step)
            self._window_FN.append(FN_step)
            self._total_det_steps += 1

            # Precision / Recall / F1 / Accuracy tu sliding window
            # LUU Y: window co maxlen nen chi giu toi da 20 buoc gan nhat.
            # Tat ca sum_* deu tu cung 1 window -> TN tinh dung la (window_size - TP - FP - FN)
            sum_TP  = sum(self._window_TP)
            sum_FP  = sum(self._window_FP)
            sum_FN  = sum(self._window_FN)
            win_len = len(self._window_TP)   # so buoc thuc su trong window (<=maxlen)
            sum_TN  = max(0, win_len - sum_TP - sum_FP - sum_FN)

            precision = sum_TP / (sum_TP + sum_FP + 1e-9)
            recall    = sum_TP / (sum_TP + sum_FN + 1e-9)
            f1        = 2 * precision * recall / (precision + recall + 1e-9)
            # Accuracy: dung win_len (khong phai total_det_steps) vi sum_TP/FN/FP/TN
            # deu lay tu window (maxlen=20), khong phai toan bo lich su
            accuracy  = (sum_TP + sum_TN) / (win_len + 1e-9)

            # FN cua buoc nay (dung cho Replay Buffer va Evasion tracking)
            FN = FN_step
            # Luu TP buoc nay de main.py tinh fn_rate = FN/(FN+TP) cho dung
            # (truoc day info thieu "TP" -> tp_count luon 0 -> fn_rate luon 1.0).
            self.last_TP = TP_step

            # SCF Impact: trung binh co trong so, chuan hoa ve [0, 1]
            scf_impact_raw = calculate_scf_impact(mab_subgraph)
            scf_impact     = min(scf_impact_raw, 1.0)

            self.last_mlp_prob   = confidence_score
            self.last_precision  = float(precision)
            self.last_recall     = float(recall)
            self.last_FN         = FN
            self.last_scf_impact = float(scf_impact)
            # FN_rate WINDOWED (= 1 - recall tren cua so 20 buoc) cho StopMonitor.
            # Per-episode fn_rate (FN/det_calls voi 1-2 call) chi nhan {0,0.5,1.0} ->
            # std qua nhieu -> Nash khong bao gio trigger. Dung windowed -> std co nghia.
            self.last_fn_rate_win = float(sum_FN / (sum_FN + sum_TP + 1e-9))

            # MAB update: thanh cong neu Detection dung (TP) hoac co anomaly cao
            mab_success = (TP_step == 1) or (anomaly_ratio > 0.3)
            self.mab.update(chosen_arm, reward_positive=mab_success)

            # 7. Prioritized Replay Buffer: luu Hard Samples khi FN=1
            if FN == 1:
                boundary_dist = 1.0 - abs(confidence_score - 0.5) * 2.0
                # Luu RAW graph (x_tensor + edge_index) thay vi latent vector.
                # Ly do: latent duoc tao boi GAT phien ban cu -> stale sau khi GAT update.
                # Khi sample de train/eval, ta re-forward qua GAT HIEN TAI -> latent luon
                # nhat quan voi khong gian GAT moi nhat.
                hard_sample   = {
                    "x_tensor":      x_tensor.detach().cpu(),     # [N, 30] raw node features
                    "edge_index":    edge_index.detach().cpu(),   # [2, E] graph structure
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

                # Luu hard sample subgraph vao file
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

                # Closed-loop feedback
                if _CLOSED_LOOP_AVAILABLE and self.last_gen_reward:
                    feedback = {
                        "graph_id":          "marl_env_graph",
                        "predicted_label":   pred_label,
                        "malicious_score":   confidence_score,
                        "confidence":        confidence_score,
                        "is_false_negative": bool(FN == 1),
                        "detector_name":     "GATs-SLOT-extension",
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
                f"[Detection GAT Metrics] "
                f"Conf={confidence_score:.3f} | "
                f"Acc={accuracy:.2%} | Prec={precision:.4f} | Recall={recall:.4f} | F1={f1:.4f} | "
                f"FN={FN} | SCF={scf_impact:.3f}"
            )
            print(
                f"[Detection GAT Window] "
                f"TP={sum_TP} FP={sum_FP} FN={sum_FN} TN={sum_TN} "
                f"(win={win_len}/{self._total_det_steps} total) | Reward={reward:.4f}"
            )
            # Fix 2: latency inference (Gap 2 — kiem chung "< 15 phut/batch")
            self.last_det_latency = _time.time() - _t0
            print(
                f"[Detection Latency] {self.last_det_latency:.3f}s "
                f"(subgraph={total_edges} edges, accum_graph={len(self.current_graph_df)} edges, "
                f"last_compression={self.last_compression_ratio:.1%})"
            )
            return float(reward)

        except Exception as e:
            print(f"[Detection Agent] Error: {e}")
            import traceback; traceback.print_exc()
            return -1.0

    # ─────────────────────────────────────────────────────────────────
    # Fix 2 helper: Bat buoc Pruning truoc Detection (kiem soat explosion)
    # ─────────────────────────────────────────────────────────────────

    def _force_prune_before_detection(self):
        """
        Goi Pruning Agent tren do thi tich luy truoc moi buoc Detection neu kich
        thuoc vuot prune_threshold. Tach roi khoi PPO action de dam bao explosion
        luon duoc kiem soat va compression ratio luon duoc do (Gap 2) — thay vi
        phu thuoc Coordinator (truoc day hoc cach khong bao gio prune).
        """
        before = len(self.current_graph_df)
        if before <= self.prune_threshold:
            self.last_compression_ratio = 0.0
            return
        # target_ratio=0.3 -> giu 30% node, nen ~70% (kiem chung muc tieu 70-80% Gap 2)
        self._pruning_agent(target_ratio=0.3)   # mutates self.current_graph_df
        after = len(self.current_graph_df)
        self.last_compression_ratio = (1.0 - after / before) if before > 0 else 0.0
        print(
            f"[Pruning->Detect] FORCED prune truoc Detection: {before} -> {after} edges "
            f"(compression={self.last_compression_ratio:.1%})"
        )

    # ─────────────────────────────────────────────────────────────────
    # Fix 4 helper: Drift detection (Wasserstein latent benign-ref vs stream)
    # ─────────────────────────────────────────────────────────────────

    def _get_benign_reference_embeddings(self, max_graphs: int = 30, max_nodes: int = 3000):
        """
        Phan phoi node-embedding tham chieu cua benign (UNICORN) qua GAT HIEN TAI.
        Cache lai; bi invalidate cung luc voi latent cache khi GAT update.
        """
        if self._benign_ref_emb is not None:
            return self._benign_ref_emb

        benign_raws = [r for r in self._get_reference_raw_graphs() if r[2] == 0][:max_graphs]
        embs = []
        self.gat.eval()
        with torch.no_grad():
            for x_t, e_t, _ in benign_raws:
                emb = self.gat(x_t.to(device), e_t.to(device))
                embs.append(emb.detach().cpu())
        if embs:
            ref = torch.cat(embs, dim=0)
            if ref.shape[0] > max_nodes:
                idx = torch.randperm(ref.shape[0])[:max_nodes]
                ref = ref[idx]
            self._benign_ref_emb = ref
        else:
            self._benign_ref_emb = torch.zeros((0, 20))
        return self._benign_ref_emb

    def _compute_latent_drift(self, node_embeddings: torch.Tensor) -> float:
        """
        Sliced-Wasserstein (trung binh Wasserstein-1 tung chieu) giua phan phoi
        node-embedding hien tai va benign reference. Re va on dinh cho [N, 20].
        """
        try:
            from scipy.stats import wasserstein_distance
        except ImportError:
            return 0.0

        ref = self._get_benign_reference_embeddings()
        cur = node_embeddings.detach().cpu()
        if ref.shape[0] == 0 or cur.shape[0] == 0:
            return 0.0

        ref_np, cur_np = ref.numpy(), cur.numpy()
        d = min(cur_np.shape[1], ref_np.shape[1])
        dists = [wasserstein_distance(cur_np[:, j], ref_np[:, j]) for j in range(d)]
        return float(np.mean(dists))

    # ─────────────────────────────────────────────────────────────────
    # Continual Learning: EWC + Prioritized Replay
    # ─────────────────────────────────────────────────────────────────

    def _joint_ewc_params(self):
        """Tham so duoc EWC bao ve: GAT (backbone) + MLP (head)."""
        params = {f"gat.{n}": p for n, p in self.gat.named_parameters() if p.requires_grad}
        params.update({f"mlp.{n}": p for n, p in self.detection_mlp.named_parameters() if p.requires_grad})
        return params

    def _compute_joint_ewc_anchor(self):
        """
        Tinh Fisher diagonal + snapshot theta* cho CA GAT lan MLP, MOT LAN, tren TAP
        THAM CHIEU DARPA (125 benign + 125 malicious) = DUNG tri thuc can GIU.

        QUAN TRONG (fix collapse): truoc day neo tren batch FN song dau tien (chi co
        live-malicious + benign), KHONG chua malicious DARPA -> Fisher khong bao ve
        tham so quan trong cho DARPA-mal -> chung troi tu do khi GAT thich nghi bien
        the song -> CATASTROPHIC FORGETTING (F1 0.96->0.06). Neo tren reference set
        bao ve CA benign LAN malicious DARPA -> GAT van hoc duoc FN moi nhung bi giu
        khong pha huy tri thuc cu.
        """
        ref = self._get_reference_raw_graphs()
        if not ref:
            print("[EWC] Khong co reference de neo Fisher -> bo qua anchor.")
            return
        self.gat.train(); self.detection_mlp.train()
        for m in self.gat.modules():
            if isinstance(m, torch.nn.BatchNorm1d):
                m.eval()
        params = self._joint_ewc_params()
        fisher = {k: torch.zeros_like(p) for k, p in params.items()}
        for x_t, e_t, lab in ref:
            self.gat.zero_grad(); self.detection_mlp.zero_grad()
            out = self.detection_mlp(self.gat.graph_latent(x_t.to(device), e_t.to(device)))
            y = torch.tensor([int(lab)], device=device)
            loss = F.nll_loss(torch.log(out + 1e-9), y)
            loss.backward()
            for k, p in params.items():
                if p.grad is not None:
                    fisher[k] += p.grad.detach() ** 2
        n = max(len(ref), 1)
        self._ewc_fisher = {k: f / n for k, f in fisher.items()}
        self._ewc_star = {k: p.detach().clone() for k, p in params.items()}
        self.gat.zero_grad(); self.detection_mlp.zero_grad()
        n_mal = sum(1 for r in ref if r[2] == 1)
        tot = sum(f.sum().item() for f in self._ewc_fisher.values())
        print(f"[EWC] Joint anchor on {n} DARPA reference graphs (mal={n_mal}, ben={n-n_mal}) "
              f"| total Fisher={tot:.4e}")

    def _joint_ewc_penalty(self):
        """L_EWC = lambda/2 * Sum_i F_i (theta_i - theta*_i)^2 tren GAT + MLP."""
        if not self._ewc_fisher:
            return torch.tensor(0.0, device=device)
        loss = torch.tensor(0.0, device=device)
        for k, p in self._joint_ewc_params().items():
            if k in self._ewc_fisher:
                loss = loss + (self._ewc_fisher[k] * (p - self._ewc_star[k]) ** 2).sum()
        return (self.ewc_lambda_joint / 2.0) * loss

    def train_detection_agent_ewc(self, run_dir: str = None, batch_size: int = 8):
        """
        Offline Batch Update trên DetectionMLP:
          1. Sample batch từ PrioritizedReplayBuffer
          2. CrossEntropyLoss + EWC penalty
          3. Backward + optimizer step
          4. Lưu best_mlp.pth nếu metric tốt hơn
          5. Recompute Fisher (chuẩn bị task tiếp theo)

        Returns:
            global_f1 (float | None): Global F1 trên tap test doc lap sau khi update.
                                      None neu buffer qua nho hoac danh gia that bai.
        """
        if len(self.replay_buffer) < 2:
            # Buffer nho -> KHONG co hard-sample de train EWC, NHUNG van phai DO Global F1
            # tren tap tham chieu DARPA (decouple eval khoi EWC update). Truoc day return
            # None -> global_f1=0 (gia) -> sau ep20 trigger "Detection collapse" FAILURE
            # sai (detector that ra dang thang, FN_rate=0). Do truc tiep, bo qua train.
            print(f"[EWC] Buffer nho ({len(self.replay_buffer)} mẫu) -> bỏ qua train, chỉ đo Global F1.")
            return self.evaluate_detection_mlp(run_dir=run_dir)

        batch = self.replay_buffer.sample(batch_size)
        if not batch:
            return self.evaluate_detection_mlp(run_dir=run_dir)

        # ── End-to-end: GAT + MLP cung train ─────────────────────────
        # Buffer luu RAW graph (x_tensor, edge_index), KHONG luu latent.
        # Ta re-forward qua GAT HIEN TAI (gradient enabled) de latent luon
        # nhat quan voi khong gian GAT moi nhat (tranh stale latent).
        #
        # Warm-up (warmup_updates update dau train MLP-only) GIO MAC DINH TAT
        # (warmup_updates=0): attention pooling cua GAT chinh la co che khang camo,
        # dong bang no = dong bang thu can hoc nhat. Co che van con neu can bat lai.
        gat_trainable = (not self.freeze_gat_cl) and (self._gat_update_count >= self.warmup_updates)
        self.detection_mlp.train()
        if gat_trainable:
            self.gat.train()
            # BatchNorm cua GAT GIU O EVAL (dung running stats tu pretrain): neu de BN
            # train, batch nho/malicious-heavy lam truot running stats -> eval benign
            # bi chuan hoa sai -> precision sup. Chi train trong so conv.
            for m in self.gat.modules():
                if isinstance(m, torch.nn.BatchNorm1d):
                    m.eval()
        else:
            self.gat.eval()   # warm-up: GAT dong bang hoan toan

        # 1. Thu thap RAW graph cho update nay: Hard Samples malicious (tu buffer)
        #    + benign anchor can bang 1:1 (tu UNICORN). Ta KHONG cache latent o day:
        #    moi buoc gradient trong inner-loop se RE-FORWARD qua GAT HIEN TAI de
        #    latent luon nhat quan voi GAT dang thay doi (tranh stale trong loop).
        train_raws = []   # list of (x_tensor, edge_index, label)
        for s in batch:
            e_t = s["edge_index"]
            if e_t.shape[1] == 0:
                continue
            train_raws.append((s["x_tensor"].to(device), e_t.to(device), int(s["label"])))

        if not train_raws:
            print("[EWC] Khong co Hard Sample hop le (edge rong). Bo qua.")
            self.gat.eval(); self.detection_mlp.eval()
            return None

        # CAN BANG 1:1 (Nguyen nhan 5): Replay Buffer chi chua Hard Samples (label=1);
        # neu chi train label=1, MLP se hoc predict all-Malicious -> FP no. Re-forward
        # benign qua GAT hien tai vua neo benign (bao ve precision) vua giam GAT drift.
        n_mal_train = len(train_raws)
        benign_raws = [r for r in self._get_reference_raw_graphs() if r[2] == 0]
        if benign_raws:
            import random
            for x_t, e_t, _ in random.choices(benign_raws, k=n_mal_train):
                train_raws.append((x_t.to(device), e_t.to(device), 0))
        else:
            print("[EWC] Khong co benign anchor (UNICORN rong) -> train mal-only (rui ro FP).")

        y_all = torch.tensor([r[2] for r in train_raws], dtype=torch.long).to(device)

        # Class weight (Fix CL): upweight malicious de day boundary ve phia RECALL,
        # chong lai benign-anchor + EWC inertia da lam FN tang khong ngung. Benign
        # anchor 1:1 van giu de bao ve precision; chi nghieng GRADIENT ve malicious.
        n_pos = max(int((y_all == 1).sum()), 1)
        n_neg = max(int((y_all == 0).sum()), 1)
        cls_w = torch.tensor(
            [1.0, (n_neg / n_pos) * self.mal_loss_boost], dtype=torch.float32, device=device
        )

        # 2. EWC anchor (Fisher + theta*) — tinh DUY NHAT 1 LAN tren trang thai con
        #    LANH MANH (gan pretrain). KHONG bao gio re-anchor vao batch camo hien tai.
        #    (Nguyen nhan 4): truoc day moi 3 update lai compute_fisher tren batch camo
        #    -> theta* = "trang thai hien tai" -> EWC bien thanh "dung thay doi gi" ->
        #    DONG BANG model dung cho no dang bo sot FN. Anchor 1 lan -> EWC chi giu
        #    ky nang cu (benign + easy-mal), con CE tren FN moi TU DO day boundary ra.
        # JOINT EWC (GAT + MLP): mo rong anchor sang GAT de backbone duoc phep thich
        # nghi voi bien the song (giam FN) MA van bi Fisher-anchor giu lai tri thuc cu
        # (chong FP creep / quen benign). Truoc day chi anchor MLP -> GAT dong bang la
        # nut that (MLP khong tach duoc cai dac trung khong tach).
        if not self._ewc_initialized and len(self.replay_buffer) >= 4:
            self._compute_joint_ewc_anchor()
            self._ewc_initialized = True

        # 3. INNER LOOP (Fix CL): K buoc gradient tren cung batch thay vi 1 buoc.
        #    1 buoc/episode dưới λ=50 truoc day khong du dich boundary cho FN camo
        #    vuot 0.5 -> TP/FP/TN dong bang, FN tang. K buoc + λ thap + class weight
        #    cho phep boundary di chuyen DU XA de HOC mau FN moi.
        self.detection_mlp.train()
        outputs = None
        last_ce = last_ewc = 0.0
        for _k in range(self.ewc_inner_steps):
            self.det_optimizer.zero_grad()
            lats = []
            for x_t, e_t, _ in train_raws:
                lat = self.gat.graph_latent(x_t, e_t)   # attention-weighted pooling [1,20]
                lats.append(lat if gat_trainable else lat.detach())
            x_batch_aug = torch.cat(lats, dim=0)

            outputs = self.detection_mlp(x_batch_aug)
            log_out = torch.log(outputs + 1e-9)
            ce_loss = F.nll_loss(log_out, y_all, weight=cls_w)
            ewc_pen = self._joint_ewc_penalty()   # EWC tren CA GAT + MLP
            loss    = ce_loss + ewc_pen

            loss.backward()
            # GAT chi co grad khi het warm-up; clip_grad_norm_ bo qua grad None.
            torch.nn.utils.clip_grad_norm_(self.gat.parameters(), max_norm=self.grad_clip_norm)
            torch.nn.utils.clip_grad_norm_(self.detection_mlp.parameters(), max_norm=self.grad_clip_norm)
            self.det_optimizer.step()
            last_ce  = ce_loss.item()
            last_ewc = ewc_pen.item() if hasattr(ewc_pen, "item") else float(ewc_pen)

        self.gat.eval()
        self.detection_mlp.eval()
        y_batch_aug = y_all   # alias cho phan metrics ben duoi
        loss_total  = last_ce + last_ewc
        _phase = "JOINT(GAT+MLP)" if gat_trainable else f"WARMUP(MLP-only {self._gat_update_count+1}/{self.warmup_updates})"
        print(f"[EWC] Update phase: {_phase} | inner_steps={self.ewc_inner_steps} | "
              f"cls_w_mal={cls_w[1].item():.2f}")

        # 4. GAT da thay doi -> invalidate cache latent UNICORN (Nguyen nhan 2).
        #    Lan eval ke tiep re-extract latent qua GAT moi -> train va eval cung
        #    1 khong gian latent.
        self._gat_update_count += 1
        if gat_trainable:
            self._invalidate_reference_latent_cache()

        # ── Tinh chinh xac cac metrics sau khi update ────────────────
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            y_np  = y_batch_aug.cpu().numpy()
            p_np  = preds.cpu().numpy()

            # Confusion matrix elements
            TP_m = int(((p_np == 1) & (y_np == 1)).sum())
            FP_m = int(((p_np == 1) & (y_np == 0)).sum())
            FN_m = int(((p_np == 0) & (y_np == 1)).sum())
            TN_m = int(((p_np == 0) & (y_np == 0)).sum())

            n_total   = len(y_np)
            accuracy  = float(accuracy_score(y_np, p_np))
            precision = float(precision_score(y_np, p_np, zero_division=0))
            recall    = float(recall_score(y_np, p_np, zero_division=0))
            f1_val    = float(f1_score(y_np, p_np, zero_division=0))

            # det_score dung F1 thay vi accuracy don thuan
            # F1 can bang giua Precision va Recall, kho bi "gian doi" boi class imbalance
            det_score = f1_val

        print(
            f"[EWC] Loss={loss_total:.4f} (CE={last_ce:.4f} + EWC={last_ewc:.4f}) | "
            f"Buffer={len(self.replay_buffer)}"
        )
        print(
            f"[EWC Metrics (Mini-batch)] Acc={accuracy:.2%} | "
            f"Prec={precision:.4f} | Recall={recall:.4f} | F1={f1_val:.4f} | "
            f"TP={TP_m} FP={FP_m} FN={FN_m} TN={TN_m} | "
            f"N={n_total} (mal={int(y_np.sum())}, ben={int((y_np==0).sum())})"
        )

        # (Fix CL — Nguyen nhan 4) KHONG re-anchor Fisher. EWC anchor da tinh DUY NHAT
        # 1 lan o trang thai lanh manh (gan pretrain). Re-anchor vao batch camo hien
        # tai chinh la loi cu lam DONG BANG model -> da loai bo.

        # ── Danh gia model tren tap test doc lap (toan cuc) ──────────
        global_f1 = self.evaluate_detection_mlp(run_dir=run_dir)

        # ── Luu best_mlp.pth vao run_dir (chi run nay) ──────────────────
        if global_f1 is not None and global_f1 > self.best_det_score:
            self.best_det_score = global_f1
            # Luu vao run_dir (moi run co file rieng, khong anh huong run khac)
            if run_dir is not None:
                os.makedirs(run_dir, exist_ok=True)
                run_path = os.path.join(run_dir, "best_mlp.pth")
                torch.save(self.detection_mlp.state_dict(), run_path)
                # QUAN TRONG: co-evolution train GAT chung voi head (freeze_gat_cl=False),
                # head duoc train tren latent cua GAT da co-evolve. Phai luu CA GAT de eval
                # dung cap (GAT+head) -> tranh mismatch latent-space khi danh gia.
                torch.save(self.gat.state_dict(), os.path.join(run_dir, "best_gat.pth"))
                print(f"[EWC] 🏆 New best Detector (GAT+MLP) -> {run_dir} (Global F1={global_f1:.4f})")

        return global_f1

    # ─────────────────────────────────────────────────────────────────
    # Evaluate DetectionMLP tren tap test doc lap sau moi EWC update
    # ─────────────────────────────────────────────────────────────────

    def evaluate_detection_mlp(self, run_dir: str = None):
        """
        Danh gia DetectionMLP sau moi lan EWC update tren tap test doc lap.

        Test set xay dung tu:
          - Malicious: latent vectors tu PrioritizedReplayBuffer (graph that tu Attack Agent)
          - Benign: latent vectors tinh tu UNICORN base graph qua GAT (graph that tu dataset)

        Metrics duoc tinh chinh xac tu inference (khong phai tu training batch):
          Accuracy, Precision, Recall, F1
        """
        self.detection_mlp.eval()

        eval_x, eval_y = [], []

        # ── 1. Malicious samples tu Replay Buffer ─────────────────────
        # Re-forward raw graph qua GAT HIEN TAI (khong dung latent cu da stale).
        if len(self.replay_buffer) > 0:
            mal_samples = self.replay_buffer.sample(min(16, len(self.replay_buffer)))
            self.gat.eval()
            with torch.no_grad():
                for s in mal_samples:
                    x_t = s["x_tensor"].to(device)
                    e_t = s["edge_index"].to(device)
                    if e_t.shape[1] == 0:
                        continue
                    latent = self.gat.graph_latent(x_t, e_t).cpu()   # attention pooling
                    eval_x.append(latent)
                    eval_y.append(1)  # Malicious

        # ── 2. Samples tu UNICORN dataset (150 do thi nhu train_mlp.py) ──
        # Day la test set chinh xac de dong bo voi pretrain
        uni_x, uni_y = self._get_reference_test_set()
        if len(uni_y) > 0:
            for i in range(len(uni_y)):
                eval_x.append(uni_x[i].unsqueeze(0))
                eval_y.append(uni_y[i].item())

        if len(eval_x) < 2:
            print("[Eval] Khong du du lieu de danh gia (can it nhat 2 mau).")
            return

        # ── 3. Inference ─────────────────────────────────────────────
        x_eval = torch.cat(eval_x, dim=0).to(device)
        y_eval = torch.tensor(eval_y, dtype=torch.long).to(device)

        with torch.no_grad():
            probs  = self.detection_mlp(x_eval)   # [N, 2] Softmax
            preds  = probs.argmax(dim=1)
            y_np   = y_eval.cpu().numpy()
            p_np   = preds.cpu().numpy()
            prob_np = probs[:, 1].cpu().numpy()   # xac suat class Malicious

        # ── 4. Confusion matrix & metrics ────────────────────────────
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
        TP = int(((p_np == 1) & (y_np == 1)).sum())
        FP = int(((p_np == 1) & (y_np == 0)).sum())
        FN = int(((p_np == 0) & (y_np == 1)).sum())
        TN = int(((p_np == 0) & (y_np == 0)).sum())

        n_total   = len(y_np)
        accuracy  = float(accuracy_score(y_np, p_np))
        precision = float(precision_score(y_np, p_np, zero_division=0))
        recall    = float(recall_score(y_np, p_np, zero_division=0))
        f1        = float(f1_score(y_np, p_np, zero_division=0))

        report = classification_report(y_np, p_np, labels=[0, 1], target_names=['Benign', 'Malicious'], zero_division=0)

        # Conf trung binh tren cac mau Malicious (do tin cay model)
        mal_idx  = (y_np == 1)
        avg_mal_conf = float(prob_np[mal_idx].mean()) if mal_idx.any() else 0.0
        ben_idx  = (y_np == 0)
        avg_ben_conf = float(prob_np[ben_idx].mean()) if ben_idx.any() else 0.0

        print(
            f"\n[Eval Post-EWC] === Detection MLP Evaluation (Real Test Set) ===\n"
            f"  Samples : {n_total} (mal={int(y_np.sum())}, ben={int((y_np==0).sum())})\n"
            f"  Accuracy : {accuracy:.4f} ({accuracy:.2%})\n"
            f"  Precision: {precision:.4f}\n"
            f"  Recall   : {recall:.4f}\n"
            f"  F1-score : {f1:.4f}\n"
            f"  Confusion: TP={TP} FP={FP} FN={FN} TN={TN}\n"
            f"  Avg conf (Malicious samples): {avg_mal_conf:.4f}\n"
            f"  Avg conf (Benign   samples) : {avg_ben_conf:.4f}\n"
            f"  Classification Report:\n"
            f"{report}"
        )

        # Luu ket qua danh gia vao file
        if run_dir is not None:
            eval_log = os.path.join(run_dir, "eval_metrics.csv")
            import csv, time
            write_header = not os.path.exists(eval_log)
            with open(eval_log, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        "timestamp", "n_total", "n_mal", "n_ben",
                        "accuracy", "precision", "recall", "f1",
                        "TP", "FP", "FN", "TN",
                        "avg_mal_conf", "avg_ben_conf"
                    ])
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    n_total, int(y_np.sum()), int((y_np == 0).sum()),
                    f"{accuracy:.4f}", f"{precision:.4f}",
                    f"{recall:.4f}", f"{f1:.4f}",
                    TP, FP, FN, TN,
                    f"{avg_mal_conf:.4f}", f"{avg_ben_conf:.4f}"
                ])
            print(f"  [Eval] Logged -> {eval_log}")
            
        return f1
    def _invalidate_reference_latent_cache(self):
        """
        Xoa cache latent reference (DARPA TC) sau khi GAT update (cache invalidation).

        Latent cache (_cached_reference_x) duoc tao boi GAT phien ban cu -> stale sau
        khi GAT thay doi. Raw graph cache (_cached_ref_raw) la input tho, KHONG
        bao gio stale -> giu lai de re-extract nhanh.
        """
        for attr in ("_cached_reference_x", "_cached_reference_y"):
            if hasattr(self, attr):
                delattr(self, attr)
        # Benign reference embeddings cung phu thuoc GAT -> recompute lazily
        self._benign_ref_emb = None

    def _get_reference_raw_graphs(self):
        """
        Doc RAW graph (x_tensor, edge_index, label) cho toan bo tap test UNICORN
        (0-149.txt) va cache lai. Raw input KHONG phu thuoc GAT -> cache mot lan,
        dung mai (khong can invalidate sau GAT update).

        - 0 -> 124 : Benign  (label=0)
        - 125 -> 149: Malicious (label=1)

        Returns:
            list[tuple(x_tensor[cpu], edge_index[cpu], label)]
        """
        if hasattr(self, '_cached_ref_raw'):
            return self._cached_ref_raw

        def _to_raw(edges_df, label, out):
            phrases, _, edge_idx, _ = prepare_graph(edges_df)
            if len(phrases) == 0:
                return
            x_t = torch.tensor(np.array([infer(p) for p in phrases]), dtype=torch.float32)
            e_t = torch.tensor(edge_idx, dtype=torch.long)
            if e_t.shape[1] == 0:
                return
            out.append((x_t, e_t, label))   # giu tren CPU

        raws = []
        # ── Nguon CHINH: DARPA TC E3 (benign + attack-localized malicious) ──
        try:
            from darpa_tc_sampler import build_darpa_corpus
            ben, mal = build_darpa_corpus(125, 125, dataset=DARPA_DATASET, seed=123)
            for b in ben:
                _to_raw(bundle_to_edges_df(b), 0, raws)
            for m in mal:
                _to_raw(bundle_to_edges_df(m), 1, raws)
            print(f"[Eval] DARPA TC/{DARPA_DATASET} reference: "
                  f"benign={sum(1 for r in raws if r[2]==0)}, malicious={sum(1 for r in raws if r[2]==1)}")
        except Exception as ex:
            print(f"[Eval] DARPA unavailable ({ex}). Fallback Benign Generator + Attack Agent JSON.")

        # ── Fallback benign: Benign Generator (neu DARPA chua co benign) ──
        if sum(1 for r in raws if r[2] == 0) == 0 and _BENIGN_GEN_AVAILABLE:
            try:
                for b in build_benign_corpus(125, source="hybrid", mimicry_level=0.5, seed=123):
                    _to_raw(bundle_to_edges_df(b), 0, raws)
            except Exception as ex:
                print(f"[Eval] Benign Generator fallback error: {ex}")

        # ── Fallback malicious: Attack Agent *_detection.json (neu DARPA chua co) ──
        if sum(1 for r in raws if r[2] == 1) == 0:
            import glob, json
            for f in sorted(glob.glob(os.path.join(base_dir, "Attack_Agent", "result_handoff", "*", "*_detection.json"))):
                try:
                    d = json.load(open(f, encoding="utf-8"))
                    _to_raw(bundle_to_edges_df({"nodes": d["nodes"], "edges": d["edges"]}), 1, raws)
                except Exception:
                    pass

        self._cached_ref_raw = raws
        n_ben = sum(1 for r in raws if r[2] == 0)
        print(f"[Eval] Reference graphs: benign={n_ben}, malicious={len(raws)-n_ben}.")
        return self._cached_ref_raw

    def _get_reference_test_set(self):
        """
        Trich xuat latent (GAT embeddings) cho toan bo tap test UNICORN.

        Latent duoc tinh tu GAT HIEN TAI tren raw graph cache. Cache lai latent
        (_cached_reference_x) cho cac lan goi trong cung 1 "phien ban GAT"; cache nay
        bi xoa boi _invalidate_reference_latent_cache() sau moi GAT update.
        """
        if hasattr(self, '_cached_reference_x'):
            return self._cached_reference_x, self._cached_reference_y

        raws  = self._get_reference_raw_graphs()
        eval_x, eval_y = [], []
        self.gat.eval()
        with torch.no_grad():
            for x_t, e_t, label in raws:
                latent = self.gat.graph_latent(x_t.to(device), e_t.to(device)).cpu()  # attention pooling
                eval_x.append(latent)
                eval_y.append(label)

        if len(eval_x) == 0:
            self._cached_reference_x = torch.zeros((0, 20))
            self._cached_reference_y = torch.zeros((0,), dtype=torch.long)
        else:
            self._cached_reference_x = torch.cat(eval_x, dim=0)
            self._cached_reference_y = torch.tensor(eval_y, dtype=torch.long)

        print(f"[Eval] Re-extract {len(eval_y)} latent UNICORN qua GAT hien tai (Benign={int((self._cached_reference_y==0).sum())}, Attack={int((self._cached_reference_y==1).sum())}).")
        return self._cached_reference_x, self._cached_reference_y

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
            "TP":                 self.last_TP,           # fix: main.py can TP de tinh fn_rate
            "fn_rate_win":        self.last_fn_rate_win,  # FN_rate windowed (cho StopMonitor)
            "scf_impact":         getattr(self, "last_scf_impact", 0.0),
            "replay_buf_size":    len(self.replay_buffer),
            "mab_stats":          self.mab.get_stats(),
            "gen_reward_total":   float(self.last_gen_reward.get("total", 0.0)),
            "closed_loop_reward": getattr(self, "last_closed_loop_reward", {}),
            "drift":              self.last_drift,        # Fix 4: Wasserstein drift
            "drift_triggered":    self.drift_triggered,
            "compression_ratio":  self.last_compression_ratio,  # Fix 2: Gap 2 metric
            "det_latency":        self.last_det_latency,
            "camo_ratio":         self.last_camo_ratio,         # Huong 1: evasion camo
            "camo_stats":         self.camo_bandit.get_stats(),
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
