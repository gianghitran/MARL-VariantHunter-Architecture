"""
networks.py
===========
Network definitions cho MARL-VariantHunter.

Kiến trúc:
  - GAT  : Graph Attention Network (thay thế GCN) — Detection Agent backbone.
            Trả về node embeddings + attention weights để giải thích.
  - DetectionMLP : MLP phân loại + EWC Continual Learning.
  - CoordinatorNetwork : PPO policy/value head.
  - PositionalEncoder  : Sinusoidal positional encoding cho node features.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv


# ─────────────────────────────────────────────────────────────────────────────
# Original Graph Convolutional Network (GCN) — SAGEConv
# ─────────────────────────────────────────────────────────────────────────────

class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, out_channels: int):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# ─────────────────────────────────────────────────────────────────────────────
# Graph Attention Network (GATs)
# ─────────────────────────────────────────────────────────────────────────────

class GAT(torch.nn.Module):
    """
    Graph Attention Network (GATs) — Detection Agent backbone.

    Kiến trúc 2 lớp:
      conv1: in_channels → hidden_dim (8 heads × 8 dim = 64)
      conv2: hidden_dim  → out_channels (1 head)

    Ưu điểm so với GCN (SAGEConv):
      - Attention mechanism học tự động để chú ý vào các edge bất thường.
      - Trả về attention_weights (alpha) cho mỗi edge — hỗ trợ explainability.
      - Phù hợp hơn với provenance graph dạng thưa (sparse), không đồng nhất.

    Tham khảo:
      Veličković et al., 2018 — "Graph Attention Networks"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int = 64,
        heads: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        # Lớp 1: Multi-head attention (heads × head_dim)
        self.conv1 = GATConv(
            in_channels,
            hidden_dim // heads,   # mỗi head output = hidden_dim / heads
            heads=heads,
            dropout=dropout,
            concat=True,           # concat → output: heads * (hidden_dim/heads) = hidden_dim
        )

        # Lớp 2: Single-head attention → out_channels
        self.conv2 = GATConv(
            hidden_dim,
            out_channels,
            heads=1,
            dropout=dropout,
            concat=False,          # out shape: [N, out_channels]
        )

        # Batch Norm sau conv1 để ổn định training
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # Top-k attention masking (chống camouflage dilution): pooling CHỈ trên các
        # node nhận attention cao nhất → loại benign-padding (node ngoại vi attention
        # thấp do attacker chèn quanh APT). Tấn công gốc camo thay vì soft-weight (vẫn
        # bị padding pha loãng nhẹ). topk_ratio = phần node giữ lại; topk_min = sàn để
        # đồ thị nhỏ không bị cắt còn quá ít node.
        self.topk_ratio = 0.5
        self.topk_min   = 12

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        Forward pass không trả về attention weights (inference nhanh).

        Args:
            x           : Node feature matrix [N, in_channels]
            edge_index  : Edge list [2, E]

        Returns:
            out : Node embeddings [N, out_channels]
        """
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        if x.size(0) > 1:          # BatchNorm cần ít nhất 2 samples
            x = self.bn1(x)
        x = F.elu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def forward_with_attention(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        Forward pass trả về kèm attention weights từ lớp conv2.
        Dùng để tính Gradient-weighted Attention Score và Confidence Score.

        Returns:
            out              : Node embeddings [N, out_channels]
            attention_weights: Tensor attention [E, 1] từ lớp 2
        """
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        if x.size(0) > 1:
            x = self.bn1(x)
        x = F.elu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x, (edge_index_out, alpha) = self.conv2(
            x, edge_index, return_attention_weights=True
        )
        return x, alpha   # alpha: [E, 1]

    def graph_latent(self, x: torch.Tensor, edge_index: torch.Tensor,
                     return_nodes: bool = False):
        """
        Graph-level latent bằng TOP-K ATTENTION-MASKED POOLING (chống camo dilution).

        Trọng số mỗi node = tổng attention mà node NHẬN ĐƯỢC (các node khác chú ý tới
        nó). Khác với soft attention-weighting (mọi node đều góp, benign-padding vẫn
        pha loãng nhẹ), ở đây ta CHỈ giữ top-k node attention cao nhất rồi pool trong
        số chúng → benign-padding (attention thấp) bị LOẠI HẲN trước MLP. Đây là cách
        tấn công gốc rễ camouflage: kẻ tấn công chèn benign quanh APT-core, nhưng các
        node đó cấu trúc ngoại vi → attention thấp → không lọt top-k.

        Returns:
            pooled [1, out_channels]  (và (node_emb, alpha) nếu return_nodes=True)
        """
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = self.conv1(h, edge_index)
        if h.size(0) > 1:
            h = self.bn1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out, (ei_out, alpha) = self.conv2(h, edge_index, return_attention_weights=True)

        a   = alpha.mean(dim=-1) if alpha.dim() > 1 else alpha   # [E']
        src = ei_out[0]                                          # node được chú ý tới
        N   = out.size(0)
        w   = torch.zeros(N, device=out.device, dtype=out.dtype)
        w   = w.index_add(0, src, a)                            # attention nhận được / node

        if float(w.sum()) <= 0:
            pooled = out.mean(dim=0, keepdim=True)              # fallback (đồ thị rỗng cạnh)
        else:
            # ── Top-k masking: giữ k node attention cao nhất, loại phần còn lại ──
            k = max(self.topk_min, int(math.ceil(self.topk_ratio * N)))
            k = min(k, N)
            if k < N:
                topv, topi = torch.topk(w, k)                  # k node trội nhất
                mask = torch.zeros_like(w)
                mask[topi] = w[topi]                           # zero-out phần đuôi
                w = mask
            wn     = (w / w.sum()).unsqueeze(-1)               # [N, 1] tổng = 1 (chỉ top-k > 0)
            pooled = (out * wn).sum(dim=0, keepdim=True)       # [1, out_channels]

        if return_nodes:
            return pooled, out, alpha
        return pooled


# ─────────────────────────────────────────────────────────────────────────────
# Positional Encoder
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoder:
    """Sinusoidal positional encoding cho node features (cùng kích thước d_model)."""

    def __init__(self, d_model: int, max_len: int = 100000):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        self.pe = torch.zeros(max_len, d_model)
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(0)]


# ─────────────────────────────────────────────────────────────────────────────
# PPO Coordinator Network
# ─────────────────────────────────────────────────────────────────────────────

class CoordinatorNetwork(nn.Module):
    """
    High-level PPO policy network (Coordinator).

    Input : Graph State Representation [state_dim]
    Output:
      - action_logits : [num_actions] — chọn Agent nào chạy tiếp
      - state_value   : [1]           — V(s) cho PPO advantage
    """

    def __init__(self, state_dim: int = 128, num_actions: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.action_head = nn.Linear(128, num_actions)
        self.value_head  = nn.Linear(128, 1)

    def forward(self, state: torch.Tensor):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.action_head(x), self.value_head(x)


# ─────────────────────────────────────────────────────────────────────────────
# Detection MLP với EWC Continual Learning
# ─────────────────────────────────────────────────────────────────────────────

class DetectionMLP(nn.Module):
    """
    DetectionMLP phân loại graph latent → Benign / Malicious.

    Bổ sung Elastic Weight Consolidation (EWC) để tránh Catastrophic Forgetting
    khi học liên tục từ các Hard Adversarial Samples mới.

    Input  : graph_latent [B, input_dim] (mean-pooled từ GAT output)
    Output : probabilities [B, 2] — Softmax (0: Benign, 1: Malicious)

    Ref: Kirkpatrick et al., 2017 — "Overcoming catastrophic forgetting in neural networks"
    """

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dim: int = 32,
        ewc_lambda: float = 8.0,
        hidden_dims: list = None,
    ):
        super().__init__()
        # hidden_dims tong quat hoa kien truc classifier (cho RQ3 architecture swap):
        #   None / [hidden_dim] -> MLP goc (mlp.pth): net.0=Linear(in,h), net.1=ReLU, net.2=Linear(h,2)
        #   []                  -> Linear classifier:   net.0=Linear(in,2)
        #   [64, 32]            -> Deep MLP:             net.0..net.4
        # Dropout KHONG nam trong Sequential (tranh shift index -> key mismatch voi checkpoint).
        if hidden_dims is None:
            hidden_dims = [hidden_dim]
        self.hidden_dims = list(hidden_dims)
        layers = []
        prev = input_dim
        for h in self.hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 2))     # head -> 2 logits
        self.net = nn.Sequential(*layers)
        self.dropout = nn.Dropout(0.2)
        self.ewc_lambda = ewc_lambda
        self.fisher_dict: dict = {}     # param_name -> Fisher diagonal tensor
        self.optimal_params: dict = {}  # param_name -> W* tensor (anchor snapshot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Chay het cac layer tru head, ap dropout ngay truoc head (giu nguyen hanh vi
        # MLP goc: Linear -> ReLU -> Dropout -> Linear), roi softmax.
        for layer in self.net[:-1]:
            x = layer(x)
        x = self.dropout(x)
        x = self.net[-1](x)         # head -> 2 logits
        return F.softmax(x, dim=-1)

    def compute_fisher(
        self,
        anchor_data_loader,
        device: torch.device,
        num_samples: int = 200,
    ):
        """
        Tính Fisher Information matrix (diagonal) trên tập anchor.

        Args:
            anchor_data_loader : iterable của (x_tensor, y_tensor)
            device             : torch.device
            num_samples        : số mẫu tối đa
        """
        self.train()
        # Snapshot trọng số hiện tại làm W* (anchor sau task cũ)
        self.optimal_params = {
            name: param.clone().detach()
            for name, param in self.named_parameters()
            if param.requires_grad
        }
        fisher = {
            name: torch.zeros_like(param)
            for name, param in self.named_parameters()
            if param.requires_grad
        }
        count = 0

        for x_batch, y_batch in anchor_data_loader:
            if count >= num_samples:
                break
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            self.zero_grad()
            # forward() tra ve SOFTMAX probs. Truoc day dung nn.CrossEntropyLoss(probs)
            # = log_softmax(softmax(logits)) -> double softmax -> gradient bi det
            # -> Fisher ~ 0 -> EWC penalty = 0 (vo nghia). Sua: dung NLL tren log-probs,
            # KHOP chinh xac voi training loss F.nll_loss(log(probs), y).
            output   = self.forward(x_batch)             # softmax probs [B, 2]
            log_prob = torch.log(output + 1e-9)
            loss     = F.nll_loss(log_prob, y_batch)
            loss.backward()
            for name, param in self.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.detach() ** 2
            count += x_batch.size(0)

        n = max(count, 1)
        self.fisher_dict = {name: f / n for name, f in fisher.items()}
        # Diagnostic: tong norm Fisher. Neu ~0 -> softmax bao hoa / anchor vo nghia.
        total_norm = sum(f.norm().item() for f in self.fisher_dict.values())
        self.eval()
        print(f"[EWC] Computed Fisher Information on {count} anchor samples "
              f"| total Fisher norm={total_norm:.6e}")

    def ewc_loss(self) -> torch.Tensor:
        """
        EWC regularization loss:
            L_EWC = λ/2 * Σ_i F_i * (θ_i − θ*_i)²
        """
        if not self.fisher_dict or not self.optimal_params:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0)
        for name, param in self.named_parameters():
            if param.requires_grad and name in self.fisher_dict:
                fisher    = self.fisher_dict[name].to(param.device)
                opt_param = self.optimal_params[name].to(param.device)
                loss = loss + (fisher * (param - opt_param) ** 2).sum()

        return (self.ewc_lambda / 2.0) * loss


# ─────────────────────────────────────────────────────────────────────────────
# Classifier factory (RQ3 architecture swap)
# ─────────────────────────────────────────────────────────────────────────────

# Kien truc classifier theo ten -> hidden_dims. Tat ca deu la DetectionMLP nen
# dung chung EWC (compute_fisher / ewc_loss) khong can sua gi.
CLASSIFIER_ARCHS = {
    "mlp":    [32],        # A0 — kien truc goc (khop mlp.pth)
    "linear": [],          # A1 — Linear(input_dim, 2)
    "deep":   [64, 32],    # A3 — MLP sau hon
}


def make_classifier(name: str = "mlp", input_dim: int = 20,
                    ewc_lambda: float = 8.0) -> DetectionMLP:
    """
    Tao DetectionMLP voi kien truc head tuong ung ten (cho RQ3).
      name in {"mlp", "linear", "deep"}; mac dinh "mlp" (khop checkpoint goc).
    """
    key = (name or "mlp").lower()
    if key not in CLASSIFIER_ARCHS:
        raise ValueError(
            f"Unknown classifier '{name}'. Choices: {list(CLASSIFIER_ARCHS)}"
        )
    return DetectionMLP(
        input_dim=input_dim,
        hidden_dims=CLASSIFIER_ARCHS[key],
        ewc_lambda=ewc_lambda,
    )
