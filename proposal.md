Ý tưởng: ĐACN/KLTN

**Ý tưởng giải quyết (Proposed Approach) - Đề tài tập trung**

Đề tài sẽ phát triển **RL-VariantHunter**: framework **Multi-Agent Reinforcement Learning (MARL)** đầu tiên trên provenance graph, tạo closed-loop liên tục giữa proactive threat variant generation, adaptive graph pruning và streaming detection. Đây là hướng RL-centric thuần provenance, kết hợp đầy đủ bốn công trình (TAGAPT + SLOT + THREATIFY + risk-based reward) mà không phụ thuộc MulVAL.

**Kiến trúc mô hình đề xuất: Multi-Agent Reinforcement Learning (MARL) - Cooperative 3-Agent: Multi-Agent RL** thay vì single-agent hierarchical vì:

- Các sub-task (generation, pruning, detection) có mục tiêu riêng biệt nhưng cần phối hợp (cooperative reward) → MARL cho phép parallel training, policy specialization, và scalability tốt hơn trên provenance graph lớn.
- Novelty cao: chưa có công trình MARL nào trên provenance APT (tính đến Q1/2026).
- Dễ ablation.

Cấu trúc 3 agents cooperative trên provenance graph:

- **Generation Agent** (dựa GraphAF + SCF từ THREATIFY)
  - State: provenance subgraph hiện tại + drift score (Wasserstein trên embedding).
  - Action: sinh aligned/evolved/composite variant (AAG → AASG → IAG).
  - Reward: VariantQuality (similarity/uniqueness/validity) + SLOT detection confidence trên IAG giả lập.
- **Pruning Agent**
  - State: sliding window provenance graph (10k-50k nodes).
  - Action: chọn node/edge quan trọng (high temporal burst + high SCF impact) để coarsening (Metis + GraphSAGE pooling).
  - Reward: CompressionRatio × (1 - loss causal relation).
- **Detection Agent** (mở rộng SLOT)
  - State: provenance subgraph đã prune + latent behavior mining.
  - Action: Bernoulli MAB neighbor selection + MLP + iForest classification.
  - Reward: (Precision + Recall - λ·FN) + risk reduction (derive từ provenance SCF impact).

**Coordinator (High-level Policy)**: Dùng PPO để điều phối 3 agents (action = chọn agent nào chạy tiếp theo).

**Multi-objective cooperative reward (chung cho 3 agents)**:

_có thể tối ưu trên các tiêu chí_

(λ = 2.0 phạt FN nặng; continual learning bằng EWC + prioritized replay buffer trên provenance subgraph).

**Cách giải quyết 3 gaps trên provenance graph**:

- Drift → Generation Agent được trigger khi Wasserstein > τ, sinh variant provenance-level mới.
- Explosion → Pruning Agent học dynamic compression → giảm 70-80% kích thước, inference < 15 phút/batch.
- FN cao → Closed-loop: Detection Agent báo cáo hard samples → Generation Agent sinh adversarial mimicry → augment training set → FN giảm 30-40% dự kiến.