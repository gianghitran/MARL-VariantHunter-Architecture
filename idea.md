Ý tưởng: ĐACN/KLTN

Tên gợi ý: **MARL-VariantHunter: Multi-Agent Reinforcement Learning for Adaptive Provenance Graph-Based APT Variant Generation and Detection**

Tên đăng kí ĐACN:

·       Anh: **Multi-Agent Reinforcement Learning Framework for Adaptive Provenance Graph-Based APT Variant Generation and Detection**

**** 

·       Việt: **Bộ Khung Học Tăng Cường Đa Tác Tử cho Việc Sinh Biến Thể APT và Phát Hiện Thích Ứng Dựa trên Đồ Thị Truy Nguyên**

 

**Vấn đề nghiên cứu (Problem Gaps) – Tập trung hoàn toàn vào Provenance Graph**

Mặc dù provenance graph là đại diện mạnh nhất cho hành vi hệ thống thời gian thực (entities: process/file/socket + temporal events), các công trình tiên tiến 2025 vẫn tồn tại **ba khoảng trống khoa học lớn chưa được giải quyết đồng thời**:

1.     **Concept drift trong provenance graph**: Kỹ thuật tấn công mới (LOTL 2.0, fileless malware, AI-generated TTP, supply-chain provenance mimicry) làm thay đổi cấu trúc temporal-causal của provenance graph. TAGAPT (GraphAF) và SLOT (latent behavior mining + Bernoulli MAB) chỉ học offline trên provenance graphs tĩnh → mô hình nhanh chóng lỗi thời, dẫn đến false negative (FN) cao khi gặp zero-day TTP chưa từng xuất hiện trong training distribution.

2.     **Bùng nổ kích thước provenance graph theo thời gian (temporal explosion)**: Provenance log tăng tuyến tính (hàng GB/ngày ở môi trường doanh nghiệp/ICS/telecom), khiến provenance graph chứa hàng triệu node/edge. SLOT và TAGAPT không có cơ chế adaptive pruning hoặc hierarchical summarization → memory overflow và inference latency không thể chấp nhận được trong streaming setting.

3.     **Thiếu closed-loop giữa generation và detection trên provenance graph**:

a.      TAGAPT sinh provenance-level IAG xuất sắc nhưng tĩnh và không biết “cái gì cần sinh” để giảm FN.

b.     SLOT đạt \~99% accuracy trên provenance graph tĩnh nhưng không tự sinh hard adversarial samples (mimic benign behaviors như scenario Firefox + DNS camouflage) để bổ sung training set.

c.      THREATIFY cung cấp Security Context Feature (SCF) và variant prediction rất mạnh, nhưng chưa áp dụng trên provenance graph thực tế (chỉ abstract graph).

d.     Automated APT Defense dùng RL + risk-based reward nhưng trên MulVAL attack graph, không phải provenance → mất mất semantic temporal-causal chi tiết.

Kết quả: chưa có framework nào xây dựng **closed-loop RL trên provenance graph** để đồng thời (i) sinh variant provenance-level theo drift, (ii) prune graph động, và (iii) giảm FN qua adversarial augmentation.

**Ý tưởng giải quyết (Proposed Approach) – Đề tài tập trung**

Đề tài sẽ phát triển **RL-VariantHunter**: framework **Multi-Agent Reinforcement Learning (MARL)** đầu tiên trên provenance graph, tạo closed-loop liên tục giữa proactive threat variant generation, adaptive graph pruning và streaming detection. Đây là hướng RL-centric thuần provenance, kết hợp đầy đủ bốn công trình (TAGAPT + SLOT + THREATIFY + risk-based reward) mà không phụ thuộc MulVAL.

**Kiến trúc mô hình đề xuất: Multi-Agent Reinforcement Learning (MARL) – Cooperative 3-Agent: Multi-Agent RL** thay vì single-agent hierarchical vì:

·       Các sub-task (generation, pruning, detection) có mục tiêu riêng biệt nhưng cần phối hợp (cooperative reward) → MARL cho phép parallel training, policy specialization, và scalability tốt hơn trên provenance graph lớn.

·       Novelty cao: chưa có công trình MARL nào trên provenance APT (tính đến Q1/2026).

·       Dễ ablation.

Cấu trúc 3 agents cooperative trên provenance graph:

1.     **Generation Agent** (dựa GraphAF + SCF từ THREATIFY)

a.      State: provenance subgraph hiện tại + drift score (Wasserstein trên embedding).

b.     Action: sinh aligned/evolved/composite variant (AAG → AASG → IAG).

c.      Reward: VariantQuality (similarity/uniqueness/validity) + SLOT detection confidence trên IAG giả lập.

**2.     Pruning Agent**

a.      State: sliding window provenance graph (10k–50k nodes).

b.     Action: chọn node/edge quan trọng (high temporal burst + high SCF impact) để coarsening (Metis + GraphSAGE pooling).

c.      Reward: CompressionRatio × (1 – loss causal relation).

3.     **Detection Agent** (mở rộng SLOT)

a.      State: provenance subgraph đã prune + latent behavior mining.

b.     Action: Bernoulli MAB neighbor selection + MLP + iForest classification.

c.      Reward: (Precision + Recall – λ·FN) + risk reduction (derive từ provenance SCF impact).

**Coordinator (High-level Policy)**: Dùng PPO để điều phối 3 agents (action \= chọn agent nào chạy tiếp theo).

**Multi-objective cooperative reward (chung cho 3 agents)**:

*có thể tối ưu trên các tiêu chí*

(λ \= 2.0 phạt FN nặng; continual learning bằng EWC + prioritized replay buffer trên provenance subgraph).

**Cách giải quyết 3 gaps trên provenance graph**:

·       Drift → Generation Agent được trigger khi Wasserstein > τ, sinh variant provenance-level mới.

·       Explosion → Pruning Agent học dynamic compression → giảm 70–80% kích thước, inference \< 15 phút/batch.

·       FN cao → Closed-loop: Detection Agent báo cáo hard samples → Generation Agent sinh adversarial mimicry → augment training set → FN giảm 30–40% dự kiến.