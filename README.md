# MARL-VariantHunter: LLM-ProvenanceGraph APT Detection

> **Multi-Agent Reinforcement Learning Framework for Adaptive Provenance Graph-Based APT Variant Generation and Detection**
>
> Bộ Khung Học Tăng Cường Đa Tác Tử cho Việc Sinh Biến Thể APT và Phát Hiện Thích Ứng Dựa trên Đồ Thị Truy Nguyên

---

## Mục lục
1. [Tổng quan Hệ thống](#1-tổng-quan-hệ-thống)
2. [Kiến trúc MARL](#2-kiến-trúc-marl)
3. [Các Tác tử (Agents)](#3-các-tác-tử-agents)
4. [Detection Agent — Cập nhật Thích ứng](#4-detection-agent--cập-nhật-thích-ứng)
5. [Hàm Reward](#5-hàm-reward)
6. [Cơ chế Dừng Sớm](#6-cơ-chế-dừng-sớm)
7. [Cấu trúc Source Code](#7-cấu-trúc-source-code)
8. [Cài đặt & Chạy](#8-cài-đặt--chạy)
9. [So sánh Thực tế vs idea.md](#9-so-sánh-thực-tế-vs-ideamd)

---

## 1. Tổng quan Hệ thống

Hệ thống giải quyết **ba khoảng trống khoa học** cốt lõi trong bài toán phát hiện APT trên Provenance Graph:

| Gap | Vấn đề | Giải pháp trong Framework |
|-----|---------|--------------------------|
| **Gap 1 — Concept Drift** | Kỹ thuật tấn công mới (zero-day TTP) làm mô hình nhanh lỗi thời, dẫn đến FN cao | **EWC Continual Learning** + **Prioritized Replay Buffer** trong Detection Agent |
| **Gap 2 — Temporal Explosion** | Provenance log tăng tuyến tính (hàng GB/ngày), gây memory overflow | **Pruning Agent** với PageRank-based compression (~30%) |
| **Gap 3 — Thiếu Closed-loop** | Không có vòng phản hồi giữa generation và detection | **PPO Coordinator** điều phối 3 agents — Generation → Pruning → Detection → lặp |

---

## 2. Kiến trúc MARL

```
┌─────────────────────────────────────────────────────────────┐
│                   PPO Coordinator (main.py)                  │
│         CoordinatorNetwork(state_dim=128, num_actions=3)     │
│         Chọn action = {0: Generation, 1: Pruning, 2: Det.}  │
└───────────────┬──────────────┬──────────────┬───────────────┘
                │              │              │
         Action=0        Action=1        Action=2
                │              │              │
    ┌───────────▼──┐   ┌───────▼──────┐  ┌───▼──────────────┐
    │  Generation  │   │   Pruning    │  │    Detection      │
    │    Agent     │   │    Agent     │  │     Agent         │
    │  (LLM/Ollama)│   │ (PageRank)   │  │ Thompson MAB +   │
    │              │   │              │  │ GCN + MLP (EWC)  │
    └──────────────┘   └──────────────┘  └──────────────────┘
                                                  │
                              ┌───────────────────▼────────┐
                              │   Prioritized Replay Buffer │
                              │   → train_detection_agent   │
                              │     _ewc()  (offline batch) │
                              └────────────────────────────┘
```

**Luồng tổng quát mỗi Episode:**
1. Load base provenance graph (`2.txt` → `current_graph_df`).
2. Encode graph state thành vector 128 chiều → đưa vào `CoordinatorNetwork`.
3. PPO chọn 1 trong 3 actions → gọi agent tương ứng → nhận reward.
4. Thu thập trajectory → tính Advantage → PPO update.
5. **Sau mỗi PPO update:** Gọi `env.train_detection_agent_ewc()` — Offline Continual Learning.
6. Lưu artefacts (đồ thị, best weights) vào `runs/<timestamp>/`.

---

## 3. Các Tác tử (Agents)

### Tổng hợp Cơ chế State - Action - Reward

| Agent | Trạng thái (State) | Hành động (Action) | Điểm thưởng (Reward) |
|---|---|---|---|
| **Generation Agent** | Đồ thị chuỗi thực thể hiện tại (Text/JSON format) | Sinh Node/Edge giả lập tấn công APT mới qua LLM | Lấy điểm rủi ro từ Detection / Phạt (-0.5) nếu sai JSON format |
| **Pruning Agent** | Đồ thị luồng hiện tại (>50 edges) | Định tuyến, cắt tỉa nhánh nhiễu qua thuật toán PageRank | Tỷ lệ nén (Compression Ratio) + 0.1 |
| **Detection Agent** | Đồ thị con đã được lấy mẫu (MAB Subgraph) | Chọn tỷ lệ mẫu (Thompson MAB) + Phân loại đồ thị (GCN+MLP/iForest) | `(Precision + Recall - λ*FN) + SCF_Impact` |

### A. Generation Agent (Action = 0)

**Mục đích:** Mô phỏng kẻ tấn công APT — sinh ra biến thể hành vi tấn công mới trên provenance graph.

**Workflow:**
1. **Serialize Graph:** Toàn bộ nodes và edges được định chỉ mục (0-indexed) và nối thành chuỗi text.
2. **Prompt LLM:** Gọi Ollama (`llm_config.py`) với prompt chứa context đồ thị + instruction chuẩn format APT.
3. **Parse JSON Output:** LLM trả về JSON `{"new_nodes": [...], "new_edges": ["src dst action"]}`.
4. **Inject Graph:** Ánh xạ index → tên node thực → `concat` vào `current_graph_df`.
5. **Visualize & Save:** Vẽ Graphviz PDF (`runs/<timestamp>/graph_visualize/`) + lưu TXT (`runs/<timestamp>/generated_graphs/`).

**Node Naming Convention trong Prompt:**

| Prefix | Ý nghĩa |
|--------|---------|
| `MP` / `MP*` | Malicious Process (root/hub nếu có `*`) |
| `MF` | Malicious File |
| `SF` | System / Sensitive File |
| `TP` | Target Process |
| `SO` | Socket / IP Address |

**Edge Actions:** `WR` (Write), `RD` (Read), `EX` (Execute), `ST` (SendTo), `RF` (ReceiveFrom), `IJ` (Inject), `FR` (Fork)

**Reward:**
- `-0.5` nếu LLM sinh sai format / không parse được JSON.
- Nếu thành công → reward = output của `_detection_agent()` (Detection chấm điểm độ khó của biến thể vừa sinh).

---

### B. Pruning Agent (Action = 1)

**Mục đích:** Kiểm soát kích thước graph — tránh memory overflow khi streaming data tích lũy theo thời gian.

**Trigger:** Chỉ hoạt động khi `len(current_graph_df) > 50`.

**Workflow (`Pruning_Agent/pruning_agent.py`):**
1. Chuyển DataFrame → `entity_list` (nodes) + `relation_list` (edges theo stage).
2. **PageRank-based scoring:** Đánh giá centrality từng node — giữ lại Hubs nguy hiểm, loại bỏ trung bình 30% nhánh nhiễu. Target = `max(50, 70% × original_nodes)`.
3. Dựng lại DataFrame từ `pruned_relations` → thay thế `current_graph_df`.
4. Fallback: Nếu `PruningAgent` lỗi → xóa cứng dòng đầu nếu vượt 1000 edges.

**Reward:** `Compression_Ratio + 0.1` (tỷ lệ nén càng cao → điểm càng cao). `-0.1` nếu cắt hỏng (kết quả rỗng).

---

### C. Detection Agent (Action = 2)

Xem chi tiết tại **[Mục 4](#4-detection-agent--cập-nhật-thích-ứng)**.

---

## 4. Detection Agent — Cập nhật Thích ứng

Detection Agent là thành phần cốt lõi và phức tạp nhất, được nâng cấp toàn diện để giải quyết **Gap 1 (Concept Drift)** và **Gap 2 (Streaming)**.

### 4.1 Thompson Sampling MAB (Online Streaming Update)

Thay thế `BernoulliMAB` tĩnh (fixed 80%) bằng `ThompsonSamplingMAB` với **3 arms** có tỷ lệ sampling khác nhau:

| Arm | Sampling Rate | Khi nào được ưu tiên |
|-----|:---:|---|
| 0 | 60% | Tiết kiệm tài nguyên — streaming constraint cao |
| 1 | 80% | Cân bằng — mặc định ban đầu |
| 2 | 95% | Toàn diện — khi anomaly ratio cao |

**Cơ chế Thompson Sampling:**
- Mỗi arm duy trì phân phối `Beta(α, β)` — `α` = số lần thành công, `β` = số lần thất bại.
- Tại mỗi bước: sample `θ_k ~ Beta(α_k, β_k)` cho cả 3 arms → chọn arm có `θ_k` lớn nhất.
- **Cập nhật sau detection:**
  - `reward_positive = True` → `α[arm] += 1` (TP được ghi nhận hoặc anomaly_ratio > 0.3)
  - `reward_positive = False` → `β[arm] += 1` (FN xảy ra)

### 4.2 Pipeline Phát hiện (mỗi Step)

```
mab_subgraph (subset of current_graph_df)
       │
       ▼
Word2Vec + PositionalEncoder   ← graph_utils.infer()
       │
       ▼
GCN(SAGEConv 30→32→20)         ← networks.GCN (frozen weights: unicorn0.pth)
       │
       ▼
graph_latent = mean_pool(latent_features)   [shape: 1×20]
       │
       ├──► DetectionMLP(20→32→2) + Softmax  →  mlp_attack_prob
       │         └── EWC-capable (compute_fisher / ewc_loss)
       │
       └──► IsolationForest (nếu mlp_attack_prob > 0.5) → anomaly_ratio
```

**Quyết định:**
- `Malicious` = `mlp_attack_prob > 0.5` **AND** `anomaly_ratio > 0.3`
- `Benign` = ngược lại
- **Ground Truth heuristic:** `total_edges > 15` → Malicious (1), ngược lại Benign (0)

### 4.3 Prioritized Replay Buffer (Offline Hard Sample Storage)

Khi `FN = 1` (Detection bỏ lọt tấn công), hard sample được thêm vào `PrioritizedReplayBuffer`:

```python
priority = 1.0 - |mlp_attack_prob - 0.5| × 2.0   # ∈ [0, 1]
# Càng gần boundary 0.5 → càng khó → priority càng cao
```

- **Capacity:** 500 mẫu (FIFO theo priority thấp nhất).
- **Sampling:** Lấy mẫu theo phân phối xác suất tỷ lệ với priority — không uniform random.
- **Cấu trúc mỗi mẫu:** `{'latent': Tensor[1×20], 'label': int, 'mlp_prob': float, 'anomaly_ratio': float}`

### 4.4 EWC Continual Learning (Offline Batch Update)

Sau mỗi PPO update episode, `main.py` gọi `env.train_detection_agent_ewc(run_dir, batch_size=8)`:

```
1. Sample batch từ PrioritizedReplayBuffer (weighted by priority)
2. Lần đầu tiên: compute_fisher(anchor_data) → lưu F_i và W*_i
3. Forward pass: outputs = DetectionMLP(x_batch)
4. Loss = NLLLoss(log(outputs), y_batch) + EWC_penalty
         EWC_penalty = λ/2 × Σ F_i × (θ_i − θ*_i)²   [λ=400.0]
5. Backward + Adam step (lr=1e-4)
6. Nếu accuracy > best_det_score → lưu best_mlp.pth vào run_dir/
7. Recompute Fisher cho EWC lần sau
```

**Ý nghĩa EWC:** Trọng số quan trọng với task cũ (Fisher cao) sẽ bị "đinh chặt" (`F_i` lớn → penalty lớn nếu thay đổi), trong khi trọng số ít quan trọng có thể thích nghi tự do với hard samples mới.

---

## 5. Hàm Reward

$$R = (Precision + Recall - \lambda_{FN} \times FN) + \text{scf\_impact}$$

- $\lambda_{FN} = 2.0$ — Phạt nặng False Negative (bỏ lọt APT = rủi ro cao nhất)
- $Precision = \frac{TP}{TP + FP + \varepsilon}$
- $Recall = \frac{TP}{TP + FN + \varepsilon}$

**Security Context Feature (SCF) Impact** — cộng thêm khi Generation Agent sinh hành vi nguy hiểm:

| Action | Risk Score |
|--------|:----------:|
| `execute` | 1.0 |
| `clone` | 0.9 |
| `mmap` | 0.8 |
| `write` | 0.6 |
| `read` | 0.2 |
| *khác* | 0.1 |

`scf_impact = mean(risk_scores của tất cả edges trong mab_subgraph)`

---

## 6. Cơ chế Dừng Sớm

Hệ thống có **hai lớp** kiểm soát vòng lặp training:

**Lớp 1 — Episode-level Peak Limit:**
Nếu tại bất kỳ step nào: `mlp_attack_prob ≥ 0.95 AND precision = 1.0 AND recall = 1.0` → trigger `done = True` ngay lập tức (không đợi đến `max_steps_per_episode = 15`).

**Lớp 2 — Global EarlyStopping:**
```python
EarlyStopping(patience=5, min_delta=0.01)
```
Nếu `avg_mlp_prob` không cải thiện trong 5 episodes liên tiếp → dừng toàn bộ training, lưu `ppo_coordinator.pth`.

---

## 7. Cấu trúc Source Code

```
LLM-ProvenanceGraph_APTDetection/
│
├── main.py                    # PPO Coordinator — luồng train chính, EWC integration
├── environment.py             # MARL Env — 3 Agents + ThompsonMAB + ReplayBuffer + EWC
├── networks.py                # Neural Architectures:
│                              #   GCN (SAGEConv), DetectionMLP (+ EWC), CoordinatorNetwork
│                              #   PositionalEncoder
│
├── env_utils.py               # Utilities: scf_impact, visualize (Graphviz/matplotlib), save TXT
├── graph_utils.py             # Graph prep: Word2Vec embedding, edge_index builder
├── llm_config.py              # Ollama client config (YAML + .env), generate_with_ollama()
├── olama.py                   # (Legacy) Ollama helper
│
├── Pruning_Agent/
│   ├── pruning_agent.py       # PageRank-based node pruning algorithm
│   └── graph_instance_patched.py  # Graph instance utilities
│
├── train_model/
│   └── train_mlp.py           # Offline pre-training script cho DetectionMLP:
│                              #   GCN feature extraction → Oversampling → BCE training
│                              #   Output: trained_weights/unicorn/mlp.pth
│
├── trained_weights/
│   └── unicorn/
│       ├── unicorn0.pth       # Pre-trained GCN weights (frozen during MARL)
│       ├── unicorn.model      # Pre-trained Word2Vec model
│       └── mlp.pth            # Pre-trained DetectionMLP weights (initial)
│
├── runs/
│   └── <timestamp>/
│       ├── best_mlp.pth       # Best DetectionMLP weights sau EWC Continual Learning
│       ├── hard_samples.txt   # File lưu các hard samples (FN=1) của riêng run này
│       ├── generated_graphs/  # TXT snapshots của graph tại mỗi generation step
│       └── graph_visualize/   # PDF visualizations (Graphviz/matplotlib)
│
├── llm-config.yaml            # LLM endpoint + model name config
├── .env                       # OLLAMA_HOST, OLLAMA_MODEL overrides
├── 2.txt                      # Base provenance graph (seed cho reset())
└── idea.md                    # Tài liệu ý tưởng gốc (MARL-VariantHunter)
```

**Các class/function quan trọng:**

| Class / Function | File | Mô tả |
|---|---|---|
| `ThompsonSamplingMAB` | `environment.py` | 3-arm MAB với Beta distribution update |
| `PrioritizedReplayBuffer` | `environment.py` | Priority queue lưu hard samples (FN=1) |
| `DetectionMLP.compute_fisher()` | `networks.py` | Tính Fisher Information matrix (diagonal) |
| `DetectionMLP.ewc_loss()` | `networks.py` | EWC regularization penalty |
| `ProvenanceGraphEnv.train_detection_agent_ewc()` | `environment.py` | Offline batch update + lưu best_mlp.pth |
| `ProvenanceGraphEnv._detection_agent()` | `environment.py` | Detection pipeline + MAB update + buffer fill |
| `CoordinatorNetwork` | `networks.py` | PPO policy (action logits + state value) |

---

## 8. Cài đặt & Chạy

### Yêu cầu

```bash
pip install torch torch-geometric gensim scikit-learn pandas networkx matplotlib graphviz ollama python-dotenv pyyaml tqdm
```

### Bước 1: Chuẩn bị Model Weights

Đảm bảo các file sau tồn tại:
```
trained_weights/unicorn/unicorn0.pth   # GCN weights (UNICORN/SLOT pre-trained)
trained_weights/unicorn/unicorn.model  # Word2Vec model
trained_weights/unicorn/mlp.pth        # DetectionMLP weights (xem Bước 1b)
```

**Bước 1b — Pre-train DetectionMLP từ đầu** (nếu chưa có `mlp.pth`):
```bash
# Chỉnh BASE_DIR trong train_model/train_mlp.py cho đúng
python train_model/train_mlp.py
# Output: trained_weights/unicorn/mlp.pth (50 epochs, oversampling applied)
```

### Bước 2: Cấu hình Ollama

Chỉnh `llm-config.yaml`:
```yaml
llm_endpoint: "http://localhost:11434"   # hoặc ngrok URL
llm_model: "gemma3:4b"                   # hoặc model khác
```

Hoặc dùng `.env`:
```
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma3:4b
```

Đảm bảo Ollama đang chạy và model đã được pull:
```bash
ollama pull gemma3:4b
ollama serve
```

### Bước 3: Chạy MARL Training

```bash
python main.py
```

**Output mẫu mỗi episode:**
```
[Main] Run artifacts will be saved to: runs/20260514_115240/
[MAB] Arm 2 selected (rate=95%, θ~Beta(1.0,1.0))
[Detection Agent] MLP Prob: 0.6823 | Anomaly: 45.00% | Pred: Malicious | GT: Malicious
[MAB] Updated Arm 2: α=2.0, β=1.0
[Detection Agent] Precision: 1.00 | Recall: 1.00 | SCF: 0.62 | FN: 0 | Reward: 2.6200
[EWC Training] Loss=0.4321 (CE=0.4321 + EWC=0.0000) | Acc=75.00% | Buffer=3
[EWC Training] 🏆 New best DetectionMLP saved → runs/20260514_115240/best_mlp.pth

Episode 1/50 - Total Reward: 2.6200 - Avg MLP Prob: 0.6823 - Actions: [0, 2]
  Detection Agent | Best Score: 0.7500 | Replay Buffer: 3 samples
  MAB Stats       | Arm0(mu=0.50) | Arm1(mu=0.50) | Arm2(mu=0.67)
Final Graph Size: 8 edges
------------------------------------------------------------
```

### Bước 4: Xem kết quả

Mỗi lần chạy sẽ tạo ra một thư mục riêng biệt tại `runs/<timestamp>/` chứa toàn bộ artifacts của lần chạy đó:

- **Best DetectionMLP:** `runs/<timestamp>/best_mlp.pth`
- **PPO Coordinator:** `ppo_coordinator.pth` (lưu ngoài thư mục gốc sau khi training kết thúc)
- **Hard Samples Buffer:** `runs/<timestamp>/hard_samples.txt` (lưu các đồ thị khó khi hệ thống bỏ lọt FN=1)
- **Graph snapshots:** `runs/<timestamp>/generated_graphs/*.txt`
- **Graph visualizations:** `runs/<timestamp>/graph_visualize/*.pdf`

---

## 9. So sánh Hệ thống Hiện tại vs idea.md Ban đầu

Dự án hiện đang duy trì kiến trúc tổng thể bám rất sát vào lý thuyết nhưng đã tùy chỉnh lại các thành phần để mang tính thực tiễn cao hơn và tối ưu khả năng chạy trên local.

### A. So sánh Cơ chế State - Action - Reward

| Thành phần | Đề xuất trong `idea.md` | Cài đặt Thực tế (Codebase) | Lý do điều chỉnh |
|---|---|---|---|
| **Generation State** | Provenance subgraph + Wasserstein drift score | Chuỗi text (danh sách Đỉnh/Cạnh) | Chuyển đổi để phù hợp đưa vào Context LLM (Ollama) thay vì Graph Deep Learning. |
| **Generation Action** | Sinh variant AAG → AASG → IAG bằng GraphAF | Prompting LLM để sinh `new_nodes` & `new_edges` | LLM mô phỏng attacker có ngữ nghĩa (semantic) tốt hơn, không cần training hội tụ tốn kém. |
| **Generation Reward** | VariantQuality + SLOT detection confidence | Lấy trực tiếp Reward từ Detection Agent (Detection confidence) | Tối giản hóa hàm mục tiêu, tạo vòng lặp Adversarial trực tiếp giữa Generation và Detection. |
| **Pruning State** | Sliding window provenance graph (10k-50k nodes) | Số lượng cạnh hiện tại (`len > 50`) | Đồ thị thực tế trong mỗi episode chưa đến 10k nodes, trigger sớm giúp giữ đồ thị tinh gọn. |
| **Pruning Action** | K-way coarsening (Metis) + GraphSAGE pooling | `PruningAgent` dùng PageRank (Centrality) để cắt | Thuật toán Centrality đồ thị truyền thống nhẹ và linh hoạt hơn cho streaming graph. |
| **Pruning Reward** | `CompressionRatio × (1 - loss)` | `CompressionRatio + 0.1` | Đơn giản hóa cơ chế tính toán, vẫn đảm bảo mục tiêu tối đa hóa nén. |
| **Detection State** | Subgraph đã prune + latent behavior | Graph latent 128-dim | Đạt sự thống nhất, Vector rút gọn giảm latency cho PPO Coordinator. |
| **Detection Action** | Bernoulli MAB + MLP + iForest | **Thompson Sampling MAB** (3 arms) + GCN + MLP + iForest | Nâng cấp MAB tĩnh (Bernoulli 80%) thành Thompson MAB thực thụ để model tự học tỷ lệ sampling tối ưu. |
| **Detection Reward** | `(Precision + Recall - λ·FN) + risk reduction` | **Giữ nguyên 100%**: `(Precision + Recall - 2.0·FN) + SCF_Impact` | Hàm Reward cốt lõi này rất mạnh, đã được code chính xác theo lý thuyết. |

### B. So sánh Cấu trúc & Khắc phục Gaps

| Tiêu chí | Đề xuất `idea.md` | Cài đặt Thực tế | Nhận xét |
|:---|:---|:---|:---|
| **Continual Learning** | EWC + Prioritized Replay Buffer để chống Concept Drift (Gap 1) | **Đã triển khai đầy đủ:** `compute_fisher()`, `ewc_loss()`, `PrioritizedReplayBuffer`, Offline batch update | Đáp ứng 100% yêu cầu về giải quyết Gap 1 cho Detection Agent. |
| **Closed-loop** | Mô tả lý thuyết giải quyết Gap 3 | **Đã hiện thực:** Generation sinh mã độc → Detection bị FN → ReplayBuffer → EWC → Update → lặp | Hoàn chỉnh vòng lặp Adversarial mimicry khép kín thực thụ. |
| **Model Saving** | Không đề cập chi tiết | **`runs/<timestamp>/best_mlp.pth`** — lưu best DetectionMLP theo accuracy sau EWC | Bảo toàn thành quả Continual Learning qua từng vòng lặp. |