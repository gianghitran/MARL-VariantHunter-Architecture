# Cơ chế State - Action - Reward của 3 Agents trong MARL-VariantHunter

Tài liệu này mô tả chi tiết luồng dữ liệu, trạng thái, hành động và các công thức/điểm số tính toán Reward cụ thể của từng Agent trong hệ thống MARL.

---

## 1. Generation Agent (Tác tử Sinh mã độc)

### State (Trạng thái)
- **Dữ liệu đầu vào:** Toàn bộ Đồ thị Provenance Graph hiện hành (`current_graph_df`).
- **Cơ chế xử lý:** Chuyển đổi Dataframe thành chuỗi văn bản (Text/JSON format). Các node được lập chỉ mục (Index) và các cạnh được serialize (ví dụ: `0 3 WR`) để phù hợp đưa vào Context Window của LLM (Ollama).

### Action (Hành động)
- **Thực thi:** Gọi API tới LLM bằng Prompt (chứa luật tạo node/edge APT).
- **Kết quả:** LLM sinh ra một khối dữ liệu định dạng JSON chứa `new_nodes` và `new_edges` để đắp nối tiếp vào đồ thị hiện hành.

### Reward (Điểm thưởng / Hình phạt)
- **Penalty (Lỗi cú pháp):** Nếu LLM sinh sai định dạng JSON, không thể parse được, hoặc cố tình sinh cạnh hỏng, không hợp lệ.
  - `Reward = -0.5`
- **Thưởng (Sinh thành công):** Nếu sinh thành công, Agent lập tức gọi hàm tính điểm từ *Detection Agent* (chạy ngầm). Điểm Reward của Generation Agent chính bằng điểm Reward mà Detection Agent đánh giá độ nguy hiểm của đồ thị vừa được cấy.
  - `Reward = (Precision + Recall - 2.0 * FN) + scf_impact` *(Chi tiết công thức xem ở phần Detection Agent)*.
  - **Sự thật:** Đồ thị sinh ra càng khó nhằn, đánh lừa được bộ phòng thủ (gây ra hiện tượng False Negative: FN=1) và chứa nhiều hành vi nguy hiểm (điểm `scf_impact` cao), thì **LLM (Generation Agent) càng nhận được nhiều điểm Reward**.

---

## 2. Pruning Agent (Tác tử Cắt tỉa)

### State (Trạng thái)
- **Dữ liệu đầu vào:** Kích thước của đồ thị hiện hành.
- **Trigger:** Agent này chỉ được phép PPO Coordinator gọi kích hoạt khi số cạnh của đồ thị vượt ngưỡng quy định (`len(current_graph_df) > 50`).

### Action (Hành động)
- **Thực thi:** Sử dụng thuật toán Centrality (cụ thể là `PageRank`) để đánh giá mức độ quan trọng / tính trung tâm của từng Node.
- **Kết quả:** Triệt tiêu trung bình 30% nhánh rác nhiễu (noise) sinh ra bởi các tiến trình nền của hệ điều hành, đồng thời đảm bảo bắt buộc phải giữ lại các Hubs quan trọng (các Node trung tâm hoặc Node có hoạt động bất thường).

### Reward (Điểm thưởng / Hình phạt)
- **Thưởng (Cắt tỉa thành công):** Nhận điểm dựa trên mức độ nén đồ thị (Compression Ratio).
  - Công thức: `Compression_Ratio = 1.0 - (Số_cạnh_sau_khi_cắt / Số_cạnh_ban_đầu)`
  - **`Reward = Compression_Ratio + 0.1`**
  - *Ví dụ:* Đồ thị từ 100 cạnh cắt xuống còn 70 cạnh $\rightarrow$ Tỷ lệ nén (Compression_Ratio) = 0.3. Hệ thống thưởng `Reward = 0.3 + 0.1 = 0.4` điểm.
- **Penalty (Cắt hỏng):** Nếu thuật toán PageRank thất bại, cắt cụt luôn cả đồ thị (trả về đồ thị rỗng 0 cạnh).
  - **`Reward = -0.1`**

---

## 3. Detection Agent (Tác tử Phát hiện)

### State (Trạng thái)
- **Dữ liệu đầu vào:** Đồ thị Provenance Graph đã qua sinh trưởng và cắt tỉa từ 2 Agent trước.
- **Cơ chế xử lý:** Đồ thị được vector hóa tự động qua thuật toán `Word2Vec` (nhúng từ) và `PositionalEncoder`, sau đó đi qua mạng chập đồ thị `GCN` để gộp thành 1 Vector đại diện cho toàn bộ đồ thị (Graph Latent).

### Action (Hành động)
- **Thực thi:**
  1. **Lấy mẫu (Thompson Sampling MAB):** Thuật toán sẽ chọn 1 trong 3 cánh tay (Arms) để quyết định giữ lại ngẫu nhiên bao nhiêu phần trăm đồ thị đưa vào đánh giá (giúp chống Overfitting):
     - Arm 0: Giữ lại 60% đồ thị
     - Arm 1: Giữ lại 80% đồ thị
     - Arm 2: Giữ lại 95% đồ thị
  2. **Phân loại:** Đưa Subgraph qua `DetectionMLP` và `IsolationForest` để tính xác suất tấn công. Phán quyết đồ thị là **Malicious (Mã độc)** nếu `mlp_prob > 0.5` VÀ `anomaly_ratio > 0.3`.

### Reward (Điểm thưởng / Hình phạt)
Đây là hàm Reward cốt lõi của toàn bộ mạng PPO Coordinator, dùng để cập nhật Policy cho bộ não trung tâm.
- **Công thức tổng quát:**
  **`Reward = (Precision + Recall - λ_FN * FN) + scf_impact`**
- **Giải thích các thành phần tính toán:**
  - `Precision` & `Recall`: Tỉ lệ phát hiện chuẩn và Không bỏ lọt. Điểm dao động từ `0.0` đến `1.0`. Khuyến khích MLP phân loại chính xác các tập Malicious và Benign.
  - `λ_FN = 2.0`: Trọng số hình phạt tàn khốc dành cho lỗi **False Negative** (Bỏ lọt mã độc).
  - `FN`: Trị số Binary. `1` nếu hệ thống phân loại nhầm đồ thị độc hại thành an toàn, `0` nếu phân loại chuẩn.
    - *Penalty Chí mạng:* Nếu FN = 1, Reward sẽ bị trừ ngay `2.0 * 1 = 2.0` điểm. Ép PPO phải học cách cảnh giác.
  - `scf_impact` (Security Context Feature): Điểm cộng rủi ro nhằm khuyến khích sinh ra / nhận diện các hành vi cực đoan. Được tính bằng cách lấy trung bình cộng rủi ro của tất cả các Action trong đồ thị:
    - `execute` (Thực thi): +1.0 điểm
    - `clone` (Nhân bản tiến trình): +0.9 điểm
    - `mmap` (Sửa đổi vùng nhớ): +0.8 điểm
    - `write` (Ghi file): +0.6 điểm
    - `read` (Đọc file): +0.2 điểm
    - *Các hành động khác*: +0.1 điểm
- **Ví dụ tính toán thực tế:** 
  - Mô hình đoán **ĐÚNG** đồ thị mã độc: Precision=1.0, Recall=1.0, FN=0. Đồ thị chứa các lệnh read và write, tính ra trung bình rủi ro `scf_impact = 0.42`.
    $\rightarrow$ **`Reward = (1.0 + 1.0 - 2.0 * 0) + 0.42 = 2.42`** điểm (Điểm rất cao, PPO sẽ tiếp tục phát huy chiến lược này).
  - Mô hình đoán **SAI** (Bỏ lọt mã độc): Precision=0.0, Recall=0.0, FN=1, `scf_impact = 0.42`.
    $\rightarrow$ **`Reward = (0.0 + 0.0 - 2.0 * 1) + 0.42 = -1.58`** điểm (Điểm âm nặng nề, PPO sẽ thay đổi chiến lược ngay lập tức).

---

> [!IMPORTANT]
> **Vòng lặp Continual Learning:** Ngay khi Detection Agent bị phạt điểm âm do bỏ lọt mã độc (`FN=1`), đồ thị State đó sẽ không bị vứt đi. Nó sẽ lập tức được lưu vào **Prioritized Replay Buffer** để tái huấn luyện với EWC (xem phần dưới).

---

## 4. Các cơ chế Bổ trợ (Continual Learning & Optimization)

Để đảm bảo hệ thống học liên tục mà không bị "quên" và tối ưu hóa thời gian huấn luyện, hai cơ chế quan trọng sau được tích hợp:

### A. Cơ chế EWC (Elastic Weight Consolidation) & Replay Buffer
Đây là giải pháp triệt để cho bài toán **"Concept Drift"** và **"Catastrophic Forgetting"** (Quên thảm họa) khi mã độc liên tục tiến hóa lẩn trốn:
- **Prioritized Replay Buffer (Bộ đệm ưu tiên):**
  - Ngay khi Detection Agent bị lừa (phân loại nhầm mã độc thành an toàn, `FN = 1`), đồ thị đó được coi là "Hard Sample" (Mẫu khó).
  - Mẫu này được đẩy ngay vào Replay Buffer kèm theo độ ưu tiên (Priority). Độ ưu tiên tính bằng khoảng cách từ `mlp_prob` đến ngưỡng 0.5 (càng sát 0.5 $\rightarrow$ mô hình càng phân vân $\rightarrow$ độ ưu tiên học lại càng cao).
- **Offline Batch Update với EWC:**
  - Kết thúc mỗi Episode, hệ thống sẽ tự động bốc một mini-batch từ Buffer này để huấn luyện lại mạng `DetectionMLP`.
  - Để tránh việc học mẫu mã độc mới làm quên mất cách nhận diện các mã độc cũ, thuật toán **EWC** tính toán ma trận **Fisher Information ($F_i$)** để đo lường "độ quan trọng" của từng trọng số.
  - **Hàm Loss cập nhật:** `Loss = NLLLoss + (λ/2) * Σ F_i * (θ_i - θ*_i)²`
    - Trọng số nào quan trọng với tri thức cũ ($F_i$ lớn) sẽ bị "đinh chặt", nếu cố tình thay đổi sẽ bị penalty (phạt loss) rất nặng. Trọng số nào ít quan trọng sẽ được nới lỏng để học cấu trúc Hard Sample mới.
  - Nhờ đó, DetectionMLP liên tục sinh ra "kháng thể" mới bảo toàn sức mạnh toàn diện. Trọng số tốt nhất được tự động sao lưu vào `runs/<timestamp>/best_mlp.pth`.

### B. Cơ chế Dừng sớm (Early Stopping)
Thuật toán MARL và LLM tiêu tốn rất nhiều thời gian. Cơ chế Early Stopping được thiết kế làm 2 lớp để tối đa hóa hiệu suất phần cứng:
- **Lớp 1 - Dừng cục bộ (Episode-level Peak Limit):**
  - Mỗi Episode giới hạn số bước sinh tối đa là `max_steps_per_episode = 15`.
  - Tuy nhiên, ở bất kỳ step nào, nếu Detection Agent đạt mức Peak Accuracy (`mlp_prob >= 0.95`, `Precision = 1.0`, `Recall = 1.0`), Đồ thị bị kết luận là đã đạt "đỉnh tinh xảo", không cần cấy thêm nữa. Hệ thống kích hoạt cờ `done = True` để kết thúc Episode ngay lập tức, tiết kiệm lượt gọi LLM.
- **Lớp 2 - Dừng toàn cục (Global Training Patience):**
  - Hệ thống liên tục theo dõi điểm số Accuracy trung bình (Avg MLP Prob) qua từng Episode với `patience = 5`.
  - Nếu sau 5 Episodes liên tiếp mà điểm số của Detection Agent chững lại, không phá được kỷ lục cũ (tức là hệ thống đã hội tụ tối đa, LLM cạn kiệt ý tưởng không thể sinh ra biến thể nào lừa được mô hình nữa), quá trình huấn luyện PPO sẽ lập tức Break.
  - Việc này tránh hiện tượng Overfitting (học vẹt) và kết thúc tiến trình đào tạo tự động, lưu mô hình `ppo_coordinator.pth` để sẵn sàng triển khai.
