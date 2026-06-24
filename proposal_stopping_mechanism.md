# Đề xuất Cơ chế Dừng cho MARL-VariantHunter
*Tài liệu: proposal_stopping_mechanism.md*

---

## 1. Bối cảnh hệ thống

MARL-VariantHunter là hệ thống **adversarial co-evolution** gồm 3 agent:

| Agent | Mục tiêu |
|---|---|
| **Generation Agent (Attack)** | Sinh variant APT có khả năng né tránh Detection |
| **Pruning Agent** | Nén đồ thị giữ nguyên quan hệ nhân quả |
| **Detection Agent (MLP + GAT)** | Phát hiện APT, giảm FN |

PPO Coordinator điều phối 3 agent trên. Hệ thống đạt **Nash Equilibrium** khi cả Detection và Attack đều không thể cải thiện thêm.

---

## 2. Phân tích cơ chế dừng hiện tại

### 2.1 Cơ chế hiện tại (`AdversarialConvergenceMonitor`)

Điều kiện 1: Detection Convergence (avg_det >= 0.7 trong 8 ep liên tiếp)
ĐỒNG THỜI VỚI
Điều kiện 2: Attack Equilibrium (std(evasion_rate) < 0.05 trong 8 ep)
HOẶC
Điều kiện 1 ĐỒNG THỜI VỚI Điều kiện 3: Reward Plateau (std < 0.15)

**Tham số hiện tại:**
- `patience = 8` episodes
- `det_thresh = 0.7` (precision + recall >= 0.7)
- `evasion_delta = 0.05`
- `reward_delta = 0.15`
- `num_episodes = 100` (hard cap)

### 2.2 Vấn đề phát hiện

| # | Vấn đề | Nguy cơ |
|---|---|---|
| 1 | `det_thresh = 0.7` được tính từ `avg_det_score = precision + recall - 2*FN + scf_impact`, **không phải F1**. Giá trị này có thể vượt 1.0 hoặc âm khi FN lớn, ngưỡng 0.7 **không có ý nghĩa cố định** | Hội tụ giả / không bao giờ dừng |
| 2 | Điều kiện dừng **không kiểm tra FN**. Hệ thống có thể dừng khi Detection agent có precision cao nhưng FN vẫn cao | FN cao không được phạt đúng mức khi dừng |
| 3 | `patience = 8` quá ngắn: với 15 steps/episode, mỗi episode chỉ có 2-3 Detection calls. Window 8 ep = ~16-24 detection steps, chưa đủ ổn định thống kê | Dừng sớm giả |
| 4 | **Không có cơ chế dừng sớm** khi FN liên tục tăng (Attack thắng hoàn toàn) hoặc Detection bị collapse | Training vô ích khi hệ thống mất cân bằng |
| 5 | `reward_plateau` dùng `std < 0.15` trên 8 episodes — quá nhỏ, noise tự nhiên của reward đã đủ để trigger | Dừng sớm không phải do hội tụ thật |

---

## 3. Đề xuất cơ chế dừng cải tiến

### 3.1 Các tiêu chí dừng được đề xuất

#### Tiêu chí A — Nash Equilibrium Thật (CHÍNH)
> **Dừng khi**: Detection đạt mục tiêu F1 thực tế **VÀ** Attack không còn cải thiện được.

```
Điều kiện A:
  (1) avg(Global_F1) >= 0.75  trong patience_nash = 12 episodes liên tiếp
  AND
  (2) std(evasion_rate) < 0.04  trong cùng window
  AND
  (3) avg(FN_rate) < 0.20  trong cùng window
```

- `Global_F1` lấy từ `evaluate_detection_mlp()` — đã được tính đúng theo sklearn
- `FN_rate = FN / (FN + TP)` — tỷ lệ thực sự bỏ sót APT

#### Tiêu chí B — Reward Plateau Có Kiểm Soát (PHỤ)
> **Dừng khi**: Cả 3 agent đều không cải thiện thêm.

```
Điều kiện B:
  (1) std(total_reward) < 0.10  trong patience_plateau = 15 episodes
  AND
  (2) best_combined_score không cải thiện trong patience_plateau episodes liên tiếp
```

#### Tiêu chí C — Early Stop Thất Bại / Suy Thoái (AN TOÀN)
> **Dừng sớm** khi hệ thống mất cân bằng: Attack thắng hoàn toàn hoặc Detection collapse.

```
Điều kiện C (bất kỳ):
  (a) avg(FN_rate) > 0.80 trong 15 episodes liên tiếp
      -> Attack thống trị, Detection không học được
  OR
  (b) avg(Global_F1) < 0.05 trong 20 episodes (sau episode 20)
      -> Detection collapse hoàn toàn
```

#### Tiêu chí D — Hard Cap (BẮT BUỘC)
```
Nếu episode >= num_episodes: dừng vô điều kiện
```

### 3.2 Thứ tự ưu tiên kiểm tra

```
mỗi episode:
  1. Kiểm tra C (Failure/Degradation) -> dừng + log cảnh báo
  2. Kiểm tra A (Nash Equilibrium)    -> dừng + log thành công
  3. Kiểm tra B (Reward Plateau)      -> dừng + log plateau
  4. Kiểm tra D (Hard Cap)            -> dừng + log timeout
  5. Tiếp tục episode tiếp theo
```

### 3.3 Tham số đề xuất

| Tham số | Giá trị cũ | Giá trị đề xuất | Lý do |
|---|---|---|---|
| `patience_nash` | 8 | **12** | Đủ ổn định thống kê (~24-36 detection calls) |
| `patience_plateau` | 8 | **15** | Tránh dừng sớm do noise |
| `patience_failure` | (không có) | **15** | Phát hiện collapse sớm |
| `det_thresh` | 0.7 (avg_det_score) | **0.75 (Global F1)** | Metric có ý nghĩa cố định [0,1] |
| `evasion_delta` | 0.05 | **0.04** | Chặt hơn một chút |
| `fn_rate_thresh` | (không có) | **0.20** | FN <= 20% mới được phép dừng |
| `failure_fn_rate` | (không có) | **0.80** | Dừng sớm nếu Attack thắng hoàn toàn |
| `reward_delta` | 0.15 | **0.10** | Tránh trigger sớm |
| `num_episodes` | 100 | **150** | Cho đủ thời gian co-evolution |

---

## 4. Pseudo-code cải tiến

```python
class ImprovedStoppingCriterion:
    def __init__(
        self,
        patience_nash    = 12,
        patience_plateau = 15,
        patience_failure = 15,
        f1_thresh        = 0.75,
        evasion_delta    = 0.04,
        fn_rate_thresh   = 0.20,
        failure_fn_rate  = 0.80,
        reward_delta     = 0.10,
    ):
        ...

    def update(self, global_f1, evasion_rate, fn_rate, total_reward):
        # --- C: Failure detection ---
        if fn_rate > self.failure_fn_rate:
            self._failure_count += 1
        else:
            self._failure_count = 0
        if self._failure_count >= self.patience_failure:
            return "FAILURE", "Attack dominated: FN rate too high"

        if global_f1 < 0.05 and self.episode > 20:
            self._collapse_count += 1
        else:
            self._collapse_count = 0
        if self._collapse_count >= 20:
            return "FAILURE", "Detection collapse: F1 < 5%"

        # --- A: Nash Equilibrium ---
        self.f1_buf.append(global_f1)
        self.evasion_buf.append(evasion_rate)
        self.fn_buf.append(fn_rate)
        if len(self.f1_buf) >= self.patience_nash:
            avg_f1   = mean(self.f1_buf)
            std_eva  = stdev(self.evasion_buf)
            avg_fn   = mean(self.fn_buf)
            if avg_f1 >= self.f1_thresh and std_eva < self.evasion_delta and avg_fn < self.fn_rate_thresh:
                return "CONVERGED", f"Nash Eq: F1={avg_f1:.3f}, EvaStd={std_eva:.3f}, FN={avg_fn:.3f}"

        # --- B: Reward Plateau ---
        self.reward_buf.append(total_reward)
        if len(self.reward_buf) >= self.patience_plateau:
            if stdev(self.reward_buf) < self.reward_delta and self._no_improvement_count >= self.patience_plateau:
                return "PLATEAU", f"Reward plateau: std={stdev(self.reward_buf):.3f}"

        return "CONTINUE", ""
```

---

## 5. Tác động đến `main.py`

Thay thế `AdversarialConvergenceMonitor` bằng `ImprovedStoppingCriterion`:

```python
# Trong vòng lặp episode, cần lấy global_f1 từ train_detection_agent_ewc():
fn_rate = fn_count / (fn_count + tp_count) if (fn_count + tp_count) > 0 else 0.0
stop_reason, stop_msg = stopping.update(global_f1, evasion_rate, fn_rate, total_rew)

if stop_reason in ("CONVERGED", "PLATEAU"):
    print(f"[STOP] {stop_reason}: {stop_msg}")
    break
elif stop_reason == "FAILURE":
    print(f"[WARNING] System degraded: {stop_msg}. Stopping.")
    break
```

**Thay đổi cần thiết ở `environment.py`:**
- `train_detection_agent_ewc()` cần trả về `global_f1` để `main.py` sử dụng.
- `evaluate_detection_mlp()` đã trả về `f1` — chỉ cần `train_detection_agent_ewc()` forward giá trị này ra ngoài.

---

## 6. Kết luận và khuyến nghị

| Ưu tiên | Hành động |
|---|---|
| CAO | Thay `avg_det_score` bằng `Global_F1` làm ngưỡng dừng chính |
| CAO | Thêm điều kiện FN_rate < 0.20 vào Nash Equilibrium |
| TRUNG BÌNH | Tăng patience từ 8 → 12 (Nash) / 15 (Plateau/Failure) |
| TRUNG BÌNH | Thêm Failure detection (Attack thống trị hoàn toàn) |
| THAP | Tăng `num_episodes` từ 100 → 150 |
| THAP | Giảm `reward_delta` từ 0.15 → 0.10 |

> **Lưu ý quan trọng**: `global_f1` đã được tính trong `evaluate_detection_mlp()` và trả về sau mỗi EWC update. Giá trị này sẵn sàng được truyền vào stopping criterion mà không cần thêm computation.
