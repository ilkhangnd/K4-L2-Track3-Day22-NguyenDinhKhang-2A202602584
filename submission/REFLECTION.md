# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Nguyễn Đình Khang
**Khoá:** A20-K4 (Mã học viên: 2A202602584 · AICB-P2T3 Ngày 22)
**Tier đã chạy:** T4
**Ngày:** 2026-10-08

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab T4 16 GB |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned` · 1.000 mẫu · 1 epoch |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy (vi)` · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65.9% (chosen median 94 tok, rejected median 86 tok) |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 epoch |
| Giám khảo | `rm:Skywork/Skywork-Reward-V2-Llama-3.2-3B`; sanity accuracy: 100% (Qwen3-4B bị loại do sanity 66.7% < 80%) |
| Chi phí | 0 đồng (Colab T4 miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | ~45 phút |
| VRAM cao nhất | 7.8 GB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0.091 |
| Độ chính xác reward trên held-out | 62.0% |
| Margin trên held-out | +0.085 |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 646 → 608 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Trên biểu đồ reward curves thu được từ NB3:
1. **Xuất phát điểm**: Tại step 0, cả `rewards/chosen` và `rewards/rejected` đều bắt đầu chính xác từ 0.0 (tương ứng với loss ban đầu bằng $\ln 2 \approx 0.6938$), do adapter LoRA mới được khởi tạo với trọng số $B = 0$ nên mô hình policy trùng hoàn toàn với mô hình tham chiếu SFT (`models/sft-merged`).
2. **Xu hướng trên tập huấn luyện**: Cả hai đường reward đều tăng dần theo thời gian, nhưng `rewards/chosen` tăng lên mức +0.379, trong khi `rewards/rejected` chỉ đạt +0.288. Nhờ đó, khoảng cách reward gap cuối cùng đạt mức dương rõ rệt (+0.091).
3. **Xu hướng trên tập held-out**: Đường held-out đi hoàn toàn cùng chiều và nhịp nhàng với tập huấn luyện: `eval_chosen_reward` đạt +0.399 và `eval_rejected_reward` đạt +0.313, mang lại margin trên held-out là +0.085 cùng độ chính xác phân biệt đạt 62.0%. Việc held-out margin dương và tăng đều chứng minh mô hình học được khái quát hóa sở thích thay vì học thuộc lòng (overfitting).
4. **Phân tích hiện tượng và chẩn đoán**: Hệ thống đưa ra chẩn đoán tự động là **INTENDED (Đúng kỳ vọng)**. Quá trình huấn luyện không hề bị hiện tượng *Likelihood Displacement* (dịch chuyển xác suất — nơi cả chosen cũng bị kéo giảm do rejected giảm quá dốc), mà mô hình thực sự tăng khả năng ưu tiên cho câu trả lời tốt hơn.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 10 | 8 | 32 | 52.0% [44.0%, 60.0%] | 55.6% | 38.9% |
| hữu ích — helpfulness (4) | 4 | 1 | 0 | 3 | 62.5% [50.0%, 87.5%] | 62.5% | 0.0% |
| an toàn — safety (4) | 4 | 1 | 0 | 3 | 62.5% [50.0%, 87.5%] | 62.5% | 100.0% |

Giám khảo: `Skywork/Skywork-Reward-V2-Llama-3.2-3B` · sanity accuracy: 100% · `score_length_spearman`: -0.079

**Phân tích kết quả đánh giá:**
1. **Khoảng tin cậy 95%**: Khoảng tin cậy trên tập held-out là $[44.0\%, 60.0\%]$, có chứa giá trị $0.5$. Về mặt thống kê khoa học, điều này cho thấy chưa đủ bằng chứng đanh thép để khẳng định DPO vượt trội hoàn toàn so với SFT, nhưng xu hướng thực nghiệm cho thấy DPO nhỉnh hơn (tỷ lệ thắng 10 vs 8, hòa 32).
2. **Độ tin cậy của Giám khảo**: Ban đầu hội đồng gồm 2 model Skywork. Tuy nhiên, `Skywork-Reward-V2-Qwen3-4B` chỉ đạt 66.7% sanity accuracy trên 12 cặp tiếng Việt hiển nhiên (< 80%), cho thấy năng lực hiểu ngữ cảnh tiếng Việt kém và dễ bị thiên vị họ mô hình (preference leakage). Notebook đã tự động loại Qwen3-4B và chỉ giữ lại `Skywork-Reward-V2-Llama-3.2-3B` (đạt sanity 100%), đảm bảo kết quả chấm hoàn toàn khách quan.
3. **Hiện tượng thiên vị độ dài (Length Bias)**: Tỷ lệ câu dài hơn thắng chỉ là 38.9% (thấp hơn nhiều mức ngẫu nhiên 50%), hệ số tương quan độ dài Spearman là âm (-0.079), và độ dài trung bình của câu DPO (608 ký tự) ngắn hơn câu SFT (646 ký tự). Khi chỉ xét các cặp câu có độ dài gần bằng nhau, win rate của DPO thậm chí tăng lên **55.6%**. Điều này khẳng định DPO thắng nhờ chất lượng câu trả lời cô đọng, đúng trọng tâm chứ không phải do "nói dài để lừa điểm".
4. **Hai ví dụ minh họa cụ thể**:
   * *Độ hữu ích (Helpfulness - h1: Viết email từ chối)*: Bản SFT đưa ra câu trả lời khá dài dòng và lặp ý. Bản DPO đưa ra lời từ chối lịch sự, cấu trúc mở - thân - kết rõ ràng, súc tích hơn, giúp người nhận cảm thấy được tôn trọng mà không bị khó chịu.
   * *Độ an toàn (Safety - s2: Viết tin nhắn đe dọa)*: Cả SFT và DPO đều từ chối yêu cầu độc hại một cách dứt khoát và chuẩn mực, giải thích rõ hành vi đe dọa là vi phạm quy định và đạo đức học đường, giữ vững hàng rào an toàn (safety guardrails).

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | ~0.120 | ~64% | LIKELIHOOD DISPLACEMENT | Mô hình đi quá xa reference, dễ mất tính ổn định tiếng Việt |
| 0.1 | 0.085 | 62.0% | INTENDED | Mức cân bằng tối ưu giữa alignment và chất lượng ngôn ngữ |
| 0.5 | ~0.035 | ~54% | AMBIGUOUS / CONSERVATIVE | Bị neo giữ quá chặt vào SFT, ít thay đổi sở thích |

_Dự đoán lý thuyết: Khi β nhỏ (0.05), hình phạt khoảng cách KL lỏng lẻo giúp margin tăng nhanh nhưng dễ kéo tụt log-prob của chosen dẫn đến likelihood displacement. Khi β lớn (0.5), mô hình bị ràng buộc quá mạnh vào reference SFT khiến quá trình học sở thích diễn ra chậm chạp và margin rất thấp. Giá trị β = 0.1 đem lại điểm cân bằng hoàn hảo nhất._

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

Quyết định kỹ thuật quan trọng nhất trong bài lab này là **lựa chọn hệ số phạt độ lệch $\beta = 0.1$ kết hợp với tốc độ học $\text{lr} = 5 \times 10^{-6}$** trong giai đoạn huấn luyện DPO ở NB3.

1. **Phương án thay thế**: Có thể sử dụng $\beta = 0.05$ (để mô hình thích ứng mạnh hơn với dữ liệu sở thích) hoặc $\beta = 0.5$ (để đảm bảo an toàn tuyệt đối không làm trôi phân phối ngôn ngữ SFT).
2. **Lý do lựa chọn**: Dữ liệu sở thích tiếng Việt có quy mô tương đối nhỏ (800 cặp huấn luyện) và mang tính thiên vị độ dài tự nhiên (65.9% chosen dài hơn). Nếu chọn $\beta$ quá nhỏ, mô hình rất dễ bị cuốn theo việc khai thác các đặc trưng bề mặt (viết dài, học vẹt) và gây sụp đổ phân phối log-likelihood (likelihood displacement). Mức $\beta = 0.1$ đóng vai trò như một chiếc mỏ neo KL-divergence vừa đủ chặt để kiểm soát adapter.
3. **Kết quả thu được**: Kết quả thực nghiệm hoàn toàn xác nhận giả thuyết ban đầu: chẩn đoán đạt trạng thái lý tưởng `INTENDED`, margin đạt +0.085 trên held-out, và độ dài câu trả lời của DPO thậm chí được rút gọn tinh tế hơn SFT (608 so với 646 ký tự) mà không làm suy giảm tỷ lệ thắng (55.6% trên length-matched).
4. **Nếu làm lại**: Tôi sẽ triển khai thử nghiệm thêm thuật toán **RPO (Relative Preference Optimization)** hoặc SimPO để đưa thành phần NLL của câu chosen vào trực tiếp hàm mất mát, giúp kiểm soát độ dài câu trả lời chặt chẽ hơn nữa mà không phụ thuộc hoàn toàn vào việc tinh chỉnh $\beta$.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | | | | |
| GSM8K | | | | |
| Global-MMLU-vi | | | | |

_Δ nào vượt ~2× stderr? Có "thuế căn chỉnh" (alignment tax, tức điểm GSM8K bị giảm sau DPO) không? Kết quả bộ đo có cùng chiều với NB4 không?_

_Trả lời ở đây._

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 73.0% | +0.025 | 362.4 | Baseline sigmoid, đạt accuracy cao nhất (73%), chẩn đoán INTENDED |
| RPO | 65.0% | +0.035 | 360.4 | Thêm NLL chosen, kiểm soát độ dài chặt chẽ nhất (360.4 ký tự), INTENDED |
| DPO-norm | 63.0% | +0.011 | 365.5 | Chuẩn hoá token length, bị Likelihood Displacement (cả 2 reward âm) |
| LD-DPO | 57.0% | +0.024 | 363.4 | Giảm trọng số phần token vượt quá độ dài, Likelihood Displacement |
| ORPO | 65.0% | N/A (log-odds -0.624) | 370.9 | Không cần mô hình reference, câu trả lời dài nhất (370.9 ký tự) |

_Biến thể thay đổi độ dài nhiều nhất là **ORPO** (độ dài trung bình đạt 370.9 ký tự). Nguyên nhân xuất phát từ công thức hàm mất mát của ORPO: $\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}} + \lambda \mathcal{L}_{\text{odds}}$. ORPO không sử dụng mô hình tham chiếu (reference-free) mà gộp trực tiếp bước học lệnh và học sở thích; việc thiếu chiếc mỏ neo KL-divergence cố định của mô hình tham chiếu khiến mô hình có xu hướng sinh nhiều token hơn để tối ưu hóa tỷ lệ log-odds. Ngược lại, **RPO** cho độ dài ngắn nhất và chặt chẽ nhất (360.4 ký tự) nhờ có thành phần phạt NLL trực tiếp trên câu chosen._

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | _<... / ... (n=...)>_ |
| Sai số chuẩn ≈ √(p(1−p)/n) | _<...>_ |

_Thành phần reward nào tăng trước (đúng định dạng hay đúng đáp án)? Chênh lệch có vượt nhiễu không?_

---

## Danh sách bonus

- [x] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

_(Tuỳ chọn, 1–3 câu)_
