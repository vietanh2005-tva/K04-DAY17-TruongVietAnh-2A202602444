# Phân tích kết quả Lab 17

## Kết quả benchmark

### Standard Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 1793 | 15752 | 0.000 | 0.200 | 0 | 0 |
| Advanced | 1386 | 20687 | 0.821 | 0.836 | 311 | 0 |

### Long-Context Stress Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 435 | 22743 | 0.000 | 0.200 | 0 | 0 |
| Advanced | 335 | 7752 | 1.000 | 1.000 | 372 | 10 |

## 1. Vì sao Advanced recall tốt hơn Baseline?

Baseline khóa lịch sử theo `thread_id` và không đọc hay ghi file bền vững. Vì các câu hỏi recall được hỏi trong thread mới, recall của Baseline bằng 0 ở cả hai bộ dữ liệu. Advanced trích fact ổn định bằng `extract_profile_updates()`, ghi theo field vào `state/profiles/<user>/User.md`, rồi đọc lại profile khi trả lời ở thread mới. Vì vậy recall tăng lên 0.821 trong Standard và 1.000 trong Stress.

## 2. Vì sao Advanced có thể tốn hơn ở hội thoại ngắn?

Trong Standard Benchmark, `Prompt tokens processed` của Advanced là 20,687, cao hơn 15,752 của Baseline. Mỗi lượt Advanced phải mang thêm `User.md`, summary và recent messages, đồng thời hội thoại ngắn chưa đủ dài để compact tạo ra lợi ích. Đây là overhead của persistent memory; recall tốt hơn không miễn phí.

## 3. Vì sao compact có lợi trong hội thoại dài?

Trong Stress Benchmark, Baseline xử lý 22,743 prompt token trong khi Advanced chỉ xử lý 7,752, giảm khoảng 65.9%. Advanced thực hiện 10 lần compaction, thay phần lịch sử cũ bằng summary có kích thước giới hạn và chỉ giữ nguyên văn các message gần nhất. Compact tối ưu lượng ngữ cảnh được xử lý (`Prompt tokens processed`), không trực tiếp hứa hẹn giảm lượng token câu trả lời (`Agent tokens only`).

## 4. Memory tăng trưởng thế nào và có rủi ro gì?

Baseline không ghi file nên memory growth bằng 0. Advanced tăng 311 byte ở Standard và 372 byte ở Stress vì phải lưu các fact bền vững. Nếu chạy lâu, file có thể tiếp tục phình, giữ fact lỗi hoặc để correction cũ xung đột với fact mới. Bài làm giảm rủi ro bằng profile có cấu trúc theo khóa, `upsert_fact()` để correction thay thế giá trị cũ, chuẩn hóa `user_id`, và bỏ qua câu hỏi cũng như các câu được đánh dấu là nhiễu.

## Bonus: structured entity extraction và conflict handling

`User.md` không được nối thêm tự do mà lưu fact theo các field như `name`, `location`, `profession`, `response_style`, `favorite_food`, `favorite_drink`, `pet` và `interests`. Khi có correction, field cũ bị thay thế nên profile không giữ đồng thời Huế/Đà Nẵng hay backend/MLOps. Guardrail bỏ qua câu hỏi giúp các câu như “mình tên gì?” không ghi đè tên thật.

Bonus này làm recall đáng tin cậy hơn và giữ file nhỏ, nhưng regex vẫn phụ thuộc cách diễn đạt. Một câu phủ định hoặc mỉa mai chưa nằm trong rule có thể bị hiểu sai; hệ thống production nên thêm confidence score, provenance và bước xác nhận khi fact mới mâu thuẫn với fact quan trọng.

## Khả năng tái lập

Benchmark chạy hoàn toàn offline và tất định, không cần API key. Trước khi đo lại, xóa `state/`, sau đó chạy:

```powershell
python src/benchmark.py
pytest src/test_agents.py -v
```

Kết quả kiểm tra cuối: cả bốn test đều pass.
