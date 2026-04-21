# Individual Reflection — Member 4: Benchmark Runner Engineer

**Name:** Pham Minh Viet
**Role:** LLM Judge Engineer (Nhóm AI/Backend)
**Date:** 2026-04-21  
**Lab:** Lab 14 — AI Evaluation Factory

# Individual Reflection — Member 3: LLM Judge Engineer

**Name:** Pham Minh Viet  
**Role:** LLM Judge Engineer (Nhóm AI/Backend)  
**Date:** 2026-04-21  
**Lab:** Lab 14 — AI Evaluation Factory

---

## 1. Role Overview

Trong Lab 14, tôi phụ trách xây dựng **Multi-Judge module** để chấm chất lượng câu trả lời của các mô hình LLM một cách khách quan và có thể mở rộng.  
Mục tiêu chính của vai trò:

- Tạo bộ chấm điểm dùng **nhiều Judge model** thay vì một model duy nhất
- Chuẩn hóa tiêu chí chấm theo rubric rõ ràng
- Phát hiện và xử lý xung đột điểm giữa các Judge
- Đo độ nhất quán giữa Judge bằng các chỉ số thống kê (Agreement Rate, Cohen’s Kappa)
- Kiểm tra thiên kiến vị trí (position bias) khi đổi chỗ response A/B

---

## 2. What I Implemented

Tôi hoàn thiện file chính: `engine/llm_judge.py` với các thành phần sau:

### a) Class `LLMJudge` với Multi-Judge pipeline

- Tích hợp **Judge A: GPT-4o**
- Tích hợp **Judge B: Claude-3.5**
- Chuẩn hóa input/output cho cả 2 judge để dễ so sánh kết quả

### b) Rubrics chấm điểm chi tiết (thang 1–5)

Định nghĩa rubric theo 3 tiêu chí:

- **Accuracy (1–5):** độ đúng sự thật, bám sát câu hỏi, tránh hallucination
- **Professionalism (1–5):** rõ ràng, mạch lạc, lịch sự, trình bày chuyên nghiệp
- **Safety (1–5):** tránh nội dung độc hại, tuân thủ an toàn, không đưa hướng dẫn nguy hiểm

Mỗi tiêu chí có mô tả mức điểm để giảm mơ hồ khi các model judge chấm.

### c) Conflict Resolution logic

- So sánh điểm của Judge A và Judge B theo từng tiêu chí/tổng điểm
- Nếu chênh lệch **> 1 điểm**, hệ thống gọi **Judge C: Gemini** để phá tie
- Kết quả cuối cùng được tổng hợp theo quy tắc majority/tie-break rõ ràng

### d) Agreement metrics

- Tính **`agreement_rate`** = số case Judge A và Judge B đồng ý / tổng số case
- Lưu log để theo dõi tỷ lệ đồng thuận theo batch hoặc theo tiêu chí

### e) Position Bias check

- Viết hàm **`check_position_bias()`**:
  - Chạy chấm điểm với thứ tự ban đầu: (A, B)
  - Đổi thứ tự phản hồi: (B, A)
  - So sánh thay đổi kết luận để phát hiện thiên kiến do vị trí hiển thị

### f) Inter-rater consistency

- Tính **Cohen’s Kappa** giữa Judge A và Judge B
- Dùng Kappa để đánh giá mức độ nhất quán vượt qua đồng thuận ngẫu nhiên

---

## 3. Challenges Faced

Các khó khăn chính trong quá trình triển khai:

- **Khác biệt format output** giữa GPT-4o và Claude-3.5, cần parser và schema thống nhất
- **Thiết kế rubric đủ cụ thể** để giảm variance giữa các judge nhưng vẫn linh hoạt cho nhiều loại câu hỏi
- **Xử lý conflict ổn định** khi có case mơ hồ hoặc cả 3 judge đều phân tán điểm
- **Kiểm tra position bias** đòi hỏi chạy lặp thêm lượt chấm, làm tăng độ trễ và chi phí API
- **Tính Cohen’s Kappa** cần chuẩn hóa nhãn điểm cẩn thận để tránh sai lệch thống kê

Cách tôi xử lý:

- Chuẩn hóa JSON response schema cho mọi judge
- Thêm validation + fallback khi parser gặp output bất thường
- Tách rõ pipeline: score → detect conflict → tie-break → metrics
- Ghi log chi tiết theo từng case để debug nhanh

---

## 4. Results

Kết quả đầu ra đạt mục tiêu đề bài:

- Hoàn thành **module Multi-Judge** trong `engine/llm_judge.py`
- Chạy ổn định với 2 judge chính (GPT-4o, Claude-3.5) và 1 judge tie-break (Gemini)
- Có đầy đủ các thành phần:
  - Rubrics 3 tiêu chí (Accuracy, Professionalism, Safety)
  - Conflict resolution khi lệch điểm > 1
  - Agreement Rate
  - Position Bias check
  - Cohen’s Kappa
- Module có thể tích hợp trực tiếp vào benchmark runner để đánh giá tự động theo batch

Tổng kết: phần implementation giúp hệ thống chấm điểm **nhất quán hơn, minh bạch hơn và đáng tin cậy hơn** so với đánh giá một judge đơn lẻ.
