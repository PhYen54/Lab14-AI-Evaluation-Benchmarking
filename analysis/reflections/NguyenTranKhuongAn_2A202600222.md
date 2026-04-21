# Báo cáo cá nhân - Lab Day 14: AI Evaluation Factory

**Họ và tên:** Nguyễn Trần Khương An
**MSSV:** 2A202600222
**Vai trò trong nhóm:** Agent Engineer + Failure Analyst

---

## 1. Đóng góp kỹ thuật (Engineering Contribution)

Trong bài Lab 14, tôi phụ trách 2 phần chính:

* Xây dựng và tinh chỉnh agent trả lời RAG trong `agent/main_agent.py`
* Phân tích kết quả benchmark và hoàn thiện báo cáo `analysis/failure_analysis.md`

Các đóng góp cụ thể:

* Hoàn thiện luồng xử lý câu hỏi trong agent:

  * Nhận câu hỏi người dùng
  * Truy vấn context liên quan
  * Tạo câu trả lời dựa trên context
  * Đóng gói output để đưa vào pipeline benchmark

* Tối ưu hành vi trả lời của agent theo hướng:

  * Giảm khẳng định mạnh khi thiếu bằng chứng
  * Tăng tính bám sát context
  * Giảm trả lời lạc đề với câu hỏi mở/nhiều nghĩa

* Tổng hợp kết quả benchmark từ `reports/summary.json` và `reports/benchmark_results.json` để viết failure analysis:

  * So sánh v1 và v2 theo score, pass rate, retrieval, cost, latency
  * Phân nhóm lỗi theo hành vi thất bại
  * Thực hiện 5 Whys cho 3 case tệ nhất
  * Đề xuất action plan P0/P1/P2 có tiêu chí đo lường rõ ràng

---

## 2. Kiến thức kỹ thuật (Technical Depth)

Qua quá trình làm việc, tôi rút ra các điểm quan trọng:

* Trong hệ thống RAG, retrieval tốt là điều kiện cần, nhưng không đủ:

  * Dù có context vẫn có thể trả lời sai nếu prompting và verification không chặt

* Cân bằng giữa 3 yếu tố:

  * Quality (độ đúng, độ đầy đủ)
  * Cost (token/API)
  * Latency (tốc độ phản hồi)

* Chỉ số đánh giá cần nhìn tổng hợp:

  * Judge score, faithfulness, relevancy, hit rate, MRR
  * Không tối ưu một metric đơn lẻ để tránh overfit benchmark

* Failure analysis có giá trị thực tế khi gắn với root-cause theo tầng hệ thống:

  * Chunking
  * Ingestion
  * Retrieval
  * Prompting

---

## 3. Giải quyết vấn đề (Problem Solving)

### Vấn đề 1: Agent trả lời tự tin trong trường hợp thiếu dữ liệu

* Triệu chứng:

  * Câu trả lời có xu hướng “đoán” thay vì nói rõ không đủ thông tin

* Tác động:

  * Tăng lỗi Hallucination/Unsupported Assertion

* Cách xử lý:

  * Điều chỉnh policy trả lời theo hướng ưu tiên grounding
  * Khuyến nghị bổ sung guardrail: nếu không có bằng chứng thì trả lời “Không rõ từ ngữ cảnh”

---

### Vấn đề 2: Câu trả lời lạc đề với câu hỏi open-ended

* Triệu chứng:

  * Câu trả lời đúng một phần nhưng sai trọng tâm

* Tác động:

  * Giảm score Accuracy và Relevancy

* Cách xử lý:

  * Đề xuất intent router (factual / open-ended / ambiguous)
  * Dùng answer template theo từng loại intent

---

### Vấn đề 3: Khó phân biệt lỗi Retrieval hay Generation khi debug

* Triệu chứng:

  * Nhật ký expected_ids/retrieved_ids chưa đầy đủ cho mỗi case

* Tác động:

  * Failure analysis không tách bạch được nguyên nhân gốc

* Cách xử lý:

  * Đề xuất nâng cấp instrumentation và logging retrieval theo từng case
  * Chạy lại benchmark sau khi sửa để xác thực improvement

---

## 4. Kết quả và nhận xét

Theo kết quả tổng hợp hiện tại:

* v1: pass 47/50, avg_judge_score 4.4428
* v2: pass 41/50, avg_judge_score 4.3048
* Regression gate: BLOCK (chất lượng giảm)

Nhận xét:

* v2 có ưu điểm về chi phí và latency
* Tuy nhiên chất lượng giảm rõ ràng nên chưa đạt điều kiện release
* Hướng đúng là fix các nhóm lỗi grounding/intent trước khi tiếp tục tối ưu cost

---

## 5. Tự đánh giá

Tôi tự đánh giá mức hoàn thành: **Tốt**

**Điểm đã làm được:**

* Hoàn thành phần agent theo đúng vai trò được giao
* Hoàn thành failure analysis có cấu trúc, có 5 Whys, có action plan
* Đưa ra được các đề xuất khả thi để tăng pass rate cho lần benchmark tiếp theo

**Điểm cần cải thiện:**

* Chưa instrument đủ retrieval-level logs cho tất cả test cases
* Cần bổ sung bộ test riêng cho hard-case để validate thay đổi prompt trước khi chạy full benchmark
* Cần tiếp tục chuẩn hóa tiêu chí pass/fail để tránh đánh giá cảm tính

---

## 6. Kế hoạch lần tiếp theo

* P0: Bổ sung grounding guardrail + self-check claim-evidence trong agent
* P1: Thêm intent router và answer templates theo loại câu hỏi
* P2: Nâng cấp retrieval instrumentation + benchmark lại + cập nhật failure clusters

**Mục tiêu:**
Nâng pass rate v2 lên ≥ 90% và chuyển regression decision từ BLOCK sang APPROVE.

