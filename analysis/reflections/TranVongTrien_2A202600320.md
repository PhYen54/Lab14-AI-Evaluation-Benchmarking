# Báo cáo cá nhân - Lab Day 14: AI Evaluation Factory

**Họ và tên:** Trần Vọng Triển
**MSSV:** 2A202600320   
**Vai trò trong nhóm:** DevOps/Analyst

---
## 1. Đóng góp kỹ thuật (Engineering Contribution)

Chịu trách nhiệm xây dựng cơ chế **Regression Release Gate** và phân tích hiệu năng hệ thống từ góc nhìn DevOps/LLMOps. Các đóng góp chính bao gồm:

- Thiết kế và triển khai logic trong `main.py` để so sánh hai phiên bản Agent (V1 vs V2) dựa trên các chỉ số:

  * `score_delta` (chênh lệch chất lượng)
  * `cost_ratio` (tỷ lệ chi phí)
  * `latency_delta` (độ trễ)

- Áp dụng rule:

  * `APPROVE` nếu chất lượng tăng và chi phí trong ngưỡng cho phép
  * `BLOCK` nếu chất lượng giảm hoặc chi phí tăng quá mức
  * `REVIEW` cho các trường hợp biên

- Viết hàm `_aggregate_metrics()` để tính toán các chỉ số quan trọng từ benchmark:

  * Avg Judge Score, Hit Rate, MRR
  * Faithfulness, Relevancy (RAGAS)
  * Cost, Token usage, Latency
  * Pass/Fail rate

- Kết nối Benchmark Runner với Regression Gate để đảm bảo toàn bộ flow:

  ```
  Agent → Benchmark → Metrics → Regression Gate → Decision
  ```

- Xuất file `reports/summary.json` và `reports/benchmark_results.json` phục vụ phân tích và chấm điểm.

---

## 2. Kiến thức kỹ thuật (Technical Depth)
- Trade-off Quality vs Cost:
    - Hiểu rằng cải thiện nhỏ về score (delta thấp) không đáng nếu cost tăng mạnh
    - Regression Gate giúp tự động hóa quyết định release thay vì cảm tính
- Thiết kế Gate thực tế:
    - Không dùng threshold cứng duy nhất
    - Kết hợp nhiều yếu tố: Quality (primary), Cost (constraint), Latency (secondary)
- Observability trong AI system:
    - Log đầy đủ metrics giúp trace nguyên nhân regression
    - Summary.json đóng vai trò như "single source of truth”

---

## 3. Giải quyết vấn đề (Problem Solving)

### Vấn đề 1: Logic Regression ban đầu không đúng yêu cầu đề bài

- Ban đầu dùng threshold 0.2 → quá cao, không thực tế
-Sửa lại thành: So sánh trực tiếp score_delta > 0 + Kết hợp thêm cost_ratio <= 1.1 và > 1.3

### Vấn đề 2: Khó đề xuất tối ưu chi phí cụ thể

- Giải pháp: Phân tích theo cost/case + tokens
- Đưa ra 4 chiến lược rõ ràng: Cache judge, Model nhỏ hơn, Batch API, Early exit
## 4. Tự đánh giá

- Tôi tự đánh giá mức hoàn thành ở mức Tốt.
- Đã xây dựng được một Regression Gate hoạt động end-to-end, có thể:
    - Tự động quyết định release
    - Phân tích chi phí rõ ràng
    - Hỗ trợ debug quality vs cost

- Điểm cần cải thiện và bổ sung thêm:
    - Threshold động (adaptive threshold)
    - Weight cho từng metric (score > cost > latency)
    - Tích hợp CI/CD để tự động chạy benchmark khi push code


---
*Ngày nộp báo cáo: 21/04/2026* **Người báo cáo** *(Ký tên)*