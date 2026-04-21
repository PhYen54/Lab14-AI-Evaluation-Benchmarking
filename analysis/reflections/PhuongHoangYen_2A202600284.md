# Báo cáo cá nhân - Lab Day 14: AI Evaluation Factory

**Họ và tên:** Phương Hoàng Yến
**MSSV:** 2A202600284
**Vai trò trong nhóm:** Data Lead / AI Data Engineer

---

## 1. Đóng góp kỹ thuật (Engineering Contribution)
Xây dựng dữ liệu cho toàn bộ hệ thống Benchmark. Các đóng góp cụ thể bao gồm:

- **Xây dựng Pipeline SDG (Synthetic Data Generation):** Thiết kế script `data/synthetic_gen.py` tích hợp OpenAI API để tự động hóa việc tạo dữ liệu kiểm thử từ các tài liệu nghiệp vụ trong thư mục `docs/`.
- **Cấu trúc hóa Golden Dataset:** Thiết kế định dạng dữ liệu chuẩn **JSONL**, đảm bảo mỗi test case đều có các trường bắt buộc: `question`, `expected_answer`, đặc biệt là `expected_retrieval_ids` để phục vụ tính toán Hit Rate và MRR.
- **Thiết kế Hard Cases:** Trực tiếp viết Prompt Engineering để ép Model sinh ra ít nhất 30% dữ liệu thuộc nhóm "khó" bao gồm: **Adversarial** (tấn công prompt), **Edge Cases** (thông tin ngoài ngữ cảnh), và **Ambiguous** (câu hỏi mập mờ).
- **Tối ưu hóa Batch Processing:** Triển khai cơ chế chia nhỏ request (batching) khi sinh dữ liệu để vượt qua giới hạn Output Token và tránh lỗi Rate Limit của API.

---

## 2. Kiến thức kỹ thuật (Technical Depth)
Qua quá trình thực hiện, em đã rút ra được các bài học chuyên sâu về kỹ thuật dữ liệu cho AI:

- **Sự quan trọng của Ground Truth:** Hiểu rõ rằng nếu không có `expected_retrieval_ids` khớp với metadata của Vector DB, mọi nỗ lực đo lường Hit Rate của nhóm Backend sẽ trở nên vô nghĩa.
- **Kỹ thuật điều khiển Output LLM:** Việc sử dụng `response_format={"type": "json_object"}` yêu cầu sự đồng bộ khắt khe giữa System Prompt và User Prompt để tránh lỗi cấu trúc khi parse dữ liệu.
- **Cost & Performance Balance:** Nhận thức được việc sinh dữ liệu hàng loạt cần sự cân bằng giữa chất lượng câu hỏi và chi phí API thông qua việc tối ưu số lượng câu hỏi trên mỗi request.

---

## 3. Giải quyết vấn đề (Problem Solving)
Em đã đối mặt và xử lý ba thách thức kỹ thuật lớn trong quá trình làm Lab:

**Vấn đề 1: Lỗi cấu trúc JSON khi sử dụng JSON Mode của OpenAI.** Ban đầu, Model trả về danh sách (List) trực tiếp khiến script bị lỗi khi parse. Em đã xử lý bằng cách bọc danh sách vào một Root Object có key là `qa_pairs` và cập nhật lại hàm xử lý hậu kỳ để đảm bảo tính ổn định của dữ liệu đầu ra.

**Vấn đề 2: Số lượng câu hỏi sinh ra bị thiếu (Yêu cầu 50 nhưng chỉ nhận được 10).** Nguyên nhân do giới hạn Output Token của Model GPT-4o. Em đã giải quyết bằng cách triển khai chiến lược "Chia để trị" (Batching) trong hàm `main()`, chia nhỏ 50 câu hỏi thành 5 đợt gọi API riêng biệt, giúp đảm bảo cả số lượng và độ chi tiết của câu trả lời.

**Vấn đề 3: Sai lệch ID tài liệu trong Golden Set.** Model thường tự bịa ra ID tài liệu khi không có hướng dẫn cụ thể. Em đã sửa lỗi bằng cách truyền trực tiếp `file_name` từ vòng lặp đọc folder vào Prompt và thêm một bước hậu kiểm (post-processing) để ép cứng ID chuẩn vào từng case, đảm bảo tính chính xác cho việc chấm điểm Retrieval.

---

## 4. Tự đánh giá
Em tự đánh giá phần hoàn thành nhiệm vụ ở mức **Tốt**. Em đã bàn giao cho nhóm một bộ `golden_set.jsonl` chất lượng cao, đủ độ khó để làm lộ diện các lỗi Hallucination của Agent.

**Điểm cần cải thiện:**
- Nếu có thêm thời gian, em sẽ tích hợp thêm thư viện `instructor` để quản lý Schema dữ liệu chặt chẽ hơn thay vì dùng `json.loads` thủ công.
- Cải tiến script để tự động nhận diện các đoạn văn bản quá dài và thực hiện "Recursive Character Text Splitting" trước khi gửi vào SDG pipeline để tránh mất mát thông tin.

---
*Ngày nộp báo cáo: 21/04/2026* **Người báo cáo** *(Ký tên)*