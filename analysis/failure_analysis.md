# Báo cáo Phân tích Thất bại (Failure Analysis Report)
Ngày: 2026-04-21 | Dataset: 50 test cases

## 1. Tổng quan Benchmark

### Kết quả v1 (MainAgent - gpt-4o)
- **Tổng số cases:** 50
- **Pass/Fail:** 47/3 (94.0% pass)
- **Điểm LLM-Judge trung bình:** 4.4428/5.0
- **Chỉ số Retrieval:**
    - Hit Rate: 0.82
    - MRR: 0.75
- **Chỉ số RAGAS:**
    - Faithfulness: 0.82
    - Relevancy: 0.82
- **Performance:**
    - Chi phí tổng: 0.000522 USD
    - Tokens tổng: 3,486
    - Độ trễ trung bình: 2.90s
- **Judge metrics:**
    - Agreement rate: 0.93
    - Cohen's Kappa: N/A

### Kết quả v2 (MainAgentV2 - gpt-4.1)
- **Tổng số cases:** 50
- **Pass/Fail:** 41/9 (82.0% pass)
- **Điểm LLM-Judge trung bình:** 4.3048/5.0 (-0.1380 vs v1)
- **Chỉ số Retrieval:**
    - Hit Rate: 0.84 (+0.02 improvement)
    - MRR: 0.73 (-0.02 vs v1)
- **Chỉ số RAGAS:**
    - Faithfulness: 0.84 (+0.02 improvement)
    - Relevancy: 0.84 (+0.02 improvement)
- **Performance:**
    - Chi phí tổng: 0.00032 USD (-38.7% vs v1)
    - Tokens tổng: 2,131 (-38.9% vs v1)
    - Độ trễ trung bình: 2.345s (-19.1% vs v1)
- **Judge metrics:**
    - Agreement rate: 0.91
    - Cohen's Kappa: N/A

### Kết luận Regression Gate
- **Quyết định:** ❌ **BLOCK**
- **Lý do chính:** 
  - Chất lượng giảm 0.1380 điểm (4.4428 → 4.3048)
  - Pass rate giảm từ 94% xuống 82% (3 lỗi → 9 lỗi, +200%)
  - Mặc dù chi phí giảm 38.7% và latency giảm 19.1%, nhưng không bù được sự sụt giảm chất lượng
  - Risk: Giảm độ tin tưởng của người dùng không chấp nhận trên v2

## 2. Phân nhóm lỗi (Failure Clustering)

Phân tích trên toàn bộ case fail của v2 (11 cases), vì đây là bản candidate cho release.

| Nhóm lỗi | Số lượng | Tỷ lệ trên fail | Biểu hiện chính | Nguyên nhân dự kiến |
|----------|----------|-----------------|------------------|---------------------|
| Unsupported Assertion khi ground truth là "Không rõ" | 4 | 36.4% | Agent khẳng định chắc chắn dù dữ liệu không đủ | Prompt chưa ép điều kiện "không đủ thông tin thì nói không rõ" |
| Off-topic / Non-answer cho câu hỏi mở | 5 | 45.5% | Trả lời lệch câu hỏi "áp dụng thế nào", "thách thức gì", "tác động gì" | Thiếu chiến lược decomposition + thiếu kiểm tra semantic alignment trước khi trả lời |
| Contradiction với fact trực tiếp trong ngữ cảnh | 2 | 18.2% | Trả lời ngược với fact cơ bản (ví dụ "quy trình xác định") | Prompt grounding yếu, không có bước self-check đối chiếu câu trả lời với context |

Quan sát quan trọng:
- Trong các case fail, trường `retrieved_ids` đều rỗng và chỉ số retrieval bằng 0.0.
- Điều này cho thấy pipeline retrieval chưa được instrument đúng để phục vụ đánh giá theo ID, hoặc bộ dữ liệu chưa gắn expected/retrieved IDs đầy đủ. Kết quả là rất khó chẩn đoán chính xác lỗi do retrieval hay generation.

## 3. Phân tích 5 Whys (3 case tệ nhất của v2)

### Case 1: Low-score failures - Pattern Analysis
1. Symptom: Agent trả lời sai/không nhất quán với ground truth.
2. Why 1: Agent không bám sát fact trực tiếp trong context dù câu hỏi rất đóng.
3. Why 2: Prompt hiện tại ưu tiên trả lời trôi chảy hơn là kiểm chứng từng mệnh đề với context.
4. Why 3: Không có bước "claim verification" trước khi phát sinh final answer.
5. Why 4: Thiết kế agent thiếu guardrail cho câu hỏi factual yes/no.
6. Root Cause: Thiếu cơ chế self-check bắt buộc giữa câu trả lời và bằng chứng context.

### Case 2: Semantic Mismatch - Off-topic Responses
1. Symptom: Câu trả lời không khớp kỳ vọng đánh giá, độ chính xác thấp.
2. Why 1: Agent diễn giải từ khóa không chuẩn (nhiễu ngôn ngữ) và đưa kết luận thiếu chắc chắn.
3. Why 2: Không có normalization cho câu hỏi tiếng Việt có biến thể diễn đạt.
4. Why 3: Pipeline không phân loại intent (factual/paraphrase/unclear) trước khi sinh câu trả lời.
5. Why 4: Không có rule xử lý ambiguity (yêu cầu làm rõ hoặc trả lời bảo thủ).
6. Root Cause: Thiếu lớp tiền xử lý semantic normalization + intent routing.

### Case 3: Model Hallucination - Inference Beyond Context
1. Symptom: Agent trả lời lệch trọng tâm, không nêu được hậu quả đúng kỳ vọng.
2. Why 1: Context không chứa trực tiếp hậu quả nhưng agent vẫn suy diễn mạnh.
3. Why 2: Prompt chưa quy định rõ khi nào được suy luận và khi nào phải trả lời "không rõ từ ngữ cảnh".
4. Why 3: Judge rubric phạt nặng việc suy diễn ngoài ground truth, nhưng agent chưa được calibrate theo rubric này.
5. Why 4: Không có vòng lặp phản hồi từ kết quả judge để tinh chỉnh policy trả lời.
6. Root Cause: Mismatch giữa policy sinh câu trả lời của agent và tiêu chí chấm điểm benchmark.

## 4. Kế hoạch cải tiến (Action Plan)

### Ưu tiên P0 (thực hiện ngay)
- [ ] Bổ sung "Grounding Guardrail" trong system prompt:
    - Nếu context không đủ, bắt buộc trả lời "Không rõ từ ngữ cảnh đã cho".
    - Cấm khẳng định chắc chắn khi thiếu bằng chứng.
- [ ] Thêm bước self-check trước khi trả lời:
    - Trích 1-2 câu bằng chứng từ context cho mỗi claim chính.
    - Nếu không tìm được bằng chứng, hạ mức chắc chắn hoặc trả lời "không rõ".
- [ ] Chuẩn hóa câu hỏi tiếng Việt (normalization/paraphrase handling) để giảm lỗi hiểu sai intent.

### Ưu tiên P1 (sprint kế tiếp)
- [ ] **Intent Router:** Phân loại câu hỏi (factual yes/no, open-ended, ambiguous, definition-seeking, counterfactual).
    - Áp dụng template trả lời khác nhau cho mỗi loại intent
    - Ví dụ: Với yes/no factual → yêu cầu một và chỉ một claim + bằng chứng
- [ ] Thêm "answer-type template" theo intent để tránh off-topic.
- [ ] **Rubric Calibration:** Đào tạo agent theo rubric judge chính thức (accuracy/professionalism/safety).
    - Cấm suy luận ngoài context nếu accuracy được chấm dưới 3/5
    - Yêu cầu explicit evidence mapping cho mỗi claim
- [ ] **Thêm verification step:** 
    - Agent tự kiểm tra: "Có bằng chứng nào từ context support claim này không?"
    - Nếu không → downgrade confidence hoặc trả lời "không rõ"

### Ưu tiên P2 (hạ tầng đánh giá)
- [ ] **Instrumentation:** Hoàn thiện logging `expected_ids` và `retrieved_ids` cho tất cả case.
    - Hiện tại nhiều case thiếu expected_ids
    - Không rõ lỗi là do Retrieval hay Generation
- [ ] **Retrieval Analysis:** Nâng cấp Hit Rate/MRR calculation.
- [ ] Sửa retrieval eval (Hit Rate/MRR) để không mặc định 0.0 khi thiếu ID.
- [ ] **Regression Testing:** Chạy lại benchmark sau mỗi P1 fix để verify improvement.

## 5. Tiêu chí thành công cho lần chạy lại
- ✅ **Quality target:** Tăng pass rate v2 từ 82% lên **>=90%** (giảm errors từ 9 xuống <=5)
- ✅ **Grounding target:** Giảm số lỗi nhóm "Hallucination/Unsupported" ít nhất **50%** (từ ~3-4 xuống <=2)
- ✅ **Intent target:** Giảm số lỗi nhóm "Off-topic" ít nhất **40%** (từ ~5 xuống <=3)
- ✅ **Cost efficiency:** Giữ chi phí v2 thấp hơn v1 (current: v2 38.7% cheaper, tốt ✓)
- ✅ **Agreement rate:** Duy trì agreement_rate >= 0.90 (current: 0.91, tốt ✓)
- ✅ **Regression gate:** Chuyển quyết định từ **BLOCK → APPROVE** khi quality delta >= 0

## 6. Recommended Next Steps
1. **Immediate (Day 1):** Implement Grounding Guardrail + self-check in system prompt (P0)
2. **Short-term (Day 2-3):** Build Intent Router + Rubric Calibration (P1)
3. **Medium-term (Week 2):** Fix instrumentation + re-run benchmark (P2)
4. **Validation:** Compare v1 vs v2 again with fixes, target APPROVE decision
