import json
import asyncio
import os
from typing import List, Dict
from openai import AsyncOpenAI
from dotenv import load_dotenv  # Thêm dòng này

# Nạp các biến từ file .env vào os.environ
load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Giả lập việc gọi LLM để tạo dữ liệu (Students will implement this)
async def generate_qa_from_text(text: str, num_pairs: int = 50) -> List[Dict]:
    """
    Sửa lỗi: Đưa danh sách vào một key cố định để khớp với json_object mode.
    """
    print(f"Generating {num_pairs} QA pairs from text...")
        
    prompt = f"""
    Phân tích đoạn văn bản sau và tạo {num_pairs} cặp câu hỏi/trả lời để benchmark hệ thống RAG.
    
    CONTEXT:
    ---
    {text}
    ---

    YÊU CẦU CHI TIẾT:
    1. Tính đa dạng: Tạo các câu hỏi về sự thật, tóm tắt và suy luận.
    2. Hard Cases (BẮT BUỘC):
        - Ít nhất 5 câu 'Adversarial'.
        - Ít nhất 5 câu 'Edge Case' (Out-of-context).
        - Ít nhất 5 câu 'Ambiguous'.
    3. Định danh tài liệu: Mỗi case PHẢI đi kèm với `expected_retrieval_ids`.

    FORMAT TRẢ VỀ (BẮT BUỘC PHẢI LÀ JSON OBJECT CÓ KEY 'qa_pairs'):
    {{
      "qa_pairs": [
        {{
          "question": "Câu hỏi",
          "expected_answer": "Câu trả lời kỳ vọng",
          "context": "Đoạn trích liên quan",
          "metadata": {{
            "difficulty": "easy/medium/hard",
            "type": "fact-check/adversarial/out-of-context/ambiguous",
            "reasoning": "Giải thích"
          }}
        }}
      ]
    }}
    """

    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Bạn là chuyên gia về AI Data Engineering. Luôn trả về JSON Object có key 'qa_pairs' chứa danh sách các câu hỏi."},
                {"role": "user", "content": prompt}
            ],
            # Mode này yêu cầu chữ 'json' xuất hiện trong prompt và kết quả phải là Object
            response_format={"type": "json_object"}
        )
        
        # Parse chuỗi JSON từ API
        raw_content = response.choices[0].message.content
        result = json.loads(raw_content)
        
        # Lấy danh sách từ key 'qa_pairs'
        qa_list = result.get("qa_pairs", [])
        
        # Debug nếu danh sách vẫn rỗng
        if not qa_list:
            print(f"⚠️ Cảnh báo: LLM trả về JSON hợp lệ nhưng key 'qa_pairs' trống hoặc sai cấu trúc.")
            
        return qa_list

    except Exception as e:
        print(f"❌ Lỗi khi chạy SDG: {e}")
        return []

async def main():
    all_qa_pairs = []
    num_batches = 5 
    
    # Giả sử bạn đang đọc nội dung từ 1 file docs
    raw_text = "AI Evaluation là một quy trình kỹ thuật nhằm đo lường chất lượng..."

    for i in range(num_batches):
        print(f"--- Batch {i+1}/{num_batches} ---")
        # Mỗi lần gọi chỉ yêu cầu 10 câu để đảm bảo chất lượng và không bị cắt token
        batch_pairs = await generate_qa_from_text(raw_text, num_pairs=10)
        all_qa_pairs.extend(batch_pairs)
        
        # Nghỉ 1 chút để tránh Rate Limit nếu cần
        await asyncio.sleep(1) 

    # Ghi toàn bộ 50 câu vào file
    with open("golden_set.jsonl", "w", encoding="utf-8") as f:
        for pair in all_qa_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            
    print(f"✅ Tổng cộng đã ghi: {len(all_qa_pairs)} dòng vào data/golden_set.jsonl")

if __name__ == "__main__":
    asyncio.run(main())