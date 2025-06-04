# ✅ llm/category_classifier.py

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ 카테고리 분류 전용 LLM 호출 함수
def classify_category_with_llm(input_text, retrieved_examples):
    """
    retrieved_examples: List of (text, category, score)
    """
    fewshot = ""
    for i, (ex, cat, score) in enumerate(retrieved_examples):
        fewshot += f"예시 {i+1}:\n텍스트: \"{ex}\"\n카테고리: {cat}\n\n"

    valid_categories = [
        "medicine",
        "treatment",
        "assist_answer",
        "assist_question",
        "internal_answer",
        "internal_question"
    ]

    prompt = f"""
당신은 텍스트를 아래 6가지 카테고리 중 하나로만 분류해야 하는 전문가입니다.

선택 가능한 카테고리:
- medicine
- treatment
- assist_answer
- assist_question
- internal_answer
- internal_question

아래는 분류된 예시들입니다:
{fewshot}

이제 입력된 텍스트를 위 카테고리 중 **하나만** 선택하여 분류하세요.
카테고리 이름만 출력하세요. 다른 설명 없이 정확히 하나의 카테고리명만 출력해야 합니다.

입력 텍스트: "{input_text}"

정답 카테고리:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()