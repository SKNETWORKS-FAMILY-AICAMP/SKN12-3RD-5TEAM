# ✅ llm/router.py

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# 전용 LLM 인스턴스 정의 (카테고리별)
llm_medicine = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
llm_treatment = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
llm_assist = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
llm_internal = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
llm_default = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# 카테고리 → LLM 매핑
target_llms = {
    "medicine": llm_medicine,
    "treatment": llm_treatment,
    "assist_answer": llm_assist,
    "assist_question": llm_assist,
    "internal_answer": llm_internal,
    "internal_question": llm_internal,
    "default":llm_default
}

# ✅ 카테고리 기반 LLM 선택 함수
def get_llm_by_category(category: str):
    return target_llms.get(category, llm_default)