# ! pip install langchain langchain-community openai sentence-transformers faiss-cpu
import os
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# 청크 불러오기
def load_chunks(chunk_path: str):
    with open(chunk_path, "r", encoding="utf-8") as f:
        return [c.strip() for c in f.read().split("\n\n") if c.strip()]

# FAISS 인덱스 불러오기
def load_faiss_index(index_path: str):
    return faiss.read_index(index_path)

# 유사한 청크 찾기
def search_faiss_index(question: str, index, chunks, model, top_k=3):
    embedding = model.encode([question])
    _, indices = index.search(np.array(embedding).astype("float32"), top_k)
    return [chunks[i] for i in indices[0]]

# RAG 체인 만들기
def build_rag_chain():
    prompt = PromptTemplate.from_template(
        "당신은 환자들에게 질병 및 약에 대해 설명해주는 전문적인 챗봇입니다"
        "다음은 사용자의 질문에 답하기 위한 문서입니다. 다음 문서를 바탕으로 대답해주세요:\n\n"
        "{context}\n\n"
        "---\n\n"
        "질문: {question}\n"
        "답변:"
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return prompt | llm | StrOutputParser()

# 최종 RAG 응답 생성
def rag_answer(question: str, index, chunks, model, rag_chain):
    top_chunks = search_faiss_index(question, index, chunks, model)
    context = "\n".join(top_chunks)
    return rag_chain.invoke({"context": context, "question": question})

# 실행 예시
if __name__ == "__main__":
    # 환경변수에서 API 키 불러오기
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")

    index = load_faiss_index("./QA_random_pair_part1_index1.faiss")
    chunks = load_chunks("./QA_random_pair_part1_chunks1.txt")
    model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    rag_chain = build_rag_chain()

    # 예시 질문
    question = "hiv는 무엇인가요?"
    answer = rag_answer(question, index, chunks, model, rag_chain)

    print("답변 :", answer)