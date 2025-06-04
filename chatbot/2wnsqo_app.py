# streamlit run app.py
import os
import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# 환경 변수 설정 (너의 키로 대체)
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# ---- 함수 정의 ----

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("jhgan/ko-sroberta-multitask")

@st.cache_resource
def load_faiss_index(index_path: str):
    return faiss.read_index(index_path)

@st.cache_data
def load_chunks(chunk_path: str):
    with open(chunk_path, "r", encoding="utf-8") as f:
        return [c.strip() for c in f.read().split("\n\n") if c.strip()]

def search_similar_chunks(question: str, index, chunks, model, top_k=3):
    embedding = model.encode([question])
    _, indices = index.search(np.array(embedding).astype("float32"), top_k)
    return [chunks[i] for i in indices[0]]

def build_rag_chain():
    prompt = PromptTemplate.from_template(
        "당신은 의학전공을 하여 저희의 질문에 대답을 잘 해주는 챗봇입니다"
        "만약 진료혹은 약, 의학과 관련이 없는 질문이면 질문이 주제와 다르다고 하면 됩니다"
        "다음은 사용자의 질문에 답하기 위한 참고 문서입니다:\n\n"
        "{context}\n\n"
        "---\n\n"
        "질문: {question}\n"
        "답변:"
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return prompt | llm | StrOutputParser()

def get_rag_answer(question: str, index, chunks, model, rag_chain):
    top_chunks = search_similar_chunks(question, index, chunks, model)
    context = "\n".join(top_chunks)
    return rag_chain.invoke({"context": context, "question": question})

# ---- Streamlit UI ----

st.set_page_config(page_title="🧠 건강 RAG 챗봇", page_icon="💬")
st.title("💬 건강 질의응답 RAG 챗봇")

# 모델 & 데이터 로드
embed_model = load_embedding_model()
index = load_faiss_index("./embedding/QA_random_pair_part2_index1.index")
chunks = load_chunks("./embedding/QA_random_pair_part2_chunks1.txt")
rag_chain = build_rag_chain()

# 사용자 입력
question = st.text_input("❓ 궁금한 건강 관련 질문을 입력하세요:", key="input")

if st.button("질문하기") and question.strip():
    with st.spinner("답변 생성 중..."):
        answer = get_rag_answer(question, index, chunks, embed_model, rag_chain)
    st.markdown("### 📌 답변:")
    st.success(answer)
