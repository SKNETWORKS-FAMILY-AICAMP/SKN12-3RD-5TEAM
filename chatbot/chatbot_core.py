import numpy as np
import config
from rag.vector_store import load_category_vector_db, load_vector_db_by_path
from rag.embedder import load_embedder
from llm.category_classifier import classify_category_with_llm
from llm.router import get_llm_by_category
from llm.responder import build_rag_chain
from llm.chatbot_llm import chatbot_response
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# ---- 초기화 ----
embedder = load_embedder()
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
path, _ = config.VECTOR_DB_PATHS.get("treatment", config.VECTOR_DB_PATHS["default"])
faiss_db = FAISS.load_local(
    folder_path=path,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True  # 직접 만든 경우에만!
)

category_texts, category_categories, category_embeddings = load_category_vector_db()

# 💡 category_embeddings (N, D), normalize
def normalize(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)
category_embeddings_norm = normalize(category_embeddings.numpy())

def run_chatbot_pipeline(user_input: str, session_id: str = "default") -> str:
    # 1. 입력 임베딩 + 정규화
    input_emb = embedder.encode([user_input])[0]  # (D,)
    input_emb_norm = input_emb / (np.linalg.norm(input_emb) + 1e-8)
    
    # 2. 코사인 유사도(0~1) top-k
    sims = np.dot(category_embeddings_norm, input_emb_norm)  # (N,)
    top_k = 3
    top_idx = np.argsort(sims)[-top_k:][::-1]
    top_sims = sims[top_idx]
    max_sim = top_sims[0]
    
    if max_sim < 0.5:
        # context 없이 바로 챗봇 LLM에 전달
        return chatbot_response(user_input, "", session_id=session_id)
    
    # 3. context 확보, 카테고리 분류 LLM
    retrieved_examples = [
        (category_texts[i], category_categories[i], float(top_sims[j]))
        for j, i in enumerate(top_idx)
        if top_sims[j] >= 0.5
    ]
    # 만약 아무것도 없으면, context 없이 바로 응답
    if not retrieved_examples:
        return chatbot_response(user_input, "", session_id=session_id)

    predicted_category = classify_category_with_llm(user_input, retrieved_examples)

    # 4. 카테고리별 벡터 DB에서 문서 재검색 (context 추출)
    if predicted_category == "treatment":
        results = faiss_db.similarity_search(user_input, k=3)
        context = "\n\n".join([doc.page_content for doc in results])
    else:
        index_file, chunks_file = config.VECTOR_DB_PATHS.get(predicted_category, config.VECTOR_DB_PATHS["default"])
        context_docs, context_index = load_vector_db_by_path(index_file, chunks_file)
        doc_emb = embedder.encode([user_input])[0]
        doc_emb_norm = doc_emb / (np.linalg.norm(doc_emb) + 1e-8)
        context_docs_norm = np.array([c / (np.linalg.norm(c) + 1e-8) for c in context_index.reconstruct_n(0, context_index.ntotal)])
        sims2 = np.dot(context_docs_norm, doc_emb_norm)

        # 유사도 0.5 이상만 추출 (최대 3개)
        selected_idx = [i for i in np.argsort(sims2)[::-1] if sims2[i] >= 0.5][:3]

        if selected_idx:
            top_docs = [context_docs[i] for i in selected_idx]
            context = "\n".join(top_docs)
        else:
            context = ""   # 유사한 문장이 없으면 context 없이!

    # 5. 카테고리별 전문 LLM → 1차 답변
    category_llm = get_llm_by_category(predicted_category)
    rag_chain = build_rag_chain(category_llm)
    expert_response = rag_chain.invoke({"question": user_input, "context": context})

    # 6. 최종 챗봇 LLM
    return chatbot_response(user_input, expert_response, session_id=session_id)
