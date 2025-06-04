import os
import json
import pickle
from tqdm import tqdm
from langchain.vectorstores import FAISS
from rag.embedder import embedding_model
from config import VECTOR_DB_DIR

def build_faiss_index(documents, output_path=VECTOR_DB_DIR):
    print("문서 임베딩 중...")

    embeddings = []
    doc_ids = []
    categories = []

    for i, doc in enumerate(tqdm(documents, desc="Embedding documents")):
        embedding = embedding_model.embed_query(doc.page_content)
        embeddings.append(embedding)
        doc_ids.append(doc.metadata.get("c_id", f"doc_{i}"))
        categories.append(doc.metadata.get("type", "기타"))

    vector_db = FAISS.from_texts(
        texts=[doc.page_content for doc in documents],
        embedding=embedding_model
    )

    # 경로 구조: vector_db/faiss_index/
    faiss_dir = os.path.join(output_path, "faiss_index")
    os.makedirs(faiss_dir, exist_ok=True)

    vector_db.save_local(faiss_dir)

    with open(os.path.join(output_path, "doc_embeddings.pkl"), "wb") as f:
        pickle.dump(embeddings, f)

    with open(os.path.join(output_path, "doc_ids.json"), "w", encoding="utf-8") as f:
        json.dump(doc_ids, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_path, "categories.json"), "w", encoding="utf-8") as f:
        json.dump(categories, f, ensure_ascii=False, indent=2)

    print(f"FAISS 벡터 DB 저장 완료: {output_path}")
    return vector_db
