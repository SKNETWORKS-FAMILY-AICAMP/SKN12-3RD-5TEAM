
from preprocessing.preprocess import load_all_documents, chunk_documents
from rag.vector_store import build_faiss_index

if __name__ == "__main__":
    # 문서 로딩 및 전처리
    docs = load_all_documents()

    # Chunking
    chunks = chunk_documents(docs)

    # FAISS 인덱싱 및 저장
    build_faiss_index(chunks)
