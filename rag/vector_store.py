# ✅ rag/vector_store.py

import os
import pickle
import torch
import faiss
from config import VECTOR_DB_PATH


def save_vector_db(save_path, texts, categories, embeddings):
    """
    텍스트, 카테고리, 임베딩 정보를 경로에 저장.
    """
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "texts.pkl"), "wb") as f:
        pickle.dump(texts, f)
    with open(os.path.join(save_path, "categories.pkl"), "wb") as f:
        pickle.dump(categories, f)
    torch.save(embeddings, os.path.join(save_path, "embeddings.pt"))


def load_vector_db(load_path):
    """
    저장된 텍스트, 카테고리, 임베딩 불러오기.
    """
    texts = pickle.load(open(os.path.join(load_path, "texts.pkl"), "rb"))
    categories = pickle.load(open(os.path.join(load_path, "categories.pkl"), "rb"))
    embeddings = torch.load(os.path.join(load_path, "embeddings.pt"), map_location=torch.device('cpu'))
    return texts, categories, embeddings


def load_category_vector_db():
    """
    카테고리 분류용 벡터 DB (category 폴더 기준)
    """
    category_path = os.path.join(VECTOR_DB_PATH, "category")
    return load_vector_db(category_path)


def load_vector_db_by_path(index_path: str, chunks_path: str):
    """
    주어진 경로의 FAISS 인덱스와 텍스트 청크 파일을 로드
    """
    import faiss

    index = faiss.read_index(index_path)
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = [c.strip() for c in f.read().split("\n\n") if c.strip()]
    return chunks, index
