import os
import torch
from sentence_transformers import SentenceTransformer
import faiss

def load_chunks(file_path: str) -> list:
    """텍스트 파일에서 줄 단위로 청크를 불러오기"""
    with open(file_path, "r", encoding="utf-8-sig") as f:
        chunks = [line.strip() for line in f if line.strip()]
    print(f"📄 총 {len(chunks)}개의 청크를 불러왔어.")
    return chunks

def get_embedding_model(model_name: str = "jhgan/ko-sroberta-multitask"):
    """임베딩 모델 로드"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 사용하는 디바이스: {device}")
    return SentenceTransformer(model_name, device=device)

def create_faiss_index(embeddings):
    """FAISS 인덱스를 생성하고 벡터를 추가"""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"✅ FAISS 인덱스에 {len(embeddings)}개의 벡터가 추가되었어.")
    return index

def save_index_and_chunks(index, index_path: str, chunks: list, chunk_path: str):
    """FAISS 인덱스와 청크를 저장"""
    faiss.write_index(index, index_path)
    with open(chunk_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n\n")
    print(f"💾 인덱스 저장: {index_path}")
    print(f"💾 청크 저장: {chunk_path}")

def embed_and_save_faiss(file_path: str, index_path: str, chunk_path: str):
    """전체 프로세스 실행: 텍스트 → 임베딩 → FAISS 인덱스 생성 → 저장"""
    chunks = load_chunks(file_path)
    model = get_embedding_model()
    print(f"🧠 임베딩 시작...")
    embeddings = model.encode(chunks, batch_size=64, show_progress_bar=True)
    index = create_faiss_index(embeddings)
    save_index_and_chunks(index, index_path, chunks, chunk_path)
    print("🎉 모든 작업이 완료되었어!")

# 사용 예시
if __name__ == "__main__":
    embed_and_save_faiss(
        file_path="./QA_random_pair_part2_preprocessed.txt",
        index_path="./embedding/QA_random_pair_part2_index1.index",
        chunk_path="./embedding/QA_random_pair_part2_chunks1.txt"
    )
