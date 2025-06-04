import torch
from langchain_huggingface import HuggingFaceEmbeddings

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"현재 디바이스: {device.upper()}")

# HuggingFace 임베딩 모델 초기화
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device}
)
