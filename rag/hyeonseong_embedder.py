# content 데이터를 임베딩하여 FAISS vector db 생성
# 생성된 벡터DB는 용량 문제로 깃허브에 못올렸으므로 vectordb_download()로 다운로드 하세요
# 최초 작성일 : 2025-06-02
# 최초 작성자 : 손현성

from sentence_transformers import SentenceTransformer
from preprocessing.hyeonseong_preprocess_jsonl import DataDownLoad
import os
import torch
import faiss
import gdown

vectordb_path = {
    'pilsu_pro_no_prepro_chunks1.txt':'https://drive.google.com/file/d/1koe7Dzj4aBqrEC4mByrksmzAoZkH-pjI/view?usp=sharing',
    'pilsu_pro_no_prepro_index1.faiss':'https://drive.google.com/file/d/1gwd8Ik_QCds6s8qBRTADTGuXMXaeo50o/view?usp=sharing'            
}
merged_data_path = './data/merged_data.txt'
vectordb_chunk_path = "./data/pilsu_pro_no_prepro_chunks1.txt"
vectordb_index_path = "./data/pilsu_pro_no_prepro_index1.faiss"

class Embedder:
    """
    임베딩하여 벡터DB 생성
    """
    def __init__(self, file_path:str=merged_data_path, model_name:str = "jhgan/ko-sroberta-multitask"):
        """
        클래스 초기화 함수

        Input:
            file_path (str): 데이터 파일 경로
            model_name (str): 임베딩에 사용할 모델 이름
        """
        self.model_name = model_name
        self.file_path = file_path
        self.index_path = vectordb_index_path
        self.chunk_path = vectordb_chunk_path
    
    def load_chunks(self)->list:
        """
        텍스트 파일에서 줄 단위로 청크를 불러오기

        Return:
            list: 청크(문장) 리스트
        """
        with open(self.file_path, "r", encoding="utf-8-sig") as f:
            chunks = [line.strip() for line in f if line.strip()]
        print(f"총 {len(chunks)}개의 청크를 불러왔습니다.")
        return chunks
    
    def get_embedding_model(self):
        """
        임베딩 모델 로드

        Return:
            SentenceTransformer: 임베딩 모델 객체
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"사용하는 디바이스 : {device}")
        return SentenceTransformer(self.model_name, device=device)
    
    def create_faiss_index(self, embeddings):
        """
        FAISS 인덱스를 생성하고 벡터를 추가

        Input:
            embeddings (np.ndarray): 임베딩 벡터 배열

        Return:
            faiss.IndexFlatL2: 생성된 FAISS 인덱스 객체
        """
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        print(f"FAISS 인덱스에 {len(embeddings)}개의 벡터가 추가되었습니다.")
        return index
    
    def save_index_and_chunks(self, index, chunks):
        """
        FAISS 인덱스와 청크를 파일로 저장

        Input:
            index (faiss.IndexFlatL2): 저장할 FAISS 인덱스 객체
            chunks (list): 텍스트 청크 리스트
        """
        faiss.write_index(index, self.index_path)
        with open(self.chunk_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(chunk + "\n\n")
        print(f"인덱스 저장 : {self.index_path}")
        print(f"청크 저장 : {self.chunk_path}")

    def vectordb_download(self, vectordb_path:str=vectordb_path):
        """
        생성된 벡터DB vector_db 디렉토리에 다운로드
        """
        download = DataDownLoad(file_name_and_path=vectordb_path, extract_base_dir='./vector_db')
        download.file_download()