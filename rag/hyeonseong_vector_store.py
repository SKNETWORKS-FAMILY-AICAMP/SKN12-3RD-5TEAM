# 용자 질문에 대해 관련 청크를 검색하고 RAG 방식으로 GPT 응답을 생성
# 최초 작성일 : 2025-06-02
# 최초 작성자 : 손현성

import os
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

class Vector_store:
    """
    벡터 저장소 클래스: 
    - FAISS 인덱스와 텍스트 청크를 불러와서 
    - 사용자 질문에 대해 관련 청크를 검색하고 
    - RAG 방식으로 GPT 응답을 생성함
    """
    def __init__(
        self,
        api_key: str,
        chunk_path: str,
        index_path: str,
        embedding_model: str = "jhgan/ko-sroberta-multitask",
        chat_model: str = "gpt-4o-mini"
    ):
        """
        클래스 초기화 함수

        Input:
            api_key (str): OpenAI API 키
            chunk_path (str): 텍스트 청크가 저장된 파일 경로
            index_path (str): FAISS 인덱스 파일 경로
            embedding_model (str): 임베딩 모델 이름
            chat_model (str): 사용할 GPT 모델 이름
        """
        self.chunks = self.load_chunks(chunk_path)
        self.index = self.load_faiss_index(index_path)
        self.embedding_model = self.load_embedding_model(embedding_model)
        self.llm = self.connect_gpt(api_key, chat_model)
        self.parser = StrOutputParser()
        self.rag_chain = self.build_rag_chain()

    def load_chunks(self, chunk_path: str):
        """
        텍스트 청크 파일을 불러와 리스트로 반환

        Input:
            chunk_path (str): 청크 텍스트 파일 경로

        Return:
            list: 청크 문자열 리스트
        """
        with open(chunk_path, "r", encoding="utf-8") as f:
            return [c.strip() for c in f.read().split("\n\n") if c.strip()]

    def load_faiss_index(self, index_path: str):
        """
        저장된 FAISS 인덱스를 불러옴

        Input:
            index_path (str): FAISS 인덱스 파일 경로

        Return:
            faiss.Index: 불러온 인덱스 객체
        """
        return faiss.read_index(index_path)

    def load_embedding_model(self, model_name: str):
        """
        임베딩 모델 로드

        Input:
            model_name (str): SentenceTransformer 모델 이름

        Return:
            SentenceTransformer: 임베딩 모델 객체
        """
        return SentenceTransformer(model_name)

    def connect_gpt(self, api_key: str, model_name: str):
        """
        GPT 언어 모델 연결

        Input:
            api_key (str): OpenAI API 키
            model_name (str): 사용할 GPT 모델 이름

        Return:
            ChatOpenAI: 연결된 언어 모델 객체
        """
        return ChatOpenAI(openai_api_key=api_key, model=model_name, temperature=0)

    def build_rag_chain(self):
        """
        RAG(Retrieval-Augmented Generation) 체인을 구성

        Return:
            RunnableSequence: LLM과 프롬프트 체인을 연결한 실행 객체
        """
        prompt = PromptTemplate.from_template(
            "당신은 환자들에게 질병 및 약에 대해 설명해주는 전문적인 챗봇입니다.\n"
            "다음은 사용자의 질문에 답하기 위한 문서입니다. 다음 문서를 바탕으로 대답해주세요:\n\n"
            "{context}\n\n"
            "---\n\n"
            "질문: {question}\n"
            "답변:"
        )
        return prompt | self.llm | self.parser

    def rag_answer(self, question: str, top_k: int = 3):
        """
        질문에 대한 관련 문서를 검색하여 LLM으로 응답 생성

        Input:
            question (str): 사용자의 질문
            top_k (int): 검색할 상위 유사 청크 수 (기본값: 3)

        Return:
            str: GPT가 생성한 응답 텍스트
        """
        embedding = self.embedding_model.encode([question])
        _, indices = self.index.search(np.array(embedding).astype("float32"), top_k)
        top_chunks = [self.chunks[i] for i in indices[0]]
        context = "\n".join(top_chunks)
        return self.rag_chain.invoke({"context": context, "question": question})