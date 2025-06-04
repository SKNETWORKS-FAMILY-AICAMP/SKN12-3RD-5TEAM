import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()  # .env → 환경변수 등록
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 환경변수 등록 (langchain, openai 내부에서 사용)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# BASE 경로 (현재 파일 기준 최상위 디렉토리)
BASE_DIR = Path(__file__).resolve().parent

# JSON 데이터 경로
JSON_PATHS = {
    "medicine": BASE_DIR / "data/row_docs/kwon/medicine/3.개방데이터/1.데이터/Training/01.원천데이터",
    "treatment": BASE_DIR / "data/row_docs/kwon/medical_treatment/09.필수의료_의학지식_데이터/3.개방데이터/1.데이터/Training/01.원천데이터",
    "assist_answer": BASE_DIR / "data/row_docs/kwon/medical_assistance_and_convergence_areas/Training/OnChwon/answer",
    "assist_question": BASE_DIR / "data/row_docs/kwon/medical_assistance_and_convergence_areas/Training/OnChwon/question",
    "internal_answer": BASE_DIR / "data/row_docs/kwon/internal_medicine_and_surgery/Training/OnChwon/answer",
    "internal_question": BASE_DIR / "data/row_docs/kwon/internal_medicine_and_surgery/Training/OnChwon/question"
}

# 벡터 DB 저장 경로
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db")
FAISS_INDEX_FILE = os.path.join(VECTOR_DB_PATH, "faiss_index.faiss")
DOC_EMBEDDINGS_FILE = os.path.join(VECTOR_DB_PATH, "doc_embeddings.pkl")
...

# 4. 카테고리별 벡터DB 경로 (확장 및 병합 지원)
VECTOR_DB_PATHS = {
    "medicine": (
         os.path.join(VECTOR_DB_PATH, "medicine/pilsu_pro_no_prepro_index1.faiss"),
         os.path.join(VECTOR_DB_PATH, "medicine/pilsu_pro_no_prepro_chunks1.txt"),
     ),
     "treatment": (
        # 보류
         os.path.join(VECTOR_DB_PATH, "treatment/RAG_Output/faiss_medical/faiss_index"),
         os.path.join(VECTOR_DB_PATH, "")
     ),
     "assist_answer": (
         os.path.join(VECTOR_DB_PATH, "QA_random_pair_part2_index1.index"),
         os.path.join(VECTOR_DB_PATH, "QA_random_pair_part2_chunks1.txt"),
     ),
     "assist_question": (
         os.path.join(VECTOR_DB_PATH, "QA_random_pair_part2_index1.index"),
         os.path.join(VECTOR_DB_PATH, "QA_random_pair_part2_chunks1.txt"),
     ),
     "internal_answer": (
         os.path.join(VECTOR_DB_PATH, "QA_random_pair_part1_index1.index"),
         os.path.join(VECTOR_DB_PATH, "QA_random_pair_part1_chunks1.txt"),
    ),
     "internal_question": (
         os.path.join(VECTOR_DB_PATH, "QA_random_pair_part1_index1.index"),
         os.path.join(VECTOR_DB_PATH, "QA_random_pair_part1_chunks1.txt"),
     ),
     "default": (
         os.path.join(VECTOR_DB_PATH, "QA_random_pair_part2_index1.index"),
         os.path.join(VECTOR_DB_PATH, "QA_random_pair_part2_chunks1.txt"),
     )
}