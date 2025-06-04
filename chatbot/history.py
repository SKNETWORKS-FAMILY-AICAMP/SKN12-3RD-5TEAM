# chatbot/history.py

# 추천 (최신 langchain-core)
from langchain_core.chat_history import InMemoryChatMessageHistory

# 세션별 대화 히스토리 저장소
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
