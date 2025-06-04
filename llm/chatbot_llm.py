# llm/chatbot_llm.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from chatbot.history import get_session_history

chatbot_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

chatbot_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    당신은 의료 질문에 전문적으로 답하는 AI 상담사입니다. 
    사용자의 질문과 history, 카테고리 전용 LLM 응답을 바탕으로 최적의 답변을 생성하세요. 
    만약 사용자의 질문이 이전 질문(history)요청, 의학, 약, 진료와 관련없는 질문을 받을 시 질문이 주제와 다르다고 말하세요."
    """),
    MessagesPlaceholder(variable_name="history"),
    ("human", """[사용자 질문]
{question}

[카테고리 전용 LLM 응답]
{draft_answer}

[최종 응답]""")
])

chatbot_chain = chatbot_prompt | chatbot_llm | StrOutputParser()

# 세션 기반 챗봇(히스토리 관리)
chatbot_with_history = RunnableWithMessageHistory(
    chatbot_chain,
    get_session_history,
    input_messages_key="question",   # input 파라미터명과 동일하게 맞추세요!
    history_messages_key="history"
)

def chatbot_response(question: str, draft_answer: str, session_id: str = "default") -> str:
    return chatbot_with_history.invoke(
        {"question": question, "draft_answer": draft_answer},
        config={"configurable": {"session_id": session_id}}
    )
