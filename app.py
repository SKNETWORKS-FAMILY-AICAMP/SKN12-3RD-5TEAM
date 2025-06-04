import streamlit as st
from chatbot.chatbot_core import run_chatbot_pipeline
from llm.chatbot_llm import chatbot_llm  # gpt-4o-mini 모델 (시스템 프롬프트 활용)

# 가운데 정렬된 제목 (큰 글씨)
st.markdown(
    """
    <h1 style='text-align: center;'>
        LangChain 및 RAG 활용<br>
        의료 LLM 개발
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown("<h1 style='text-align: center;'>MediChain</h1>", unsafe_allow_html=True)

# 1. 세션ID 관리
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = "default"
if 'all_histories' not in st.session_state:
    st.session_state['all_histories'] = {}

session_id = st.text_input("세션 아이디를 입력하세요", st.session_state['session_id'])
if session_id != st.session_state['session_id']:
    st.session_state['session_id'] = session_id
    if session_id not in st.session_state['all_histories']:
        st.session_state['all_histories'][session_id] = []

chat_history = st.session_state['all_histories'].setdefault(session_id, [])

user_input = st.text_input("질문을 입력하세요:", key="input_box")
submit = st.button("질문하기")

if submit and user_input:
    with st.spinner("답변 생성 중..."):
        answer = run_chatbot_pipeline(user_input, session_id=session_id)
        chat_history.append(("질문", user_input))
        chat_history.append(("응답", answer))
        # st.session_state['input_box'] = ""

# 2. 히스토리 출력 (최신이 아래쪽)
for role, text in chat_history:
    if role == "질문":
        st.markdown(f"**👤 질문:** {text}")
    else:
        st.markdown(f"**🤖 응답:** {text}")

# 3. 히스토리 다운로드 버튼
if chat_history:
    history_str = ""
    for role, text in chat_history:
        history_str += f"[{role}]\n{text}\n\n"
    st.download_button(
        label="💾 이 세션 대화 다운로드",
        data=history_str,
        file_name=f"chat_{session_id}.txt",
        mime="text/plain"
    )

# 4. 세션 요약 기능
def summarize_history(history):
    # 질문/응답을 한 문단으로 이어붙임
    convo = ""
    for role, text in history:
        if role == "질문":
            convo += f"Q: {text}\n"
        else:
            convo += f"A: {text}\n"
    # LLM에게 요약 요청 (시스템 프롬프트)
    prompt = (
        "다음은 의료 챗봇과 사용자의 대화 내역입니다.\n"
        "중요한 정보, 맥락, 사용자 질문 의도, 챗봇의 주요 답변을 핵심 위주로 5줄 이내로 요약해 주세요.\n\n"
        f"{convo}"
    )
    response = chatbot_llm.invoke(prompt)
    return response.content if hasattr(response, "content") else response

if chat_history and st.button("📝 이 세션 대화 요약"):
    with st.spinner("요약 생성 중..."):
        summary = summarize_history(chat_history)
        st.markdown(f"#### 📄 요약\n{summary}")

st.info("세션 ID를 바꾸면 별도의 대화 히스토리가 저장되고 전환됩니다.\n'요약' 버튼으로 대화의 핵심 내용을 빠르게 확인할 수 있습니다.")
