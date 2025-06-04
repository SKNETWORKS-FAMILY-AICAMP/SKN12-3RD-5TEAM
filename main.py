# main.py - 실행 진입점 (chatbot_core에서 pipeline 호출)

from chatbot.chatbot_core import run_chatbot_pipeline

if __name__ == "__main__":
    question = input("질문을 입력하세요: ")
    answer = run_chatbot_pipeline(question)
    print("\n💬 답변:", answer)