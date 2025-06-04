from chatbot.chatbot_core import run_chatbot_pipeline

def main():
    session_id = input("세션 아이디를 입력하세요: ") or "default"
    print(f"[INFO] 세션 ID: {session_id} (exit 입력시 종료)\n")
    while True:
        user_input = input("질문을 입력하세요 (종료: exit): ")
        if user_input.strip().lower() == "exit":
            break
        answer = run_chatbot_pipeline(user_input, session_id=session_id)  # 세션 ID 전달
        print(f"\n[{session_id} 응답]\n{answer}\n")

if __name__ == "__main__":
    main()