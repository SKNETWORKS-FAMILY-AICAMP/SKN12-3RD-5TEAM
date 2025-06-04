from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def build_rag_chain(llm):
    prompt = ChatPromptTemplate.from_template("""
[전문 LLM] 아래 질문과 context(관련 문서)를 참고하여 전문적인 답변을 생성하라.
질문: {question}
context: {context}
(문서 내용이 반드시 포함되어야 하며, 논리적이고 친절하게 답변할 것.)
""")
    chain = prompt | llm | StrOutputParser()
    return chain