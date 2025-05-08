import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import tempfile
import os

# ✅ 환경 변수에서 OpenAI API 키 불러오기
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# ✅ Streamlit 설정
st.set_page_config(page_title="PDF 대화형 챗봇", page_icon="🤖")
st.title("PDF 기반 GPT 챗봇")
st.markdown("PDF 파일을 업로드하고 자유롭게 질문해보세요!")

# ✅ PDF 업로드
uploaded_file = st.sidebar.file_uploader("PDF 파일 업로드", type="pdf")

# ✅ 파일 업로드 시 처리
if uploaded_file:
    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # 1️⃣ 문서 로드
    loader = PyPDFLoader(tmp_file_path)
    data = loader.load()

    # 2️⃣ 문서 임베딩 및 벡터 DB 생성
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vectors = FAISS.from_documents(data, embeddings)

    # 3️⃣ Conversational QA 체인 구성
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-4", temperature=0.0, api_key=openai_api_key),
        retriever=vectors.as_retriever()
    )

    # 4️⃣ 질의응답 처리 함수
    def conversational_chat(query):
        result = chain({
            "question": query,
            "chat_history": st.session_state['history']
        })
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]

    # ✅ 세션 상태 초기화
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["안녕하세요! 업로드한 문서에 대해 질문해보세요."]
    if "past" not in st.session_state:
        st.session_state["past"] = ["안녕하세요!"]

    # ✅ 사용자 입력 영역
    with st.container():
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input("질문을 입력하세요:", placeholder="예: 2페이지 요약해줘")
            submit_button = st.form_submit_button(label="Send")

        if submit_button and user_input:
            output = conversational_chat(user_input)
            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(output)

    # ✅ 대화 이력 출력
    if st.session_state["generated"]:
        with st.container():
            for i in range(len(st.session_state["generated"])):
                message(st.session_state["past"][i], is_user=True, key=f"{i}_user", avatar_style="fun-emoji")
                message(st.session_state["generated"][i], key=f"{i}_bot", avatar_style="bottts")
