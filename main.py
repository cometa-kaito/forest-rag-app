import streamlit as st
from langchain_community.document_loaders import CSVLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

# --- 設定 ---
st.set_page_config(page_title="森林ナレッジチャットボット(Gemini版)", page_icon="🌲")
st.title("🌲 森林経営ナレッジボット (Gemini)")

# APIキーの取得（UI入力またはSecretsから）
if "GOOGLE_API_KEY" not in st.session_state:
    st.session_state.GOOGLE_API_KEY = ""

# StreamlitのSecretsに設定されているか確認、なければサイドバーで入力
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    api_key = st.sidebar.text_input("Google API Key", type="password")

if not api_key:
    st.info("左のサイドバーにGoogle APIキーを入力してください")
    st.stop()

# 環境変数にセット
os.environ["GOOGLE_API_KEY"] = api_key

# --- RAG構築 (キャッシュ化して高速化) ---
@st.cache_resource
def build_vector_store():
    # データの読み込み
    loader = CSVLoader(
        file_path="data/森林ナレッジ.csv",
        encoding="utf-8",
        source_column="質問 (Question)"
    )
    docs = loader.load()
    
    # ベクトル化（GeminiのEmbeddingモデルを使用）
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # ベクトルストアの作成
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

try:
    vectorstore = build_vector_store()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
except Exception as e:
    st.error(f"データの読み込みに失敗しました: {e}")
    st.stop()

# --- プロンプト定義 ---
prompt_template = """あなたは森林経営の専門家です。以下の「参照情報」のみに基づいて質問に回答してください。
もし参照情報に答えが含まれていない場合は、正直に「情報がありません」と答えてください。

参照情報:
{context}

質問:
{question}

回答:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# --- Geminiモデルの設定 ---
# gemini-1.5-flash は高速でコスト効率が良いモデルです
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT}
)

# --- チャットUIの実装 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("質問を入力してください..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Geminiが思考中..."):
            try:
                response = qa_chain.invoke({"query": prompt})
                answer = response['result']
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")