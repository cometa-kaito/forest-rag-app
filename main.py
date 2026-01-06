import streamlit as st
import asyncio
import os

# --- 【重要】Streamlitで非同期処理エラーを防ぐためのおまじない ---
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

from langchain_community.document_loaders import CSVLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- 設定 ---
st.set_page_config(page_title="森林ナレッジチャットボット(Gemini版)", page_icon="🌲", layout="wide")
st.title("🌲 森林経営ナレッジボット (Gemini 2.5)")

# APIキーの取得
if "GOOGLE_API_KEY" not in st.session_state:
    st.session_state.GOOGLE_API_KEY = ""

if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    api_key = st.sidebar.text_input("Google API Key", type="password")

if not api_key:
    st.info("左のサイドバーにGoogle APIキーを入力してください")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

# --- RAG構築 ---
@st.cache_resource
def build_vector_store():
    # 【修正1】エンコーディングを 'utf-8-sig' に変更 (BOM付きCSVに対応)
    loader = CSVLoader(
        file_path="data/森林ナレッジ.csv",
        encoding="utf-8-sig", 
        source_column="質問 (Question)",
        csv_args={
            'delimiter': ',',
            'quotechar': '"'
        }
    )
    docs = loader.load()
    
    # デバッグ情報の表示（サイドバー）
    if len(docs) > 0:
        st.sidebar.success(f"📚 データ読み込み成功: {len(docs)}件")
        with st.sidebar.expander("データの先頭を確認"):
            st.text(docs[0].page_content)
    else:
        st.sidebar.error("⚠️ データが読み込めませんでした")

    # Embeddingモデル
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # ベクトルストアの作成
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

try:
    vectorstore = build_vector_store()
    # 検索数 k=5
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
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
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True 
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
        with st.spinner("Gemini 2.5 が思考中..."):
            try:
                response = qa_chain.invoke({"query": prompt})
                answer = response['result']
                source_docs = response['source_documents']

                st.markdown(answer)
                
                # 参照データの確認エリア
                with st.expander("🔍 参照したデータを確認する"):
                    for i, doc in enumerate(source_docs):
                        st.markdown(f"**ランク {i+1}**")
                        # page_contentを表示して、検索が正しいか確認
                        st.text(doc.page_content)

                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")