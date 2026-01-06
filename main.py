import streamlit as st
from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

# --- 設定 ---
st.set_page_config(page_title="森林ナレッジチャットボット", page_icon="🌲")
st.title("🌲 森林経営ナレッジボット")

# APIキーの取得（Streamlit Secretsから読み込む安全な方法）
# ローカルで動かす場合は .streamlit/secrets.toml が必要ですが、
# UI上で入力させる簡易版として以下のように書くことも可能です。
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state.OPENAI_API_KEY = ""

api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if not api_key:
    st.info("左のサイドバーにOpenAI APIキーを入力してください")
    st.stop()
else:
    os.environ["OPENAI_API_KEY"] = api_key

# --- RAG構築 (キャッシュ化して高速化) ---
@st.cache_resource
def build_vector_store():
    # データの読み込み
    loader = CSVLoader(
        file_path="data/森林ナレッジ.csv",
        encoding="utf-8",
        source_column="質問 (Question)" # 検索精度向上のため質問文を検索対象にする
    )
    docs = loader.load()
    
    # ベクトル化と保存
    embeddings = OpenAIEmbeddings()
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

# --- LLMとChainの定義 ---
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT}
)

# --- チャットUIの実装 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# 過去の履歴を表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ユーザー入力の処理
if prompt := st.chat_input("質問を入力してください..."):
    # ユーザーのメッセージを表示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AIの回答生成
    with st.chat_message("assistant"):
        with st.spinner("資料を検索中..."):
            try:
                # invokeを使用して回答を取得
                response = qa_chain.invoke({"query": prompt})
                answer = response['result']
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")