import streamlit as st
import asyncio
import os
import pandas as pd
from typing import List, Tuple

# --- 【重要】Streamlitで非同期処理エラーを防ぐためのおまじない ---
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
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


# --- 【改善1】CSVを直接読み込み、質問のみをEmbedding対象にする ---
@st.cache_resource
def build_vector_store():
    # CSVを直接読み込み（pandasで制御しやすくする）
    df = pd.read_csv("data/森林ナレッジ.csv", encoding="utf-8-sig")
    
    st.sidebar.success(f"📚 データ読み込み成功: {len(df)}件")
    
    # 【重要】質問文のみをpage_contentとし、回答とカテゴリはメタデータに格納
    docs = []
    for idx, row in df.iterrows():
        doc = Document(
            page_content=row["質問 (Question)"],  # 質問のみをEmbedding対象
            metadata={
                "category": row["カテゴリ"],
                "question": row["質問 (Question)"],
                "answer": row["回答 (Answer)"],
                "index": idx
            }
        )
        docs.append(doc)
    
    with st.sidebar.expander("読み込んだデータ（先頭5件）"):
        for i, doc in enumerate(docs[:5]):
            st.text(f"{i+1}. {doc.page_content[:50]}...")
    
    # Embeddingモデル
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # ベクトルストアの作成
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore, df


# --- 【改善2】BM25キーワード検索を追加 ---
def keyword_search(query: str, df: pd.DataFrame, top_k: int = 5) -> List[Tuple[int, float]]:
    """
    シンプルなキーワードマッチング検索
    質問文と回答文の両方を検索対象にする
    """
    from collections import Counter
    import re
    
    # クエリを単語に分割（日本語対応の簡易版）
    query_terms = set(re.findall(r'\w+', query.lower()))
    
    scores = []
    for idx, row in df.iterrows():
        # 質問と回答を結合してテキスト化
        text = f"{row['質問 (Question)']} {row['回答 (Answer)']}".lower()
        text_terms = set(re.findall(r'\w+', text))
        
        # マッチする単語数をスコアに
        match_count = len(query_terms & text_terms)
        if match_count > 0:
            scores.append((idx, match_count))
    
    # スコア順にソート
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


# --- 【改善3】ハイブリッド検索の実装 ---
def hybrid_search(query: str, vectorstore, df: pd.DataFrame, k: int = 5) -> List[dict]:
    """
    ベクトル検索とキーワード検索を組み合わせたハイブリッド検索
    """
    # 1. ベクトル検索（スコア付き）
    vector_results = vectorstore.similarity_search_with_score(query, k=k)
    
    # 2. キーワード検索
    keyword_results = keyword_search(query, df, top_k=k)
    
    # 3. スコアを統合（RRF: Reciprocal Rank Fusion）
    combined_scores = {}
    
    # ベクトル検索結果のスコア追加
    for rank, (doc, score) in enumerate(vector_results):
        idx = doc.metadata.get("index")
        if idx is not None:
            # FAISSのスコアは距離なので、小さいほど良い → 変換
            combined_scores[idx] = combined_scores.get(idx, 0) + 1 / (rank + 1)
    
    # キーワード検索結果のスコア追加（重み付け：キーワードマッチを重視）
    for rank, (idx, _) in enumerate(keyword_results):
        combined_scores[idx] = combined_scores.get(idx, 0) + 1.5 / (rank + 1)  # キーワードマッチに1.5倍の重み
    
    # スコア順にソート
    sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    
    # 結果を整形
    results = []
    for idx, score in sorted_results:
        row = df.iloc[idx]
        results.append({
            "category": row["カテゴリ"],
            "question": row["質問 (Question)"],
            "answer": row["回答 (Answer)"],
            "score": score
        })
    
    return results


try:
    vectorstore, df = build_vector_store()
except Exception as e:
    st.error(f"データの読み込みに失敗しました: {e}")
    st.stop()


# --- Geminiモデルの設定 ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


# --- 【改善4】検索結果を使った回答生成 ---
def generate_answer(query: str, search_results: List[dict]) -> str:
    """検索結果をコンテキストとして回答を生成"""
    
    # コンテキストの作成
    context_parts = []
    for i, result in enumerate(search_results):
        context_parts.append(
            f"【情報{i+1}】\n"
            f"カテゴリ: {result['category']}\n"
            f"質問: {result['question']}\n"
            f"回答: {result['answer']}"
        )
    context = "\n\n".join(context_parts)
    
    prompt = f"""あなたは森林経営の専門家です。以下の「参照情報」に基づいて質問に回答してください。

【重要なルール】
1. 参照情報に直接関連する内容がある場合は、その情報を元に回答してください
2. 参照情報に答えが含まれていない場合は、「申し訳ありませんが、ナレッジベースにその情報がありません」と答えてください
3. 回答は簡潔に、しかし必要な情報は漏らさないようにしてください

参照情報:
{context}

ユーザーの質問:
{query}

回答:"""
    
    response = llm.invoke(prompt)
    return response.content


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
                # 【改善】ハイブリッド検索を実行
                search_results = hybrid_search(prompt, vectorstore, df, k=5)
                
                # 回答を生成
                answer = generate_answer(prompt, search_results)
                
                st.markdown(answer)
                
                # 参照データの確認エリア（スコア付き）
                with st.expander("🔍 参照したデータを確認する"):
                    for i, result in enumerate(search_results):
                        st.markdown(f"**ランク {i+1}** (スコア: {result['score']:.2f})")
                        st.markdown(f"- **カテゴリ**: {result['category']}")
                        st.markdown(f"- **質問**: {result['question']}")
                        st.markdown(f"- **回答**: {result['answer']}")
                        st.divider()

                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
                import traceback
                st.code(traceback.format_exc())