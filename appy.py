from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# --- 1. LLM呼び出し関数 (要件③) ---
def get_llm_response(input_text: str, expert_type: str) -> str:
    """
    入力テキストと専門家の種類を引数に受け取り、LLMの回答を返します。
    
    Args:
        input_text (str): ユーザーからの質問テキスト。
        expert_type (str): 選択された専門家の種類 ('IT技術者' または '歴史学者')。
        
    Returns:
        str: LLMからの回答テキスト。
    """
    
    # 専門家の種類に応じたシステムメッセージの定義 (要件②)
    # ここでLLMに振る舞わせるペルソナを設定します。
    system_messages = {
        "IT技術者": (
            "あなたは世界で最も優れたIT技術コンサルタントです。最新の技術トレンド、プログラミング、"
            "システム設計について、正確かつ分かりやすく解説してください。回答は専門的ですが、"
            "誰にでも理解できるように優しい言葉遣いを心がけてください。"
        ),
        "歴史学者": (
            "あなたは歴史に精通したベテランの歴史学者です。世界史、日本史、文化史に関する質問に対し、"
            "多角的な視点から詳細かつ洞察に満ちた回答を提供してください。回答は出典に基づき、"
            "常に客観的であることを徹底してください。"
        )
    }

    # 選択された専門家のシステムメッセージを取得
    system_message_content = system_messages.get(
        expert_type, 
        "あなたは一般的なアシスタントです。質問に丁寧に答えてください。" # 該当なしのフォールバック
    )
    
    # LangChainのChatモデルとプロンプトの構成
    try:
        # Chat modelsを使用 (要件①)
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        
        # システムメッセージとユーザー入力を組み合わせてプロンプトを作成
        messages = [
            SystemMessage(content=system_message_content),
            HumanMessage(content=input_text)
        ]
        
        # LLMの呼び出し
        response = llm.invoke(messages)
        return response.content

    except Exception as e:
        return f"LLM呼び出し中にエラーが発生しました: {e}\n(ヒント: 環境変数 OPENAI_API_KEY が正しく設定されているか確認してください。)"


# --- 2. Streamlit UIの構築 (要件①, ②, ④) ---

st.set_page_config(page_title="専門家ペルソナ切替チャット", layout="centered")

st.title("💡 LangChain 専門家ペルソナ切替チャット")

# アプリの概要と操作方法 (要件④)
st.markdown("""
---
**アプリの概要**:
このWebアプリは、LangChainを用いてLLM（大規模言語モデル）の**振る舞い（ペルソナ）**をラジオボタンで切り替えることができます。
選択した専門家に応じて、LLMが専門的な知識を活かして回答します。

**操作方法**:
1. **専門家の選択**: 左のラジオボタンで、LLMに回答させたい専門家を選んでください。
2. **質問の入力**: 下の入力フォームに質問内容を入力してください。
3. **回答の取得**: 「LLMに質問を送信」ボタンを押すと、選択した専門家としてLLMが回答を生成します。
---
""")

# --- サイドバー (ラジオボタンの配置 - 要件②) ---
with st.sidebar:
    st.header("👤 専門家選択")
    expert_type = st.radio(
        "LLMに振る舞ってほしい専門家を選んでください:",
        ("IT技術者", "歴史学者"),
        index=0, # デフォルトはIT技術者
        key="expert_selector"
    )
    
    st.info(f"現在選択中の専門家: **{expert_type}**")


# --- メインコンテンツ (入力フォームと送信ボタン - 要件①) ---

# 質問入力フォーム (要件①)
input_text = st.text_area(
    "💬 質問を入力してください:",
    placeholder="例: 量子コンピュータの最新動向について教えてください。" if expert_type == "IT技術者" else "例: 産業革命が社会に与えた影響を説明してください。",
    height=150,
    key="user_input"
)

# LLMへの質問送信ボタン
if st.button("🚀 LLMに質問を送信", type="primary"):
    
    if not os.getenv("OPENAI_API_KEY"):
        st.error("🚨 エラー: 環境変数 `OPENAI_API_KEY` が設定されていません。コード実行前に設定してください。")
    elif not input_text:
        st.warning("⚠️ 質問テキストを入力してください。")
    else:
        # ローディング表示
        with st.spinner(f"**{expert_type}** が回答を作成中です... しばらくお待ちください。"):
            
            # 定義した関数を呼び出し
            response_text = get_llm_response(input_text, expert_type)
        
        # 回答結果の表示 (要件①)
        st.subheader(f"✅ {expert_type} からの回答")
        st.markdown(response_text)

# 初期状態のメッセージ
if not st.session_state.get('last_response'):
    st.info("↑ 専門家を選択し、質問を入力して「LLMに質問を送信」ボタンを押してください。")
