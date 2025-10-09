"""
このファイルは、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4
from dotenv import load_dotenv
import streamlit as st
import tiktoken
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain import SerpAPIWrapper
from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent
import utils
import constants as ct



############################################################
# 設定関連
############################################################
load_dotenv()


############################################################
# 関数定義
############################################################

def initialize():
    """
    画面読み込み時に実行する初期化処理
    """
    # 初期化データの用意
    initialize_session_state()
    # ログ出力用にセッションIDを生成
    initialize_session_id()
    # ログ出力の設定
    initialize_logger()
    # Agent Executor 等の重いリソースは起動時に作成せず遅延初期化とする
    # initialize.initialize_heavy_resources() を必要時に呼んでください


def initialize_session_state():
    """
    初期化データの用意
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []
        # 会話履歴の合計トークン数を加算する用の変数
        st.session_state.total_tokens = 0

        # フィードバックボタンで「はい」を押下した後にThanksメッセージを表示するためのフラグ
        st.session_state.feedback_yes_flg = False
        # フィードバックボタンで「いいえ」を押下した後に入力エリアを表示するためのフラグ
        st.session_state.feedback_no_flg = False
        # LLMによる回答生成後、フィードバックボタンを表示するためのフラグ
        st.session_state.answer_flg = False
        # フィードバックボタンで「いいえ」を押下後、フィードバックを送信するための入力エリアからの入力を受け付ける変数
        st.session_state.dissatisfied_reason = ""
        # フィードバック送信後にThanksメッセージを表示するためのフラグ
        st.session_state.feedback_no_reason_send_flg = False


def initialize_session_id():
    """
    セッションIDの作成
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex


def initialize_logger():
    """
    ログ出力の設定
    """
    os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)

    logger = logging.getLogger(ct.LOGGER_NAME)

    if logger.hasHandlers():
        return

    log_handler = TimedRotatingFileHandler(
        os.path.join(ct.LOG_DIR_PATH, ct.LOG_FILE),
        when="D",
        encoding="utf8"
    )
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, session_id={st.session_state.session_id}: %(message)s"
    )
    log_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(log_handler)


def initialize_agent_executor():
    """
    画面読み込み時にAgent Executor（AIエージェント機能の実行を担当するオブジェクト）を作成
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにAgent Executorが作成済みの場合、後続の処理を中断
    if "agent_executor" in st.session_state:
        return
    
    # 軽量な部分のみを初期化する: トークンエンコードとLLMクライアント
    # 重いChain/Agentの作成は initialize_heavy_resources() で行う
    # ENCODING_KIND may be missing; use getattr with default and guard tiktoken errors
    encoding_kind = getattr(ct, 'ENCODING_KIND', None)
    try:
        if encoding_kind:
            st.session_state.enc = tiktoken.get_encoding(encoding_kind)
        else:
            raise Exception("ENCODING_KIND not defined")
    except Exception as e:
        logger = logging.getLogger(ct.LOGGER_NAME)
        logger.warning(f"tiktoken encoding unavailable ({e}), using SimpleEncoder fallback")
        class SimpleEncoder:
            def encode(self, text):
                # fallback: split on whitespace as rough token approximation
                try:
                    return text.split()
                except Exception:
                    return [text]
        st.session_state.enc = SimpleEncoder()

    st.session_state.llm = ChatOpenAI(model_name=ct.MODEL, temperature=ct.TEMPERATURE, streaming=True)


def initialize_heavy_resources():
    """
    重いリソース（RAGチェーン、ツール、Agent Executor）を作成する遅延初期化関数
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    # 既に作成済みなら何もしない
    if "agent_executor" in st.session_state and "rag_chain" in st.session_state:
        return

    # 各Tool用のChainを作成（重い処理）
    st.session_state.customer_doc_chain = utils.create_rag_chain(ct.DB_CUSTOMER_PATH)
    st.session_state.service_doc_chain = utils.create_rag_chain(ct.DB_SERVICE_PATH)
    st.session_state.company_doc_chain = utils.create_rag_chain(ct.DB_COMPANY_PATH)
    st.session_state.rag_chain = utils.create_rag_chain(ct.DB_ALL_PATH)

    # Web検索用のToolを設定するためのオブジェクトを用意
    search = SerpAPIWrapper()
    # Agent Executorに渡すTool一覧を用意
    tools = [
        Tool(
            name=ct.SEARCH_COMPANY_INFO_TOOL_NAME,
            func=utils.run_company_doc_chain,
            description=ct.SEARCH_COMPANY_INFO_TOOL_DESCRIPTION
        ),
        Tool(
            name=ct.SEARCH_SERVICE_INFO_TOOL_NAME,
            func=utils.run_service_doc_chain,
            description=ct.SEARCH_SERVICE_INFO_TOOL_DESCRIPTION
        ),
        Tool(
            name=ct.SEARCH_CUSTOMER_COMMUNICATION_INFO_TOOL_NAME,
            func=utils.run_customer_doc_chain,
            description=ct.SEARCH_CUSTOMER_COMMUNICATION_INFO_TOOL_DESCRIPTION
        ),
        Tool(
            name = ct.SEARCH_WEB_INFO_TOOL_NAME,
            func=search.run,
            description=ct.SEARCH_WEB_INFO_TOOL_DESCRIPTION
        ),
        Tool(
            name=ct.SEARCH_BUSINESS_HOURS_TOOL_NAME,
            func=utils.run_business_hours_tool,
            description=ct.SEARCH_BUSINESS_HOURS_TOOL_DESCRIPTION
        )
    ]

    # Agent Executorの作成（重い）
    st.session_state.agent_executor = initialize_agent(
        llm=st.session_state.llm,
        tools=tools,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        max_iterations=ct.AI_AGENT_MAX_ITERATIONS,
        early_stopping_method="generate",
        handle_parsing_errors=True
    )