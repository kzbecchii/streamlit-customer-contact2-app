"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
from dotenv import load_dotenv
import streamlit as st
import logging
import sys
import unicodedata
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from typing import List
from sudachipy import tokenizer, dictionary
from langchain_community.agent_toolkits import SlackToolkit
from langchain.agents import AgentType, initialize_agent
from slack_sdk import WebClient
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from docx import Document
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain import LLMChain
import datetime
import constants as ct
import sqlite3
from pathlib import Path


############################################################
# 設定関連
############################################################
load_dotenv()


############################################################
# 関数定義
############################################################

def build_error_message(message):
    """
    エラーメッセージと管理者問い合わせテンプレートの連結

    Args:
        message: 画面上に表示するエラーメッセージ

    Returns:
        エラーメッセージと管理者問い合わせテンプレートの連結テキスト
    """
    return "\n".join([message, ct.COMMON_ERROR_MESSAGE])


def create_rag_chain(db_name):
    """
    引数として渡されたDB内を参照するRAGのChainを作成

    Args:
        db_name: RAG化対象のデータを格納するデータベース名
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    docs_all = []
    # AIエージェント機能を使わない場合の処理
    if db_name == ct.DB_ALL_PATH:
        folders = os.listdir(ct.RAG_TOP_FOLDER_PATH)
        # 「data」フォルダ直下の各フォルダ名に対して処理
        for folder_path in folders:
            if folder_path.startswith("."):
                continue
            # フォルダ内の各ファイルのデータをリストに追加
            add_docs(f"{ct.RAG_TOP_FOLDER_PATH}/{folder_path}", docs_all)
    # AIエージェント機能を使う場合の処理
    else:
        # データベース名に対応した、RAG化対象のデータ群が格納されているフォルダパスを取得
        folder_path = ct.DB_NAMES[db_name]
        # フォルダ内の各ファイルのデータをリストに追加
        add_docs(folder_path, docs_all)

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])

    # ロードしたドキュメントの要約ログ
    logger.info({"create_rag_chain_loaded_docs": len(docs_all)})
    
    text_splitter = CharacterTextSplitter(
        chunk_size=ct.CHUNK_SIZE,
        chunk_overlap=ct.CHUNK_OVERLAP,
        separator="\n",
    )
    splitted_docs = text_splitter.split_documents(docs_all)

    embeddings = OpenAIEmbeddings()

    # DB ごとに固有の persist_directory を使う（db_name がパスのケースを想定）
    # 明示的な定数がある場合はそれを優先して使う
    if db_name == ct.DB_SERVICE_PATH:
        persist_dir = os.path.normpath(ct.DB_SERVICE_PERSIST_DIR)
    elif db_name == ct.DB_COMPANY_PATH:
        persist_dir = os.path.normpath(ct.DB_COMPANY_PERSIST_DIR)
    elif db_name == ct.DB_CUSTOMER_PATH:
        persist_dir = os.path.normpath(ct.DB_CUSTOMER_PERSIST_DIR)
    elif db_name == ct.DB_ALL_PATH:
        persist_dir = os.path.normpath(ct.DB_ALL_PERSIST_DIR)
    else:
        # フォールバック: 引数の db_name をそのままパスと見なす
        persist_dir = os.path.normpath(db_name)
    try:
        os.makedirs(persist_dir, exist_ok=True)
    except Exception:
        # 冗長だが、作成に失敗したらフォールバックで共通 .db を使う
        persist_dir = ".db"

    # すでに対象のデータベースが作成済みの場合は読み込み、未作成の場合は新規作成する
    if os.path.isdir(persist_dir) and any(Path(persist_dir).iterdir()):
        logger.info({"chroma_load_from": persist_dir})
        db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        logger.info({"chroma_create_at": persist_dir, "doc_count": len(splitted_docs)})
        db = Chroma.from_documents(splitted_docs, embedding=embeddings, persist_directory=persist_dir)

    retriever = db.as_retriever(search_kwargs={"k": ct.TOP_K})

    question_generator_template = ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT
    question_generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_generator_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_template = ct.SYSTEM_PROMPT_INQUIRY
    question_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_answer_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        st.session_state.llm, retriever, question_generator_prompt
    )

    question_answer_chain = create_stuff_documents_chain(st.session_state.llm, question_answer_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


def add_docs(folder_path, docs_all):
    """
    フォルダ内のファイル一覧を取得

    Args:
        folder_path: フォルダのパス
        docs_all: 各ファイルデータを格納するリスト
    """
    try:
        files = os.listdir(folder_path)
    except FileNotFoundError:
        logger = logging.getLogger(ct.LOGGER_NAME)
        logger.warning({"add_docs_missing_folder": folder_path})
        return

    for file in files:
        # ファイルの拡張子を取得・小文字化して判定（例: .PDF を見逃さない）
        file_extension = os.path.splitext(file)[1].lower()
        file_path = f"{folder_path}/{file}"
        # 想定していたファイル形式の場合のみ読み込む
        if file_extension in ct.SUPPORTED_EXTENSIONS:
            logger = logging.getLogger(ct.LOGGER_NAME)
            logger.info({"add_docs_loading": file_path})
            # ファイルの拡張子に合ったdata loaderを使ってデータ読み込み
            loader = ct.SUPPORTED_EXTENSIONS[file_extension](file_path)
        else:
            continue
        try:
            docs = loader.load()
            docs_all.extend(docs)
        except Exception as e:
            logger = logging.getLogger(ct.LOGGER_NAME)
            logger.warning({"add_docs_load_failed": file_path, "error": str(e)})


def adjust_reference_data(docs, docs_history):
    """
    Slack通知用の参照先データの整形

    Args:
        docs: 従業員情報ファイルの読み込みデータ
        docs_history: 問い合わせ対応履歴ファイルの読み込みデータ

    Returns:
        従業員情報と問い合わせ対応履歴の結合テキストを持つドキュメントリスト
    """

    docs_all = []
    for row in docs:
        # 従業員IDの取得（ページコンテンツを行ごとに分割して key: value 形式を期待）
        row_lines = row.page_content.split("\n")
        row_dict = {}
        for item in row_lines:
            if ": " in item:
                k, v = item.split(": ", 1)
                row_dict[k] = v
        employee_id = row_dict.get("従業員ID", "")

        # 取得した従業員IDに紐づく問い合わせ対応履歴を取得
        same_employee_inquiries = []
        for row_history in docs_history:
            row_history_lines = row_history.page_content.split("\n")
            row_history_dict = {}
            for item in row_history_lines:
                if ": " in item:
                    k, v = item.split(": ", 1)
                    row_history_dict[k] = v
            if row_history_dict.get("従業員ID") == employee_id:
                same_employee_inquiries.append(row_history_dict)

        new_doc = Document()
        if same_employee_inquiries:
            doc_text = "【従業員情報】\n"
            doc_text += "\n".join(row_lines) + "\n=================================\n"
            doc_text += "【この従業員の問い合わせ対応履歴】\n"
            for inquiry_dict in same_employee_inquiries:
                for key, value in inquiry_dict.items():
                    doc_text += f"{key}: {value}\n"
                doc_text += "---------------\n"
            new_doc.page_content = doc_text
        else:
            new_doc.page_content = row.page_content

        new_doc.metadata = {}
        docs_all.append(new_doc)

    return docs_all


def run_company_doc_chain(param):
    """
    会社に関するデータ参照に特化したTool設定用の関数

    Args:
        param: ユーザー入力値

    Returns:
        LLMからの回答
    """
    # 会社に関するデータ参照に特化したChainを実行してLLMからの回答取得
    ai_msg = st.session_state.company_doc_chain.invoke({"input": param, "chat_history": st.session_state.chat_history})
    # run_*_doc_chain は回答のみを返す。会話履歴への追加は呼び出し元で一元管理する。

    return ai_msg["answer"]

def run_service_doc_chain(param):
    """
    サービスと商品に関するデータ参照に特化したTool設定用の関数

    Args:
        param: ユーザー入力値

    Returns:
        LLMからの回答
    """
    # サービスに関するデータ参照に特化したChainを実行してLLMからの回答取得
    ai_msg = st.session_state.service_doc_chain.invoke({"input": param, "chat_history": st.session_state.chat_history})

    return ai_msg["answer"]

def run_customer_doc_chain(param):
    """
    顧客とのやり取りに関するデータ参照に特化したTool設定用の関数

    Args:
        param: ユーザー入力値
    
    Returns:
        LLMからの回答
    """
    # 顧客とのやり取りに関するデータ参照に特化したChainを実行してLLMからの回答取得
    ai_msg = st.session_state.customer_doc_chain.invoke({"input": param, "chat_history": st.session_state.chat_history})

    return ai_msg["answer"]


def delete_old_conversation_log(result):
    """
    古い会話履歴の削除

    Args:
        result: LLMからの回答
    """
    # LLMからの回答テキストのトークン数を取得
    response_tokens = len(st.session_state.enc.encode(result))
    # 過去の会話履歴の合計トークン数に加算
    st.session_state.total_tokens += response_tokens

    # トークン数が上限値を下回るまで、順に古い会話履歴を削除
    while st.session_state.total_tokens > ct.MAX_ALLOWED_TOKENS:
        # 最も古い会話履歴を削除
        # safety: ensure there are at least 2 messages (index 1 exists) before popping
        chat_history = getattr(st.session_state, 'chat_history', None)
        logger = logging.getLogger(ct.LOGGER_NAME)
        if not isinstance(chat_history, list):
            logger.warning("chat_history not a list or not present; aborting deletion loop")
            break
        if len(chat_history) <= 1:
            logger.warning("chat_history too short to pop; aborting deletion loop")
            break
        try:
            removed_message = chat_history.pop(1)
        except IndexError:
            logger.warning("IndexError while popping chat_history; aborting deletion loop")
            break
        # 最も古い会話履歴のトークン数を取得
        removed_tokens = len(st.session_state.enc.encode(removed_message.content))
        # 過去の会話履歴の合計トークン数から、最も古い会話履歴のトークン数を引く
        st.session_state.total_tokens -= removed_tokens


def execute_agent_or_chain(chat_message):
    """
    AIエージェントもしくはAIエージェントなしのRAGのChainを実行

    Args:
        chat_message: ユーザーメッセージ
    
    Returns:
        LLMからの回答
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    # AIエージェント機能を利用する場合
    if st.session_state.agent_mode == ct.AI_AGENT_MODE_ON:
        # LLMによる回答をストリーミング出力するためのオブジェクトを用意
        try:
            st_callback = StreamlitCallbackHandler(st.container())
            # Agent Executorの実行（AIエージェント機能を使う場合は、Toolとして設定した関数内で会話履歴への追加処理を実施）
            result = st.session_state.agent_executor.invoke({"input": chat_message}, {"callbacks": [st_callback]})
            response = result["output"]
        except Exception as e:
            # Streamlit のセッションコンテキストが別スレッドで利用できない等の例外が出た場合、
            # コールバック無しで実行し直す（ストリーミング表示は行われないが処理は継続する）
            logger = logging.getLogger(ct.LOGGER_NAME)
            logger.warning(f"StreamlitCallbackHandler failed ({e}), retrying without callbacks")
            result = st.session_state.agent_executor.invoke({"input": chat_message})
            response = result.get("output") if isinstance(result, dict) else result
    # AIエージェントを利用しない場合
    else:
        # まずは全体用のRAGチェーンを試す（st.session_state.rag_chain が存在しない場合は次に移る）
        response = None
        tried_chains = []
        chains_to_try = []
        # 優先順: 全体 -> 会社 -> サービス -> 顧客
        # Use persisted DB paths as a source of truth and allow on-demand recreation
        db_map = {
            'all': ct.DB_ALL_PATH,
            'company': ct.DB_COMPANY_PATH,
            'service': ct.DB_SERVICE_PATH,
            'customer': ct.DB_CUSTOMER_PATH,
        }
        # build list of (name, chain_obj, db_path) for attempts; chain_obj may be None
        chains_to_try = [
            ('all', st.session_state.get('rag_chain', None), db_map['all']),
            ('company', st.session_state.get('company_doc_chain', None), db_map['company']),
            ('service', st.session_state.get('service_doc_chain', None), db_map['service']),
            ('customer', st.session_state.get('customer_doc_chain', None), db_map['customer']),
        ]
        # Keep a candidate answer that may have no sources (use only if no sourced answers found)
        no_source_candidate = None
        for name, chain, db_path in chains_to_try:
            tried_chains.append(name)
            try:
                # If chain object is missing for this session/tab, recreate it from persisted DB
                if chain is None:
                    try:
                        logger.info({f"recreating_chain_for_session": name, 'db_path': db_path})
                        chain = create_rag_chain(db_path)
                        # save into session_state so subsequent calls in same session reuse it
                        if name == 'all':
                            st.session_state.rag_chain = chain
                        else:
                            st.session_state[f"{name}_doc_chain"] = chain
                    except Exception as e:
                        logger.warning(f"Failed to recreate chain '{name}' on-demand: {e}")
                        # continue to next chain
                        continue

                result = chain.invoke({"input": chat_message, "chat_history": st.session_state.chat_history})
                # Try to extract answer
                ans = result.get("answer") if isinstance(result, dict) else getattr(result, 'answer', None)

                # Extract source documents for logging/debugging (LangChain chains often return 'source_documents' or 'sources')
                sources = None
                if isinstance(result, dict):
                    sources = result.get('source_documents') or result.get('sources') or result.get('source_documents')
                else:
                    # some chain implementations attach source_documents attribute
                    sources = getattr(result, 'source_documents', None) or getattr(result, 'sources', None)

                # Log top sources for this chain
                try:
                    if sources:
                        src_list = []
                        for i, doc in enumerate(sources[: ct.TOP_K]):
                            snippet = getattr(doc, 'page_content', None) or str(doc)
                            snippet = snippet.replace('\n', ' ')[:200]
                            meta = getattr(doc, 'metadata', {})
                            src_list.append({
                                'index': i,
                                'snippet': snippet,
                                'metadata': meta,
                            })
                        logger.info({f"rag_sources_{name}": src_list})
                    else:
                        logger.info({f"rag_sources_{name}": "no_source_documents"})
                        # If no sources were returned, try to recreate chain once (fresh retriever)
                        # This helps when another session/tab created/updated the persisted DB earlier
                        try:
                            logger.info({f"recreate_retry_chain": name, 'db_path': db_path})
                            fresh_chain = create_rag_chain(db_path)
                            # update session_state and re-invoke once
                            if name == 'all':
                                st.session_state.rag_chain = fresh_chain
                            else:
                                st.session_state[f"{name}_doc_chain"] = fresh_chain
                            # re-invoke
                            result2 = fresh_chain.invoke({"input": chat_message, "chat_history": st.session_state.chat_history})
                            ans2 = result2.get("answer") if isinstance(result2, dict) else getattr(result2, 'answer', None)
                            sources2 = result2.get('source_documents') or result2.get('sources') if isinstance(result2, dict) else getattr(result2, 'source_documents', None) or getattr(result2, 'sources', None)
                            if sources2:
                                # log and adopt this result in place of previous
                                src_list = []
                                for i, doc in enumerate(sources2[: ct.TOP_K]):
                                    snippet = getattr(doc, 'page_content', None) or str(doc)
                                    snippet = snippet.replace('\n', ' ')[:200]
                                    meta = getattr(doc, 'metadata', {})
                                    src_list.append({
                                        'index': i,
                                        'snippet': snippet,
                                        'metadata': meta,
                                    })
                                logger.info({f"rag_sources_{name}": src_list})
                                ans = ans2
                                sources = sources2
                                result = result2
                        except Exception as e:
                            logger.warning(f"Recreate+reinvoke for chain '{name}' failed: {e}")
                except Exception as e:
                    logger.warning(f"Failed to log source documents for chain '{name}': {e}")

                # None や 空文字の場合は次へ
                if not ans:
                    continue
                # RAG が NO_DOC_MATCH_MESSAGE を返した場合は次のチェーンを試す
                if ans == ct.NO_DOC_MATCH_MESSAGE:
                    continue

                # ソースが存在する回答を最優先で採用
                if sources and len(sources) > 0:
                    st.session_state.chat_history.extend([HumanMessage(content=chat_message), AIMessage(content=ans)])
                    response = ans
                    break

                # ソース無しだが有効な回答は二次候補として保持（ただし直ちに返さない）
                if no_source_candidate is None:
                    no_source_candidate = ans

                # さらに良い候補を探すために次のチェーンへ進む
                continue
            except Exception as e:
                logger.warning(f"RAG chain '{name}' invocation failed: {e}; trying next chain")
                continue
        # いずれのチェーンにもソース付き回答が無かった場合は、ソース無しで得られた回答を使う
        if response is None and no_source_candidate is not None:
            logger.info(f"Using no-source candidate answer after trying chains (tried: {tried_chains})")
            response = no_source_candidate
            try:
                if isinstance(st.session_state.chat_history, list):
                    st.session_state.chat_history.extend([HumanMessage(content=chat_message), AIMessage(content=response)])
                else:
                    st.session_state.chat_history = [HumanMessage(content=chat_message), AIMessage(content=response)]
            except Exception:
                pass

        # どのチェーンでも全く候補が得られなかった場合、軽量LLMへフォールバックして回答を生成する
        if response is None:
            logger.info(f"No RAG chain returned any valid answer (tried: {tried_chains}); falling back to simple LLM generation")
            try:
                response = generate_simple_answer(chat_message)
            except Exception as e:
                logger.exception(f"Fallback simple generation failed: {e}")
                response = ct.NO_DOC_MATCH_MESSAGE

            # 会話履歴への追加（失敗しないように安全に実行）
            try:
                if isinstance(st.session_state.chat_history, list):
                    st.session_state.chat_history.extend([HumanMessage(content=chat_message), AIMessage(content=response)])
                else:
                    st.session_state.chat_history = [HumanMessage(content=chat_message), AIMessage(content=response)]
            except Exception:
                # 履歴更新に失敗しても回答は返す
                pass

    # LLMから参照先のデータを基にした回答が行われた場合のみ、フィードバックボタンを表示
    if response != ct.NO_DOC_MATCH_MESSAGE:
        st.session_state.answer_flg = True
    
    return response


def generate_simple_answer(chat_message: str) -> str:
    """
    重いリソース（RAG/Agent）が使えない時のために、軽量なLLMで回答を生成するフォールバック関数

    Args:
        chat_message: ユーザーメッセージ

    Returns:
        生成された回答またはエラーメッセージ
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    # まずセッションの LLM を優先
    llm = getattr(st.session_state, 'llm', None)
    if llm is None:
        try:
            llm = ChatOpenAI(model_name=ct.MODEL, temperature=ct.TEMPERATURE, streaming=False)
            logger.info("Created local ChatOpenAI instance for fallback simple answer generation")
        except Exception as e:
            logger.exception(f"Failed to create local ChatOpenAI for fallback: {e}")
            return "申し訳ございません。回答の生成中にエラーが発生しました。しばらく経ってから再度お試しください。"

    try:
        # まず chat-style の呼び出しを試す
        try:
            gen = llm([
                {"role": "system", "content": "あなたは社内文書を元に顧客の問い合わせに回答するアシスタントです。社内データが参照できない場合は一般的な知見に基づいて回答してください。出力は日本語で記載してください。"},
                {"role": "user", "content": chat_message},
            ])
            text = getattr(gen, 'content', None) or (gen[0] if isinstance(gen, (list, tuple)) and gen else None) or str(gen)
        except Exception:
            gen = llm(chat_message)
            text = getattr(gen, 'content', None) or str(gen)

        # 空や None の場合はエラーメッセージにフォールバック
        if not text:
            return "申し訳ございません。回答の生成中にエラーが発生しました。しばらく経ってから再度お試しください。"

        return text
    except Exception as e:
        logger.exception(f"Simple LLM generation failed: {e}")
        return "申し訳ございません。回答の生成中にエラーが発生しました。しばらく経ってから再度お試しください。"


def notice_slack(chat_message, requester_override=None):
    """
    問い合わせ内容のSlackへの通知

    Args:
        chat_message: ユーザーメッセージ

    Returns:
        問い合わせサンクスメッセージ
    """

    # Slack通知用のAgent Executorを作成（ただしバックグラウンド実行時は st.session_state が利用できないことがある）
    agent_executor = None
    try:
        # session_state.llm がないと例外になるためガード
        llm_for_agent = getattr(st.session_state, "llm", None)
        if llm_for_agent is not None:
            toolkit = SlackToolkit()
            tools = toolkit.get_tools()
            agent_executor = initialize_agent(
                llm=llm_for_agent,
                tools=tools,
                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
            )
        else:
            # session の LLM が取得できない場合は agent を使わず直接 WebClient にフォールバックする
            logger = logging.getLogger(ct.LOGGER_NAME)
            logger.info("st.session_state.llm not available in background; skipping agent creation and using WebClient fallback")
    except Exception as e:
        logger = logging.getLogger(ct.LOGGER_NAME)
        logger.warning(f"Failed to create agent_executor: {e}; will fallback to WebClient")

    # 担当者割り振りに使う用の「従業員情報」と「問い合わせ対応履歴」の読み込み
    loader = CSVLoader(ct.EMPLOYEE_FILE_PATH, encoding=ct.CSV_ENCODING)
    docs = loader.load()
    loader = CSVLoader(ct.INQUIRY_HISTORY_FILE_PATH, encoding=ct.CSV_ENCODING)
    docs_history = loader.load()

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    for doc in docs:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])
    for doc in docs_history:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])

    # 問い合わせ内容と関連性が高い従業員情報を取得するために、参照先データを整形
    docs_all = adjust_reference_data(docs, docs_history)
    
    # 形態素解析による日本語の単語分割を行うため、参照先データからテキストのみを抽出
    docs_all_page_contents = []
    for doc in docs_all:
        docs_all_page_contents.append(doc.page_content)

    # Retrieverの作成
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(docs_all, embedding=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": ct.TOP_K})
    bm25_retriever = BM25Retriever.from_texts(
        docs_all_page_contents,
        preprocess_func=preprocess_func,
        k=ct.TOP_K
    )
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, retriever],
        weights=ct.RETRIEVER_WEIGHTS
    )

    # 問い合わせ内容と関連性の高い従業員情報を取得
    employees = retriever.invoke(chat_message)
    
    # プロンプトに埋め込むための従業員情報テキストを取得
    context = get_context(employees)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", ct.SYSTEM_PROMPT_EMPLOYEE_SELECTION)
    ])
    # フォーマット文字列を生成
    output_parser = CommaSeparatedListOutputParser()
    format_instruction = output_parser.get_format_instructions()

    # 問い合わせ内容と関連性が高い従業員のID一覧を取得
    messages = prompt_template.format_prompt(context=context, query=chat_message, format_instruction=format_instruction).to_messages()
    # バックグラウンド実行時には st.session_state.llm が存在しない可能性があるためガード
    llm_for_agent = getattr(st.session_state, "llm", None)
    if llm_for_agent is not None:
        try:
            employee_id_response = llm_for_agent(messages)
            employee_ids = output_parser.parse(employee_id_response.content)
        except Exception as e:
            logger = logging.getLogger(ct.LOGGER_NAME)
            logger.warning(f"Failed to get employee ids via llm: {e}; proceeding without mentions")
            employee_ids = []
    else:
        logger = logging.getLogger(ct.LOGGER_NAME)
        logger = logging.getLogger(ct.LOGGER_NAME)

        # 1) Agent executor を試す（セッション LLM がバックグラウンドでは使えないことがある）
        agent_executor = None
        try:
            session_llm = getattr(st.session_state, "llm", None)
            if session_llm is not None:
                toolkit = SlackToolkit()
                tools = toolkit.get_tools()
                agent_executor = initialize_agent(
                    llm=session_llm,
                    tools=tools,
                    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                )
            else:
                logger.info("st.session_state.llm not available in background; will fallback to WebClient/Local LLM")
        except Exception as e:
            logger.warning(f"Failed to initialize agent executor: {e}; continuing with fallback flows")

        # 2) CSV から従業員情報と履歴を読み込む
        loader = CSVLoader(ct.EMPLOYEE_FILE_PATH, encoding=ct.CSV_ENCODING)
        docs = loader.load()
        loader = CSVLoader(ct.INQUIRY_HISTORY_FILE_PATH, encoding=ct.CSV_ENCODING)
        docs_history = loader.load()

        for doc in docs:
            doc.page_content = adjust_string(doc.page_content)
            for key in doc.metadata:
                doc.metadata[key] = adjust_string(doc.metadata[key])
        for doc in docs_history:
            doc.page_content = adjust_string(doc.page_content)
            for key in doc.metadata:
                doc.metadata[key] = adjust_string(doc.metadata[key])

        docs_all = adjust_reference_data(docs, docs_history)
        docs_all_page_contents = [d.page_content for d in docs_all]

        # Retriever を作成して候補従業員を取得
        embeddings = OpenAIEmbeddings()
        db = Chroma.from_documents(docs_all, embedding=embeddings)
        vector_retriever = db.as_retriever(search_kwargs={"k": ct.TOP_K})
        bm25_retriever = BM25Retriever.from_texts(docs_all_page_contents, preprocess_func=preprocess_func, k=ct.TOP_K)
        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=ct.RETRIEVER_WEIGHTS)

        employees = ensemble_retriever.invoke(chat_message)

        # LLM による候補絞り込み（可能なら実行）
        employee_ids = []
        try:
            prompt_template = ChatPromptTemplate.from_messages([("system", ct.SYSTEM_PROMPT_EMPLOYEE_SELECTION)])
            output_parser = CommaSeparatedListOutputParser()
            format_instruction = output_parser.get_format_instructions()
            messages = prompt_template.format_prompt(context=get_context(employees), query=chat_message, format_instruction=format_instruction).to_messages()

            session_llm = getattr(st.session_state, "llm", None)
            if session_llm is not None:
                resp = session_llm(messages)
                employee_ids = output_parser.parse(getattr(resp, 'content', '') or str(resp))
        except Exception as e:
            logger.warning(f"Failed to get employee ids via session LLM: {e}")

        # LLM が何も返さなかった場合は retriever 上位を代替採用
        if not employee_ids:
            logger.info("No employee_ids from LLM; falling back to top retriever candidates for mentions")
            try:
                target_employees = employees[: ct.TOP_K]
            except Exception:
                target_employees = []
        else:
            target_employees = get_target_employees(employees, employee_ids)

        slack_ids = get_slack_ids(target_employees)
        slack_id_text = create_slack_id_text(slack_ids)
        context_for_prompt = get_context(target_employees)

        now_datetime = get_datetime()
        # requester precedence: caller override -> session -> default
        requester = requester_override or getattr(st.session_state, 'user_name', None) or '山田太郎'

        prompt = PromptTemplate(
            input_variables=["slack_id_text", "query", "context", "now_datetime", "requester"],
            template=ct.SYSTEM_PROMPT_NOTICE_SLACK,
        )
        prompt_message = prompt.format(slack_id_text=slack_id_text, query=chat_message, context=context_for_prompt, now_datetime=now_datetime, requester=requester)

        # Agent executor があっても、バックグラウンド実行時の挙動差異で二重投稿や
        # 不要なメンションが発生することがあるため、ここでは agent_executor による直接投稿は行わず
        # 以降の LLM/WebClient フローで必ず本文を生成・後処理して投稿します。
        if agent_executor is not None:
            logger.info("Agent executor available but skipping direct agent posting to avoid duplicate mentions; using LLM/WebClient flow")

        # 次に LLM でメッセージ本体を生成（セッション LLM、なければローカル ChatOpenAI を作成）
        llm_for_generation = getattr(st.session_state, "llm", None)
        if llm_for_generation is None:
            try:
                llm_for_generation = ChatOpenAI(model_name=ct.MODEL, temperature=ct.TEMPERATURE, streaming=False)
                logger.info("Created local ChatOpenAI instance for background generation")
            except Exception as e:
                logger.exception(f"Failed to create local ChatOpenAI for background generation: {e}")
                llm_for_generation = None

        generated_text = None
        if llm_for_generation is not None:
            try:
                # try chat-style call first
                try:
                    gen_resp = llm_for_generation([
                        {"role": "system", "content": "あなたはSlackで投稿するための文章を生成するアシスタントです。出力はプレーンテキストのみとしてください。"},
                        {"role": "user", "content": prompt_message},
                    ])
                    generated_text = getattr(gen_resp, 'content', None) or (gen_resp[0] if isinstance(gen_resp, (list, tuple)) and gen_resp else None) or str(gen_resp)
                except Exception:
                    gen_resp = llm_for_generation(prompt_message)
                    generated_text = getattr(gen_resp, 'content', None) or str(gen_resp)
            except Exception as e:
                logger.warning(f"LLM generation failed: {e}; falling back to concise summary")

        # メンションテキスト（SlackのユーザーIDで <@U...> 形式）、重複排除
        mention_list = []
        seen = set()
        for s in slack_ids:
            if not s:
                continue
            if s in seen:
                continue
            seen.add(s)
            mention_list.append(f"<@{s}>")
        mention_text = " ".join(mention_list)

        if generated_text:
            post_body = generated_text
            # 生成済みテキストに既にメンション行や生の SlackID 行、あるいは @表示（表示名含む）が含まれている場合は削除
            cleaned_lines = []
            for line in post_body.splitlines():
                stripped = line.strip()
                # Skip lines that are only mentions like '<@U...' or raw IDs like 'U09...'
                if not stripped:
                    continue
                if stripped.startswith('<@') and '>' in stripped:
                    continue
                if stripped.startswith('@'):
                    # Any line starting with @ (display mention) should be removed to avoid duplicate mentions
                    continue
                # Remove inline bare U... tokens anywhere in line
                # Replace occurrences of tokens that look like U0.. with empty string
                import re
                line = re.sub(r'\bU[0-9A-Z]{6,}\b', '', line)
                # After removals, skip if line is empty/only punctuation
                if line.strip() == '' or all(c in ' ,。、.-–—()[]{}' for c in line.strip()):
                    continue
                cleaned_lines.append(line)
            post_body = '\n'.join(cleaned_lines).strip()
        else:
            safe_query = (chat_message or "(内容がありません)").strip()
            safe_requester = (requester or "不明な依頼者")
            safe_datetime = now_datetime or ""
            safe_context = (context_for_prompt or "").strip()
            if len(safe_context) > 800:
                safe_context = safe_context[:800] + "... (省略)"

            post_body_lines = [
                "以下のお問い合わせを受け付けました。担当者の確認をお願いします。",
                "",
                f"・問い合わせ内容: {safe_query}",
                f"・問い合わせ者: {safe_requester}",
                f"・受付日時: {safe_datetime}",
            ]
            if safe_context:
                post_body_lines.append("")
                post_body_lines.append("・参照用（該当従業員情報の抜粋）:")
                post_body_lines.append(safe_context)

            post_body = "\n".join(post_body_lines)

        # メンションは先頭に一度だけ付与し、本文に問い合わせ者情報があるか確認
        # もしpromptや生成文が問い合わせ者を固定の "山田太郎" のまま出力してしまう場合は
        # セッションの requester を優先して置換する
        if requester and post_body:
            # 強制的に「問い合わせ者」行を置換する: '・問い合わせ者: ...' を探して上書き
            import re
            def replace_requester_line(text, name):
                pattern = r"(^\s*・問い合わせ者:\s*).*$"
                if re.search(pattern, text, flags=re.MULTILINE):
                    return re.sub(pattern, rf"\1{name}", text, flags=re.MULTILINE)
                # フォールバック: 単純置換
                return text.replace('山田太郎', name).replace('{requester}', name)

            post_body = replace_requester_line(post_body, requester)

        post_text = f"{mention_text}\n\n{post_body}" if mention_text else post_body

        # 投稿先チャンネルは環境変数で上書き可能。なければ 'general' を使う。
        channel = os.getenv("SLACK_CHANNEL", "general")
        try:
            token = os.getenv("SLACK_USER_TOKEN")
            client = WebClient(token=token)

            logger.info(f"Attempting WebClient.auth_test with token set={bool(token)}")
            try:
                auth = client.auth_test()
                logger.info(f"auth_test: {auth}")
            except Exception as e:
                logger.warning(f"auth_test failed: {e}")

            logger.info(f"Attempting WebClient.chat_postMessage to channel={channel} with token set={bool(token)}")
            resolved_channel = channel
            try:
                if not channel.startswith(("C", "G", "U", "D", "#")):
                    convs = client.conversations_list(types="public_channel,private_channel")
                    channels = convs.get("channels", []) if isinstance(convs, dict) else []
                    for ch in channels:
                        if ch.get("name") == channel or ch.get("name_normalized") == channel:
                            resolved_channel = ch.get("id")
                            break
            except Exception as e:
                logger.warning(f"Could not resolve channel name to id: {e}; will try posting with given channel value")

            # Final safeguard: replace any remaining hardcoded placeholders with the requester
            if requester:
                post_text = post_text.replace('山田太郎', requester).replace('{requester}', requester)

            logger.info(f"Final Slack post_text (truncated 1000 chars): {post_text[:1000]}")
            logger.info(f"Requester used for notification: {requester}")
            resp = client.chat_postMessage(channel=resolved_channel, text=post_text)
            try:
                ok = resp.get("ok", None) if hasattr(resp, 'get') else getattr(resp, 'ok', None)
            except Exception:
                ok = None
            logger.info(f"WebClient.postMessage response ok={ok}")
        except Exception as e:
            logger.exception(f"Slack WebClient fallback failed: {e}")

        return ct.CONTACT_THANKS_MESSAGE


def get_target_employees(employees, employee_ids):
    """
    問い合わせ内容と関連性が高い従業員情報一覧の取得

    Args:
        employees: 問い合わせ内容と関連性が高い従業員情報一覧
        employee_ids: 問い合わせ内容と関連性が「特に」高い従業員のID一覧

    Returns:
        問い合わせ内容と関連性が「特に」高い従業員情報一覧
    """

    target_employees = []
    duplicate_check = []
    target_text = "従業員ID"
    for employee in employees:
        # 従業員IDの取得
        num = employee.page_content.find(target_text)
        employee_id = employee.page_content[num+len(target_text)+2:].split("\n")[0]
        # 問い合わせ内容と関連性が高い従業員情報を、IDで照合して取得（重複除去）
        if employee_id in employee_ids:
            if employee_id in duplicate_check:
                continue
            duplicate_check.append(employee_id)
            target_employees.append(employee)

    return target_employees


def get_slack_ids(target_employees):
    """
    SlackIDの一覧を取得

    Args:
        target_employees: 問い合わせ内容と関連性が高い従業員情報一覧

    Returns:
        SlackIDの一覧
    """

    target_text = "SlackID"
    slack_ids = []
    for employee in target_employees:
        num = employee.page_content.find(target_text)
        slack_id = employee.page_content[num+len(target_text)+2:].split("\n")[0]
        slack_ids.append(slack_id)
    
    return slack_ids


def create_slack_id_text(slack_ids):
    """
    SlackIDの一覧を取得

    Args:
        slack_ids: SlackIDの一覧

    Returns:
        SlackIDを「と」で繋いだテキスト
    """
    slack_id_text = ""
    for i, id in enumerate(slack_ids):
        slack_id_text += f"「{id}」"
        # 最後のSlackID以外、連結後に「と」を追加
        if not i == len(slack_ids)-1:
            slack_id_text += "と"
    
    return slack_id_text


def get_context(docs):
    """
    プロンプトに埋め込むための従業員情報テキストの生成
    Args:
        docs: 従業員情報の一覧

    Returns:
        生成した従業員情報テキスト
    """

    context = ""
    for i, doc in enumerate(docs, start=1):
        context += "===========================================================\n"
        context += f"{i}人目の従業員情報\n"
        context += "===========================================================\n"
        context += doc.page_content + "\n\n"

    return context


def get_datetime():
    """
    現在日時を取得

    Returns:
        現在日時
    """

    dt_now = datetime.datetime.now()
    now_datetime = dt_now.strftime('%Y年%m月%d日 %H:%M:%S')

    return now_datetime


def search_faq(q, top_k=3):
    """
    FAQ DB を検索して上位候補を返す（MATCH → LIKE フォールバック）

    Args:
        q: クエリ文字列
        top_k: 取得件数

    Returns:
        [{id, category, question, snippet, url}, ...]
    """
    db_path = Path(__file__).resolve().parents[0] / ".db_faq" / "faq.db"
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    # try FTS MATCH
    try:
        cur.execute("SELECT id, category, question, answer, url FROM faq WHERE faq MATCH ? LIMIT ?", (q, top_k))
        rows = cur.fetchall()
    except Exception:
        rows = []
    # fallback to LIKE if no rows
    if not rows:
        like_q = f"%{q}%"
        cur.execute("SELECT id, category, question, answer, url FROM faq WHERE question LIKE ? OR answer LIKE ? LIMIT ?",
                    (like_q, like_q, top_k))
        rows = cur.fetchall()

    results = []
    for r in rows:
        combined = (r[2] or "") + "\n" + (r[3] or "")
        # simple snippet: first 200 chars or around query
        snippet = combined
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        results.append({
            'id': r[0],
            'category': r[1],
            'question': r[2],
            'snippet': snippet,
            'url': r[4]
        })
    conn.close()
    return results


def get_customer_history_summary(customer_id=None, customer_name=None, limit=5):
    """
    問い合わせ対応履歴CSVから顧客の過去履歴を抽出して簡易サマリを返す

    Args:
        customer_id: 顧客ID（優先）
        customer_name: 顧客名（顧客IDが無い場合の代替）
        limit: 取得する最近の履歴件数

    Returns:
        {found: bool, summary: str, recent_interactions: [{date, subject, excerpt, ref_row}], count: int}
    """
    import csv
    from pathlib import Path

    csv_path = Path(__file__).resolve().parents[0] / "data" / "slack" / "問い合わせ対応履歴.csv"
    if not csv_path.exists():
        return {"found": False, "summary": "履歴ファイルが見つかりません。", "recent_interactions": [], "count": 0}

    records = []
    with open(csv_path, newline='', encoding=ct.CSV_ENCODING) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # CSV のカラム名に依存するため、存在しない場合はスキップ
            records.append(row)

    # フィルタリング
    matched = []
    for r in records:
        # try customer_id match
        if customer_id and (r.get('顧客ID') == customer_id or r.get('customer_id') == customer_id):
            matched.append(r)
            continue
        # try customer_name match
        if customer_name and (r.get('顧客名') == customer_name or r.get('name') == customer_name):
            matched.append(r)

    if not matched:
        return {"found": False, "summary": "該当する顧客の履歴は見つかりませんでした。", "recent_interactions": [], "count": 0}

    # 最近の履歴順（CSV内の日時カラム名がわからないため、降順はそのまま最後の方を最近と仮定）
    recent = matched[-limit:][::-1]

    recent_interactions = []
    combined_texts = []
    for r in recent:
        date = r.get('日時') or r.get('date') or r.get('timestamp') or ''
        subject = r.get('件名') or r.get('タイトル') or r.get('subject') or ''
        body = r.get('内容') or r.get('本文') or r.get('message') or ''
        excerpt = (body[:200] + '...') if len(body) > 200 else body
        combined_texts.append(f"{date} {subject} {excerpt}")
        recent_interactions.append({"date": date, "subject": subject, "excerpt": excerpt, "ref_row": r})

    # 簡易サマリは最近の件名と抜粋をまとめたもの（後で LLM 要約に差し替え可能）
    summary = "\n".join(combined_texts)

    return {"found": True, "summary": summary, "recent_interactions": recent_interactions, "count": len(matched)}


def preprocess_func(text):
    """
    形態素解析による日本語の単語分割
    Args:
        text: 単語分割対象のテキスト

    Returns:
        単語分割を実施後のテキスト
    """

    tokenizer_obj = dictionary.Dictionary(dict="full").create()
    mode = tokenizer.Tokenizer.SplitMode.A
    tokens = tokenizer_obj.tokenize(text ,mode)
    words = [token.surface() for token in tokens]
    words = list(set(words))

    return words


def adjust_string(s):
    """
    Windows環境でRAGが正常動作するよう調整
    
    Args:
        s: 調整を行う文字列
    
    Returns:
        調整を行った文字列
    """
    # 調整対象は文字列のみ
    if type(s) is not str:
        return s

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    if sys.platform.startswith("win"):
        s = unicodedata.normalize('NFC', s)
        s = s.encode("cp932", "ignore").decode("cp932")
        return s
    
    # OSがWindows以外の場合はそのまま返す
    return s


def run_business_hours_tool(param: str):
    """
    営業時間やSLAに関する定型回答を返すツールラッパー

    Args:
        param: ユーザークエリ（自然文）

    Returns:
        str: 人間向けの回答テキスト
    """
    q = (param or '').lower()
    hours = ct.BUSINESS_HOURS
    sla = ct.DEFAULT_SLA_HOURS

    # キーワード判定
    if '営業時間' in q or '何時' in q or 'open' in q:
        return f"平日: {hours['weekday']}、土日: {hours['weekend']}、祝日: {hours['holidays']}。"
    if '初動' in q or '対応時間' in q or 'sla' in q:
        return f"初動の目安は原則{sla}時間（約3営業日）です。緊急の場合はお問い合わせください。"
    if '土曜' in q or '土日' in q or '週末' in q:
        return f"週末の営業時間は {hours['weekend']} です。"

    # デフォルトの案内
    return f"営業時間は平日 {hours['weekday']}、週末 {hours['weekend']} です。初動の目安は原則{sla}時間です。"