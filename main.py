"""
このファイルは、Webアプリのメイン処理が記述されたファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
from dotenv import load_dotenv
import logging
import streamlit as st
import utils
from initialize import initialize
import initialize as initialize_module
import components as cn
import constants as ct


############################################################
# 設定関連
############################################################
st.set_page_config(
    page_title=ct.APP_NAME
)

load_dotenv()

logger = logging.getLogger(ct.LOGGER_NAME)


############################################################
# 初期化処理
############################################################
try:
    initialize()
except Exception as e:
    logger.error(f"{ct.INITIALIZE_ERROR_MESSAGE}\n{e}")
    st.error(utils.build_error_message(ct.INITIALIZE_ERROR_MESSAGE), icon=ct.ERROR_ICON)
    st.stop()

# アプリ起動時のログ
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    logger.info(ct.APP_BOOT_MESSAGE)

############################################################
# 初期表示
############################################################
# タイトル表示
cn.display_app_title()

# サイドバー表示
cn.display_sidebar()

# AIメッセージの初期表示
cn.display_initial_ai_message()


############################################################
# スタイリング処理
############################################################
# 画面装飾を行う「CSS」を記述
st.markdown(ct.STYLE, unsafe_allow_html=True)


############################################################
# チャット入力の受け付け
############################################################
chat_message = st.chat_input(ct.CHAT_INPUT_HELPER_TEXT)


############################################################
# 会話ログの表示
############################################################
try:
    cn.display_conversation_log(chat_message)
except Exception as e:
    logger.error(f"{ct.CONVERSATION_LOG_ERROR_MESSAGE}\n{e}")
    st.error(utils.build_error_message(ct.CONVERSATION_LOG_ERROR_MESSAGE), icon=ct.ERROR_ICON)
    st.stop()


############################################################
# チャット送信時の処理
############################################################
if chat_message:
    # ==========================================
    # 会話履歴の上限を超えた場合、受け付けない
    # ==========================================
    # ユーザーメッセージのトークン数を取得
    # 'enc' は遅延初期化される可能性があるため、存在しない場合は
    # 軽量な初期化（エンコーダと簡易LLM）を行ってから使用する。
    if "enc" not in st.session_state:
        initialize_module.initialize_agent_executor()

    input_tokens = len(st.session_state.enc.encode(chat_message))
    # トークン数が、受付上限を超えている場合にエラーメッセージを表示
    if input_tokens > ct.MAX_ALLOWED_TOKENS:
        with st.chat_message("assistant", avatar=ct.AI_ICON_FILE_PATH):
            st.error(ct.INPUT_TEXT_LIMIT_ERROR_MESSAGE)
            st.stop()
    # トークン数が受付上限を超えていない場合、会話ログ全体のトークン数に加算
    st.session_state.total_tokens += input_tokens

    # ==========================================
    # 1. ユーザーメッセージの表示
    # ==========================================
    logger.info({"message": chat_message})

    res_box = st.empty()
    with st.chat_message("user", avatar=ct.USER_ICON_FILE_PATH):
        st.markdown(chat_message)
    
    # ==========================================
    # 2. LLMからの回答取得 or 問い合わせ処理
    # ==========================================
    # まずFAQを検索して候補を提示（該当があればユーザーがFAQで解決できるか選べる）
    faq_results = utils.search_faq(chat_message)
    use_faq = False
    if faq_results:
        with st.expander("該当しそうなFAQが見つかりました。こちらで解決しましたか？"):
            for i, r in enumerate(faq_results, start=1):
                st.markdown(f"**{i}. {r['question']}**  ")
                st.markdown(r['snippet'])
                if r.get('url'):
                    st.markdown(f"参照: {r['url']}")
                if st.button(f"このFAQで解決しました ({i})"):
                    use_faq = True
                    chosen_faq = r
                    break

    res_box = st.empty()
    if use_faq:
        # FAQで解決した場合は、その内容を回答として表示し、LLM呼び出しはスキップ
        result = f"FAQで解決しました:\n{chosen_faq['question']}\n{chosen_faq['snippet']}\n参照: {chosen_faq.get('url','') }"
    else:
        if st.session_state.contact_mode == ct.CONTACT_MODE_OFF:
            # 重いリソース（RAGチェーンやAgent）が未作成の場合は遅延初期化
            try:
                if "rag_chain" not in st.session_state or "agent_executor" not in st.session_state:
                    initialize_module.initialize_heavy_resources()
                with st.spinner(ct.SPINNER_TEXT):
                    result = utils.execute_agent_or_chain(chat_message)
            except Exception as e:
                # 初期化や実行で失敗した場合は詳しくログに残し、まずは軽量なLLMでの回答生成を試みる
                logger.exception(f"Failed to generate answer via agent/chain: {e}")
                try:
                    result = utils.generate_simple_answer(chat_message)
                except Exception as e2:
                    logger.exception(f"Fallback simple generation also failed: {e2}")
                    result = "申し訳ございません。回答の生成中にエラーが発生しました。しばらく経ってから再度お試しください。"
        else:
            # 問い合わせモード: ユーザーには即時にサンクスメッセージを表示し、
            # Slack 送信はバックグラウンドで非同期に実行してユーザーを待たせない。
            result = ct.CONTACT_THANKS_MESSAGE
            # 非同期に通知を送る（失敗時はログのみ）
            import threading
            # capture requester now to avoid accessing st.session_state from background thread
            requester_for_thread = st.session_state.get('user_name', None)
            threading.Thread(target=utils.notice_slack, args=(chat_message, requester_for_thread), daemon=True).start()
    
    # ==========================================
    # 3. 古い会話履歴を削除
    # ==========================================
    utils.delete_old_conversation_log(result)

    # ==========================================
    # 4. LLMからの回答表示
    # ==========================================
    with st.chat_message("assistant", avatar=ct.AI_ICON_FILE_PATH):
        try:
            cn.display_llm_response(result)

            logger.info({"message": result})
        except Exception as e:
            logger.error(f"{ct.DISP_ANSWER_ERROR_MESSAGE}\n{e}")
            st.error(utils.build_error_message(ct.DISP_ANSWER_ERROR_MESSAGE), icon=ct.ERROR_ICON)
            st.stop()
    # 回答の直後に重いリソース（RAGチェーン等）を再初期化して、別タブや次の問い合わせで最新の persisted DB を利用できるようにする
    try:
        initialize_module.initialize_heavy_resources()
    except Exception as e:
        logger.warning(f"Reinitialize heavy resources after response failed: {e}")
    
    # ==========================================
    # 5. 会話ログへの追加
    # ==========================================
    st.session_state.messages.append({"role": "user", "content": chat_message})
    st.session_state.messages.append({"role": "assistant", "content": result})


############################################################
# 6. ユーザーフィードバックのボタン表示
############################################################
if st.session_state.contact_mode == ct.CONTACT_MODE_OFF:
    cn.display_feedback_button()