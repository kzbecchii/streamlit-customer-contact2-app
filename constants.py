"""
このファイルは、固定の文字列や数値などのデータを変数として一括管理するファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader

# ログ出力系（モジュールインポート時に参照されるため早めに定義）
LOG_DIR_PATH = "./logs"
LOGGER_NAME = "ApplicationLog"
LOG_FILE = "application.log"


############################################################
# 共通変数の定義
############################################################

# ==========================================
# 画面表示系
# ==========================================
APP_NAME = "問い合わせ対応自動化AIエージェント"
CHAT_INPUT_HELPER_TEXT = "こちらからメッセージを送信してください。"
APP_BOOT_MESSAGE = "アプリが起動されました。"
USER_ICON_FILE_PATH = "./images/user_icon.jpg"
AI_ICON_FILE_PATH = "./images/ai_icon.jpg"
WARNING_ICON = ":material/warning:"
ERROR_ICON = ":material/error:"
SYSTEM_PROMPT_NOTICE_SLACK = """
【問い合わせ情報】
・問い合わせ内容: {query}
・カテゴリ: 
・問い合わせ者: {requester}
・日時: {now_datetime}

【回答・対応案】
＜1つ目＞
●内容: 
●根拠: 

＜2つ目＞
●内容: 
●根拠: 

＜3つ目＞
●内容: 
●根拠: 

【メンション先の選定理由】
ここには問い合わせ内容に基づき、誰に連絡するべきかを「です・ます調」で具体的に記載してください。氏名を含めてください。

【参照資料】
・従業員情報.csv
・問い合わせ履歴.csv
"""

# RAG データのトップフォルダと DB パスの定義（不足していた値を補完）
RAG_TOP_FOLDER_PATH = "./data/rag"

# 各種 DB パス
DB_ALL_PATH = "./.db_all"
DB_COMPANY_PATH = "./.db_company"

# トークン関連のデフォルト
MAX_ALLOWED_TOKENS = 1000

# エンコーディング種別（tiktoken 用）
ENCODING_KIND = "cl100k_base"

# LLM モデル設定（初期値）
# 実運用では .env や設定ファイルで上書きしてください
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0


# ログ出力系
LOG_DIR_PATH = "./logs"
LOGGER_NAME = "ApplicationLog"
LOG_FILE = "application.log"


RAG_TOP_FOLDER_PATH = "./data/rag"

DB_ALL_PATH = "./.db_all"
DB_COMPANY_PATH = "./.db_company"
DB_SERVICE_PATH = "./.db_service"
DB_CUSTOMER_PATH = "./.db_customer"

DB_NAMES = {
    DB_COMPANY_PATH: f"{RAG_TOP_FOLDER_PATH}/company",
    DB_SERVICE_PATH: f"{RAG_TOP_FOLDER_PATH}/service",
    DB_CUSTOMER_PATH: f"{RAG_TOP_FOLDER_PATH}/customer"
}

# 問い合わせモード用の即時サンクスメッセージ
CONTACT_THANKS_MESSAGE = (
    "\n    このたびはお問い合わせいただき、誠にありがとうございます。\n"
    "    担当者が内容を確認し、3営業日以内にご連絡いたします。\n"
    "    ただし問い合わせ内容によっては、ご連絡いたしかねる場合がございます。\n"
    "    もしお急ぎの場合は、お電話にてご連絡をお願いいたします。\n"

)

AI_AGENT_MODE_ON = "利用する"
AI_AGENT_MODE_OFF = "利用しない"

CONTACT_MODE_ON = "ON"
CONTACT_MODE_OFF = "OFF"

SEARCH_COMPANY_INFO_TOOL_NAME = "search_company_info_tool"
SEARCH_COMPANY_INFO_TOOL_DESCRIPTION = "自社「株式会社EcoTee」に関する情報を参照したい時に使う"
SEARCH_SERVICE_INFO_TOOL_NAME = "search_service_info_tool"
SEARCH_SERVICE_INFO_TOOL_DESCRIPTION = "自社「株式会社EcoTee」のサービス、商品に関する情報を参照したい時に使う"
SEARCH_CUSTOMER_COMMUNICATION_INFO_TOOL_NAME = "search_customer_communication_tool"
SEARCH_CUSTOMER_COMMUNICATION_INFO_TOOL_DESCRIPTION = "顧客とのやりとりに関する情報を参照したい時に使う"
SEARCH_WEB_INFO_TOOL_NAME = "search_web_tool"
SEARCH_WEB_INFO_TOOL_DESCRIPTION = "自社サービス「HealthX」に関する質問で、Web検索が必要と判断した場合に使う"
SEARCH_CUSTOMER_HISTORY_TOOL_NAME = "search_customer_history_tool"
SEARCH_CUSTOMER_HISTORY_TOOL_DESCRIPTION = "顧客の過去問い合わせ履歴を要約して参照したい時に使う"
SEARCH_BUSINESS_HOURS_TOOL_NAME = "business_hours_tool"
SEARCH_BUSINESS_HOURS_TOOL_DESCRIPTION = "営業時間や初動目安（SLA）に関する定型回答を返すツール"


# ==========================================
# Slack連携関連

# ==========================================
EMPLOYEE_FILE_PATH = "./data/slack/従業員情報.csv"
INQUIRY_HISTORY_FILE_PATH = "./data/slack/問い合わせ対応履歴.csv"
CSV_ENCODING = "utf-8-sig"


# ==========================================
# プロンプトテンプレート
# ==========================================
SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT = "会話履歴と最新の入力をもとに、会話履歴なしでも理解できる独立した入力テキストを生成してください。"

NO_DOC_MATCH_MESSAGE = "回答に必要な情報が見つかりませんでした。弊社に関する質問・要望を、入力内容を変えて送信してください。"

SYSTEM_PROMPT_INQUIRY = """
    あなたは社内文書を基に、顧客からの問い合わせに対応するアシスタントです。
    以下の条件に基づき、ユーザー入力に対して回答してください。

    【条件】
    1. ユーザー入力内容と以下の文脈との間に関連性がある場合のみ、以下の文脈に基づいて回答してください。
    2. ユーザー入力内容と以下の文脈との関連性が明らかに低い場合、「回答に必要な情報が見つかりませんでした。弊社に関する質問・要望を、入力内容を変えて送信してください。」と回答してください。
    3. 憶測で回答せず、あくまで以下の文脈を元に回答してください。
    4. できる限り詳細に、マークダウン記法を使って回答してください。
    5. マークダウン記法で回答する際にhタグの見出しを使う場合、最も大きい見出しをh3としてください。
    6. 複雑な質問の場合、各項目についてそれぞれ詳細に回答してください。
    7. 必要と判断した場合は、以下の文脈に基づかずとも、一般的な情報を回答してください。

    {context}
"""

SYSTEM_PROMPT_EMPLOYEE_SELECTION = """
    # 命令
    以下の「顧客からの問い合わせ」に対して、社内のどの従業員が対応するかを
    判定する生成AIシステムを作ろうとしています。

    以下の「従業員情報」は、問い合わせに対しての一人以上の対応者候補のデータです。
    しかし、問い合わせ内容との関連性が薄い従業員情報が含まれている可能性があります。
    以下の「条件」に従い、従業員情報の中から、問い合わせ内容との関連性が特に高いと思われる
    従業員の「ID」をカンマ区切りで返してください。

    # 顧客からの問い合わせ
    {query}

    # 条件
    - 全ての従業員が、問い合わせ内容との関連性が高い（対応者候補である）と判断した場合は、
    全ての従業員の従業員IDをカンマ区切りで返してください。ただし、関連性が低い（対応者候補に含めるべきでない）
    と判断した場合は省いてください。
    - 特に、「過去の問い合わせ対応履歴」と、「対応可能な問い合わせカテゴリ」、また「現在の主要業務」を元に判定を
    行ってください。
    - 一人も対応者候補がいない場合、空文字を返してください。
    - 判定は厳しく行ってください。

    # 従業員情報
    {context}

    # 出力フォーマット
    {format_instruction}
"""

SYSTEM_PROMPT_NOTICE_SLACK = """
    # 役割
    具体的で分量の多いメッセージの作成と、指定のメンバーにメンションを当ててSlackへの送信を行うアシスタント


    # 命令
    Slackの「テスト」チャンネルで、メンバーIDが{slack_id_text}のメンバーに一度だけメンションを当て、生成したメッセージを送信してください。


    # 送信先のチャンネル名
    テスト


    # メッセージの通知先
    メンバーIDが{slack_id_text}のメンバー


    # メッセージ通知（メンション付け）のルール
    - メッセージ通知（メンション付け）は、メッセージの先頭で「一度だけ」行ってください。
    - メンション付けの行は、メンションのみとしてください。


    # メッセージの生成条件
    - 各項目について、できる限り長い文章量で、具体的に生成してください。

        - 【メンション先の選定理由】は必ず「です・ます調（敬体／丁寧体）」で記述し、
            指示文そのもの（例：「以下の〜で記述してください」など）を出力しないでください。

    - 「メッセージフォーマット」を使い、以下の各項目の文章を生成してください。
        - 【問い合わせ情報】の「カテゴリ」
        - 【問い合わせ情報】の「日時」
        - 【回答・対応案とその根拠】

    - 「顧客から弊社への問い合わせ内容」と「従業員情報と過去の問い合わせ対応履歴」を基に文章を生成してください。

    - 【問い合わせ情報】の「カテゴリ」は、【問い合わせ情報】の「問い合わせ内容」を基に適切なものを生成してください。

    - 【回答・対応案】について、以下の条件に従って生成してください。
        - 回答・対応案の内容と、それが良いと判断した根拠を、それぞれ3つずつ生成してください。


    # 顧客から弊社への問い合わせ内容
    {query}


    # 従業員情報と過去の問い合わせ対応履歴
    {context}


    # メッセージフォーマット
    こちらは顧客問い合わせに対しての「担当者割り振り」と「回答・対応案の提示」を自動で行うAIアシスタントです。
    担当者は問い合わせ内容を確認し、対応してください。

    ================================================

    【問い合わせ情報】
    ・問い合わせ内容: {query}
    ・カテゴリ: 
        ・問い合わせ者: {requester}
    ・日時: {now_datetime}

    --------------------

    【回答・対応案】
    ＜1つ目＞
    ●内容: 
    ●根拠: 

    ＜2つ目＞
    ●内容: 
    ●根拠: 

    ＜3つ目＞
    ●内容: 
    ●根拠: 

    --------------------

    【メンション先の選定理由】
    （ここに、問い合わせ内容に基づくメンション先の選定理由を「です・ます調」で記載してください。氏名も記載してください。）

    --------------------

    【参照資料】
    ・従業員情報.csv
    ・問い合わせ履歴.csv
"""


# ==========================================
# エラー・警告メッセージ
# ==========================================
COMMON_ERROR_MESSAGE = "このエラーが繰り返し発生する場合は、管理者にお問い合わせください。"
INITIALIZE_ERROR_MESSAGE = "初期化処理に失敗しました。"
CONVERSATION_LOG_ERROR_MESSAGE = "過去の会話履歴の表示に失敗しました。"
MAIN_PROCESS_ERROR_MESSAGE = "ユーザー入力に対しての処理に失敗しました。"
DISP_ANSWER_ERROR_MESSAGE = "回答表示に失敗しました。"
INPUT_TEXT_LIMIT_ERROR_MESSAGE = f"入力されたテキストの文字数が受付上限値（{MAX_ALLOWED_TOKENS}）を超えています。受付上限値を超えないよう、再度入力してください。"


# ==========================================
# スタイリング
# ==========================================
STYLE = """
<style>
    .stHorizontalBlock {
        margin-top: -14px;
    }
    .stChatMessage + .stHorizontalBlock {
        margin-left: 56px;
    }
    .stChatMessage + .stHorizontalBlock .stColumn:nth-of-type(2) {
        margin-left: -24px;
    }
    @media screen and (max-width: 480px) {
        .stChatMessage + .stHorizontalBlock {
            flex-wrap: nowrap;
            margin-left: 56px;
        }
        .stChatMessage + .stHorizontalBlock .stColumn:nth-of-type(2) {
            margin-left: -206px;
        }
    }
</style>
"""


# ==========================================
# フィードバック関連表示文言
# ==========================================
FEEDBACK_YES = "はい"
FEEDBACK_NO = "いいえ"
FEEDBACK_BUTTON_LABEL = "送信"
FEEDBACK_THANKS_MESSAGE = "ご意見を送信いただき、ありがとうございました。"
FEEDBACK_YES_MESSAGE = "ご評価ありがとうございます。"
FEEDBACK_NO_MESSAGE = "ご意見の詳細を入力してください。"
FEEDBACK_REQUIRE_MESSAGE = "この回答は参考になりましたか？ よろしければフィードバックをお願いします。"
 
# フィードバックログ用定数
SATISFIED = "satisfied"
DISSATISFIED = "dissatisfied"


# ------------------------------------------
# RAG / Retriever 関連のデフォルト設定
# ------------------------------------------
TOP_K = 3
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Agent の最大反復回数（initialize_heavy_resources で参照）
AI_AGENT_MAX_ITERATIONS = 3

# 営業時間関連のデフォルト設定
BUSINESS_HOURS = {
    "weekday": "午前9時〜午後6時",
    "weekend": "午前10時〜午後4時",
    "holidays": "休業"
}

# 初動対応の目安（時間）
DEFAULT_SLA_HOURS = 72

# Document loader のマッピング（拡張子 -> ローダー呼び出し）
SUPPORTED_EXTENSIONS = {
    ".pdf": PyMuPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
}


# Ensemble Retriever の重み（BM25 と ベクトルの重み付け）
RETRIEVER_WEIGHTS = [0.6, 0.4]

# スピナー表示テキスト
SPINNER_TEXT = "回答を生成しています... 少々お待ちください。"