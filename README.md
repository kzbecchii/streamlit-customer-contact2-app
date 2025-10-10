# streamlit-customer-contact2-app

## 永続化ディレクトリ (Persist directories)

このリポジトリでアプリ実行時に使用する永続化ディレクトリは以下の通りです（`constants.py` に定義されています）。

- `DB_SERVICE_PERSIST_DIR` : `./.db_service_index`  
	- サービス／商品情報用の Chroma ベクトルストアの永続ディレクトリ
- `DB_COMPANY_PERSIST_DIR` : `./.db_company`  
	- 会社情報用の Chroma 永続ディレクトリ
- `DB_CUSTOMER_PERSIST_DIR` : `./.db_customer`  
	- 顧客関連ドキュメント用の Chroma 永続ディレクトリ
- `DB_ALL_PERSIST_DIR` : `./.db_all`  
	- 全ドキュメント（company + service + customer）をまとめた Chroma 永続ディレクトリ
- `DB_FAQ_PATH` : `./.db_faq/faq.db`  
	- FAQ 用の SQLite DB パス

これらのパスは `constants.py` で一元管理されており、`utils.create_rag_chain` などが参照します。

## 運用手順（重要）

1. デプロイ先での永続化
	 - 本番サーバやコンテナでは上記のディレクトリを永続ボリュームとしてマウントしてください。
	 - 例（Docker Compose）: `./data/chroma:/app/.db_service_index`

2. パーミッション
	 - アプリ実行ユーザーがこれらのディレクトリに読み書きできることを確認してください。

3. 既存 DB の移行
	 - リポジトリ内に旧ディレクトリ（例: `./.db_service`）が存在する場合、運用で使用する新しいディレクトリ名（`./.db_service_index`）へ移動またはコピーしてください。
	 - 例: `mv ./.db_service/* ./.db_service_index/`

4. バックアップ
	 - 永続化ディレクトリは定期的にバックアップしてください（Chroma の内部形式のためファイル単位でのスナップショット推奨）。

5. ローカル開発
	 - `.env` に API キー等を設定してからアプリを起動してください。仮想環境使用を推奨します。

## 参照とメンテナンス
- `constants.py` に定義された定数を修正することで、参照先のディレクトリを変更できます。
- 将来的には `langchain-chroma` への移行を検討してください（LangChain の API 変更に伴う推奨移行）。
