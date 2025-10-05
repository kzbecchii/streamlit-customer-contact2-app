import csv
import sqlite3
import os
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "rag" / "FAQ" / "FAQファイル.csv"
DB_DIR = Path(__file__).resolve().parents[1] / ".db_faq"
DB_PATH = DB_DIR / "faq.db"


def build_faq_db(csv_path=CSV_PATH, db_path=DB_PATH):
    DB_DIR.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    # Create FTS5 virtual table
    # Use a simple column definition; concatenated content is not required for basic MATCH queries
    cur.execute("CREATE VIRTUAL TABLE faq USING fts5(id UNINDEXED, category, question, answer, url)")

    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        idx = 1
        for r in reader:
            qid = f"faq-{idx}"
            category = r.get('カテゴリ', '')
            question = r.get('質問', '')
            answer = r.get('回答', '')
            url = r.get('参考資料（出典）', '')
            cur.execute("INSERT INTO faq (id, category, question, answer, url) VALUES (?, ?, ?, ?, ?)",
                        (qid, category, question, answer, url))
            idx += 1
    conn.commit()
    conn.close()
    print(f"Built FAQ DB at: {db_path}")


if __name__ == '__main__':
    build_faq_db()
