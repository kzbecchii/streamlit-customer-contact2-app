import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / ".db_faq" / "faq.db"


def snippet(text, q, width=120):
    # 簡易抜粋: クエリ語を含む最初の位置を見つけて前後を切る
    pos = text.find(q)
    if pos == -1:
        return text[:width] + ("..." if len(text) > width else "")
    start = max(0, pos - 30)
    end = min(len(text), pos + len(q) + 90)
    return ("..." if start>0 else "") + text[start:end] + ("..." if end < len(text) else "")


def query_faq(q, top_k=3):
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found: {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    # Simple MATCH search
    cur.execute("SELECT id, category, question, answer, url FROM faq WHERE faq MATCH ? LIMIT ?", (q, top_k))
    rows = cur.fetchall()
    # FTS tokenization for Japanese may not match; fall back to LIKE search if no rows
    if not rows:
        like_q = f"%{q}%"
        cur.execute("SELECT id, category, question, answer, url FROM faq WHERE question LIKE ? OR answer LIKE ? LIMIT ?",
                    (like_q, like_q, top_k))
        rows = cur.fetchall()
    results = []
    for r in rows:
        qtext = r[2] or ""
        atext = r[3] or ""
        combined = qtext + "\n" + atext
        results.append({
            'id': r[0],
            'category': r[1],
            'question': qtext,
            'snippet': snippet(combined, q),
            'url': r[4]
        })
    conn.close()
    return results


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python query_faq.py <query>")
        sys.exit(1)
    q = sys.argv[1]
    for r in query_faq(q):
        print('---')
        print(r['id'], r['category'])
        print('Q:', r['question'])
        print('Snippet:', r['snippet'])
        print('URL:', r['url'])
