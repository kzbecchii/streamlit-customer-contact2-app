from concurrent.futures import ThreadPoolExecutor
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import constants as ct


def query_once(name, q):
    try:
        emb = OpenAIEmbeddings()
        db = Chroma(persist_directory=ct.DB_SERVICE_PERSIST_DIR, embedding_function=emb)
        retriever = db.as_retriever(search_kwargs={"k":3})
        docs = retriever.get_relevant_documents(q)
        print(f"[{name}] retrieved docs: {len(docs)}")
        for i,d in enumerate(docs,1):
            print(f"[{name}] -- Doc #{i} preview:")
            print(d.page_content[:200].replace('\n',' '))
            print(f"[{name}] ---")
    except Exception as e:
        print(f"[{name}] error: {e}")


if __name__ == '__main__':
    q = '商品の価格は？'
    with ThreadPoolExecutor(max_workers=2) as ex:
        ex.submit(query_once, 'session-A', q)
        ex.submit(query_once, 'session-B', q)
