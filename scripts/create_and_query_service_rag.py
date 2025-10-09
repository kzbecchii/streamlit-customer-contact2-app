from pathlib import Path
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

SERVICE_FOLDER = Path('./data/rag/service')
PERSIST_DIR = Path('./.db_service_index')
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3

print(f"Service folder: {SERVICE_FOLDER.resolve()}")
if not SERVICE_FOLDER.exists():
    print("Service folder does not exist")
    raise SystemExit(1)

# collect docs
from langchain.schema import Document

docs = []
for f in SERVICE_FOLDER.iterdir():
    if f.suffix.lower() == '.pdf':
        print(f"Loading {f.name}")
        loader = PyMuPDFLoader(str(f))
        try:
            loaded = loader.load()
            docs.extend(loaded)
            print(f" -> loaded {len(loaded)} chunks from {f.name}")
        except Exception as e:
            print(f" -> failed: {e}")

print(f"Total raw docs: {len(docs)}")

# split
text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separator='\n')
splitted = text_splitter.split_documents(docs)
print(f"Splitted into {len(splitted)} chunks")

# create embeddings and chroma
embeddings = OpenAIEmbeddings()
if PERSIST_DIR.exists():
    print(f"Removing existing persist dir {PERSIST_DIR}")
    import shutil
    shutil.rmtree(PERSIST_DIR)

PERSIST_DIR.mkdir(parents=True, exist_ok=True)
print(f"Creating Chroma at {PERSIST_DIR}")
db = Chroma.from_documents(splitted, embedding=embeddings, persist_directory=str(PERSIST_DIR))
print("Chroma created and persisted")

retriever = db.as_retriever(search_kwargs={"k": TOP_K})

# query test
queries = [
    '商品の価格は？',
    'ベーシックTシャツの価格は？',
    'オーガニックカラフルTシャツの説明'
]

for q in queries:
    print('='*40)
    print('Query:', q)
    docs = retriever.get_relevant_documents(q)
    print(f'Retrieved {len(docs)} docs')
    for i, d in enumerate(docs, start=1):
        preview = (d.page_content or '')[:400]
        print(f'-- Doc #{i} preview:\n{preview}\n')

print('Done')
