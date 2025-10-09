import os
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
import constants as ct

service_folder = Path(ct.DB_NAMES.get(ct.DB_SERVICE_PATH))
print(f"Service folder: {service_folder}")
if not service_folder.exists():
    print("Service folder does not exist")
    exit(1)

files = list(service_folder.iterdir())
print(f"Files in folder ({len(files)}): {[f.name for f in files]}")

loaded = []
for f in files:
    ext = f.suffix.lower()
    if ext in ct.SUPPORTED_EXTENSIONS:
        print(f"Attempting to load: {f.name} (ext={ext})")
        loader_cls = ct.SUPPORTED_EXTENSIONS[ext]
        try:
            loader = loader_cls(str(f))
            docs = loader.load()
            print(f" -> loaded {len(docs)} docs from {f.name}")
            for d in docs:
                preview = (d.page_content or '')[:400]
                print(f" ---- preview:\n{preview}\n")
            loaded.extend(docs)
        except Exception as e:
            print(f" -> failed to load {f.name}: {e}")
    else:
        print(f"Skipping unsupported file: {f.name}")

print(f"Total loaded docs: {len(loaded)}")
print('Contains 商品情報.pdf:', any(f.name == '商品情報.pdf' for f in files))
