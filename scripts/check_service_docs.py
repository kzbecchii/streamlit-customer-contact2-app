import logging
import constants as ct
import utils
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("check_service_docs")

service_folder = ct.DB_NAMES.get(ct.DB_SERVICE_PATH)
print(f"Service folder: {service_folder}")
if not service_folder or not os.path.isdir(service_folder):
    print("Service folder does not exist")
    exit(1)

files = os.listdir(service_folder)
print(f"Files in folder ({len(files)}): {files}")

# Try to load documents via add_docs
loaded_docs = []
utils.add_docs(service_folder, loaded_docs)
print(f"Loaded docs count: {len(loaded_docs)}")

for i, d in enumerate(loaded_docs, start=1):
    meta = getattr(d, 'metadata', {})
    content_preview = getattr(d, 'page_content', '')[:500]
    print(f"--- Doc #{i} metadata: {meta}")
    print(f"--- Doc #{i} content preview:\n{content_preview}\n")

# Check for 商品情報.pdf existence explicitly
target = '商品情報.pdf'
print(f"Explicit file exists: {target in files}")
