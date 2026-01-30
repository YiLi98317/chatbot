import os
import sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load .env so QDRANT_URL / QDRANT_API_KEY can be picked up without exporting
load_dotenv()

qdrant_url = os.getenv("QDRANT_URL")
if not qdrant_url:
    print("QDRANT_URL is not set. Set it in your .env or export it.")
    sys.exit(1)

qdrant_api_key = os.getenv("QDRANT_API_KEY")
collection = os.getenv("COLLECTION", "chatbot_docs")

c = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
try:
    c.delete_collection(collection)
    print(f"Deleted {collection}")
except Exception as e:
    print("Delete skipped:", e)