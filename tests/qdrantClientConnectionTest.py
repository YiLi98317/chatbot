import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load environment variables from a local .env file if present.
load_dotenv()

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

if not qdrant_url or not qdrant_api_key:
    missing_names = [
        name
        for name, value in [("QDRANT_URL", qdrant_url), ("QDRANT_API_KEY", qdrant_api_key)]
        if not value
    ]
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing_names)}")

qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

print(qdrant_client.get_collections())
