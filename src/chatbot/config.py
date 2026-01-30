import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    qdrant_url: str
    qdrant_api_key: Optional[str]
    ollama_base_url: str
    default_collection: str
    default_top_k: int
    data_dir: str
    db_uri: Optional[str]
    sql_table: Optional[str]
    sql_updated_at: str
    sql_pk: str
    embed_provider: str
    embed_model: str
    chat_model: str
    enable_query_planner: bool
    enable_legacy_trailing_trim: bool
    dev_mode: bool
    # Feature flags for retrieval layers
    enable_bm25_layer: bool
    enable_prf_layer: bool
    enable_qexp_layer: bool
    # Observability/SLA
    obs_metrics_enabled: bool
    sla_p95_latency_ms: int


def get_settings() -> Settings:
    load_dotenv()
    qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_url:
        raise RuntimeError("QDRANT_URL is required but was not found in environment.")

    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    default_collection = os.getenv("QDRANT_COLLECTION", "chatbot_docs")
    default_top_k = int(os.getenv("TOP_K", "4"))
    # Determine project root (two levels up from this file: src/chatbot/)
    project_root = Path(__file__).resolve().parents[2]
    default_data_dir = str(project_root / "data")
    data_dir = os.getenv("DATA_DIR", default_data_dir)

    db_uri = os.getenv("DB_URI")  # e.g., sqlite:///data/knowledge.db
    if not db_uri:
        # Convenience: allow MySQL config via MYSQL_* env vars (matches reingest_chinook_mysql.py).
        mysql_user = os.getenv("MYSQL_USER")
        mysql_password = os.getenv("MYSQL_PASSWORD")
        mysql_db = os.getenv("MYSQL_DB")
        mysql_host = os.getenv("MYSQL_HOST", "localhost")
        mysql_port = os.getenv("MYSQL_PORT", "3306")
        if mysql_user and mysql_password and mysql_db:
            # URL-encode username/password to be safe.
            u = quote_plus(str(mysql_user))
            p = quote_plus(str(mysql_password))
            db_uri = f"mysql+pymysql://{u}:{p}@{mysql_host}:{mysql_port}/{mysql_db}?charset=utf8mb4"
    sql_table = os.getenv("SQL_TABLE")
    sql_updated_at = os.getenv("SQL_UPDATED_AT", "updated_at")
    sql_pk = os.getenv("SQL_PK", "id")

    embed_provider = (os.getenv("EMBED_PROVIDER", "ollama") or "ollama").strip().lower()
    if embed_provider not in {"ollama", "sentence_transformers"}:
        embed_provider = "ollama"
    embed_model = os.getenv("EMBED_MODEL", "nomic-embed-text")
    chat_model = os.getenv("CHAT_MODEL", "llama3.1")
    enable_query_planner = os.getenv("ENABLE_QUERY_PLANNER", "true").lower() not in {"0", "false", "no"}
    enable_legacy_trailing_trim = os.getenv("ENABLE_LEGACY_TRAILING_TRIM", "false").lower() not in {"0", "false", "no"}
    dev_mode = os.getenv("CHATBOT_DEV_MODE", "false").lower() not in {"0", "false", "no"}
    enable_bm25_layer = os.getenv("ENABLE_BM25_LAYER", "true").lower() not in {"0", "false", "no"}
    enable_prf_layer = os.getenv("ENABLE_PRF_LAYER", "true").lower() not in {"0", "false", "no"}
    enable_qexp_layer = os.getenv("ENABLE_QEXP_LAYER", "true").lower() not in {"0", "false", "no"}
    obs_metrics_enabled = os.getenv("OBS_METRICS_ENABLED", "true").lower() not in {"0", "false", "no"}
    sla_p95_latency_ms = int(os.getenv("SLA_P95_LATENCY_MS", "1500"))

    return Settings(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        ollama_base_url=ollama_base_url,
        default_collection=default_collection,
        default_top_k=default_top_k,
        data_dir=data_dir,
        db_uri=db_uri,
        sql_table=sql_table,
        sql_updated_at=sql_updated_at,
        sql_pk=sql_pk,
        embed_provider=embed_provider,
        embed_model=embed_model,
        chat_model=chat_model,
        enable_query_planner=enable_query_planner,
        enable_legacy_trailing_trim=enable_legacy_trailing_trim,
        dev_mode=dev_mode,
        enable_bm25_layer=enable_bm25_layer,
        enable_prf_layer=enable_prf_layer,
        enable_qexp_layer=enable_qexp_layer,
        obs_metrics_enabled=obs_metrics_enabled,
        sla_p95_latency_ms=sla_p95_latency_ms,
    )
