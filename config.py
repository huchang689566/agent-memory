"""
统一配置模块
-----------
从 .env 文件和环境变量加载配置，供所有子系统使用。
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# ── 全局设置（必须在任何模型加载前执行） ──
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

# ── 加载 .env ──
try:
    load_dotenv(Path(__file__).parent / ".env")
except Exception:
    pass

# ── 日志 ──
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
)

# 关掉三方库的噪音日志
for noisy in ["httpx", "httpcore", "faiss", "sentence_transformers",
              "huggingface_hub", "urllib3", "openai", "asyncio",
              "langgraph", "langchain", "langchain_core"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

# 关掉弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langgraph.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain.*")

CFG = {
    # ── LLM ──
    "api_key": os.getenv("OPENAI_API_KEY"),
    "base_url": os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
    "llm_model": os.getenv("LLM_MODEL", "deepseek-chat"),
    "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),

    # ── Redis ──
    "redis_host": os.getenv("REDIS_HOST", "localhost"),
    "redis_port": int(os.getenv("REDIS_PORT", "6379")),
    "redis_db": int(os.getenv("REDIS_DB", "0")),

    # ── 记忆参数 ──
    "max_turns": int(os.getenv("MAX_TURNS", "10")),
    "top_k_memories": int(os.getenv("TOP_K_MEMORIES", "5")),
    "duplicate_threshold": float(os.getenv("DUPLICATE_THRESHOLD", "0.85")),

    # ── 高德地图 ──
    "gaode_api_key": os.getenv("GAODE_API_KEY", ""),

    # ── 服务端口 ──
    "rag_host": os.getenv("RAG_HOST", "127.0.0.1"),
    "rag_port": int(os.getenv("RAG_PORT", "8002")),
    "mcp_host": os.getenv("MCP_HOST", "127.0.0.1"),
    "mcp_port": int(os.getenv("MCP_PORT", "8001")),
}

# ── 数据目录 ──
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

METADATA_PATH = DATA_DIR / "memories.json"
FAISS_INDEX_PATH = DATA_DIR / "index.faiss"
SESSIONS_DIR = DATA_DIR / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)
