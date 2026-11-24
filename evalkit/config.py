"""All settings in one place. Override any of them with an environment variable or a line in .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _first_existing(*paths: Path) -> Path:
    """Windows ignores letter case in folder names, Linux (GitHub CI) does not: accept either spelling."""
    return next((p for p in paths if p.exists()), paths[0])


DATA_DIR = _first_existing(ROOT / "data", ROOT / "DATA")
CORPUS_DIR = _first_existing(DATA_DIR / "corpus", DATA_DIR / "CORPUS")
INDEX_DIR = DATA_DIR / "index"
GOLDEN_PATH = DATA_DIR / "golden_set.jsonl"
JUDGE_LABELS_PATH = DATA_DIR / "judge_labels.csv"
RESULTS_DIR = ROOT / "results"

# Embeddings run locally (no API key). Generation and judging call an LLM (OpenAI, or any
# OpenAI-compatible server such as Ollama via EVALKIT_LLM_BASE_URL).
EMBED_MODEL = os.getenv("EVALKIT_EMBED_MODEL", "all-MiniLM-L6-v2")
GEN_MODEL = os.getenv("EVALKIT_GEN_MODEL", "gpt-4o-mini")
JUDGE_MODEL = os.getenv("EVALKIT_JUDGE_MODEL", "gpt-4o")  # different from GEN_MODEL on purpose
LLM_BASE_URL = os.getenv("EVALKIT_LLM_BASE_URL") or None

# Retrieval knobs: the "one improvement" for the before/after proof is a change to these.
CHUNK_STRATEGY = os.getenv("EVALKIT_CHUNK_STRATEGY", "fixed")
CHUNK_SIZE = int(os.getenv("EVALKIT_CHUNK_SIZE", "60"))
CHUNK_OVERLAP = int(os.getenv("EVALKIT_CHUNK_OVERLAP", "0"))
TOP_K = int(os.getenv("EVALKIT_TOP_K", "5"))
