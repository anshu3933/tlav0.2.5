"""Utility functions for verifying and initializing vector stores."""

from typing import Optional
from config.logging_config import get_module_logger
from config.app_config import config
from core.embeddings.vector_store import FAISSVectorStore

logger = get_module_logger("fix_vector_store")

VALID_STORE_TYPES = {"faiss", "chroma", "vertex"}


def verify_store_type(store_type: str) -> bool:
    """Validate the requested vector store type.

    Args:
        store_type: Store type string

    Returns:
        True if the store type is recognised, False otherwise
    """
    if store_type.lower() not in VALID_STORE_TYPES:
        logger.warning(f"Unknown vector store type: {store_type}")
        return False
    return True


def create_empty_faiss_index(index_dir: Optional[str] = None) -> bool:
    """Create an empty FAISS index if one does not already exist."""
    try:
        store = FAISSVectorStore(index_dir=index_dir or config.vector_store.index_dir)
        if store._index_exists():
            logger.debug("FAISS index already exists; skipping creation")
            return True
        store.build_index([])
        logger.info(f"Created empty FAISS index at {store.index_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to create FAISS index: {str(e)}", exc_info=True)
        return False
