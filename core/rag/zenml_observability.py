from typing import Any, Callable
from datetime import datetime
try:
    from zenml.logger import get_logger
except Exception:  # pragma: no cover - ZenML optional
    get_logger = None

class ZenMLObservability:
    """Observability callbacks using ZenML logging."""

    def __init__(self):
        if get_logger is None:
            raise ImportError("zenml is not installed")
        self.logger = get_logger(__name__)

    def rag_step_callback(self) -> Callable:
        """Return a callback to log RAG steps to ZenML."""

        def callback(step: str, input: Any, output: Any) -> None:
            ts = datetime.now().isoformat()
            self.logger.info(f"{ts} - step={step}")
            self.logger.debug(f"input_type={type(input).__name__}")
            if step == "retrieval":
                count = len(output) if isinstance(output, list) else 0
                self.logger.info(f"retrieved_documents={count}")
            elif step == "generation":
                self.logger.info(f"output_length={len(str(output))}")
        return callback
