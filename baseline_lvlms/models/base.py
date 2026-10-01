from abc import ABC, abstractmethod
from typing import Optional


class BaseVLM(ABC):
    """Common interface every baseline model backend must implement.

    A backend only has to turn (prompt text, optional image path) into the
    model's raw text output. Everything else — prompt construction, JSON
    parsing/retry, batching, resuming, output format — lives in
    run_baseline_eval.py and is shared across all models.
    """

    name: str = "base"

    @abstractmethod
    def generate(self, prompt: str, image_path: Optional[str] = None) -> str:
        """Return the raw text output from the model for a given prompt
        (and optional image). Let exceptions (network errors, API errors)
        propagate — run_baseline_eval.py handles retries."""
        raise NotImplementedError
