import os
import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "").strip()

def ask_local_llm(prompt: str, timeout: int = 45) -> str | None:
    """Call a local Ollama model if configured; otherwise return None."""
    if not OLLAMA_MODEL:
        return None
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip() or None
    except (requests.RequestException, ValueError, KeyError):
        return None
