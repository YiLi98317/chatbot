from __future__ import annotations

import requests


def _generate_endpoint(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/api/generate"


def generate(prompt: str, model: str, base_url: str) -> str:
    resp = requests.post(
        _generate_endpoint(base_url),
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        # Make the error actionable (common causes: wrong base URL, model not pulled).
        body_head = ""
        try:
            body_head = (resp.text or "")[:400]
        except Exception:
            body_head = ""
        raise RuntimeError(
            f"Ollama /api/generate failed: status={resp.status_code} url={resp.url}. "
            f"Check OLLAMA_BASE_URL and that the model is pulled (ollama list). "
            f"Response head: {body_head!r}"
        ) from e

    data = resp.json()
    output = data.get("response")
    if not isinstance(output, str):
        raise RuntimeError(f"Unexpected generate response: {data}")
    return output
