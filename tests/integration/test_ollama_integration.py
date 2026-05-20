"""Integration tests for Ollama provider via LiteLLM Chat Completions API.

These tests verify the dual-path routing in litellm_client.py works against a real
Ollama server. They were added after the original Ollama support PR shipped with
working unit tests but broken end-to-end behavior — `litellm.aresponses()` does
NOT support Ollama, but the mocked unit tests never caught it.

Skip conditions:
- RUN_E2E=1 must be set (matches the rest of tests/integration/)
- Ollama daemon must be reachable at OLLAMA_API_BASE (default localhost:11434)
- The model under test must be available locally (`ollama pull <model>` or signed in to cloud)
"""

import os

import httpx
import pytest

from multi_mcp.models.config import load_models_config
from multi_mcp.models.litellm_client import LiteLLMClient
from multi_mcp.models.resolver import ModelResolver
from multi_mcp.settings import settings

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_E2E") != "1",
    reason="Integration tests require RUN_E2E=1 (real Ollama server needed)",
)


def _ollama_reachable() -> bool:
    """Return True if the Ollama HTTP API responds within 2 seconds."""
    base = settings.ollama_api_base or "http://localhost:11434"
    try:
        resp = httpx.get(f"{base}/api/tags", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


def _model_available(model_name: str) -> bool:
    """Check whether `model_name` can be used through the local Ollama daemon.

    Cloud models (`:cloud` suffix) are streamed on-demand from Ollama's cloud
    and never appear in `/api/tags` — they are available if the daemon is
    reachable AND the user is signed in (`ollama signin`). For local models,
    require an exact tag match. For untagged names, match any tag with the
    same base name (avoids the trap where `qwen3:8b` would seem available
    when only `qwen3:4b` is installed).
    """
    # Cloud models: skip the /api/tags check entirely.
    # If the daemon is reachable (already verified by caller), they are usable.
    if model_name.endswith(":cloud"):
        return True

    base = settings.ollama_api_base or "http://localhost:11434"
    try:
        resp = httpx.get(f"{base}/api/tags", timeout=2.0)
        if resp.status_code != 200:
            return False
        models = {m.get("name", "") for m in resp.json().get("models", [])}
        if model_name in models:
            return True
        if ":" not in model_name:
            return any(m.split(":", 1)[0] == model_name for m in models)
        return False
    except Exception:
        return False


@pytest.mark.asyncio
async def test_ollama_end_to_end_via_litellm_client():
    """Smoke test: actually call a local Ollama model through LiteLLMClient.

    This test is the gap closer — it would have caught the original
    `litellm.aresponses()` bug because it makes a real HTTP call to Ollama.

    Picks the FIRST Ollama model in the resolver config that's available locally.
    Skips with a clear message if Ollama isn't running or no configured model is available.
    """
    if not _ollama_reachable():
        pytest.skip("Ollama not reachable at OLLAMA_API_BASE (start with `ollama serve`)")

    # Find a configured Ollama model that's actually available
    config = load_models_config()
    resolver = ModelResolver(config=config)

    ollama_models = [
        (name, cfg)
        for name, cfg in config.models.items()
        if cfg.litellm_model and cfg.litellm_model.split("/", 1)[0] in ("ollama", "ollama_chat")
    ]
    if not ollama_models:
        pytest.skip("No Ollama models defined in current config — add one to ~/.multi_mcp/config.yaml")

    # Try each configured Ollama model until we find one that's available
    test_model = None
    for name, cfg in ollama_models:
        # Extract the bare model name (strip ollama_chat/ prefix)
        model_id = cfg.litellm_model.split("/", 1)[1]
        if _model_available(model_id):
            test_model = (name, cfg)
            break

    if test_model is None:
        configured = [m[1].litellm_model for m in ollama_models]
        pytest.skip(f"No configured Ollama models available locally. Configured: {configured}. Pull one with `ollama pull <model>`.")

    name, cfg = test_model
    client = LiteLLMClient(resolver=resolver)

    canonical_name, model_config = resolver.resolve(name)
    result = await client.execute(
        canonical_name=canonical_name,
        model_config=model_config,
        messages=[{"role": "user", "content": "Reply with exactly one word: hello"}],
    )

    # The critical assertion: this used to fail with BadRequestError because
    # litellm.aresponses() doesn't route to Ollama.
    assert result.status == "success", f"Ollama call failed: {result.error}"
    assert result.content, "Ollama returned empty content"
    assert result.metadata.model == canonical_name
    assert result.metadata.latency_ms > 0
