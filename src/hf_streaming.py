from __future__ import annotations

import inspect
import requests
from typing import Any, Generator, Optional

from huggingface_hub import InferenceClient

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

RECOMMENDED_FREE_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3.5-mini-instruct",
    "google/gemma-2-2b-it",
    "tiiuae/falcon-7b-instruct",
    "bigscience/bloom-560m",
    "HuggingFaceH4/zephyr-7b-beta",
]


def fetch_free_inference_models(
    token: str | None = None,
    task: str = "text-generation",
    limit: int = 50,
) -> list[dict]:
    """
    Fetch models from the HF Hub that support the free Inference API.
    Returns a list of dicts with 'id', 'likes', 'downloads', 'pipeline_tag'.
    """
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    params = {
        "filter": task,
        "inference": "warm",   # only models with a warm/free inference endpoint
        "sort": "likes",
        "direction": "-1",
        "limit": limit,
    }

    try:
        response = requests.get(
            "https://huggingface.co/api/models",
            headers=headers,
            params=params,
            timeout=15,
        )
        response.raise_for_status()
        models = response.json()
        return [
            {
                "id": m.get("modelId") or m.get("id", ""),
                "likes": m.get("likes", 0),
                "downloads": m.get("downloads", 0),
                "pipeline_tag": m.get("pipeline_tag", task),
            }
            for m in models
            if (m.get("modelId") or m.get("id"))
        ]
    except Exception as exc:  # noqa: BLE001
        return [{"id": mid, "likes": 0, "downloads": 0, "pipeline_tag": task} for mid in RECOMMENDED_FREE_MODELS]


# Models confirmed free & accessible without any license acceptance
RECOMMENDED_FREE_MODELS = [
    "HuggingFaceH4/zephyr-7b-beta",
    "HuggingFaceH4/zephyr-7b-alpha",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "tiiuae/falcon-7b-instruct",
    "bigscience/bloom-560m",
]

# Models that require accepting a license gate on huggingface.co
GATED_MODELS = {
    # Meta Llama
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    # Google Gemma
    "google/gemma-7b-it",
    "google/gemma-2b-it",
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",
    # Microsoft Phi (ALL phi models are gated)
    "microsoft/phi-4",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3-medium-4k-instruct",
    "microsoft/Phi-3-medium-128k-instruct",
    "microsoft/Phi-3.5-mini-instruct",
    # Mistral gated variants
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mistral-Large-Instruct-2407",
    # Apple
    "apple/OpenELM-3B-Instruct",
    # 01.AI
    "01-ai/Yi-34B-Chat",
}


def validate_hf_token(token: str) -> tuple[bool, str]:
    """Validate a Hugging Face token via whoami. Returns (is_valid, message)."""
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        user_info = api.whoami()
        username = user_info.get("name", "Unknown")
        return True, f"✅ Authenticated as **{username}**"
    except Exception as exc:
        return False, f"❌ Authentication failed: {exc}"


def check_model_access(token: str, model: str) -> tuple[bool, str]:
    """
    Verify whether the token can actually access the model.
    Uses model_info gated flag + a live HEAD request as fallback.
    Returns (accessible, message).
    """
    # Fast path: known gated list
    if model in GATED_MODELS:
        # Still try to confirm via API in case user accepted the license
        pass

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        info = api.model_info(model, token=token)
        gated = getattr(info, "gated", False)
        if gated:
            return False, (
                f"🔒 **`{model}`** is a **gated model**.\n\n"
                f"Accept the license at https://huggingface.co/{model} "
                f"while logged in, then re-validate your token."
            )
        return True, f"✅ Model `{model}` is accessible."
    except Exception as exc:
        err = str(exc)
        if "403" in err or "gated" in err.lower():
            return False, (
                f"🔒 **`{model}`** is gated — license not accepted.\n\n"
                f"Visit https://huggingface.co/{model} and click **Agree and access repository**."
            )
        if "404" in err:
            hint = (
                f"\n\n> 💡 `{model}` is in the known gated list. "
                f"Visit https://huggingface.co/{model} and accept the license."
            ) if model in GATED_MODELS else (
                f"\n\n> 💡 Double-check the model ID at https://huggingface.co/models"
            )
            return False, f"❌ Model `{model}` not found or not accessible (404).{hint}"
        return False, f"❌ Could not access model `{model}`: {exc}"


def _build_messages(
    prompt: str,
    system_prompt: str = "",
) -> list[dict[str, str]]:
    """Build a well-formed messages list for chat completions."""
    messages: list[dict[str, str]] = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": prompt.strip()})
    return messages


def _make_client(token: str, model: str, provider: Optional[str]) -> tuple[InferenceClient, str]:
    """
    Always resolve to a concrete provider — never 'auto'.
    Default: 'hf-inference' (avoids Cerebras / limited providers).
    """
    resolved = provider if (provider and provider != "auto") else "hf-inference"
    init_params = inspect.signature(InferenceClient.__init__).parameters
    if "provider" in init_params:
        client = InferenceClient(model=model, token=token, provider=resolved)
    else:
        client = InferenceClient(model=model, token=token)
    return client, resolved


def stream_chat_completion(
    token: str,
    model: str,
    prompt: str,
    system_prompt: str = "",
    max_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9,
    provider: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    Stream chat completion tokens from a Hugging Face hosted model.
    Runs a pre-flight access check and raises a clear error for gated / 404 models.
    """
    # ── Pre-flight: check model accessibility ─────────────────────────────────
    accessible, access_msg = check_model_access(token, model)
    if not accessible:
        raise RuntimeError(access_msg)

    client, resolved_provider = _make_client(token, model, provider)
    messages = _build_messages(prompt, system_prompt)

    # Build kwargs
    chat_kwargs: dict[str, Any] = dict(
        messages=messages,
        max_tokens=max_tokens,
        stream=True,
        top_p=top_p,
    )
    if temperature > 0.0:
        chat_kwargs["temperature"] = temperature

    # If provider was NOT injected at client level (older lib), pass per-call
    chat_sig  = inspect.signature(client.chat_completion).parameters
    init_sig  = inspect.signature(InferenceClient.__init__).parameters
    if "provider" in chat_sig and "provider" not in init_sig:
        chat_kwargs["provider"] = resolved_provider

    # ── Attempt chat_completion ───────────────────────────────────────────────
    try:
        stream = client.chat_completion(**chat_kwargs)
        for chunk in stream:
            delta = getattr(getattr(chunk, "choices", [None])[0], "delta", None)
            if delta:
                content = getattr(delta, "content", None)
                if content:
                    yield content
        return
    except Exception as chat_exc:
        chat_err = str(chat_exc)
        # 404 after access check passed → surface immediately, no fallback
        if "404" in chat_err:
            raise RuntimeError(
                f"❌ **404 Not Found** — `{model}` via `{resolved_provider}`.\n\n"
                f"The model exists but the **hf-inference endpoint is not available** for it.\n\n"
                f"**Try one of these confirmed-free models:**\n"
                + "\n".join(f"- `{m}`" for m in RECOMMENDED_FREE_MODELS)
            ) from chat_exc

    # ── Fallback: text_generation ─────────────────────────────────────────────
    full_prompt = (f"{system_prompt.strip()}\n\n" if system_prompt.strip() else "") + prompt.strip()
    gen_kwargs: dict[str, Any] = dict(
        prompt=full_prompt,
        max_new_tokens=max_tokens,
        stream=True,
    )
    if temperature > 0.0:
        gen_kwargs["temperature"] = temperature

    gen_sig = inspect.signature(client.text_generation).parameters
    if "provider" in gen_sig and "provider" not in init_sig:
        gen_kwargs["provider"] = resolved_provider

    try:
        for chunk in client.text_generation(**gen_kwargs):
            if isinstance(chunk, str):
                yield chunk
            else:
                text = (
                    getattr(chunk, "token", None)
                    or getattr(chunk, "generated_text", None)
                    or getattr(chunk, "text", None)
                )
                if isinstance(text, str):
                    yield text
        return
    except Exception as gen_exc:
        raise RuntimeError(
            f"Both endpoints failed for `{model}` (provider=`{resolved_provider}`).\n\n"
            f"chat_completion: {chat_err}\n"
            f"text_generation: {gen_exc}\n\n"
            f"**Confirmed free models to try:**\n"
            + "\n".join(f"- `{m}`" for m in RECOMMENDED_FREE_MODELS)
        ) from gen_exc
