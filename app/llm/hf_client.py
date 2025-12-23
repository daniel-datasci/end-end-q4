import os
import time
import requests
import httpx
import asyncio
from typing import Dict, Any

from app.core.config import HF_API_TOKEN

HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}

MODEL_FALLBACKS = [
    "deepseek-ai/DeepSeek-V3.2",
    "openai/gpt-oss-20b",
    "Qwen/Qwen2.5-7B-Instruct"
]

class LLMError(Exception):
    pass


async def _call_hf_router(
    model: str,
    messages: list,
    temperature: float = 0.3,
    max_tokens: int = 300,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(HF_ROUTER_URL, headers=HEADERS, json=payload)

    if response.status_code != 200:
        raise LLMError(f"HF API error {response.status_code}: {response.text}")

    data = response.json()
    return data["choices"][0]["message"]["content"]


async def explain_churn(
    features: Dict[str, Any],
    churn_score: float,
    retries: int = 2,
) -> str:
    """
    Robust async LLM explanation with retry + fallback models
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior data scientist explaining "
                "customer churn predictions to business stakeholders."
            )
        },
        {
            "role": "user",
            "content": (
                f"The churn probability is {churn_score:.2f}.\n\n"
                f"Customer features:\n{features}\n\n"
                "Explain the main drivers of churn clearly and concisely."
            )
        }
    ]

    last_error = None

    for model in MODEL_FALLBACKS:
        for attempt in range(retries):
            try:
                return await _call_hf_router(model=model, messages=messages)
            except Exception as e:
                last_error = e
                await asyncio.sleep(1)  # small backoff
                continue

    raise LLMError(f"All LLM attempts failed: {last_error}")