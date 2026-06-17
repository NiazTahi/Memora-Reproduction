from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv

from memora_repro.llm.cache import DiskCache


class OpenAIClient:
    def __init__(
        self,
        cache_dir: str | Path,
        seed: int | None = 42,
        env_file: str | Path | None = ".env",
    ):
        if env_file is not None and Path(env_file).exists():
            load_dotenv(env_file)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to .env or export it in the shell."
            )
        self.client = OpenAI(api_key=api_key)
        self.cache = DiskCache(cache_dir)
        self.seed = seed

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        response_format: dict[str, str] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "response_format": response_format,
            "seed": self.seed,
        }
        cached = self.cache.get("chat", payload)
        if cached is not None:
            return str(cached["content"])

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if response_format is not None:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        self.cache.set("chat", payload, {"content": content})
        return content

    def chat_json(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        content = self.chat(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start >= 0 and end > start:
                return json.loads(content[start : end + 1])
            raise

    def embed(self, *, model: str, texts: list[str]) -> list[list[float]]:
        payload = {"model": model, "texts": texts}
        cached = self.cache.get("embeddings", payload)
        if cached is not None:
            return cached["embeddings"]

        response = self.client.embeddings.create(model=model, input=texts)
        embeddings = [item.embedding for item in response.data]
        self.cache.set("embeddings", payload, {"embeddings": embeddings})
        return embeddings
