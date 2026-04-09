from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
        self.enabled = bool(self.api_key)
        self._client = None
        if self.enabled:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key)
            except Exception:
                self.enabled = False
                self._client = None

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> Optional[str]:
        if not self.enabled or self._client is None:
            return None
        try:
            resp = self._client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            return getattr(resp, "output_text", None)
        except Exception:
            return None
