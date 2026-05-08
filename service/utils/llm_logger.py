# utils/llm_logger.py
"""
Прозрачный прокси над LLMClient, который пишет в logger «llm.calls»
КАЖДЫЙ вызов LLM: полный промпт, параметры и полный ответ.

Что попадает в лог:
  • session_id, request_id  — сквозная трассировка
  • step                    — имя шага pipeline (если установлен через llm_step())
  • method                  — «complete» или «chat_completion»
  • system_prompt / messages — полный входной контекст
  • user_prompt             — пользовательская часть (для complete)
  • temperature, max_tokens — гиперпараметры вызова
  • response                — полный ответ LLM (не обрезанный)
  • response_len            — длина ответа в символах
  • latency_ms              — время генерации

Подключение (в main.py):
    from utils.llm_logger import LoggingLLMClient
    _llm = LoggingLLMClient(LLMClient(api_key=...))
"""

import logging
import time
from typing import Any

from llm.client import LLMClient
from utils.context import request_id_var, session_id_var, step_name_var

# Отдельный логгер — можно направить в отдельный файл через logging_config
llm_call_log = logging.getLogger("llm.calls")


class LoggingLLMClient:
    """
    Обёртка над LLMClient: перехватывает complete() и chat_completion(),
    логирует полный промпт + ответ + параметры, затем проксирует вызов дальше.
    """

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    # ------------------------------------------------------------------
    # Основные методы
    # ------------------------------------------------------------------

    async def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        t0 = time.monotonic()
        response: str = await self._client.complete(
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)

        llm_call_log.info(
            "llm_call",
            extra={
                "session_id":    session_id_var.get("—"),
                "request_id":    request_id_var.get("—"),
                "step":          step_name_var.get("—"),
                "method":        "complete",
                # --- промпт ---
                "system_prompt": system,
                "user_prompt":   user,
                # --- параметры ---
                "temperature":   temperature,
                "max_tokens":    max_tokens,
                # --- ответ ---
                "response":      response,
                "response_len":  len(response),
                # --- метрика ---
                "latency_ms":    latency_ms,
            },
        )
        return response

    async def chat_completion(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs: Any,
    ) -> dict:
        t0 = time.monotonic()
        raw: dict = await self._client.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)

        # Извлекаем текст ответа из стандартной структуры OpenAI-совместимого API
        response_text = ""
        try:
            response_text = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            pass

        llm_call_log.info(
            "llm_call",
            extra={
                "session_id":   session_id_var.get("—"),
                "request_id":   request_id_var.get("—"),
                "step":         step_name_var.get("—"),
                "method":       "chat_completion",
                # --- промпт ---
                "messages":     messages,
                # --- параметры ---
                "temperature":  temperature,
                "max_tokens":   max_tokens,
                # --- ответ ---
                "response":     response_text,
                "response_len": len(response_text),
                # --- метрика ---
                "latency_ms":   latency_ms,
            },
        )
        return raw

    # ------------------------------------------------------------------
    # Служебные методы
    # ------------------------------------------------------------------

    async def close(self) -> None:
        await self._client.close()

    def __getattr__(self, name: str) -> Any:
        """Все остальные атрибуты/методы проксируются напрямую к клиенту."""
        return getattr(self._client, name)
