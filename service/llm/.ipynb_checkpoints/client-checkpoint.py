import asyncio
import logging
import httpx
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"


class LLMClient:
    """
    Асинхронный клиент для DeepSeek API.
    Поддерживает повторные попытки (до 3) с экспоненциальной задержкой.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    async def complete(
        self,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float = 0.3,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        client = await self._get_client()
        last_exc: Exception | None = None

        for attempt in range(3):
            try:
                resp = await client.post(DEEPSEEK_API_URL, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                last_exc = exc
                wait = 2 ** attempt
                logger.warning(
                    "LLM запрос не удался (попытка %d/3): %s — повтор через %ds",
                    attempt + 1, exc, wait,
                )
                await asyncio.sleep(wait)

        raise RuntimeError(f"LLM запрос провалился после 3 попыток: {last_exc}") from last_exc

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float = 0.3,
    ) -> Dict[str, any]:
        """
        Удобный метод, имитирующий интерфейс OpenAI.
        Принимает список сообщений с ролями "system" / "user" / "assistant",
        возвращает словарь с ключом "choices".
        """
        # Извлекаем system-сообщение (если есть) и последнее user-сообщение
        system_prompt = ""
        user_prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                user_prompt = msg["content"]  # берём последнее (перезапишется, но подойдёт для простоты)
        # Если нужна полная поддержка истории – лучше использовать другой метод,
        # но для текущей задачи простого RAG-запроса достаточно.
        content = await self.complete(
            system=system_prompt,
            user=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return {"choices": [{"message": {"content": content}}]}

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()