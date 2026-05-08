"""
query_gen.py — generate_queries(clause_text, context_str, api_key) -> list[str]

Вызывает DeepSeek API для генерации 5 практических вопросов по пункту закона.
Retry: 3 попытки с exponential backoff 1/2/4 сек.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

SYSTEM_PROMPT = (
    "Ты юридический эксперт. Сгенерируй ровно 5 разных практических вопросов, "
    "на которые отвечает следующий пункт законодательства. Вопросы должны быть "
    "такими, которые реально задают юристы или обычные люди. Отвечай только списком "
    "вопросов, по одному на строке, без нумерации и лишних символов."
    " Вопросы должны быть разными, не переформулировками друг друга."
)


def generate_queries(
    clause_text: str,
    context_str: str,
    api_key: str,
    max_retries: int = 3,
) -> list[str]:
    """
    Возвращает список из ~5 вопросов. При ошибке — [].
    """
    user_message = f"{context_str}\n\n{clause_text}"
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.7,
        "max_tokens": 512,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    delay = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                DEEPSEEK_URL,
                json=payload,
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            content: str = data["choices"][0]["message"]["content"].strip()
            questions = [q.strip() for q in content.splitlines() if q.strip()]
            return questions
        except Exception as exc:
            logger.warning(
                "DeepSeek attempt %d/%d failed: %s", attempt, max_retries, exc
            )
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2

    logger.warning("generate_queries: all retries exhausted, returning []")
    return []
