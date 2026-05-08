"""
Утилиты для работы с ответами LLM: парсинг JSON и обработка плейсхолдеров.
"""
import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


def _parse_llm_json(raw: str, fallback: Optional[dict] = None) -> dict:
    """Безопасный парсинг JSON из ответа LLM (убирает markdown-обёртку)."""
    cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Не удалось распарсить JSON от LLM: %s", cleaned[:200])
        return fallback or {}


def _extract_placeholders(markdown: str) -> list[str]:
    """
    Находит все незаполненные поля вида {{field_name}}.

    Паттерн `[\\w\\s]+` матчит любые слова и пробелы внутри двойных фигурных скобок.
    """
    return re.findall(r"\{\{([\w\s]+)\}\}", markdown)


def _format_clarification_for_history(questions: list[str], context: str = "анализа") -> str:
    """Форматирует уточняющие вопросы для сохранения в историю."""
    numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    return f"Для полноценного {context} уточните, пожалуйста:\n{numbered}"
