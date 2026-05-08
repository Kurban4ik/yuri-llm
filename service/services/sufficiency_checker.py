# services/sufficiency_checker.py
"""
Проверка достаточности данных для анализа.
"""
import json
import logging
import time
from typing import Optional

from llm.client import LLMClient
from schemas import ChatMessage
from utils.context import llm_step

logger = logging.getLogger(__name__)


class SufficiencyChecker:
    """Проверка достаточности данных для юридического анализа."""
    
    # Компактный системный промпт
    SYSTEM_PROMPT = """Юрист-ассистент. Достаточно данных? Верни JSON: {"sufficient": true/false, "questions": [...]}.
⚠️ НИКОГДА не запрашивай персональные данные (ФИО, паспорт, адрес, телефон, email, ИНН, СНИЛС, банковские реквизиты).
Если пользователь уже предоставил информацию — считай её собранной. Если можно обойтись без уточнения — обходись.
Вопросы max 12 слов. Без лишних слов."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    async def check(
        self,
        query: str,
        document_text: str,
        chat_history: list[ChatMessage],
        collected_variables: Optional[list[str]] = None,
    ) -> tuple[bool, list[str]]:
        """
        Проверяет достаточность данных для анализа.
        Возвращает (sufficient: bool, questions: list[str]).
        """
        recent = chat_history[-8:]
        history_block = "\n".join(f"{m.role.upper()}: {m.content}" for m in recent)
        
        known_block = ""
        if collected_variables:
            items = "\n".join(f"  - {v}" for v in collected_variables)
            known_block = (
                "УЖЕ ПРЕДОСТАВЛЕНО (не спрашивай повторно):\n" 
                f"{items}\n\n"
            )
        
        parts: list[str] = []
        if history_block:
            parts.append(f"История:\n{history_block}")
        if document_text:
            parts.append(f"Договор (первые 800):\n{document_text[:800]}")
        parts.append(f"Запрос: {query}")
        
        user_prompt = known_block + "\n\n".join(parts)
        
        logger.debug(
            "SufficiencyChecker.check: вызов LLM",
            extra={
                "query_preview": query[:80],
                "history_msgs": len(recent),
                "collected_vars": len(collected_variables or []),
            },
        )
        
        t0 = time.monotonic()
        with llm_step("check_sufficiency"):
            raw = await self.llm.complete(
                system=self.SYSTEM_PROMPT,
                user=user_prompt,
                max_tokens=400,
            )
        latency_ms = int((time.monotonic() - t0) * 1000)
        
        try:
            data = json.loads(raw)
            sufficient = bool(data.get("sufficient", True))
            questions = data.get("questions", [])
            
            logger.info(
                "SufficiencyChecker.check: результат",
                extra={
                    "sufficient": sufficient,
                    "questions_count": len(questions),
                    "questions": questions,
                    "latency_ms": latency_ms,
                },
            )
            return sufficient, questions
        except (json.JSONDecodeError, AttributeError):
            logger.warning(
                "SufficiencyChecker.check: не удалось распарсить JSON — считаем sufficient=True",
                extra={"raw_response": raw[:200], "latency_ms": latency_ms},
            )
            return True, []
