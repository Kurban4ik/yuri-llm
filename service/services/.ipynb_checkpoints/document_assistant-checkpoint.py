# services/document_assistant.py
"""
Полный пайплайн генерации юридических документов.
"""
import json
import logging
import re
import time
from textwrap import dedent
from typing import Any, Optional

from llm.client import LLMClient
from rag.retriever import Retriever
from schemas import (
    ChatMessage,
    InteractRequest,
    InteractResponse,
    ClarificationPayload,
    DocumentGenerationPayload,
    AtomSource,
)
from utils.llm_utils import _parse_llm_json, _extract_placeholders
from utils.history_utils import _extract_user_provided_facts
from utils.context import llm_step

logger = logging.getLogger(__name__)


class DocumentAssistant:
    """Полный пайплайн генерации документов с короткими промптами."""
    
    # --- Компактные системные промпты ---
    
    TYPE_SYSTEM = "Помощник-классификатор. Ответь одной строкой."
    
    STRUCTURE_SYSTEM = """Эксперт по юридическим документам. Верни строго JSON.
Не включай персональные данные (ФИО, паспорт, адрес, телефон, email, ИНН, СНИЛС) в required_fields."""
    
    EXTRACT_SYSTEM = """Точный экстрактор данных из диалога.
Ищи по смыслу, не по точным словам. "Нет", "не предусмотрен" — это ЗАПОЛНЕННЫЕ поля. 
Строгий JSON."""
    
    SUFFICIENCY_SYSTEM = """Вежливый ассистент для уточняющих вопросов.
Не переспрашивай о сообщённом. ⚠️ ЗАПРЕЩЕНО запрашивать персональные данные (ФИО, паспорт, адрес, телефон, email, ИНН, СНИЛС). 
Если не хватает только таких полей → ГОТОВО. Вопросы max 12 слов."""
    
    FORM_SYSTEM = "Эксперт по юридическому делопроизводству. Ответь ровно одним словом из списка."
    
    MARKDOWN_SYSTEM = "Старший юрист-делопроизводитель. Генерируй только контент документа в Markdown."
    
    def __init__(self, llm_client: LLMClient, retriever: Retriever):
        self.llm = llm_client
        self.retriever = retriever
    
    # --- Приватные методы: каждый шаг пайплайна ---
    
    async def _determine_type(
        self,
        query: str,
        history: list[ChatMessage],
    ) -> str:
        """Шаг 1: Определение типа документа."""
        history_block = (
            "\n".join(f"{m.role}: {m.content}" for m in history[-6:])
            if history else ""
        )
        
        prompt = (
            "Пользователь хочет составить юридический документ.\n\n"
            f"История:\n{history_block}\n\n"
            f"Запрос:\n{query}\n\n"
            "Определи точный тип документа (кратко, одной фразой). "
            "Например: 'исковое заявление о взыскании задолженности' или 'апелляционная жалоба'. "
            "Если тип уже упоминался — используй его."
        )
        
        logger.debug("DocumentAssistant: шаг 1 — определение типа")
        t0 = time.monotonic()
        response = await self.llm.complete(
            system=self.TYPE_SYSTEM,
            user=prompt,
            max_tokens=100,
            temperature=0.0,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        doc_type = response.strip()
        
        logger.info(
            "DocumentAssistant [1/8] тип определён",
            extra={"doc_type": doc_type, "latency_ms": latency_ms},
        )
        return doc_type
    
    async def _generate_structure(
        self,
        doc_type: str,
        query: str,
        history: list[ChatMessage],
    ) -> dict:
        """Шаг 2: Структура документа."""
        prompt = (
            f"Тип: {doc_type}\n"
            "Составь JSON:\n"
            '- "document_title": полное название,\n'
            '- "sections": массив названий разделов (≥3),\n'
            '- "required_fields": необходимые юридические/фактические данные. '
            "НЕ ВКЛЮЧАЙ персональные (ФИО, паспорт, адрес, телефон, email, ИНН, СНИЛС).\n"
            "Только JSON, без пояснений."
        )
        
        logger.debug("DocumentAssistant: шаг 2 — структура")
        t0 = time.monotonic()
        response = await self.llm.complete(
            system=self.STRUCTURE_SYSTEM,
            user=prompt,
            max_tokens=500,
            temperature=0.0,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        
        structure = _parse_llm_json(response, fallback={
            "document_title": doc_type,
            "sections": ["Шапка", "Описательная часть", "Просительная часть", "Приложения"],
            "required_fields": ["Истец", "Ответчик", "Сумма", "Основание требования"],
        })
        
        logger.info(
            "DocumentAssistant [2/8] структура",
            extra={
                "sections": len(structure.get("sections", [])),
                "fields": len(structure.get("required_fields", [])),
                "latency_ms": latency_ms,
            },
        )
        return structure
    
    async def _extract_data(
        self,
        current_message: str,
        history: list[ChatMessage],
        required_fields: list[str],
    ) -> tuple[dict, list[str]]:
        """Шаг 3: Извлечение данных из диалога."""
        fields_list = "\n".join(f"- {f}" for f in required_fields)
        history_text = (
            "\n".join(f"{m.role}: {m.content}" for m in history)
            if history else "(история пуста)"
        )
        
        prompt = (
            f"Поля для документа:\n{fields_list}\n\n"
            f"История:\n{history_text}\n\n"
            f"Текущее сообщение: {current_message}\n\n"
            "Правила:\n"
            "1. Ищи по СМЫСЛУ, не по точным словам.\n"
            "2. «Нет», «не предусмотрен», «не требуется» → заполненное поле (значение: 'не предусмотрен').\n"
            "3. Игнорируй несвязанные темы.\n"
            "4. Ключи JSON = точные имена полей.\n\n"
            'Верни JSON: {"data": {...}, "missing": [...]}. Только JSON.'
        )
        
        logger.debug("DocumentAssistant: шаг 3 — извлечение данных")
        t0 = time.monotonic()
        response = await self.llm.complete(
            system=self.EXTRACT_SYSTEM,
            user=prompt,
            max_tokens=600,
            temperature=0.0,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        
        data = _parse_llm_json(response, fallback={"data": {}, "missing": required_fields})
        collected = data.get("data", {})
        missing = [f for f in required_fields if not collected.get(f)]
        
        logger.info(
            "DocumentAssistant [3/8] данные извлечены",
            extra={
                "total_fields": len(required_fields),
                "collected": len(collected),
                "missing": len(missing),
                "latency_ms": latency_ms,
            },
        )
        return collected, missing
    
    async def _check_sufficiency(
        self,
        required_fields: list[str],
        collected_data: dict,
        user_context: Optional[list[str]] = None,
    ) -> tuple[bool, list[str]]:
        """Шаг 4: Проверка достаточности данных."""
        missing = [f for f in required_fields if not collected_data.get(f)]
        if not missing:
            logger.debug("DocumentAssistant: все поля заполнены")
            return True, []
        
        fields_str = "\n".join(f"- {f}" for f in missing)
        
        user_context_block = ""
        if user_context:
            items = "\n".join(f"  • {v}" for v in user_context)
            user_context_block = (
                "\n\nДАННЫЕ, УЖЕ СООБЩЁННЫЕ ПОЛЬЗОВАТЕЛЕМ:\n"
                f"{items}\n"
                "Если поле покрыто этими данными — не задавай вопрос.\n"
            )
        
        
        prompt = (
            "Формально отсутствуют данные по полям:\n"
            f"{fields_str}\n"
            f"{user_context_block}"
            "Задача: вопросы ТОЛЬКО по тем полям, данные для которых НЕ были предоставлены.\n"
            "⚠️ НЕ ЗАПРАШИВАЙ персональные (ФИО, паспорт, адрес, телефон, ИНН).\n"
            "Если все поля покрыты или остались только персональные → ответь: ГОТОВО\n"
            "Иначе — краткие вопросы (один на строку, без нумерации)."
        )
        
        logger.debug("DocumentAssistant: шаг 4 — проверка достаточности")
        t0 = time.monotonic()
        response = await self.llm.complete(
            system=self.SUFFICIENCY_SYSTEM,
            user=prompt,
            max_tokens=300,
            temperature=0.3,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        stripped = response.strip()
        
        if stripped.upper().startswith("ГОТОВО") or stripped == "":
            logger.info(
                "DocumentAssistant [4/8] достаточно данных",
                extra={"latency_ms": latency_ms},
            )
            return True, []
        
        questions = [q.strip() for q in stripped.split("\n") if q.strip()]
        if not questions:
            return True, []
        
        logger.info(
            "DocumentAssistant [4/8] нужны уточнения",
            extra={"questions_count": len(questions), "latency_ms": latency_ms},
        )
        return False, questions
    
    async def _determine_form(
        self,
        doc_type: str,
        collected_data: dict,
        attached_docs_count: int,
    ) -> str:
        """Шаг 5: Определение формы оформления."""
        prompt = dedent(f"""
            Тип: {doc_type}
            Данные: {', '.join(collected_data.keys())}
            Прикреплено: {attached_docs_count} шт.
            
            Выбери ОДНУ форму:
            - court_standard_form (суды общей юрисдикции)
            - arbitration_template (арбитраж)
            - notarial_template (нотариус/доверенности)
            - free_form (претензии, письма, жалобы)
            - custom (нестандартная)
            
            Ответь СТРОГО одним значением без кавычек.
        """)
        
        logger.debug("DocumentAssistant: шаг 5 — форма оформления")
        t0 = time.monotonic()
        response = await self.llm.complete(
            system=self.FORM_SYSTEM,
            user=prompt,
            max_tokens=50,
            temperature=0.0,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        
        form = response.strip().lower()
        allowed = {"court_standard_form", "arbitration_template", "notarial_template", "free_form", "custom"}
        if form not in allowed:
            logger.warning(f"Неизвестная форма {form}, используем free_form")
            form = "free_form"
        
        logger.info(
            "DocumentAssistant [5/8] форма определена",
            extra={"doc_form": form, "latency_ms": latency_ms},
        )
        return form
    
    async def _generate_markdown(
        self,
        structure: dict,
        collected_data: dict,
        doc_form: str,
        rag_atoms: list[dict],
    ) -> str:
        """Шаг 7: Генерация документа в Markdown."""
        title = structure.get("document_title", "Документ")
        sections = structure.get("sections", [])
        
        fields_str = "\n".join(
            f"- {k}: {v or '{' + k + '}'}" for k, v in collected_data.items()
        )
        sections_str = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sections))
        
        norms_block = ""
        if rag_atoms:
            norms_items = []
            for a in rag_atoms[:5]:
                meta = a["metadata"]
                norms_items.append(
                    f"- {meta.get('act_name', '')}, "
                    f"ст. {meta.get('article_number', '')}: "
                    f"{meta.get('clause_text', '')[:200]}"
                )
            norms_block = "**Применимые нормы:**\n" + "\n".join(norms_items)
        
        prompt = dedent(f"""
            Создай полный текст документа в Markdown.
            Заголовок: # {title}
            Форма: {doc_form}
            Разделы:
            {sections_str}
            
            Данные:
            {fields_str}
            
            {norms_block}
            
            ⚠️ ПРАВИЛА:
            1. Только Markdown, никакого HTML.
            2. Заголовки: ##, подразделы: ###.
            3. Ключевые данные: **жирным**.
            4. Отсутствующие данные: {{{{Название поля}}}}.
            5. Персональные → плейсхолдеры (пользователь заполнит).
            6. Списки приложений: `- [ ] `.
            7. Юридически грамотный, сухой, структурированный текст.
            8. Без вводных фраз. Начинай с #.
            9. Без markdown-таблиц, если это не список сравнения.
            10. Ссылки на нормы в тексте (например, «согласно ст. 15 ГК РФ»).
            11. Краткость: только необходимые формулировки.
        """)
        
        logger.debug("DocumentAssistant: шаг 7 — генерация Markdown")
        t0 = time.monotonic()
        response = await self.llm.complete(
            system=self.MARKDOWN_SYSTEM,
            user=prompt,
            max_tokens=4000,
            temperature=0.4,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        result = response.strip()
        
        logger.info(
            "DocumentAssistant [7/8] документ сгенерирован",
            extra={"markdown_len": len(result), "latency_ms": latency_ms},
        )
        return result
    
    # --- Публичный метод: точка входа ---
    
    async def handle(
        self,
        request: InteractRequest,
    ) -> InteractResponse:
        """
        Точка входа: полный пайплайн генерации документа.
        Возвращает либо clarification, либо document_generated.
        """
        ctx = {
            "session_id": request.session_id,
            "history_len": len(request.chat_history or []),
        }
        pipeline_start = time.monotonic()
        
        logger.info("DocumentAssistant.handle: старт", extra=ctx)
        
        try:
            # Шаг 1: Определение типа
            t0 = time.monotonic()
            doc_type = await self._determine_type(
                query=request.message,
                history=request.chat_history,
            )
            
            # Шаг 2: Структура
            t0 = time.monotonic()
            with llm_step("generate_document_structure"):
                structure = await self._generate_structure(
                    doc_type=doc_type,
                    query=request.message,
                    history=request.chat_history,
                )
            
            required_fields = structure.get("required_fields", [])
            
            # Шаг 3: Сбор данных
            t0 = time.monotonic()
            with llm_step("extract_collected_data"):
                collected_data, missing = await self._extract_data(
                    current_message=request.message,
                    history=request.chat_history,
                    required_fields=required_fields,
                )
            
            # Шаг 4: Проверка достаточности
            user_context = _extract_user_provided_facts(request.chat_history)
            t0 = time.monotonic()
            with llm_step("check_document_data_sufficiency"):
                sufficient, questions = await self._check_sufficiency(
                    required_fields=required_fields,
                    collected_data=collected_data,
                    user_context=user_context,
                )
            
            if not sufficient:
                logger.info(
                    "DocumentAssistant.handle: нужны уточнения",
                    extra={**ctx, "questions": questions},
                )
                return InteractResponse(
                    session_id=request.session_id,
                    type="clarification",
                    clarification=ClarificationPayload(questions=questions),
                )
            
            # Шаг 5: Форма оформления
            t0 = time.monotonic()
            with llm_step("determine_document_form"):
                doc_form = await self._determine_form(
                    doc_type=doc_type,
                    collected_data=collected_data,
                    attached_docs_count=len(request.attached_documents or []),
                )
            
            # Шаг 6: RAG retrieval
            t0 = time.monotonic()
            rag_query = f"{doc_type} {request.message}"
            raw_atoms = await self.retriever.retrieve(rag_query)
            
            logger.info(
                "DocumentAssistant [6/8] RAG retrieval",
                extra={**ctx, "atoms_count": len(raw_atoms)},
            )
            
            # Шаг 7: Генерация Markdown
            t0 = time.monotonic()
            with llm_step("generate_document_markdown"):
                markdown = await self._generate_markdown(
                    structure=structure,
                    collected_data=collected_data,
                    doc_form=doc_form,
                    rag_atoms=raw_atoms,
                )
            
            # Шаг 8: Финализация
            placeholders = _extract_placeholders(markdown)
            warnings = [
                f"Поле «{p}» не было найдено в диалоге и оставлено шаблоном. Проверьте перед печатью."
                for p in placeholders
            ]
            safe_name = re.sub(r"[^\w\s-]", "", doc_type).lower().replace(" ", "_")[:50]
            
            total_ms = int((time.monotonic() - pipeline_start) * 1000)
            logger.info(
                "DocumentAssistant [8/8] готово",
                extra={
                    **ctx,
                    "doc_type": doc_type,
                    "doc_form": doc_form,
                    "sections": len(structure.get("sections", [])),
                    "filled_fields": len(collected_data),
                    "total_fields": len(required_fields),
                    "placeholders": len(placeholders),
                    "total_latency_ms": total_ms,
                },
            )
            
            return InteractResponse(
                session_id=request.session_id,
                type="document_generated",
                document=DocumentGenerationPayload(
                    markdown_content=markdown,
                    document_title=structure["document_title"],
                    suggested_filename=f"{safe_name}_{request.session_id[:8]}.md",
                    structure_summary={
                        "sections": structure.get("sections", []),
                        "filled_fields_count": len(collected_data),
                        "total_fields": len(required_fields),
                    },
                    warnings=warnings,
                ),
            )
        
        except Exception as e:
            total_ms = int((time.monotonic() - pipeline_start) * 1000)
            logger.error(
                "DocumentAssistant.handle: ошибка",
                extra={**ctx, "error": str(e), "total_latency_ms": total_ms},
                exc_info=True,
            )
            raise
