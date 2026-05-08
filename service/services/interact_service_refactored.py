# services/interact_service_refactored.py
"""
Сервис обработки /interact: маршрутизация с использованием классов-сервисов.
Каждый этап инкапсулирован в отдельный класс.
"""
import logging
import time

from llm.client import LLMClient
from rag.retriever import Retriever
from schemas import (
    ChatMessage,
    InteractRequest,
    InteractResponse,
    ClarificationPayload,
    AnswerPayload,
    AtomSource,
)
from services.sufficiency_checker import SufficiencyChecker
from services.rag_query_builder import RAGQueryBuilder
from services.contract_analyzer import ContractAnalyzer
from services.document_assistant import DocumentAssistant
from utils.history_utils import _extract_user_provided_facts
from utils.context import llm_step

logger = logging.getLogger(__name__)


def _session_ctx(request: InteractRequest) -> dict:
    """Словарь общих extra-полей сессии."""
    return {
        "session_id": request.session_id,
        "interact_type": getattr(request, "interact_type", "—"),
        "history_len": len(request.chat_history or []),
        "has_docs": bool(request.attached_documents),
    }


# --- Построение промпта для general-запросов ---

def build_general_prompt(
    query: str,
    rag_atoms: list[dict],
    chat_history: list[ChatMessage],
) -> tuple[str, str]:
    """Строит промпт для общих запросов."""
    # Системный промпт
    system_prompt = """Юридический ассистент. Анализируй по российскому законодательству.
Используй предоставленные нормы. Отвечай конкретно, структурированно, без лишних слов."""
    
    # История диалога
    history_block = ""
    if chat_history:
        history_block = "История диалога:\n" + "\n".join(
            f"{m.role}: {m.content}" for m in chat_history[-5:]
        ) + "\n\n"
    
    # Нормы из RAG
    norms_block = ""
    if rag_atoms:
        norms_items = []
        for a in rag_atoms:
            meta = a["metadata"]
            norms_items.append(
                f"[{meta.get('act_name', '')}, ст.{meta.get('article_number', '')}] "
                f"{meta.get('article_full_text') or meta.get('clause_text', '')}"
            )
        norms_block = "НОРМЫ ПРАВА:\n" + "\n\n".join(norms_items) + "\n\n"
    
    user_prompt = f"{history_block}{norms_block}ЗАПРОС: {query}"
    
    return system_prompt, user_prompt


# --- Главный обработчик ---

async def handle_interact(
    request: InteractRequest,
    llm_client: LLMClient,
    retriever: Retriever,
) -> InteractResponse:
    """
    Основная точка входа: маршрутизация между генерацией документа и анализом.
    Теперь использует классы-сервисы вместо функций.
    """
    ctx = _session_ctx(request)
    pipeline_start = time.monotonic()
    
    logger.info(
        "handle_interact: вход",
        extra={**ctx, "msg_preview": (request.message or "")[:100]},
    )
    
    try:
        # --- Генерация документа ---
        if request.interact_type == "generate_doc":
            logger.info("handle_interact: маршрут → generate_doc", extra=ctx)
            
            # Используем DocumentAssistant — весь пайплайн инкапсулирован
            doc_assistant = DocumentAssistant(llm_client, retriever)
            return await doc_assistant.handle(request)
        
        # --- Анализ (contract или general) ---
        
        document_text = ""
        if request.attached_documents:
            document_text = request.attached_documents[0]
            logger.debug(
                "handle_interact: прикреплён документ",
                extra={**ctx, "doc_len": len(document_text)},
            )
        
        # Проверка достаточности данных
        collected_variables = _extract_user_provided_facts(request.chat_history)
        
        sufficiency_checker = SufficiencyChecker(llm_client)
        t0 = time.monotonic()
        sufficient, questions = await sufficiency_checker.check(
            query=request.message,
            document_text=document_text[:800],
            chat_history=request.chat_history,
            collected_variables=collected_variables,
        )
        logger.info(
            "handle_interact: sufficiency check",
            extra={
                **ctx,
                "sufficient": sufficient,
                "questions_count": len(questions),
                "questions": questions,
                "latency_ms": int((time.monotonic() - t0) * 1000),
            },
        )
        
        if not sufficient:
            logger.info(
                "handle_interact: возвращаем clarification",
                extra={**ctx, "questions": questions},
            )
            return InteractResponse(
                session_id=request.session_id,
                type="clarification",
                clarification=ClarificationPayload(questions=questions),
            )
        
        # RAG retrieval
        t0 = time.monotonic()
        raw_atoms = await _retrieve_atoms_for_analyze(
            query=request.message,
            document_text=document_text,
            chat_history=request.chat_history,
            retriever=retriever,
            llm_client=llm_client,
            query_type=request.interact_type,
        )
        logger.info(
            "handle_interact: RAG retrieval",
            extra={
                **ctx,
                "atoms_count": len(raw_atoms),
                "latency_ms": int((time.monotonic() - t0) * 1000),
            },
        )
        
        # Построение промпта
        t0 = time.monotonic()
        if request.interact_type == "contract":
            # Используем ContractAnalyzer и RAGQueryBuilder
            query_builder = RAGQueryBuilder(llm_client)
            with llm_step("build_contract_questions"):
                questions = await query_builder.build_contract_questions(
                    document_text=document_text,
                    query=request.message,
                    chat_history=request.chat_history,
                )
            
            analyzer = ContractAnalyzer(llm_client)
            sys_p, usr_p = await analyzer.build_prompt(
                document_text=document_text,
                rag_atoms=raw_atoms,
                questions=questions,
            )
        else:
            sys_p, usr_p = build_general_prompt(
                query=request.message,
                rag_atoms=raw_atoms,
                chat_history=request.chat_history,
            )
        
        logger.debug(
            "handle_interact: промпт построен",
            extra={
                **ctx,
                "sys_prompt_len": len(sys_p),
                "usr_prompt_len": len(usr_p),
                "latency_ms": int((time.monotonic() - t0) * 1000),
            },
        )
        
        # LLM финальный ответ
        t0 = time.monotonic()
        with llm_step("final_answer"):
            answer_text = await llm_client.complete(
                system=sys_p,
                user=usr_p,
                max_tokens=request.max_tokens,
            )
        llm_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "handle_interact: LLM ответ получен",
            extra={**ctx, "answer_len": len(answer_text), "latency_ms": llm_ms},
        )
        
        # Формируем источники
        sources = [
            AtomSource(
                atom_id=str(a["metadata"]["id"]),
                act_name=a["metadata"].get("act_name", ""),
                article=str(a["metadata"].get("article_number", "")),
                clause_text=a["metadata"].get("clause_text", ""),
                article_full_text=a["metadata"].get("article_full_text", ""),
                relevance_score=a.get("relevance_score", 0.0),
            )
            for a in raw_atoms
        ]
        
        total_ms = int((time.monotonic() - pipeline_start) * 1000)
        logger.info(
            "handle_interact: ответ отправлен",
            extra={
                **ctx,
                "atoms_count": len(sources),
                "answer_len": len(answer_text),
                "total_latency_ms": total_ms,
            },
        )
        
        return InteractResponse(
            session_id=request.session_id,
            type="answer",
            answer=AnswerPayload(
                answer=answer_text,
                sources=sources,
                query_type=request.interact_type,
            ),
        )
    
    except Exception as e:
        total_ms = int((time.monotonic() - pipeline_start) * 1000)
        logger.error(
            "handle_interact: необработанная ошибка",
            extra={**ctx, "error": str(e), "total_latency_ms": total_ms},
            exc_info=True,
        )
        return InteractResponse(
            session_id=request.session_id,
            type="answer",
            answer=AnswerPayload(
                answer="Произошла техническая ошибка при обработке запроса. Попробуйте переформулировать или повторите позже.",
                sources=[],
                query_type="general",
            ),
        )


# --- Вспомогательная функция для retrieval ---

async def _retrieve_atoms_for_analyze(
    query: str,
    document_text: str,
    chat_history: list[ChatMessage],
    retriever: Retriever,
    llm_client: LLMClient,
    query_type: str,
) -> list[dict]:
    """
    RAG-retrieval с учётом типа запроса.
    Для contract: генерирует вопросы и запрашивает по ним.
    Для general: использует переформулированный запрос.
    """
    query_builder = RAGQueryBuilder(llm_client)
    
    if query_type == "contract" and document_text:
        # Генерируем вопросы по договору
        questions = await query_builder.build_contract_questions(
            document_text=document_text,
            query=query,
            chat_history=chat_history,
        )
        
        # Retrieval по вопросам
        all_atoms = []
        for q in questions[:7]:
            atoms = await retriever.retrieve(q)
            all_atoms.extend(atoms)
        
        # Дедупликация
        seen_ids = set()
        unique_atoms = []
        for a in all_atoms:
            atom_id = a["metadata"]["id"]
            if atom_id not in seen_ids:
                seen_ids.add(atom_id)
                unique_atoms.append(a)
        
        return unique_atoms[:20]
    else:
        # Для general: переформулируем запрос с учётом истории
        rag_query = await query_builder.build_search_query(
            query=query,
            chat_history=chat_history,
        )
        
        return await retriever.retrieve(rag_query)
