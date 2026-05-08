# main_refactored.py
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException

from config import settings
from logging_config import setup_logging
from llm.client import LLMClient
from utils.llm_logger import LoggingLLMClient
from rag.store import AtomStore
from rag.retriever import Retriever
from schemas import (
    RagRequest, RagResponse, RagAtom,
    IndexRequest,
    InteractResponse,
    InteractRequest,
)
from services.interact_service_refactored import handle_interact
from middleware.request_logging import RequestLoggingMiddleware

# Настройка логирования
setup_logging()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Глобальные синглтоны сервисов
# ---------------------------------------------------------------------------
_store: Optional[AtomStore] = None
_retriever: Optional[Retriever] = None
_llm: Optional[LoggingLLMClient] = None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _store, _retriever, _llm

    logger.info("Запуск inference service…")

    logger.info("Инициализация LLMClient")
    _llm = LoggingLLMClient(LLMClient(api_key=settings.deepseek_api_key))

    logger.info(
        "Инициализация AtomStore",
        extra={"db_path": settings.db_path, "embed_model": settings.embed_model},
    )
    _store = AtomStore(
        db_path=settings.db_path,
        embed_model=settings.embed_model,
        embed_device=settings.embed_device,
    )
    await _store.load_model()
    logger.info("AtomStore готов", extra={"atoms_count": _store.count()})

    logger.info("Инициализация Retriever + reranker")
    _retriever = Retriever(store=_store, config=settings)
    await _retriever.load_reranker()

    if os.path.exists(settings.atoms_jsonl) and _store.count() == 0:
        logger.info("Коллекция пуста — индексируем", extra={"path": settings.atoms_jsonl})
        n = _store.index_atoms(settings.atoms_jsonl)
        logger.info("Индексирование завершено", extra={"indexed": n})

    logger.info("✓ Сервис готов", extra={"atoms_count": _store.count()})
    yield

    logger.info("Остановка сервиса")
    await _llm.close()
    logger.info("Сервис остановлен")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Legal Inference Service",
    description="RAG-based legal analysis & document generation with DeepSeek",
    version="5.0.0",  # Обновлённая версия с рефакторингом
    lifespan=lifespan,
)

app.add_middleware(RequestLoggingMiddleware)


# ---------------------------------------------------------------------------
# Эндпоинты
# ---------------------------------------------------------------------------

@app.post("/generate-rag", response_model=RagResponse)
async def handle_generate_rag(request: RagRequest) -> RagResponse:
    if _retriever is None:
        raise HTTPException(status_code=503, detail="RAG retriever не инициализирован")

    logger.info("RAG retrieval requested", extra={"query_preview": request.query[:120]})
    raw_atoms: list[dict] = await _retriever.retrieve(request.query)

    atoms = [
        RagAtom(
            atom_id=str(a["metadata"]["id"]),
            act_name=a["metadata"].get("act_name", ""),
            article=str(a["metadata"].get("article_number", "")),
            clause_text=a["metadata"].get("clause_text", ""),
            article_full_text=a["metadata"].get("article_full_text", ""),
            relevance_score=a.get("relevance_score", 0.0),
        )
        for a in raw_atoms
    ]

    logger.info("RAG retrieval done", extra={"atoms_returned": len(atoms)})
    return RagResponse(atoms=atoms)


@app.post("/interact", response_model=InteractResponse)
async def handle_interact_endpoint(request: InteractRequest) -> InteractResponse:
    """
    Обработка /interact с использованием новых сервисных классов.
    """
    print('\n'*10)
    print(request)
    print('\n'*10)
    if _llm is None or _retriever is None:
        raise HTTPException(status_code=503, detail="Сервис не инициализирован")
    
    # Используем рефакторенный handle_interact с классами-сервисами
    return await handle_interact(request, llm_client=_llm, retriever=_retriever)


@app.post("/index")
async def handle_index(request: IndexRequest) -> dict:
    if _store is None:
        raise HTTPException(status_code=503, detail="Store не инициализирован")
    logger.info("Index requested", extra={"path": request.atoms_jsonl_path})
    n = _store.index_atoms(request.atoms_jsonl_path)
    logger.info("Index completed", extra={"indexed": n})
    return {"indexed": n}


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "atoms_count": _store.count() if _store else 0,
    }
