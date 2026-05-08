# schemas.py

from pydantic import BaseModel, Field
from typing import Optional, Literal, Union


# ===========================================================================
# 1.  Общие вспомогательные модели
# ===========================================================================

class AtomSource(BaseModel):
    """Один атом из RAG, приложенный к ответу как источник."""
    atom_id: str
    act_name: str
    article: str
    clause_text: str
    article_full_text: str = ""   # v2: полный текст статьи (small-to-big retrieval)
    relevance_score: float        # итоговый скор после всех этапов фильтрации


# ===========================================================================
# 2.  Сценарий /generate-rag — чистый RAG-поиск без LLM
# ===========================================================================

class RagRequest(BaseModel):
    """Запрос на поиск релевантных атомов по пользовательскому тексту."""
    query: str


class RagAtom(BaseModel):
    """Атом знаний, возвращаемый эндпоинтом /generate-rag."""
    atom_id: str
    act_name: str
    article: str
    clause_text: str
    article_full_text: str = ""
    relevance_score: float


class RagResponse(BaseModel):
    """Результат RAG-поиска: список найденных атомов без ответа LLM."""
    atoms: list[RagAtom]


# ===========================================================================
# 3.  Сценарий /generate-document — генерация документа (заглушка)
# ===========================================================================

class DocumentRequest(BaseModel):
    """Запрос на генерацию документа (заглушка)."""
    query: str


class DocumentResponse(BaseModel):
    """Ответ заглушки для /generate-document."""
    status: str
    message: str


# ===========================================================================
# 4.  Сценарий /analyze — полный пайплайн RAG + LLM
# ===========================================================================

class AnalyzeRequest(BaseModel):
    """Запрос на полный анализ: RAG-поиск + промпт-инжиниринг + вызов LLM."""
    query: str
    document_text: Optional[str] = None   # текст договора для двухпроходного анализа
    max_tokens: int = 2000


class AnalyzeResponse(BaseModel):
    """Итоговый ответ LLM с перечнем источников и типом запроса."""
    answer: str
    sources: list[AtomSource]
    query_type: Literal["general", "contract"]   # "contract" | "general"


# ===========================================================================
# 5.  Служебный эндпоинт /index
# ===========================================================================

class IndexRequest(BaseModel):
    """Запрос на переиндексацию базы знаний."""
    atoms_jsonl_path: str


# ===========================================================================
# 6.  Модели для диалогового /interact эндпоинта (новый универсальный API)
# ===========================================================================

class ChatMessage(BaseModel):
    """Одно сообщение в истории диалога."""
    role: Literal["user", "assistant"]
    content: str
    is_clarification: bool = False
    is_clarification_answer: bool = False
    
    variables: dict = {}  # должен содержать вытянутые сообщения


class InteractRequest(BaseModel):
    """
    Единый запрос для обработки одного хода диалога.

    Backend 1 передаёт полную историю диалога (без текущего сообщения —
    оно идёт отдельным полем `message`), чтобы Backend 2 мог учесть контекст.
    """
    session_id: str
    message: str
    chat_history: list[ChatMessage] = Field(default_factory=list)
    
    attached_documents: Optional[list[str]] = None   # список текстов прикреплённых документов
    interact_type: Literal["generate_doc", "analysis"] = "analysis"
    max_tokens: int = 2000


class ClarificationPayload(BaseModel):
    """Содержимое ответа-уточнения."""
    questions: list[str]


class DocumentGenerationPayload(BaseModel):
    """Содержимое ответа с сгенерированным документом."""
    markdown_content: str           # готовый к конвертации Markdown
    document_title: str             # название для UI
    suggested_filename: str         # например: "iskovoe_zayavlenie_dolg_20260428.md"
    structure_summary: dict         # {"sections": [...], "filled_fields": {...}}
    warnings: list[str] = Field(default_factory=list)   # незаполненные {{placeholder}}


class AnswerPayload(BaseModel):
    """Содержимое финального ответа (анализ/общий вопрос)."""
    answer: str
    sources: list[AtomSource]
    query_type: Literal["general", "contract", "document", 'analysis']


class InteractResponse(BaseModel):
    """Унифицированный ответ Backend 2 на один ход диалога."""
    session_id: str
    type: Literal["clarification", "answer", "document_generated"]
    clarification: Optional[ClarificationPayload] = None
    answer: Optional[AnswerPayload] = None
    document: Optional[DocumentGenerationPayload] = None