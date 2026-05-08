# utils/context.py
"""
Контекстные переменные для сквозной трассировки запросов.

Устанавливаются ОДИН РАЗ при входе в запрос и автоматически
наследуются во всех async-вызовах (asyncio propagates ContextVar).

Использование:
    from utils.context import session_id_var, request_id_var, llm_step

    # В точке входа в запрос:
    session_id_var.set(request.session_id)
    request_id_var.set(some_uuid)

    # Для пометки текущего шага (опционально):
    with llm_step("rag_query_build"):
        response = await llm.complete(...)
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Generator

# Сессия пользователя (из тела запроса)
session_id_var: ContextVar[str] = ContextVar("session_id", default="—")

# UUID сквозного запроса (из middleware)
request_id_var: ContextVar[str] = ContextVar("request_id", default="—")

# Имя текущего шага pipeline (для корреляции промптов)
step_name_var: ContextVar[str] = ContextVar("step_name", default="—")


@contextmanager
def llm_step(name: str) -> Generator[None, None, None]:
    """
    Контекстный менеджер: помечает все LLM-вызовы внутри блока именем шага.

    Пример:
        with llm_step("check_query_sufficiency"):
            result = await llm.complete(system=..., user=...)
        # В логах: step = "check_query_sufficiency"
    """
    token = step_name_var.set(name)
    try:
        yield
    finally:
        step_name_var.reset(token)
