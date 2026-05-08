# middleware/request_logging.py
"""
FastAPI-middleware: логирует каждый входящий запрос и его ответ.

Что фиксируется:
  • method, path, query_string
  • session_id, interact_type (если есть в теле)
  • длина тела запроса, размер ответа
  • HTTP-статус ответа
  • полное время обработки (latency_ms)
  • трассировочный request_id (UUID4), прокидывается в заголовок X-Request-ID ответа

Контекстные переменные (utils.context):
  • request_id_var  — устанавливается здесь для всех LLM-вызовов цепочки
  • session_id_var  — устанавливается здесь из тела запроса
"""

import json
import logging
import time
import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from utils.context import request_id_var, session_id_var

logger = logging.getLogger("api.request")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Логирует каждый HTTP-запрос: вход, выход и задержку."""

    SKIP_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())

        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        body_bytes: bytes = await request.body()
        body_meta = self._parse_body_meta(body_bytes)

        # --- Устанавливаем контекстные переменные ---
        # Они автоматически наследуются во всех async-вызовах этого запроса,
        # и LoggingLLMClient будет видеть их при каждом вызове LLM.
        rid_token = request_id_var.set(request_id)
        sid_token = session_id_var.set(body_meta.get("session_id", "—"))

        started_at = time.monotonic()

        logger.info(
            "Request received",
            extra={
                "request_id":    request_id,
                "method":        request.method,
                "path":          request.url.path,
                "session_id":    body_meta.get("session_id", "—"),
                "interact_type": body_meta.get("interact_type", "—"),
                "msg_len":       body_meta.get("msg_len", 0),
                "history_len":   body_meta.get("history_len", 0),
                "has_docs":      body_meta.get("has_docs", False),
                "body_bytes":    len(body_bytes),
            },
        )

        request.state.request_id = request_id

        try:
            response: Response = await call_next(request)
        except Exception as exc:
            elapsed_ms = int((time.monotonic() - started_at) * 1000)
            logger.error(
                "Request failed with unhandled exception",
                extra={
                    "request_id": request_id,
                    "path":       request.url.path,
                    "session_id": body_meta.get("session_id", "—"),
                    "latency_ms": elapsed_ms,
                    "exc":        str(exc),
                },
                exc_info=True,
            )
            raise
        finally:
            # Сбрасываем контекстные переменные после завершения запроса
            request_id_var.reset(rid_token)
            session_id_var.reset(sid_token)

        elapsed_ms = int((time.monotonic() - started_at) * 1000)

        log_level = logging.WARNING if response.status_code >= 400 else logging.INFO
        logger.log(
            log_level,
            "Request completed",
            extra={
                "request_id":   request_id,
                "method":       request.method,
                "path":         request.url.path,
                "session_id":   body_meta.get("session_id", "—"),
                "status_code":  response.status_code,
                "latency_ms":   elapsed_ms,
            },
        )

        response.headers["X-Request-ID"] = request_id
        return response

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_body_meta(body: bytes) -> dict:
        if not body:
            return {}
        try:
            data = json.loads(body)
            return {
                "session_id":    data.get("session_id", "—"),
                "interact_type": data.get("interact_type", "—"),
                "msg_len":       len(data.get("message", "") or ""),
                "history_len":   len(data.get("chat_history", []) or []),
                "has_docs":      bool(data.get("attached_documents")),
            }
        except Exception:
            return {}
