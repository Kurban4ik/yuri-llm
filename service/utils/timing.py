# utils/timing.py
"""
Вспомогательные инструменты для замера времени шагов pipeline.

Использование:
    from utils.timing import StepTimer

    with StepTimer("rag_retrieval", logger, extra={"session_id": sid}) as t:
        atoms = await retriever.retrieve(query)
    # После блока t.elapsed_ms содержит задержку
"""

import logging
import time
from contextlib import contextmanager
from typing import Any, Generator


class StepTimer:
    """Контекстный менеджер: фиксирует время выполнения шага и пишет INFO/ERROR в лог."""

    def __init__(
        self,
        step_name: str,
        logger: logging.Logger,
        extra: dict[str, Any] | None = None,
        log_entry: bool = False,   # писать ли лог при ВХОДЕ в блок
    ):
        self.step_name = step_name
        self.logger = logger
        self.extra = extra or {}
        self.log_entry = log_entry
        self.elapsed_ms: int = 0
        self._start: float = 0.0

    def __enter__(self) -> "StepTimer":
        self._start = time.monotonic()
        if self.log_entry:
            self.logger.debug(
                f"Step started: {self.step_name}",
                extra={"step": self.step_name, **self.extra},
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.elapsed_ms = int((time.monotonic() - self._start) * 1000)
        if exc_type is not None:
            self.logger.error(
                f"Step failed: {self.step_name}",
                extra={
                    "step":       self.step_name,
                    "latency_ms": self.elapsed_ms,
                    "error":      str(exc_val),
                    **self.extra,
                },
                exc_info=True,
            )
        else:
            self.logger.debug(
                f"Step done: {self.step_name}",
                extra={
                    "step":       self.step_name,
                    "latency_ms": self.elapsed_ms,
                    **self.extra,
                },
            )
        return False  # не подавляем исключение
