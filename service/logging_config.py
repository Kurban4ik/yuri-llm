# logging_config.py
"""
Централизованная настройка логирования сервиса.

Два потока вывода:
  1. Основной (stdout) — все логи сервиса в JSON или PRETTY.
  2. LLM-вызовы (файл) — отдельный JSONL-файл только для logger «llm.calls».
     Содержит полные промпты, параметры и ответы каждого LLM-вызова.

Переменные окружения:
  LOG_LEVEL       — уровень основного лога (DEBUG / INFO / WARNING / ERROR).
                    По умолчанию: INFO.
  LOG_FORMAT      — "json" (продакшн) или "pretty" (разработка).
                    По умолчанию: json.
  LLM_LOG_FILE    — путь к JSONL-файлу с LLM-вызовами.
                    По умолчанию: logs/llm_calls.jsonl.
                    Передайте "" (пустую строку) — запись в файл отключается,
                    вызовы всё равно идут в stdout через корневой обработчик.
"""

import json
import logging
import logging.handlers
import os
import pathlib
import sys
import time
from typing import Any


# ---------------------------------------------------------------------------
# JSON-форматтер
# ---------------------------------------------------------------------------

class JsonFormatter(logging.Formatter):
    """Форматирует записи лога в однострочный JSON с фиксированными полями."""

    # Поля из LogRecord, которые уже вынесены явно или не нужны
    _SKIP = frozenset({
        "message", "asctime", "levelname", "name", "exc_info", "exc_text",
        "stack_info",
        "name", "msg", "args", "levelname", "levelno", "pathname",
        "filename", "module", "exc_info", "exc_text", "stack_info",
        "lineno", "funcName", "created", "msecs", "relativeCreated",
        "thread", "threadName", "processName", "process", "taskName",
    })

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts":     self._ts(record),
            "level":  record.levelname,
            "logger": record.name,
            "msg":    record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key.startswith("_") or key in self._SKIP or key in logging.LogRecord.__dict__:
                continue
            payload[key] = value

        return json.dumps(payload, ensure_ascii=False, default=str)

    @staticmethod
    def _ts(record: logging.LogRecord) -> str:
        return (
            time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z"
        )


# ---------------------------------------------------------------------------
# Pretty-форматтер (локальная разработка)
# ---------------------------------------------------------------------------

class PrettyFormatter(logging.Formatter):
    COLORS = {
        "DEBUG":    "\033[36m",
        "INFO":     "\033[32m",
        "WARNING":  "\033[33m",
        "ERROR":    "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    _SKIP = frozenset({
        "name", "msg", "args", "levelname", "levelno", "pathname",
        "filename", "module", "exc_info", "exc_text", "stack_info",
        "lineno", "funcName", "created", "msecs", "relativeCreated",
        "thread", "threadName", "processName", "process", "taskName",
        "message", "asctime",
    })

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        base = (
            f"{color}[{record.levelname:8s}]{self.RESET} "
            f"\033[90m{self._ts(record)}\033[0m "
            f"\033[34m{record.name}\033[0m — {record.getMessage()}"
        )
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in self._SKIP
            and not k.startswith("_")
            and k not in logging.LogRecord.__dict__
        }
        if extras:
            # Длинные поля (промпты, ответы) выводим полностью — это и есть цель
            parts = []
            for k, v in extras.items():
                val_str = str(v)
                # Переносим длинные значения на новую строку с отступом
                if len(val_str) > 120:
                    parts.append(f"\033[90m{k}=\033[0m\n      {val_str}")
                else:
                    parts.append(f"\033[90m{k}=\033[0m{val_str}")
            base += "\n    ↳ " + "  ".join(parts)
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base

    @staticmethod
    def _ts(record: logging.LogRecord) -> str:
        return (
            time.strftime("%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}"
        )


# ---------------------------------------------------------------------------
# Публичная функция настройки
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    """
    Вызывать ОДИН РАЗ при старте приложения.

    Настраивает:
      • Корневой логгер → stdout (JSON или PRETTY).
      • logger «llm.calls» → дополнительно в LLM_LOG_FILE (всегда JSON),
        чтобы промпты и ответы были изолированы в отдельном файле.
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    fmt_mode = os.getenv("LOG_FORMAT", "json").lower()
    stdout_formatter: logging.Formatter = (
        PrettyFormatter() if fmt_mode == "pretty" else JsonFormatter()
    )

    # --- Корневой обработчик (stdout) ---
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(stdout_formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(stdout_handler)

    # --- Заглушить шумные библиотеки ---
    for noisy in ("uvicorn.access", "httpx", "httpcore", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # --- Отдельный файловый обработчик для LLM-вызовов ---
    llm_log_file = os.getenv("LLM_LOG_FILE", "logs/llm_calls.jsonl")
    llm_logger = logging.getLogger("llm.calls")

    if llm_log_file:
        log_path = pathlib.Path(llm_log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # RotatingFileHandler: не даём файлу расти бесконечно
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_path,
            maxBytes=100 * 1024 * 1024,  # 100 MB на файл
            backupCount=10,
            encoding="utf-8",
        )
        # LLM-вызовы ВСЕГДА в JSON — независимо от LOG_FORMAT,
        # чтобы их можно было парсить программно
        file_handler.setFormatter(JsonFormatter())
        file_handler.setLevel(logging.DEBUG)

        # Не наследуем stdout-обработчик от корневого логгера —
        # LLM-вызовы будут и в stdout (через root), и в файл
        llm_logger.addHandler(file_handler)
        # propagate=True оставляем, чтобы вызовы шли и в stdout

        logging.getLogger(__name__).info(
            "LLM call logging configured",
            extra={
                "llm_log_file":    str(log_path.resolve()),
                "max_bytes":       "100MB",
                "backup_count":    10,
            },
        )
    else:
        logging.getLogger(__name__).info(
            "LLM call file logging disabled (LLM_LOG_FILE is empty)"
        )

    logging.getLogger(__name__).info(
        "Logging configured",
        extra={"log_level": level_name, "log_format": fmt_mode},
    )
