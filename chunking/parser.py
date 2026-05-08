"""
parser.py — parse_rtf(filepath) -> LegalAct

Иерархия: Акт → [Раздел/Глава] → Статья → Часть → Пункт

Изменения v2:
  - Исправлен баг с RE_CLAUSE / RE_PART: \S заменён на lookahead (?=\S),
    из-за чего первый символ текста пункта больше не обрубается.
  - При разборе каждой статьи накапливается current_article_lines;
    по завершении статьи весь текст записывается в article.full_text —
    используется в atom_builder для small-to-big retrieval.
"""
from __future__ import annotations

import os
import re
import subprocess
import logging
from pathlib import Path
from typing import Optional

from models import LegalAct, Section, Article, Part, Clause, detect_act_type

logger = logging.getLogger(__name__)

# ── регулярки ────────────────────────────────────────────────────────────────

RE_DATE = re.compile(
    r"от\s+(\d{1,2}[\.\s]\d{1,2}[\.\s]\d{4}|\d{1,2}\s+\w+\s+\d{4})",
    re.IGNORECASE,
)
RE_SECTION = re.compile(
    r"^\s*(Раздел|Глава)\s+([\wIVXивхлсмдк]+\.?)\s*(.*)?$",
    re.IGNORECASE | re.UNICODE,
)
RE_ARTICLE = re.compile(
    r"^\s*Статья\s+(\d+[\.\d]*)\s*\.?\s*(.*)?$",
    re.IGNORECASE | re.UNICODE,
)

# FIX: \S → (?=\S) — первый символ текста больше не поглощается матчем
RE_PART   = re.compile(r"^\s*(\d+)\.\s+(?=\S)")
RE_CLAUSE = re.compile(
    r"^\s*(\d+\)|[а-яёa-z]\)|[а-яёa-z]\.|\d+\.)\s+(?=\S)",
    re.UNICODE,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _rtf_to_text(filepath: str) -> str:
    """Конвертирует RTF → plain text. Пробуем striprtf, затем pandoc."""
    try:
        from striprtf.striprtf import rtf_to_text
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
        return rtf_to_text(raw)
    except Exception as e:
        logger.debug("striprtf failed (%s), trying pandoc", e)

    result = subprocess.run(
        ["pandoc", filepath, "-t", "plain", "--wrap=none"],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode == 0:
        return result.stdout
    raise RuntimeError(f"Cannot convert RTF '{filepath}': {result.stderr}")


def _slug(name: str) -> str:
    return re.sub(r"[^\w]+", "_", name).strip("_").lower()


def _make_clause_id(file_slug: str, art_num: str, part_num: str, cl_num: str) -> str:
    return f"{file_slug}_{art_num}_{part_num}_{_slug(cl_num)}"


# ── основной парсер ───────────────────────────────────────────────────────────

def parse_rtf(filepath: str) -> LegalAct:
    path = Path(filepath)
    text = _rtf_to_text(filepath)
    lines = text.splitlines()

    filename = path.stem
    act_type = detect_act_type(filename)
    file_slug = _slug(filename)

    # Пытаемся вытащить дату
    date = ""
    for line in lines[:30]:
        m = RE_DATE.search(line)
        if m:
            date = m.group(1).strip()
            break

    # Пытаемся вытащить название акта (первые непустые строки)
    act_name_lines = []
    for line in lines[:20]:
        stripped = line.strip()
        if stripped:
            act_name_lines.append(stripped)
        if len(act_name_lines) >= 3:
            break
    act_name = " ".join(act_name_lines) or filename

    # ── state machine ──
    sections: list[Section] = []
    articles: list[Article] = []
    current_section: Optional[Section] = None
    current_article: Optional[Article] = None
    current_part: Optional[Part] = None
    current_clause_lines: list[str] = []
    current_clause_num: str = ""

    # NEW: накапливаем сырые строки текущей статьи для full_text
    current_article_lines: list[str] = []

    def flush_clause():
        nonlocal current_clause_lines, current_clause_num
        if current_clause_lines and current_part and current_article:
            full_text = " ".join(current_clause_lines).strip()
            cl_id = _make_clause_id(
                file_slug,
                current_article.number,
                current_part.number,
                current_clause_num,
            )
            clause = Clause(
                number=current_clause_num,
                full_text=full_text,
                id=cl_id,
            )
            current_part.clauses.append(clause)
        current_clause_lines = []
        current_clause_num = ""

    def flush_part():
        flush_clause()
        nonlocal current_part
        current_part = None

    def flush_article_full_text():
        """Записывает накопленные строки в article.full_text и сбрасывает буфер."""
        nonlocal current_article_lines
        if current_article is not None and current_article_lines:
            current_article.full_text = "\n".join(current_article_lines).strip()
        current_article_lines = []

    def new_article(number: str, title: str):
        nonlocal current_article, current_part
        flush_part()
        flush_article_full_text()          # NEW: сохраняем full_text предыдущей статьи
        art = Article(number=number, title=title.strip())
        articles.append(art)
        if current_section is not None:
            current_section.articles.append(art)
        current_article = art
        current_part = None

    def new_section(name: str):
        nonlocal current_section
        flush_part()
        flush_article_full_text()          # NEW: тоже сохраняем при смене раздела
        sec = Section(name=name.strip() or None)
        sections.append(sec)
        current_section = sec

    # Гарантируем «безымянный» раздел для статей вне разделов
    default_section = Section(name=None)
    sections.append(default_section)
    current_section = default_section

    part_counter = 0

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped:
            continue

        # Раздел / Глава
        m = RE_SECTION.match(line)
        if m:
            sec_name = f"{m.group(1)} {m.group(2)} {m.group(3)}".strip()
            new_section(sec_name)
            continue

        # Статья
        m = RE_ARTICLE.match(line)
        if m:
            new_article(m.group(1), m.group(2) or "")
            part_counter = 0
            continue

        if current_article is None:
            continue  # до первой статьи — пропускаем

        # NEW: любая строка внутри статьи попадает в буфер full_text
        current_article_lines.append(stripped)

        # Часть (явная нумерация: "1. Текст")
        m_part = RE_PART.match(line)
        if m_part and not RE_CLAUSE.match(line):
            flush_clause()
            part_num = m_part.group(1)
            new_part = Part(number=part_num)
            current_article.parts.append(new_part)
            current_part = new_part
            rest = re.sub(r"^\s*\d+\.\s+", "", line)
            m_cl = RE_CLAUSE.match(rest)
            if m_cl:
                flush_clause()
                current_clause_num = m_cl.group(1)
                # FIX: используем m_cl.end() вместо len(group(0).strip())
                current_clause_lines = [rest[m_cl.end():].strip()]
            else:
                current_clause_num = "0"
                current_clause_lines = [rest.strip()] if rest.strip() else []
            continue

        # Пункт
        m_cl = RE_CLAUSE.match(line)
        if m_cl:
            flush_clause()
            if current_part is None:
                part_counter += 1
                new_part = Part(number=str(part_counter))
                current_article.parts.append(new_part)
                current_part = new_part
            current_clause_num = m_cl.group(1)
            # FIX: используем m_cl.end() — позиция после маркера без \S
            current_clause_lines = [stripped[m_cl.end():].strip()]
            continue

        # Продолжение текущего пункта / части
        if current_part is None:
            part_counter += 1
            new_part = Part(number=str(part_counter))
            current_article.parts.append(new_part)
            current_part = new_part
            current_clause_num = "0"

        if current_clause_num:
            current_clause_lines.append(stripped)
        else:
            current_clause_num = "0"
            current_clause_lines = [stripped]

    # Финальный флаш
    flush_part()
    flush_article_full_text()   # NEW: последняя статья

    # Убираем пустой default_section если нет статей
    sections = [s for s in sections if s.articles or s is default_section and articles]
    if not default_section.articles:
        sections = [s for s in sections if s is not default_section]

    return LegalAct(
        act_name=act_name,
        act_type=act_type,
        date=date,
        sections=sections,
        articles=articles,
    )