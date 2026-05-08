"""
atom_builder.py — build_atoms(act, generated_queries) -> list[dict]

Каждый clause порождает атомы:
  1. type="atom"       — текст пункта (или чанк) + контекст + ожидаемые запросы
  2. type="query_atom" — только вопросы (для отдельного embedding)

Изменения v2:
  - Длинные клозы (> MAX_CLAUSE_CHARS) нарезаются sliding window с перекрытием.
    Каждый чанк получает отдельный атом с id вида {clause_id}_c0, _c1, ...
  - Все атомы содержат metadata.article_full_text — полный текст статьи
    (small-to-big retrieval: ищем по короткому чанку, в LLM отдаём всю статью).
  - metadata дополнена chunk_index, chunk_total, parent_clause_id.
  - Добавлено поле plain_text — чистый текст чанка без обёрток (v3).
"""
from __future__ import annotations

from typing import Optional

from models import LegalAct, Section, Article, Part, Clause

# ── константы ────────────────────────────────────────────────────────────────

MAX_CLAUSE_CHARS: int = 1300   # клозы длиннее этого порога будут нарезаны
CHUNK_OVERLAP: int    = 200    # перекрытие между соседними чанками (символов)


# ── helpers ──────────────────────────────────────────────────────────────────

def _section_for_article(act: LegalAct, article: Article) -> Optional[str]:
    for section in act.sections:
        if article in section.articles:
            return section.name
    return None


def _neighbor_ids(article: Article, current_clause_id: str) -> list[str]:
    """
    Возвращает id всех клозов той же статьи, кроме текущего.
    Для чанкованных атомов возвращает базовые clause_id (без суффикса _cN),
    чтобы при retrieval можно было найти соседей независимо от нарезки.
    """
    ids = []
    for part in article.parts:
        for clause in part.clauses:
            if clause.id != current_clause_id:
                ids.append(clause.id)
    return ids


def split_text(text: str, max_chars: int = MAX_CLAUSE_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Нарезает text на чанки длиной не более max_chars с перекрытием overlap.
    Если текст короче max_chars — возвращает [text] без изменений.
    Нарезка идёт по границам слов, чтобы не рвать слова на середине.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + max_chars

        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        boundary = text.rfind(" ", start, end)
        if boundary == -1 or boundary <= start:
            boundary = end

        chunks.append(text[start:boundary].strip())
        next_start = boundary - overlap
        start = max(next_start, boundary - overlap, start + 1)

    return [c for c in chunks if c]


# ── основной builder ──────────────────────────────────────────────────────────

def build_atoms(act: LegalAct, generated_queries: dict[str, list[str]]) -> list[dict]:
    atoms: list[dict] = []

    for article in act.articles:
        section_name  = _section_for_article(act, article)
        section_display = section_name or "—"

        article_full_text = getattr(article, "full_text", "") or ""

        for part in article.parts:
            for clause in part.clauses:
                queries      = generated_queries.get(clause.id, [])
                queries_text = "\n".join(queries)

                chunks      = split_text(clause.full_text)
                chunk_total = len(chunks)

                for chunk_index, chunk_text in enumerate(chunks):

                    # ── формируем id ──────────────────────────────────────
                    if chunk_total == 1:
                        atom_id = clause.id
                    else:
                        atom_id = f"{clause.id}_c{chunk_index}"

                    # ── embedding text с обёртками ────────────────────────
                    embedding_text = (
                        f"[АТОМ]\n"
                        f"{chunk_text}\n\n"
                        f"[КОНТЕКСТ]\n"
                        f"Нормативный акт: {act.act_name} ({act.act_type}, принят {act.date})\n"
                        f"Статья: {article.number} {article.title}\n"
                        f"Часть: {part.number}\n"
                        f"Раздел: {section_display}\n\n"
                        f"[ОЖИДАЕМЫЕ_ЗАПРОСЫ]\n"
                        f"{queries_text}"
                    )

                    # ── plain text: чистая вырезка (без обёрток) ─────────
                    plain_text = chunk_text

                    neighbor_ids = _neighbor_ids(article, clause.id)

                    atom: dict = {
                        "id":   atom_id,
                        "type": "atom",
                        "embedding_text": embedding_text,
                        "plain_text": plain_text,          # NEW
                        "metadata": {
                            "act_name":  act.act_name,
                            "act_type":  act.act_type,
                            "date":      act.date,
                            "article_number":    article.number,
                            "article_title":     article.title,
                            "article_full_text": article_full_text,
                            "part_number":    part.number,
                            "section_name":   section_display,
                            "clause_number":  clause.number,
                            "clause_text":    clause.full_text,
                            "chunk_index":      chunk_index,
                            "chunk_total":      chunk_total,
                            "chunk_text":       chunk_text,
                            "parent_clause_id": clause.id,
                            "neighbor_ids": neighbor_ids,
                            "atom_type":    "atom",
                        },
                    }
                    atoms.append(atom)

                # query_atom остаются без plain_text (или можно добавить при необходимости)
                for q_index, query in enumerate(queries):
                    query_atom: dict = {
                        "id":   f"{clause.id}_q_{q_index}",
                        "type": "query_atom",
                        "embedding_text": query,
                        "metadata": {
                            "parent_atom_id": clause.id,
                            "clause_text":    clause.full_text,
                            "atom_type":      "query_atom",
                        },
                    }
                    atoms.append(query_atom)

    return atoms