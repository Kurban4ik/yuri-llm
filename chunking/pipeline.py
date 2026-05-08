"""
pipeline.py — сборка всего пайплайна + CLI

Запуск:
  python -m chunking.pipeline \\
    --input-dir ./laws_rtf \\
    --output ./data/atoms.jsonl \\
    --api-key sk-... \\
    [--migrate-legacy ./data/laws.jsonl] \\
    [--concurrency 5]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from atom_builder import build_atoms
from models import LegalAct, Article, Part, Clause, Section
from parser import parse_rtf, _slug
from query_gen import generate_queries

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_existing_ids(output_file: str) -> set[str]:
    """Загружает id атомов, уже записанных в output_file (идемпотентность)."""
    existing: set[str] = set()
    path = Path(output_file)
    if not path.exists():
        return existing
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "id" in obj:
                    existing.add(obj["id"])
            except json.JSONDecodeError:
                pass
    return existing


def _context_str(act: LegalAct, article: "Article", part: "Part", section_name: Optional[str]) -> str:
    return (
        f"Нормативный акт: {act.act_name} ({act.act_type}, принят {act.date})\n"
        f"Статья: {article.number} {article.title}\n"
        f"Часть: {part.number}\n"
        f"Раздел: {section_name or '—'}"
    )


def _section_for_article(act: LegalAct, article: Article) -> Optional[str]:
    for section in act.sections:
        if article in section.articles:
            return section.name
    return None


# ── async query generation ────────────────────────────────────────────────────

async def _generate_queries_async(
    clause_id: str,
    clause_text: str,
    context: str,
    api_key: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, list[str]]:
    async with semaphore:
        loop = asyncio.get_event_loop()
        queries = await loop.run_in_executor(
            None, generate_queries, clause_text, context, api_key
        )
        return clause_id, queries


# ── main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    input_dir: str,
    output_file: str,
    api_key: str,
    concurrency: int = 30,
) -> None:
    rtf_files = sorted(Path(input_dir).glob("**/*.rtf"))
    if not rtf_files:
        logger.warning("No .rtf files found in %s", input_dir)
        return

    existing_ids = _load_existing_ids(output_file)
    logger.info("Found %d RTF files; %d atom ids already in output", len(rtf_files), len(existing_ids))

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "a", encoding="utf-8") as out_fh:
        for rtf_path in tqdm(rtf_files, desc="RTF files"):
            try:
                act = parse_rtf(str(rtf_path))
            except Exception as exc:
                logger.error("Failed to parse %s: %s", rtf_path, exc)
                continue

            # Собираем все clause для которых нужны запросы
            pending: list[tuple[str, str, str]] = []  # (clause_id, clause_text, context)
            for article in act.articles:
                section_name = _section_for_article(act, article)
                for part in article.parts:
                    for clause in part.clauses:
                        main_id = clause.id
                        if main_id in existing_ids or f"{main_id}_q" in existing_ids:
                            continue
                        ctx = _context_str(act, article, part, section_name)
                        pending.append((clause.id, clause.full_text, ctx))
            pending = pending[:5]  # safety limit for testing
            if not pending:
                continue

            # Async генерация запросов
            semaphore = asyncio.Semaphore(concurrency)

            async def gather_queries():
                tasks = [
                    _generate_queries_async(cid, ctxt, ctx, api_key, semaphore)
                    for cid, ctxt, ctx in pending
                ]
                results = []
                for coro in tqdm(
                    asyncio.as_completed(tasks),
                    total=len(tasks),
                    desc=f"  Queries ({rtf_path.stem})",
                    leave=False,
                ):
                    results.append(await coro)
                return results

            query_results = asyncio.run(gather_queries())
            generated_queries: dict[str, list[str]] = dict(query_results)

            atoms = build_atoms(act, generated_queries)

            for atom in atoms:
                if atom["id"] in existing_ids:
                    continue
                out_fh.write(json.dumps(atom, ensure_ascii=False) + "\n")
                existing_ids.add(atom["id"])

    logger.info("Done. Output: %s", output_file)


# ── legacy migration ──────────────────────────────────────────────────────────

def migrate_legacy_jsonl(
    legacy_path: str,
    output_path: str,
    api_key: str,
    concurrency: int = 5,
) -> None:
    """
    Конвертирует старый laws.jsonl (поля: text, metadata.{chapter,article,point,file_source})
    в новый формат атомов, вызывая generate_queries для каждого пункта.
    """
    existing_ids = _load_existing_ids(output_path)
    records = []
    with open(legacy_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    logger.info("Loaded %d legacy records from %s", len(records), legacy_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(concurrency)

    async def process_record(rec: dict) -> list[dict]:
        meta = rec.get("metadata", {})
        text: str = rec.get("text", "").strip()
        file_source: str = meta.get("file_source", "unknown")
        chapter: str = meta.get("chapter", "") or ""
        article: str = meta.get("article", "") or ""
        point: str = meta.get("point", "") or ""

        file_slug = _slug(file_source.replace(".rtf", "").replace(".RTF", ""))
        art_slug = _slug(article)
        pt_slug = _slug(point) if point else "0"
        clause_id = f"{file_slug}_{art_slug}_{pt_slug}"

        if clause_id in existing_ids:
            return []

        context_str = (
            f"Нормативный акт: {file_source}\n"
            f"Глава: {chapter}\n"
            f"Статья: {article}\n"
            f"Пункт: {point}"
        )

        async with semaphore:
            loop = asyncio.get_event_loop()
            queries = await loop.run_in_executor(
                None, generate_queries, text, context_str, api_key
            )

        queries_text = "\n".join(queries)
        embedding_text = (
            f"[АТОМ]\n{text}\n\n"
            f"[КОНТЕКСТ]\n{context_str}\n\n"
            f"[ОЖИДАЕМЫЕ_ЗАПРОСЫ]\n{queries_text}"
        )

        atom = {
            "id": clause_id,
            "type": "atom",
            "embedding_text": embedding_text,
            "metadata": {
                "act_name": file_source,
                "act_type": "иное",
                "date": "",
                "article_number": article,
                "article_title": "",
                "part_number": "1",
                "section_name": chapter or "—",
                "clause_number": point,
                "clause_text": text,
                "neighbor_ids": [],
                "atom_type": "atom",
            },
        }
        query_atoms = []
        counter = 0
        for i in queries_text.splitlines():
            counter += 1
            query_atom = {
                "id": f"{clause_id}_q_{counter}",
                "type": "query_atom",
                "embedding_text": i,
                "metadata": {
                    "parent_atom_id": clause_id,
                    "clause_text": "",
                    "atom_type": "query_atom",
                },
            }
            query_atoms.append(query_atom)
        return [atom, *query_atoms]

    async def run_all():
        tasks = [process_record(r) for r in records]
        all_atoms: list[dict] = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Migrating"):
            result = await coro
            all_atoms.extend(result)
        return all_atoms

    all_atoms = asyncio.run(run_all())

    with open(output_path, "a", encoding="utf-8") as fh:
        for atom in all_atoms:
            if atom["id"] not in existing_ids:
                fh.write(json.dumps(atom, ensure_ascii=False) + "\n")
                existing_ids.add(atom["id"])

    logger.info("Migration done. Output: %s", output_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Legal chunking pipeline: RTF → atoms JSONL"
    )
    parser.add_argument("--input-dir", default="./laws_rtf", help="Directory with .rtf files")
    parser.add_argument("--output", default="./data/atoms.jsonl", help="Output JSONL path")
    parser.add_argument("--api-key", required=True, help="DeepSeek API key")
    parser.add_argument("--concurrency", type=int, default=5, help="Async concurrency limit")
    parser.add_argument(
        "--migrate-legacy",
        metavar="LEGACY_JSONL",
        default=None,
        help="Path to legacy laws.jsonl to migrate instead of parsing RTFs",
    )
    args = parser.parse_args()

    if args.migrate_legacy:
        migrate_legacy_jsonl(
            legacy_path=args.migrate_legacy,
            output_path=args.output,
            api_key=args.api_key,
            concurrency=args.concurrency,
        )
    else:
        run_pipeline(
            input_dir=args.input_dir,
            output_file=args.output,
            api_key=args.api_key,
            concurrency=args.concurrency,
        )


if __name__ == "__main__":
    main()
