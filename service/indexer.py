"""
indexer.py — standalone script for indexing atoms.jsonl into ChromaDB.

Usage:
    python -m inference.indexer
    python -m inference.indexer --atoms-jsonl ./data/atoms.jsonl
    python -m inference.indexer --atoms-jsonl ./data/atoms.jsonl --db-path ./rag_db
"""

import argparse
import asyncio
import logging
import os
import sys

from config import settings
from rag.store import AtomStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("indexer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index atoms.jsonl into ChromaDB for the inference service."
    )
    parser.add_argument(
        "--atoms-jsonl",
        default=settings.atoms_jsonl,
        help=f"Path to atoms.jsonl (default: {settings.atoms_jsonl})",
    )
    parser.add_argument(
        "--db-path",
        default=settings.db_path,
        help=f"ChromaDB directory (default: {settings.db_path})",
    )
    parser.add_argument(
        "--embed-model",
        default=settings.embed_model,
        help=f"SentenceTransformer model name (default: {settings.embed_model})",
    )
    parser.add_argument(
        "--embed-device",
        default=settings.embed_device,
        help=f"Device for embedding model, e.g. cuda / cpu (default: {settings.embed_device})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-index even if the collection is not empty",
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
    atoms_path = args.atoms_jsonl

    if not os.path.exists(atoms_path):
        logger.error("atoms.jsonl not found: %s", atoms_path)
        return 1

    store = AtomStore(
        db_path=args.db_path,
        embed_model=args.embed_model,
        embed_device=args.embed_device,
    )

    existing = store.count()
    if existing > 0 and not args.force:
        logger.info(
            "Collection already contains %d atoms. Use --force to re-index.", existing
        )
        return 0

    logger.info("Loading embedding model: %s on %s", args.embed_model, args.embed_device)
    await store.load_model()

    logger.info("Indexing: %s → %s", atoms_path, args.db_path)
    n = store.index_atoms(atoms_path)
    logger.info("Done. Indexed %d atoms.", n)
    return 0


def main():
    args = parse_args()
    code = asyncio.run(run(args))
    sys.exit(code)


if __name__ == "__main__":
    main()
