import json
import logging
import gc
import os
from typing import Optional, Iterator, Dict, Any

import chromadb
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Настройка окружения
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

COLLECTION_NAME = "atoms"
MAX_TEXT_LENGTH = 2000      # максимальная длина embedding_text в символах
BATCH_SIZE = 4            # размер батча для отправки в ChromaDB


def merge_by_article(atoms: list[dict]) -> list[dict]:
    """
    Объединяет атомы с одинаковым article_full_text.
    Для каждой статьи оставляет clause_text с наибольшим relevance_score.
    
    Args:
        atoms: список словарей с ключами 'article_full_text', 'relevance_score',
               'clause_text' и другими метаданными.
    
    Returns:
        Список уникальных статей (с лучшим clause_text), отсортированный
        по убыванию relevance_score.
    """
    # Группировка по полному тексту статьи
    best_by_article = {}
    for atom in atoms:
        article = atom.get('article_full_text')
        if not article:
            # Если по какой-то причине нет полного текста, используем atom_id или parent_clause_id
            article = atom.get('atom_id') or atom.get('metadata', {}).get('parent_clause_id')
        
        score = atom.get('relevance_score')
        # Если score вдруг нет, можно использовать расстояние (меньше – лучше)
        if score is None and 'distance' in atom:
            score = -atom['distance']  # превращаем расстояние в "скор" (чем меньше dist, тем выше скор)
        
        # Если статьи ещё нет в словаре или у текущего атома скор выше – заменяем
        if article not in best_by_article or score > best_by_article[article]['relevance_score']:
            best_by_article[article] = atom
    
    # Возвращаем список уникальных статей, отсортированный по убыванию скора
    merged = list(best_by_article.values())
    merged.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
    return merged


class AtomStore:
    def __init__(self, db_path: str, embed_model: str, embed_device: str):
        self.db_path = db_path
        self.embed_model_name = embed_model
        self.embed_device = embed_device
        self._model: Optional[SentenceTransformer] = None

        self._chroma = chromadb.PersistentClient(path=db_path)
        self._collection = self._chroma.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    async def load_model(self):
        import asyncio
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(
            None,
            lambda: SentenceTransformer(
                self.embed_model_name,
                device=self.embed_device,
                local_files_only=True,
                            model_kwargs={"torch_dtype": torch.float16}   # ← ключевое изменение
            ),
        )
        self._model.max_seq_length = 4096

        logger.info("✓ Embedding model loaded: %s", self.embed_model_name)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            raise RuntimeError("Embedding model not loaded — call load_model() first")
        with torch.no_grad():
            vecs = self._model.encode(
                texts,
                convert_to_numpy=True,
                batch_size=8,
                show_progress_bar=False
            )
        if self.embed_device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()
        return vecs.tolist()

    @staticmethod
    def _iter_atoms(jsonl_path: str) -> Iterator[Dict[str, Any]]:
        """Генератор, возвращающий атомы по одному из файла."""
        with open(jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def index_atoms(self, atoms_jsonl_path: str) -> int:
        """Индексация атомов с потоковым чтением и строгой очисткой памяти."""
        indexed = 0
        skipped = 0
        batch_docs = []
        batch_metas = []
        batch_ids = []

        # Подсчёт общего числа строк для прогресса (без загрузки в память)
        total_lines = 0
        with open(atoms_jsonl_path, encoding="utf-8") as f:
            for _ in f:
                total_lines += 1
        logger.info("Total lines in file: %d", total_lines)

        for atom in self._iter_atoms(atoms_jsonl_path):
            atom_type = atom.get("type", "atom")
            if atom_type not in ("atom", "query_atom"):
                continue

            doc_text = atom.get("embedding_text", "")
            if len(doc_text) > MAX_TEXT_LENGTH:
                logger.warning(
                    "Skipping atom id=%s type=%s: length %d > %d chars",
                    atom.get("id"), atom_type, len(doc_text), MAX_TEXT_LENGTH
                )
                skipped += 1
                continue

            nested_meta = atom.get("metadata", {})
            if isinstance(nested_meta, str):
                try:
                    nested_meta = json.loads(nested_meta)
                except (json.JSONDecodeError, TypeError):
                    nested_meta = {}
            
            meta = {
                k: v for k, v in atom.items()
                if k not in ("embedding_text", "metadata")
            }
            meta.update(nested_meta)
            
            neighbor_ids = meta.pop("neighbor_ids", [])
            meta["neighbor_ids_json"] = json.dumps(neighbor_ids)
            for key, val in list(meta.items()):
                if isinstance(val, (list, dict)):
                    meta[key] = json.dumps(val)

            batch_docs.append(doc_text)
            batch_metas.append(meta)
            batch_ids.append(str(atom["id"]))

            if len(batch_docs) >= BATCH_SIZE:
                indexed += self._process_batch(batch_docs, batch_metas, batch_ids)
                batch_docs.clear()
                batch_metas.clear()
                batch_ids.clear()

                gc.collect()
                if self.embed_device.startswith("cuda"):
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                logger.info("Indexed %d atoms so far (skipped: %d)", indexed, skipped)

        if batch_docs:
            indexed += self._process_batch(batch_docs, batch_metas, batch_ids)

        gc.collect()
        if self.embed_device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        logger.info("✓ Total indexed: %d, skipped due to length: %d", indexed, skipped)
        return indexed

    def _process_batch(self, docs: list, metas: list, ids: list) -> int:
        """Обработка одного батча: эмбеддинг + запись в ChromaDB."""
        # Статистика по длинам текстов в батче
        total_chars = sum(len(doc) for doc in docs)
        avg_chars = total_chars / len(docs) if docs else 0
        max_chars = max(len(doc) for doc in docs) if docs else 0
        min_chars = min(len(doc) for doc in docs) if docs else 0

        logger.info(
            "Processing batch: size=%d, total chars=%d, avg=%.1f, min=%d, max=%d",
            len(docs), total_chars, avg_chars, min_chars, max_chars
        )

        embs = self._embed(docs)
        self._collection.add(
            documents=docs,
            metadatas=metas,
            ids=ids,
            embeddings=embs,
        )
        del embs
        return len(docs)

    def search(self, query: str, n: int, deduplicate_by_parent: bool = True) -> list[dict]:
        emb = self._embed([query])
        # Запрашиваем больше записей, чтобы после дедупликации набрать n уникальных
        fetch_n = min(n * 3, self._collection.count() or 1) if deduplicate_by_parent else n
        results = self._collection.query(
            query_embeddings=emb,
            n_results=fetch_n,
            include=["metadatas", "documents", "distances"],
        )
        print(fetch_n)
        out: list[dict] = []
        metas = results["metadatas"][0]
        docs = results["documents"][0]
        dists = results["distances"][0]
    
        for meta, doc, dist in zip(metas, docs, dists):
            if meta.get("type") != "atom":
                continue
            out.append({"metadata": meta, "document": doc, "distance": dist})
    
        # Дедупликация по parent_clause_id
        if deduplicate_by_parent:
            seen = set()
            unique_out = []
            for item in out:
                parent_id = item['metadata'].get('parent_clause_id')
                if parent_id and parent_id not in seen:
                    seen.add(parent_id)
                    unique_out.append(item)
                    if len(unique_out) >= n:
                        break
            out = unique_out
        else:
            out = out[:n]
    
        if self.embed_device.startswith("cuda"):
            torch.cuda.empty_cache()
        gc.collect()
        print(len(out))
        return out
    def get_by_ids(self, ids: list[str]) -> list[dict]:
        if not ids:
            return []
        results = self._collection.get(
            ids=ids,
            include=["metadatas", "documents"],
        )
        out: list[dict] = []
        for meta, doc in zip(results["metadatas"], results["documents"]):
            out.append({"metadata": meta, "document": doc})
        return out

    def count(self) -> int:
        return self._collection.count()