import json
import logging
import asyncio
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .store import AtomStore
import functools
logger = logging.getLogger(__name__)


class Retriever:
    """
    Пайплайн поиска атомов — только embedding + reranking, без LLM.

    Этапы retrieve():
      1. Векторный поиск (ChromaDB)
      2. Расширение соседними атомами
      3. CrossEncoder реранкинг
    """

    def __init__(self, store: AtomStore, config):
        # LLM-клиент намеренно отсутствует: Retriever работает
        # исключительно с векторной БД и локальным CrossEncoder.
        # Любая логика с вызовами LLM принадлежит слою main.py, не сюда.
        self.store = store
        self.config = config
        self._reranker = None  # загружается отдельно через load_reranker()

    async def load_reranker(self) -> None:
        """Загружает модель реранкинга в bfloat16 (без flash_attention)."""
        if not self.config.rerank_model:
            logger.info("Реранкинг отключён (rerank_model не задан)")
            return

        loop = asyncio.get_event_loop()
        def _load():
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.rerank_model, padding_side="left"
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.config.rerank_model,
                torch_dtype=torch.bfloat16,          # загрузка в bf16
                attn_implementation="eager",          # отключаем flash_attention
            ).eval()

            device = self.config.rerank_device or "cpu"
            model = model.to(device)

            token_false_id = tokenizer.convert_tokens_to_ids("no")
            token_true_id = tokenizer.convert_tokens_to_ids("yes")

            prefix = (
                "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. "
                "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
                "<|im_start|>user\n"
            )
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
            suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

            max_length = 8192

            return {
                "tokenizer": tokenizer,
                "model": model,
                "token_false_id": token_false_id,
                "token_true_id": token_true_id,
                "prefix_tokens": prefix_tokens,
                "suffix_tokens": suffix_tokens,
                "max_length": max_length,
                "device": device,
            }

        self._reranker = await loop.run_in_executor(None, _load)
        logger.info(f"✓ {self.config.rerank_model} загружен (bfloat16): %s", self.config.rerank_model)

    async def retrieve(self, query: str) -> list[dict]:
        """
        Основной метод: возвращает отранжированные атомы.

        Порядок: векторный поиск → расширение соседями → CrossEncoder реранкинг.
        """
        # 1. Начальный векторный поиск по embedding
        
        loop = asyncio.get_event_loop()
        search_wrapped = functools.partial(self.store.search, query, n=self.config.search_top_k)
        results = await loop.run_in_executor(None, search_wrapped)

        if not results:
            logger.warning("Векторный поиск вернул 0 результатов для запроса: %s", query)
            return []

        # 2. Собираем уникальные ID атомов + их соседей (до 2 на результат)
        seen_ids: set[str] = set()
        ordered_ids: list[str] = []

        for r in results:
            atom_id = str(r["metadata"]["id"])
            if atom_id not in seen_ids:
                seen_ids.add(atom_id)
                ordered_ids.append(atom_id)

            # Соседние атомы из метаданных
            neighbor_ids_json = r["metadata"].get("neighbor_ids_json", "[]")
            try:
                neighbors = json.loads(neighbor_ids_json)
            except (json.JSONDecodeError, TypeError):
                neighbors = []

            for nid in neighbors[:2]:
                nid = str(nid)
                if nid not in seen_ids:
                    seen_ids.add(nid)
                    ordered_ids.append(nid)

        # 3. Загружаем тексты атомов из ChromaDB
        all_atoms = self.store.get_by_ids(ordered_ids)

        if not all_atoms:
            logger.warning("get_by_ids вернул пустой список для %d id", len(ordered_ids))
            return []

        # Сохраняем порядок из ordered_ids
        id_to_atom: dict[str, dict] = {
            str(a["metadata"]["id"]): a for a in all_atoms
        }
        ordered_atoms = [id_to_atom[oid] for oid in ordered_ids if oid in id_to_atom]

        # 4. CrossEncoder реранкинг → топ search_final_k
        reranked = await loop.run_in_executor(None, self._rerank, query, ordered_atoms)

        return reranked[: self.config.search_final_k]

    # ------------------------------------------------------------------
    # Реранкинг (локальная модель, без сети и LLM)
    # ------------------------------------------------------------------


    def _rerank(self, query: str, atoms: list[dict]) -> list[dict]:
        """
        Реранкинг с помощью self.config.rerank_model.
        Обработка идёт по батчам, размер батча задаётся в self.config.rerank_batch_size.
        """
        if not atoms:
            return []

        if self._reranker is None:
            logger.info("Реранкер не загружен, возвращаем атомы в порядке векторного поиска")
            return [dict(a) | {"relevance_score": 0.0} for a in atoms]

        # Компоненты модели
        tok = self._reranker["tokenizer"]
        model = self._reranker["model"]
        prefix_tokens = self._reranker["prefix_tokens"]
        suffix_tokens = self._reranker["suffix_tokens"]
        max_length = self._reranker["max_length"]
        token_true_id = self._reranker["token_true_id"]
        token_false_id = self._reranker["token_false_id"]
        device = self._reranker["device"]

        instruction = "Given a web search query, retrieve relevant passages that answer the query"

        # Размер батча (по умолчанию 32, если не задано в конфиге)
        batch_size = getattr(self.config, "rerank_batch_size", 32)

        all_scores = []

        # Обработка по батчам
        for i in range(0, len(atoms), batch_size):
            batch_atoms = atoms[i:i + batch_size]
            len_atoms = sum([len(a['document']) for a in batch_atoms])
            print(len_atoms)
            # Формируем текстовые пары для батча
            pairs = [
                self._format_instruction(instruction, query, a.get("document", ""))
                for a in batch_atoms
            ]

            inputs = self._process_inputs_batch(
                tok, prefix_tokens, suffix_tokens, max_length, device, pairs
            )

            with torch.inference_mode():
                outputs = model(**inputs, use_cache=False)   # ← ключевое изменение
                logits = outputs.logits[:, -1, :]
                true_scores = logits[:, token_true_id]
                false_scores = logits[:, token_false_id]
                stacked = torch.stack([false_scores, true_scores], dim=1)
                probs = torch.nn.functional.log_softmax(stacked, dim=1).exp()
                batch_scores = probs[:, 1].cpu().tolist()
                del outputs.past_key_values   # удаляем кэш вручную
                del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            all_scores.extend(batch_scores)

        # Логирование статистики
        if all_scores:
            logger.debug(
                "Qwen3-Reranker скоры: min=%.3f max=%.3f mean=%.3f",
                min(all_scores), max(all_scores), sum(all_scores) / len(all_scores),
            )

        # Сортировка по убыванию relevance_score
        ranked = sorted(
            [dict(a) | {"relevance_score": s} for a, s in zip(atoms, all_scores)],
            key=lambda x: x["relevance_score"],
            reverse=True,
        )
        return ranked[: self.config.rerank_top_k]

    @staticmethod
    def _format_instruction(instruction: str, query: str, doc: str) -> str:
        if instruction is None:
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    @staticmethod
    def _process_inputs_batch(tokenizer, prefix_tokens, suffix_tokens, max_length, device, texts):
        """Токенизация батча с добавлением префикса/суффикса и паддингом."""
        # Токенизация самих текстов
        encoded = tokenizer(
            texts,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
        )
        # Добавляем префикс и суффикс
        for idx, ids in enumerate(encoded["input_ids"]):
            encoded["input_ids"][idx] = prefix_tokens + ids + suffix_tokens
        # Паддинг
        encoded = tokenizer.pad(
            encoded, padding=True, return_tensors="pt", max_length=max_length
        )
        return {k: v.to(device) for k, v in encoded.items()}