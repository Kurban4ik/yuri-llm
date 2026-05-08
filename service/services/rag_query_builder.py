# services/rag_query_builder.py
"""
Построение RAG-запросов и юридических вопросов по договору.
"""
import logging
import time
from textwrap import dedent

from llm.client import LLMClient
from schemas import ChatMessage
from utils.context import llm_step

logger = logging.getLogger(__name__)


class RAGQueryBuilder:
    """Перефразирование запросов для векторного поиска и генерация вопросов."""
    
    # Системный промпт для поискового запроса
    SEARCH_QUERY_SYSTEM = """Помощник для преобразования диалога в точный RAG-запрос. 
Учитывай юридический контекст. Игнорируй персональные данные. Только строка запроса."""
    
    # Системный промпт для генерации вопросов по договору
    CONTRACT_QUESTIONS_SYSTEM = """Юрист. Придумай 7 юридических вопросов для проверки договора. Один на строку."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    async def build_search_query(
        self,
        query: str,
        chat_history: list[ChatMessage],
    ) -> str:
        """
        Формирует оптимизированный поисковый запрос для RAG.
        """
        history_str = "\n".join(
            f"{msg.role}: {msg.content}" 
            for msg in chat_history[-5:]
        )
        
        user_prompt = dedent(f"""
            История диалога:
            {history_str}
            
            Текущий запрос:
            {query}
            
            Напиши один поисковый запрос для векторного поиска по юридическим документам.
            Запрос на русском, с ключевыми юридическими терминами, максимально релевантный.
        """)
        
        logger.debug(
            "RAGQueryBuilder.build_search_query: генерация",
            extra={"history_msgs": len(chat_history)},
        )
        
        t0 = time.monotonic()
        with llm_step("build_search_query"):
            response = await self.llm.chat_completion(
                messages=[
                    {"role": "system", "content": self.SEARCH_QUERY_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=200,
            )
        latency_ms = int((time.monotonic() - t0) * 1000)
        
        rag_query = response["choices"][0]["message"]["content"].strip()
        
        logger.debug(
            "RAGQueryBuilder.build_search_query: готово",
            extra={"rag_query_preview": rag_query[:100], "latency_ms": latency_ms},
        )
        return rag_query
    
    async def build_contract_questions(
        self,
        document_text: str,
        query: str,
        chat_history: list[ChatMessage],
    ) -> list[str]:
        """
        Генерирует до 7 юридически значимых вопросов по договору.
        """
        history_block = (
            "\n".join(f"{m.role}: {m.content}" for m in chat_history[-8:])
            if chat_history else ""
        )
        
        user_prompt = dedent(f"""
            История диалога:
            {history_block}
            
            Текущий запрос:
            {query}
            
            Текст договора:
            {document_text[:3000]}
            
            Сформулируй до 7 юридически значимых вопросов для анализа этого договора.
        """)
        
        logger.debug(
            "RAGQueryBuilder.build_contract_questions: генерация",
            extra={"doc_len": len(document_text), "history_msgs": len(chat_history)},
        )
        
        t0 = time.monotonic()
        with llm_step("build_contract_questions"):
            response = await self.llm.complete(
                system=self.CONTRACT_QUESTIONS_SYSTEM,
                user=user_prompt,
                max_tokens=500,
                temperature=0.3,
            )
        latency_ms = int((time.monotonic() - t0) * 1000)
        
        # Парсим вопросы
        questions = [
            line.strip(" \t1234567890.)- ")
            for line in response.strip().splitlines()
            if line.strip()
        ]
        
        logger.info(
            "RAGQueryBuilder.build_contract_questions: готово",
            extra={"questions_count": len(questions), "latency_ms": latency_ms},
        )
        return questions
