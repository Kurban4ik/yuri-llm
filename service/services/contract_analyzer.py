# services/contract_analyzer.py
"""
Построение финального промпта для анализа договора.
"""
import logging
import time

from llm.client import LLMClient
from utils.context import llm_step

logger = logging.getLogger(__name__)


class ContractAnalyzer:
    """Построение финального промпта для анализа договора с двухпроходной логикой."""
    
    # Системный промпт для анализа
    SYSTEM_PROMPT = """Опытный юрист-эксперт. 
Анализируй строго по российскому законодательству. Используй только предоставленные нормы. 
Отвечай на русском, конкретно."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    async def build_prompt(
        self,
        document_text: str,
        rag_atoms: list[dict],
        questions: list[str],
    ) -> tuple[str, str]:
        """
        Строит финальные промпты для анализа договора.
        
        Args:
            document_text: Текст договора
            rag_atoms: Нормы из RAG
            questions: Юридические вопросы (сгенерированы RAGQueryBuilder)
        
        Returns:
            (system_prompt, user_prompt)
        """
        logger.debug(
            "ContractAnalyzer.build_prompt: построение",
            extra={
                "doc_len": len(document_text),
                "atoms_count": len(rag_atoms),
                "questions_count": len(questions),
            },
        )
        
        # Формируем блок норм (предпочитаем article_full_text)
        norms_block = "\n\n".join(
            f"[{a['metadata'].get('act_name', '')}, ст.{a['metadata'].get('article_number', '')}] "
            f"{a['metadata'].get('article_full_text') or a['metadata'].get('clause_text', '')}"
            for a in rag_atoms
        )
        
        # Формируем пронумерованные вопросы
        numbered_questions = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))
        
        user_prompt = (
            f"НОРМЫ ПРАВА:\n{norms_block}\n\n"
            f"ДОГОВОР:\n{document_text[:6000]}\n\n"
            f"ПРОВЕРЬ по вопросам:\n{numbered_questions}\n\n"
            "Для каждого риска: уровень (низкий/средний/высокий/критический), "
            "пункт договора, норму закона, рекомендацию."
        )
        
        logger.debug(
            "ContractAnalyzer.build_prompt: готово",
            extra={
                "sys_prompt_len": len(self.SYSTEM_PROMPT),
                "user_prompt_len": len(user_prompt),
            },
        )
        
        return self.SYSTEM_PROMPT, user_prompt
