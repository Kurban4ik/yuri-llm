from llm.client import LLMClient


async def build_contract_prompt(
    document_text: str,
    rag_atoms: list[dict],
    llm_client: LLMClient,
) -> tuple[str, str, list[str]]:
    """
    Строит промпты для анализа договора.

    Двухпроходная логика:
      Проход 1 (rag_atoms=[]): LLM генерирует 7 юридических вопросов по договору.
      Проход 2 (rag_atoms заполнены): строится финальный промпт с нормами из RAG.

    Возвращает (system_prompt, user_prompt, questions).
    """
    # --- Проход 1: генерация юридических вопросов ---
    questions_raw = await llm_client.complete(
        system=(
            "Ты юрист. Придумай 7 юридических вопросов для проверки договора. "
            "Один вопрос на строку."
        ),
        user=document_text[:3000],
        max_tokens=500,
        temperature=0.3,
    )
    # Парсим ответ: убираем нумерацию и лишние символы
    questions: list[str] = [
        line.strip(" \t1234567890.)- ")
        for line in questions_raw.strip().splitlines()
        if line.strip()
    ]

    # --- Проход 2: финальный промпт с нормами ---
    # v2: предпочитаем article_full_text (вся статья) вместо clause_text (один пункт).
    norms_block = "\n\n".join(
        f"[{a['metadata'].get('act_name', '')}, ст.{a['metadata'].get('article_number', '')}] "
        f"{a['metadata'].get('article_full_text') or a['metadata'].get('clause_text', '')}"
        for a in rag_atoms
    )

    numbered_questions = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))

    system_prompt = (
        "Ты опытный юрист-эксперт. "
        "Анализируй строго по российскому законодательству. "
        "Используй только предоставленные нормы. "
        "Отвечай на русском, конкретно."
    )

    user_prompt = (
        f"НОРМЫ ПРАВА:\n{norms_block}\n\n"
        f"ДОГОВОР:\n{document_text[:6000]}\n\n"
        f"ПРОВЕРЬ по вопросам:\n{numbered_questions}\n\n"
        "Для каждого риска: уровень (низкий/средний/высокий/критический), "
        "пункт договора, норму закона, рекомендацию."
    )

    return system_prompt, user_prompt, questions
