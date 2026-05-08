"prompts/document_generation.py"
def build_general_prompt(
    query: str,
    rag_atoms: list[dict],
    chat_history: list[dict] = None,   # добавлен параметр, по умолчанию None
) -> tuple[str, str]:
    """
    Строит промпты для ответа на общий юридический вопрос.
    Возвращает (system_prompt, user_prompt).

    v2: использует article_full_text вместо clause_text там, где он доступен
    (small-to-big retrieval: поиск по чанку, в LLM идёт вся статья целиком).

    v3: добавлена поддержка chat_history для учёта контекста диалога.
    """
    # Формируем блок найденных норм (без изменений)
    norms_block_parts = []
    for a in rag_atoms:
        meta = a["metadata"]
        act_name     = meta.get("act_name", "")
        article_num  = meta.get("article_number", "")
        clause_num   = meta.get("clause_number", "")
        display_text = meta.get("article_full_text") or meta.get("clause_text", "")
        norms_block_parts.append(
            f"[{act_name}, ст.{article_num}, п.{clause_num}]\n{display_text}"
        )
    norms_block = "\n\n".join(norms_block_parts)

    # Подготовка исторического контекста, если он передан
    history_block = ""
    if chat_history and isinstance(chat_history, list):
        # Берём последние 5 сообщений для контекста, чтобы не перегружать промпт
        recent_history = chat_history[-5:]
        history_lines = []
        for msg in recent_history:
            role = msg.role.capitalize()
            content = msg.content
            history_lines.append(f"{role}: {content}")
        if history_lines:
            history_block = "ИСТОРИЯ ДИАЛОГА:\n" + "\n".join(history_lines) + "\n\n"

    system_prompt = (
        "Ты юридический консультант. "
        "Отвечай точно, со ссылками на нормы российского права. "
        "Если норм недостаточно — говори об этом явно. "
        "Учитывай историю диалога, если она есть."
    )

    user_prompt = (
        f"{history_block}"
        f"НОРМЫ ПРАВА:\n{norms_block}\n\n"
        f"ВОПРОС: {query}\n\n"
        "Дай развёрнутый ответ со ссылками на конкретные нормы."
    )

    return system_prompt, user_prompt