"""
prompts/document_generation.py
Промпты для генерации юридических документов.
"""

def build_document_generation_prompt(
    query: str,
    rag_atoms: list[dict],
    structure: dict,
    collected_data: dict,
    chat_history: list[dict] = None,
) -> tuple[str, str]:
    """Возвращает (system_prompt, user_prompt) для генерации документа."""

    # Блок нормативных источников
    norms_parts = []
    for a in rag_atoms:
        meta = a["metadata"]
        act = meta.get("act_name", "")
        art = meta.get("article_number", "")
        cl = meta.get("clause_number", "")
        text = meta.get("article_full_text") or meta.get("clause_text", "")
        norms_parts.append(f"[{act}, ст.{art}, п.{cl}]\n{text}")
    norms_block = "\n\n".join(norms_parts)

    # Историческая справка (последние 5 сообщений)
    history_lines = []
    if chat_history:
        for msg in chat_history[-5:]:
            role = msg.role.capitalize()
            content = msg.content
            history_lines.append(f"{role}: {content}")
    history_block = "ИСТОРИЯ ДИАЛОГА:\n" + "\n".join(history_lines) + "\n\n" if history_lines else ""

    # Структура документа и данные
    structure_str = f"Тип документа: {structure.get('document_title', 'Юридический документ')}\n"
    structure_str += "Структура (разделы):\n" + "\n".join(f"- {s}" for s in structure["sections"]) + "\n"
    structure_str += "Доступные данные для заполнения:\n" + "\n".join(f"- {k}: {v}" for k, v in collected_data.items())

    system_prompt = (
        "Ты — квалифицированный юрист, специализирующийся на составлении юридических документов. "
        "Твоя задача — на основе предоставленной структуры, данных пользователя и норм права "
        "составить полный, юридически грамотный текст документа."
    )

    user_prompt = (
        f"{history_block}"
        f"НОРМЫ ПРАВА, РЕГУЛИРУЮЩИЕ ФОРМУ И СОДЕРЖАНИЕ ДОКУМЕНТА:\n{norms_block}\n\n"
        f"ОПИСАНИЕ ДОКУМЕНТА И ДАННЫЕ:\n{structure_str}\n\n"
        f"ПОЖЕЛАНИЯ ПОЛЬЗОВАТЕЛЯ: {query}\n\n"
        "Напиши полный текст документа, соблюдая структуру и используя все предоставленные данные. "
        "Если для какого-то раздела информации не хватает, используй стандартные формулировки, "
        "но отметь это в сноске. Ссылайся на конкретные нормы права в обосновании."
    )

    return system_prompt, user_prompt