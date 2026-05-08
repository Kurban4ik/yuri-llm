# Ключевые слова, указывающие что запрос касается договора
CONTRACT_KEYWORDS = ["договор", "контракт", "соглашение", "оферт"]


def detect_query_type(query: str, document_text: str | None) -> str:
    # TODO: придумать как определяем детект куэри

    return "general"
