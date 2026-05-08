from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # DeepSeek API key
    deepseek_api_key: str = 'sk-'
    
    # Путь до ChromaDB
    db_path: str = "./rag_db"

    # Путь до atoms.jsonl для индексации (v2: перекрывающиеся чанки + article_full_text)
    atoms_jsonl: str = "../chunking/data/laws_v3.jsonl"

    # Модель эмбеддингов (SentenceTransformer)
    embed_model: str = "intfloat/e5-mistral-7b-instruct"
    embed_device: str = "cpu"

    # Сколько документов достаём из ChromaDB на первом шаге
    search_top_k: int = 3

    # Сколько атомов отдаём в LLM после всех фильтров
    search_final_k: int = 10

    # CrossEncoder для реранкинга (пустая строка — реранкинг отключён)
    rerank_model: str = "Qwen/Qwen3-Reranker-4B"
    rerank_device: str = 'cuda'
    rerank_batch_size: int = 10
    
    # Сколько атомов остаётся после реранкинга (перед LLM-валидацией)
    rerank_top_k: int = 10

    # Лог LLM-валидации релевантности
    validation_log: str = "../chunking/data/validation_log.jsonl"

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
