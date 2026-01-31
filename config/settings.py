from pydantic_settings import BaseSettings, SettingsConfigDict

class DBSettings(BaseSettings):

    MONGO_DB: str = "llm_project"
    MONGO_COL_PAGE: str = "financial_reports"
    MONGO_COL_EVAL: str = "evaluation_store"
    QDRANT_EMBED: str = "financial_reports"
    BM25_INDEX_NAME: str = "text_content_bm25_index"

class GenAISettings(BaseSettings):
    EMBED_MODEL: str = "gemini-embedding-001"
    EMBED_SIZE: int = 768
    GEN_OCR: str = "gemini-2.5-flash"
    GEN_CHAT: str = "gemini-2.5-flash"
    GEN_CHAT_LITE: str = "gemini-2.5-flash-lite"
    GEN_JUDGE: str = "gemini-2.5-flash"
    GEN_RETRIEVAL: str = "gemini-2.5-flash"

class Settings(BaseSettings):
    """The Root Settings Object"""

    db: DBSettings = DBSettings()
    ai: GenAISettings = GenAISettings()

    MONGODB_HOST: str 
    QDRANT_CLOUD_URL: str
    QDRANT_APIKEY: str  
    
    USE_QDRANT_CLOUD: bool = False 

    HF_APIKEY: str
    GEMINI_APIKEY: str

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        case_sensitive=True,
        extra="ignore" 
    )


settings = Settings()