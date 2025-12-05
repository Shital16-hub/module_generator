"""
Configuration module for Training Generator Agent
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Central configuration"""
    
    # Azure OpenAI
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_DEPLOYMENT_NAME: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1-mini")
    OPENAI_API_VERSION: str = os.getenv("OPENAI_API_VERSION", "2024-10-01-preview")
    
    # Qdrant
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY") or None
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "tasconnect_knowledge_base")
    
    # Neo4j
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "")
    
    # Model config
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gpt-4.1-mini")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))
    
    # Embeddings
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    EMBEDDING_MODEL_TYPE: str = os.getenv("EMBEDDING_MODEL_TYPE", "sentence-transformers")
    
    # Agent config
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "8"))
    SEARCH_TOP_K: int = int(os.getenv("SEARCH_TOP_K", "10"))
    MIN_RELEVANCE_SCORE: float = float(os.getenv("MIN_RELEVANCE_SCORE", "0.7"))
    
    # LangSmith (optional)
    LANGSMITH_API_KEY: Optional[str] = os.getenv("LANGSMITH_API_KEY") or None
    LANGSMITH_TRACING: bool = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "training_generator_agent")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def validate(cls) -> tuple[bool, list[str]]:
        """Validate configuration"""
        errors = []
        
        if not cls.AZURE_OPENAI_API_KEY:
            errors.append("AZURE_OPENAI_API_KEY is required")
        
        if not cls.AZURE_OPENAI_ENDPOINT:
            errors.append("AZURE_OPENAI_ENDPOINT is required")
        
        return len(errors) == 0, errors


# Singleton instance
config = Config()