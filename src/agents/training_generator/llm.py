"""
LLM initialization module for Azure OpenAI

This module handles the initialization of Azure OpenAI models
for both chat completion and structured output generation.
"""

from langchain_openai import AzureChatOpenAI
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel
from typing import Type, TypeVar

from .config import config

# Type variable for Pydantic models
T = TypeVar('T', bound=BaseModel)


def get_llm(temperature: float = None, max_tokens: int = None) -> AzureChatOpenAI:
    """
    Get Azure OpenAI chat model instance.
    
    Args:
        temperature: Temperature for generation (0.0 to 1.0)
                    Defaults to config.LLM_TEMPERATURE
        max_tokens: Maximum tokens to generate
                   Defaults to config.LLM_MAX_TOKENS
    
    Returns:
        AzureChatOpenAI instance
    """
    return AzureChatOpenAI(
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_API_KEY,
        api_version=config.OPENAI_API_VERSION,
        azure_deployment=config.AZURE_OPENAI_DEPLOYMENT_NAME,
        temperature=temperature or config.LLM_TEMPERATURE,
        max_tokens=max_tokens or config.LLM_MAX_TOKENS,
    )


def get_structured_llm(
    pydantic_model: Type[T],
    temperature: float = None,
    max_tokens: int = None
) -> BaseChatModel:
    """
    Get Azure OpenAI chat model with structured output.
    
    This ensures the LLM returns data conforming to the Pydantic model.
    
    Args:
        pydantic_model: Pydantic model class for structured output
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
    
    Returns:
        AzureChatOpenAI instance with structured output
    
    Example:
        >>> from models import PlannerDecision
        >>> llm = get_structured_llm(PlannerDecision)
        >>> result = llm.invoke("What should I do next?")
        >>> # result is a PlannerDecision instance
    """
    llm = get_llm(temperature=temperature, max_tokens=max_tokens)
    return llm.with_structured_output(pydantic_model)


# Create default LLM instances for convenience
default_llm = get_llm()


if __name__ == "__main__":
    # Test LLM initialization
    print("Testing Azure OpenAI LLM initialization...")
    
    try:
        llm = get_llm()
        print("✅ LLM initialized successfully!")
        print(f"  Model: {config.AZURE_OPENAI_DEPLOYMENT_NAME}")
        print(f"  Endpoint: {config.AZURE_OPENAI_ENDPOINT}")
        print(f"  Temperature: {config.LLM_TEMPERATURE}")
        print(f"  Max Tokens: {config.LLM_MAX_TOKENS}")
        
        # Test a simple invocation
        print("\n🧪 Testing LLM invocation...")
        response = llm.invoke("Say 'Hello, I am working!' and nothing else.")
        print(f"✅ Response: {response.content}")
        
    except Exception as e:
        print(f"❌ Error initializing LLM: {str(e)}")
        print("\nPlease check your .env file configuration.")