"""
LLM Factory - Centralized LLM instance creation and management
Eliminates duplicate LLM initialization across nodes
"""

from typing import Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.utils import convert_to_secret_str

from src.config import config_manager
from src.models import OpenRouterConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


class LLMFactory:
    """
    Factory class for creating and caching LLM instances.
    Eliminates duplicate initialization and provides centralized configuration.
    """
    
    def __init__(self):
        self._instances: Dict[str, ChatOpenAI] = {}
        self._config: Optional[OpenRouterConfig] = None
    
    def _get_config(self) -> OpenRouterConfig:
        """Get OpenRouter configuration (cached)"""
        if self._config is None:
            self._config = config_manager.get_openrouter_config()
        return self._config
    
    def _create_instance_key(self, model: str, temperature: float) -> str:
        """Create unique key for LLM instance"""
        return f"{model}_{temperature}"
    
    def get_classification_llm(self) -> ChatOpenAI:
        """
        Get or create classification LLM instance
        
        Returns:
            ChatOpenAI instance configured for classification tasks
        """
        config = self._get_config()
        key = self._create_instance_key(
            config.classification.model, 
            config.classification.temperature
        )
        
        if key not in self._instances:
            logger.info("Creating new classification LLM instance", 
                       model=config.classification.model,
                       temperature=config.classification.temperature)
            
            self._instances[key] = ChatOpenAI(
                model=config.classification.model,
                api_key=convert_to_secret_str(config.api_key),
                base_url=config.base_url,
                temperature=config.classification.temperature,
                request_timeout=15.0,  # Reduce to 15 seconds
                max_retries=0,         # No retries for faster response
                timeout=15.0,          # OpenAI client timeout
                http_client=None,      # Use default HTTP client with keep-alive
            )
        
        return self._instances[key]
    
    def get_response_llm(self) -> ChatOpenAI:
        """
        Get or create response generation LLM instance
        
        Returns:
            ChatOpenAI instance configured for response generation
        """
        config = self._get_config()
        key = self._create_instance_key(
            config.response.model, 
            config.response.temperature
        )
        
        if key not in self._instances:
            logger.info("Creating new response LLM instance", 
                       model=config.response.model,
                       temperature=config.response.temperature)
            
            self._instances[key] = ChatOpenAI(
                model=config.response.model,
                api_key=convert_to_secret_str(config.api_key),
                base_url=config.base_url,
                temperature=config.response.temperature,
                request_timeout=15.0,  # Reduce to 15 seconds
                max_retries=0,         # No retries for faster response
                timeout=15.0,          # OpenAI client timeout
                http_client=None,      # Use default HTTP client with keep-alive
            )
        
        return self._instances[key]
    
    def get_custom_llm(self, model: str, temperature: float = 0.7) -> ChatOpenAI:
        """
        Get or create custom LLM instance with specific parameters
        
        Args:
            model: Model name
            temperature: Temperature setting
            
        Returns:
            ChatOpenAI instance with custom configuration
        """
        config = self._get_config()
        key = self._create_instance_key(model, temperature)
        
        if key not in self._instances:
            logger.info("Creating new custom LLM instance", 
                       model=model,
                       temperature=temperature)
            
            self._instances[key] = ChatOpenAI(
                model=model,
                api_key=convert_to_secret_str(config.api_key),
                base_url=config.base_url,
                temperature=temperature,
                request_timeout=30.0,  # 30 second timeout
                max_retries=1,         # Reduce retries for faster failure
                timeout=30.0,          # OpenAI client timeout
            )
        
        return self._instances[key]
    
    def clear_cache(self):
        """Clear all cached LLM instances"""
        logger.info("Clearing LLM instance cache", instances_count=len(self._instances))
        self._instances.clear()
        self._config = None
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            "cached_instances": len(self._instances),
            "instance_keys": list(self._instances.keys())
        }


# Global factory instance
llm_factory = LLMFactory()