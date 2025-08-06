"""
LLM Factory - Centralized LLM instance creation and management
Eliminates duplicate LLM initialization across nodes
"""

import time
from typing import Dict, Optional
import warnings
# Suppress langchain warnings first
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core.globals")

from langchain_openai import ChatOpenAI
from langchain_core.utils import convert_to_secret_str
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import config_manager
from src.models import OpenRouterConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


class LLMFactory:
    """
    Factory class for creating and caching LLM instances.
    Eliminates duplicate initialization and provides centralized configuration.
    Includes warmup functionality to solve cold start issues.
    """
    
    def __init__(self):
        self._instances: Dict[str, ChatOpenAI] = {}
        self._config: Optional[OpenRouterConfig] = None
        self._warmup_status: Dict[str, bool] = {}  # Track warmup status per model
    
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
                max_retries=0,         # No retries for faster response
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
                max_retries=0,         # No retries for faster response
                http_client=None,      # Use default HTTP client with keep-alive
            )
        
        return self._instances[key]
    
    def warmup_response_llm(self) -> bool:
        """
        Warmup Response LLM to eliminate cold start delays
        
        Returns:
            bool: True if warmup successful, False otherwise
        """
        try:
            logger.info("Starting Response LLM warmup...")
            start_time = time.time()
            
            # Get LLM instance (creates if not exists)
            llm = self.get_response_llm()
            config = self._get_config()
            key = self._create_instance_key(config.response.model, config.response.temperature)
            
            # Skip if already warmed up
            if self._warmup_status.get(key, False):
                logger.info("Response LLM already warmed up", model=config.response.model)
                return True
            
            # Minimal warmup messages - Thai computer store context
            warmup_messages = [
                SystemMessage(content="You are a Thai computer store assistant."),
                HumanMessage(content="สวัสดี")
            ]
            
            # Send warmup request
            response = llm.invoke(warmup_messages)
            
            warmup_time = (time.time() - start_time) * 1000
            self._warmup_status[key] = True
            
            logger.info("Response LLM warmup completed", 
                       model=config.response.model,
                       warmup_time_ms=round(warmup_time, 2),
                       response_length=len(response.content) if response.content else 0)
            
            print(f"🔥 Response LLM warmed up in {warmup_time:.1f}ms")
            return True
            
        except Exception as e:
            logger.error("Response LLM warmup failed", error=str(e))
            print(f"❌ Response LLM warmup failed: {str(e)}")
            return False
    
    def warmup_classification_llm(self) -> bool:
        """
        Warmup Classification (NLU) LLM to eliminate cold start delays
        
        Returns:
            bool: True if warmup successful, False otherwise
        """
        try:
            logger.info("Starting Classification LLM warmup...")
            start_time = time.time()
            
            # Get LLM instance (creates if not exists)
            llm = self.get_classification_llm()
            config = self._get_config()
            key = self._create_instance_key(config.classification.model, config.classification.temperature)
            
            # Skip if already warmed up
            if self._warmup_status.get(key, False):
                logger.info("Classification LLM already warmed up", model=config.classification.model)
                return True
            
            # Minimal warmup message for NLU
            warmup_messages = [
                HumanMessage(content="test")
            ]
            
            # Send warmup request
            response = llm.invoke(warmup_messages)
            
            warmup_time = (time.time() - start_time) * 1000
            self._warmup_status[key] = True
            
            logger.info("Classification LLM warmup completed", 
                       model=config.classification.model,
                       warmup_time_ms=round(warmup_time, 2),
                       response_length=len(response.content) if response.content else 0)
            
            print(f"🔥 Classification LLM warmed up in {warmup_time:.1f}ms")
            return True
            
        except Exception as e:
            logger.error("Classification LLM warmup failed", error=str(e))
            print(f"❌ Classification LLM warmup failed: {str(e)}")
            return False
    
    def warmup_all_llms(self) -> bool:
        """
        Warmup all LLM models in parallel for maximum efficiency
        
        Returns:
            bool: True if all warmups successful, False otherwise
        """
        logger.info("Starting parallel LLM warmup...")
        print("🔥 Warming up LLM models...")
        
        start_time = time.time()
        
        # Run warmups in parallel using threads with timeout
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both warmup tasks
            future_response = executor.submit(self.warmup_response_llm)
            future_classification = executor.submit(self.warmup_classification_llm)
            
            # Wait for both to complete with timeout
            try:
                response_success = future_response.result(timeout=30)  # 30s timeout
                classification_success = future_classification.result(timeout=30)  # 30s timeout
            except concurrent.futures.TimeoutError:
                logger.error("LLM warmup timed out after 30 seconds")
                print("⏰ LLM warmup timed out - continuing anyway...")
                return False
        
        total_time = (time.time() - start_time) * 1000
        success = response_success and classification_success
        
        if success:
            logger.info("All LLM warmups completed successfully", total_time_ms=round(total_time, 2))
            print(f"✅ All LLMs warmed up successfully in {total_time:.1f}ms")
        else:
            logger.warning("Some LLM warmups failed", total_time_ms=round(total_time, 2))
            print(f"⚠️  Some LLM warmups failed (total time: {total_time:.1f}ms)")
        
        return success
    
    def is_warmed_up(self, llm_type: str = "all") -> bool:
        """
        Check if LLM(s) are warmed up
        
        Args:
            llm_type: "response", "classification", or "all"
            
        Returns:
            bool: True if specified LLM(s) are warmed up
        """
        config = self._get_config()
        
        if llm_type == "response":
            key = self._create_instance_key(config.response.model, config.response.temperature)
            return self._warmup_status.get(key, False)
        elif llm_type == "classification":
            key = self._create_instance_key(config.classification.model, config.classification.temperature)
            return self._warmup_status.get(key, False)
        elif llm_type == "all":
            response_key = self._create_instance_key(config.response.model, config.response.temperature)
            classification_key = self._create_instance_key(config.classification.model, config.classification.temperature)
            return (self._warmup_status.get(response_key, False) and 
                   self._warmup_status.get(classification_key, False))
        else:
            raise ValueError(f"Invalid llm_type: {llm_type}")
    
    def get_warmup_status(self) -> Dict[str, bool]:
        """Get current warmup status for all models"""
        return self._warmup_status.copy()


# Global factory instance
llm_factory = LLMFactory()