import yaml
from pathlib import Path
from typing import Optional
from src.models import Config, OpenRouterConfig, MemoryConfig, LLMModelConfig, NLUConfig
from .env_loader import env_loader
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ConfigManager:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config: Optional[Config] = None
    
    def load_config(self) -> Config:
        if self._config is None:
            # Try environment variables first, then fallback to YAML
            try:
                self._config = self._load_from_environment()
                logger.info("Configuration loaded from environment variables")
            except Exception as env_error:
                logger.info("Environment configuration failed, falling back to YAML", error=str(env_error))
                self._config = self._load_from_yaml()
                logger.info("Configuration loaded from YAML file")
        
        return self._config
    
    def _load_from_environment(self) -> Config:
        """Load configuration from environment variables"""
        # Validate required credentials
        env_loader.validate_required_credentials([
            'OPENROUTER_API_KEY',
            'REDIS_URL'
        ])
        
        # Load OpenRouter config from environment
        classification_config = LLMModelConfig(
            model=env_loader.get_str('CLASSIFICATION_MODEL', 'google/gemini-2.5-flash-lite'),
            temperature=env_loader.get_float('CLASSIFICATION_TEMPERATURE', 0.1),
            max_tokens=env_loader.get_int('CLASSIFICATION_MAX_TOKENS', 500)
        )
        
        response_config = LLMModelConfig(
            model=env_loader.get_str('RESPONSE_MODEL', 'google/gemini-2.5-flash-lite'),
            temperature=env_loader.get_float('RESPONSE_TEMPERATURE', 0.7),
            max_tokens=env_loader.get_int('RESPONSE_MAX_TOKENS', 2000)
        )
        
        openrouter_config = OpenRouterConfig(
            api_key=env_loader.get_str('OPENROUTER_API_KEY', required=True),
            base_url=env_loader.get_str('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1'),
            classification=classification_config,
            response=response_config
        )
        
        # Load Memory config from environment
        memory_config = MemoryConfig(
            redis_url=env_loader.get_str('REDIS_URL', required=True),
            sm_ttl=env_loader.get_int('SM_TTL', 240),
            lm_base_path=env_loader.get_str('LM_BASE_PATH', 'data/longterm')
        )
        
        # Load NLU config from environment (with defaults)
        nlu_config = NLUConfig(
            default_intent=env_loader.get_str('NLU_DEFAULT_INTENT', "purchase_intent:0.8, inquiry_intent:0.7, support_intent:0.6, complain_intent:0.6, past_purchase:0.7"),
            additional_intent=env_loader.get_str('NLU_ADDITIONAL_INTENT', "greet:0.3, complaint:0.5, cancel_order:0.4, ask_price:0.6, compare_product:0.5"),
            default_entity=env_loader.get_str('NLU_DEFAULT_ENTITY', "product, quantity, brand, price"),
            additional_entity=env_loader.get_str('NLU_ADDITIONAL_ENTITY', "color, model, spec, budget, warranty, delivery"),
            tuple_delimiter=env_loader.get_str('NLU_TUPLE_DELIMITER', "<||>"),
            record_delimiter=env_loader.get_str('NLU_RECORD_DELIMITER', "##"),
            completion_delimiter=env_loader.get_str('NLU_COMPLETION_DELIMITER', "<|COMPLETE|>"),
            enable_robust_parsing=env_loader.get_bool('NLU_ENABLE_ROBUST_PARSING', True),
            fallback_to_simple_json=env_loader.get_bool('NLU_FALLBACK_TO_SIMPLE_JSON', True),
            importance_threshold=env_loader.get_float('NLU_IMPORTANCE_THRESHOLD', 0.7)
        )
        
        return Config(
            openrouter=openrouter_config,
            memory=memory_config,
            nlu=nlu_config
        )
    
    def _load_from_yaml(self) -> Config:
        """Load configuration from YAML file (fallback method)"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # Parse LLM model configs
        classification_config = LLMModelConfig(**config_data['openrouter']['classification'])
        response_config = LLMModelConfig(**config_data['openrouter']['response'])
        
        openrouter_config = OpenRouterConfig(
            api_key=env_loader.get_str('OPENROUTER_API_KEY', required=True),
            base_url=env_loader.get_str('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1'),
            classification=classification_config,
            response=response_config
        )
        
        memory_config = MemoryConfig(
            redis_url=env_loader.get_str('REDIS_URL', required=True),
            sm_ttl=config_data['memory'].get('sm_ttl', 240),
            lm_base_path=config_data['memory'].get('lm_base_path', 'data/longterm')
        )
        nlu_config = NLUConfig(**config_data['nlu'])
        
        return Config(
            openrouter=openrouter_config,
            memory=memory_config,
            nlu=nlu_config
        )
    
    def get_openrouter_config(self) -> OpenRouterConfig:
        return self.load_config().openrouter
    
    def get_memory_config(self) -> MemoryConfig:
        return self.load_config().memory
    
    def get_nlu_config(self) -> NLUConfig:
        return self.load_config().nlu
    
    def get_config(self) -> Config:
        return self.load_config()
    
    def is_using_environment_config(self) -> bool:
        """Check if configuration is loaded from environment variables"""
        return env_loader.has_credential('OPENROUTER_API_KEY') and env_loader.has_credential('REDIS_URL')


# Global config instance
config_manager = ConfigManager()