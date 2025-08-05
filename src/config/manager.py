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
            # Use YAML as primary, environment for API keys only
            try:
                self._config = self._load_from_yaml()
                logger.info("Configuration loaded from YAML file")
            except Exception as yaml_error:
                logger.info("YAML configuration failed, falling back to environment", error=str(yaml_error))
                self._config = self._load_from_environment()
                logger.info("Configuration loaded from environment variables")
        
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
        
        # Load NLU config from YAML file (NLU should be dynamic, not hardcoded in ENV)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file required for NLU configuration: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        nlu_config = NLUConfig(**config_data['nlu'])
        
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