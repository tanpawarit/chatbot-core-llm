import yaml
from pathlib import Path
from typing import Optional
from src.models import Config, OpenRouterConfig, MemoryConfig, LLMModelConfig


class ConfigManager:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config: Optional[Config] = None
    
    def load_config(self) -> Config:
        if self._config is None:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Parse LLM model configs
            classification_config = LLMModelConfig(**config_data['openrouter']['classification'])
            response_config = LLMModelConfig(**config_data['openrouter']['response'])
            
            openrouter_config = OpenRouterConfig(
                api_key=config_data['openrouter']['api_key'],
                base_url=config_data['openrouter']['base_url'],
                classification=classification_config,
                response=response_config
            )
            
            memory_config = MemoryConfig(**config_data['memory'])
            
            self._config = Config(
                openrouter=openrouter_config,
                memory=memory_config
            )
        
        return self._config
    
    def get_openrouter_config(self) -> OpenRouterConfig:
        return self.load_config().openrouter
    
    def get_memory_config(self) -> MemoryConfig:
        return self.load_config().memory


# Global config instance
config_manager = ConfigManager()