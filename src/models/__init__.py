# Re-export all models for backward compatibility
from src.models.base import MessageRole, MediaType
from src.models.message import MediaContent, Message
from src.models.nlu import (
    NLUIntent, 
    NLUEntity, 
    NLULanguage, 
    NLUSentiment, 
    NLUResult,
    ImportanceScoringConfig,
    NLUConfig
)
from src.models.conversation import Conversation, LongTermMemory
from src.models.config import (
    LLMModelConfig,
    OpenRouterConfig,
    MemoryConfig,
    Config
)

# Make all models available at package level
__all__ = [
    # Base
    'MessageRole',
    'MediaType',
    
    # Message
    'MediaContent',
    'Message',
    
    # NLU
    'NLUIntent',
    'NLUEntity', 
    'NLULanguage',
    'NLUSentiment',
    'NLUResult',
    'ImportanceScoringConfig',
    'NLUConfig',
    
    # Conversation
    'Conversation',
    'LongTermMemory',
    
    # Config
    'LLMModelConfig',
    'OpenRouterConfig',
    'MemoryConfig',
    'Config'
]