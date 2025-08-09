from typing import Optional
from pydantic import BaseModel, Field, field_validator


class LLMModelConfig(BaseModel):
    model: str
    temperature: float = Field(ge=0.0, le=2.0, default=0.7)
    max_tokens: Optional[int] = Field(ge=1, le=8192, default=None)


class NLUConfig(BaseModel):
    default_intent: str
    additional_intent: str
    default_entity: str
    additional_entity: str
    tuple_delimiter: str = "<||>"
    record_delimiter: str = "##"
    completion_delimiter: str = "<|COMPLETE|>"
    enable_robust_parsing: bool = True
    importance_threshold: float = Field(ge=0.0, le=1.0, default=0.7)


class OpenRouterConfig(BaseModel):
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    classification: LLMModelConfig  # LLM for NLU analysis (was event classification)
    response: LLMModelConfig        # LLM for chat responses


class MemoryConfig(BaseModel):
    redis_url: str
    sm_ttl: int = Field(ge=60, default=1800)  # 30 minutes default
    lm_base_path: str = "data/longterm"
    extend_ttl_on_activity: bool = True  # Extend TTL when users send messages


class Config(BaseModel):
    openrouter: OpenRouterConfig
    memory: MemoryConfig
    nlu: NLUConfig
    
    @field_validator('memory')
    @classmethod
    def validate_memory_config(cls, v):
        if not v.redis_url:
            raise ValueError("Redis URL is required")
        return v
    
    @field_validator('nlu')
    @classmethod
    def validate_nlu_config(cls, v):
        if not v.default_intent or not v.additional_intent:
            raise ValueError("Intent configurations are required")
        if not v.default_entity or not v.additional_entity:
            raise ValueError("Entity configurations are required")
        return v