from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field, validator


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class EventType(str, Enum):
    INQUIRY = "INQUIRY"
    FEEDBACK = "FEEDBACK"  
    REQUEST = "REQUEST"
    COMPLAINT = "COMPLAINT"
    TRANSACTION = "TRANSACTION"
    SUPPORT = "SUPPORT"
    INFORMATION = "INFORMATION"
    GENERIC_EVENT = "GENERIC_EVENT"


class EventClassification(BaseModel):
    event_type: EventType
    importance_score: float = Field(ge=0.0, le=1.0)
    intent: str = ""
    reasoning: str = ""


class Event(BaseModel):
    event_type: EventType
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    classification: EventClassification
    context: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def importance_score(self) -> float:
        return self.classification.importance_score
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Conversation(BaseModel):
    user_id: str
    messages: List[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def add_message(self, message: Message) -> None:
        self.messages.append(message)
        self.updated_at = datetime.now(timezone.utc)
    
    def get_recent_messages(self, limit: int = 10) -> List[Message]:
        return self.messages[-limit:] if self.messages else []
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class LongTermMemory(BaseModel):
    user_id: str
    events: List[Event] = Field(default_factory=list)
    summary: str = ""
    context: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    def add_event(self, event: Event) -> None:
        self.events.append(event)
        self.updated_at = datetime.now(timezone.utc)
    
    def get_important_events(self, threshold: float = 0.7) -> List[Event]:
        return [event for event in self.events if event.importance_score >= threshold]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class LLMModelConfig(BaseModel):
    model: str
    temperature: float = Field(ge=0.0, le=2.0, default=0.7)
    max_tokens: Optional[int] = Field(ge=1, le=8192, default=None)


class OpenRouterConfig(BaseModel):
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    classification: LLMModelConfig  # LLM for event classification
    response: LLMModelConfig        # LLM for chat responses


class MemoryConfig(BaseModel):
    redis_url: str
    sm_ttl: int = Field(ge=60, default=1800)  # 30 minutes default
    lm_base_path: str = "data/longterm"


class Config(BaseModel):
    openrouter: OpenRouterConfig
    memory: MemoryConfig
    
    @validator('memory')
    def validate_memory_config(cls, v):
        if not v.redis_url:
            raise ValueError("Redis URL is required")
        return v
