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


# NLU Models for Natural Language Understanding

class NLUIntent(BaseModel):
    name: str
    confidence: float = Field(ge=0.0, le=1.0)
    priority_score: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NLUEntity(BaseModel):
    type: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NLULanguage(BaseModel):
    code: str = Field(min_length=3, max_length=3)  # ISO 3166 Alpha-3
    confidence: float = Field(ge=0.0, le=1.0)
    is_primary: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NLUSentiment(BaseModel):
    label: str = Field(pattern="^(positive|negative|neutral)$")
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NLUResult(BaseModel):
    content: str  # Original message content
    intents: List[NLUIntent] = Field(default_factory=list)
    entities: List[NLUEntity] = Field(default_factory=list)
    languages: List[NLULanguage] = Field(default_factory=list)
    sentiment: Optional[NLUSentiment] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    parsing_metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def importance_score(self) -> float:
        """Calculate importance score from NLU analysis."""
        try:
            importance = 0.0
            
            # Intent contribution (60% weight)
            if self.intents:
                top_intent = max(self.intents, key=lambda x: x.confidence)
                intent_weight = top_intent.confidence * top_intent.priority_score
                importance += intent_weight * 0.6
            
            # Entity contribution (25% weight)
            if self.entities:
                entity_confidence_avg = sum(e.confidence for e in self.entities) / len(self.entities)
                entity_count_bonus = min(len(self.entities) * 0.1, 0.3)  # Max 0.3 bonus
                importance += (entity_confidence_avg + entity_count_bonus) * 0.25
            
            # Sentiment contribution (15% weight)
            if self.sentiment:
                sentiment_weight = self.sentiment.confidence
                if self.sentiment.label in ["positive", "negative"]:  # Strong emotions get higher weight
                    sentiment_weight *= 1.2
                importance += sentiment_weight * 0.15
            
            return max(0.0, min(1.0, importance))
            
        except Exception:
            return 0.5  # Default fallback
    
    @property
    def primary_intent(self) -> Optional[str]:
        """Get the highest confidence intent name."""
        if self.intents:
            return max(self.intents, key=lambda x: x.confidence).name
        return None
    
    @property
    def primary_language(self) -> Optional[str]:
        """Get the primary language code."""
        primary = next((l for l in self.languages if l.is_primary), None)
        return primary.code if primary else None
    
    @property
    def extracted_entities(self) -> Dict[str, List[str]]:
        """Get entities grouped by type."""
        grouped = {}
        for entity in self.entities:
            if entity.type not in grouped:
                grouped[entity.type] = []
            grouped[entity.type].append(entity.value)
        return grouped
    
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
    nlu_analyses: List[NLUResult] = Field(default_factory=list)
    summary: str = ""
    context: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    def add_nlu_analysis(self, nlu_result: NLUResult) -> None:
        """Add NLU analysis result to long-term memory."""
        self.nlu_analyses.append(nlu_result)
        self.updated_at = datetime.now(timezone.utc)
    
    def get_important_analyses(self, threshold: float = 0.7) -> List[NLUResult]:
        """Get NLU analyses with importance score above threshold."""
        return [analysis for analysis in self.nlu_analyses if analysis.importance_score >= threshold]
    
    def get_analyses_by_intent(self, intent_name: str) -> List[NLUResult]:
        """Get analyses that contain a specific intent."""
        return [
            analysis for analysis in self.nlu_analyses 
            if any(intent.name == intent_name for intent in analysis.intents)
        ]
    
    def get_customer_preferences(self) -> Dict[str, Any]:
        """Extract customer preferences from NLU analyses."""
        preferences = {
            "preferred_language": "thai",
            "common_intents": [],
            "product_interests": [],
            "communication_style": "neutral",
            "urgency_patterns": []
        }
        
        if not self.nlu_analyses:
            return preferences
        
        try:
            # Language preference
            language_counts = {}
            for analysis in self.nlu_analyses:
                if analysis.primary_language:
                    lang = analysis.primary_language
                    language_counts[lang] = language_counts.get(lang, 0) + 1
            
            if language_counts:
                preferences["preferred_language"] = max(language_counts, key=language_counts.get)
            
            # Common intents
            intent_counts = {}
            for analysis in self.nlu_analyses:
                if analysis.primary_intent:
                    intent = analysis.primary_intent
                    intent_counts[intent] = intent_counts.get(intent, 0) + 1
            
            preferences["common_intents"] = sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Product interests
            product_interests = set()
            for analysis in self.nlu_analyses:
                for entity in analysis.entities:
                    if entity.type in ["product", "brand", "model"]:
                        product_interests.add(entity.value)
            preferences["product_interests"] = list(product_interests)[:10]
            
            # Communication style (based on sentiment patterns)
            sentiment_scores = [
                analysis.sentiment.confidence 
                for analysis in self.nlu_analyses 
                if analysis.sentiment
            ]
            if sentiment_scores:
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                if avg_sentiment > 0.7:
                    preferences["communication_style"] = "expressive"
                elif avg_sentiment < 0.4:
                    preferences["communication_style"] = "reserved"
                else:
                    preferences["communication_style"] = "neutral"
            
        except Exception:
            pass  # Return default preferences on error
        
        return preferences
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


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
    fallback_to_simple_json: bool = True
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


class Config(BaseModel):
    openrouter: OpenRouterConfig
    memory: MemoryConfig
    nlu: NLUConfig
    
    @validator('memory')
    def validate_memory_config(cls, v):
        if not v.redis_url:
            raise ValueError("Redis URL is required")
        return v
    
    @validator('nlu')
    def validate_nlu_config(cls, v):
        if not v.default_intent or not v.additional_intent:
            raise ValueError("Intent configurations are required")
        if not v.default_entity or not v.additional_entity:
            raise ValueError("Entity configurations are required")
        return v
