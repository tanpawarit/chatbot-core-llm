from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


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


class ImportanceScoringConfig(BaseModel):
    """Configuration for importance scoring in NLU results."""
    short_message_threshold: int = Field(ge=1, default=10)
    generic_intent_penalty: float = Field(ge=0.0, le=1.0, default=0.5)
    generic_intents: List[str] = Field(default_factory=lambda: ["purchase_intent", "inquiry_intent"])
    length_penalties: Dict[str, Union[int, float]] = Field(default_factory=lambda: {
        "very_short_threshold": 3,
        "very_short_penalty": 0.3,
        "short_threshold": 10,
        "short_penalty": 0.6,
        "medium_threshold": 20,
        "medium_penalty": 0.8
    })


class NLUResult(BaseModel):
    content: str  # Original message content
    intents: List[NLUIntent] = Field(default_factory=list)
    entities: List[NLUEntity] = Field(default_factory=list)
    languages: List[NLULanguage] = Field(default_factory=list)
    sentiment: Optional[NLUSentiment] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    parsing_metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    config: Optional[ImportanceScoringConfig] = None  # Optional config for importance scoring
    
    @property
    def importance_score(self) -> float:
        """Calculate importance score from NLU analysis with smart filtering."""
        try:
            importance = 0.0
            message_length = len(self.content.strip())
            
            # Get config or use defaults
            if self.config:
                scoring_config = self.config
            else:
                # Fallback to default config
                scoring_config = ImportanceScoringConfig()
            
            # Length penalty for very short messages using dynamic config
            length_penalty = 1.0
            length_penalties = scoring_config.length_penalties
            
            if message_length <= length_penalties.get("very_short_threshold", 3):
                length_penalty = length_penalties.get("very_short_penalty", 0.3)
            elif message_length <= length_penalties.get("short_threshold", 10):
                length_penalty = length_penalties.get("short_penalty", 0.6)
            elif message_length <= length_penalties.get("medium_threshold", 20):
                length_penalty = length_penalties.get("medium_penalty", 0.8)
            
            # Intent contribution (60% weight)
            if self.intents:
                top_intent = max(self.intents, key=lambda x: x.confidence)
                intent_weight = top_intent.confidence * top_intent.priority_score
                
                # Extra penalty for generic intents on short messages using dynamic config
                if (message_length <= scoring_config.short_message_threshold and 
                    top_intent.name in scoring_config.generic_intents):
                    intent_weight *= scoring_config.generic_intent_penalty
                
                importance += intent_weight * 0.6
            
            # Entity contribution (25% weight) - with validation
            if self.entities:
                # Check if entities actually exist in the message text
                valid_entities = []
                for entity in self.entities:
                    # Simple check: if entity value appears in message content
                    if entity.value.lower() in self.content.lower():
                        valid_entities.append(entity)
                
                if valid_entities:
                    entity_confidence_avg = sum(e.confidence for e in valid_entities) / len(valid_entities)
                    entity_count_bonus = min(len(valid_entities) * 0.1, 0.3)  # Max 0.3 bonus
                    importance += (entity_confidence_avg + entity_count_bonus) * 0.25
                else:
                    # No valid entities found - likely hallucination, significant penalty
                    importance *= 0.3
            
            # Sentiment contribution (15% weight)
            if self.sentiment:
                sentiment_weight = self.sentiment.confidence
                if self.sentiment.label in ["positive", "negative"]:  # Strong emotions get higher weight
                    sentiment_weight *= 1.2
                importance += sentiment_weight * 0.15
            
            # Apply length penalty
            importance *= length_penalty
            
            # Business importance boost for specific patterns
            if any(word in self.content.lower() for word in ["ซื้อ", "เท่าไหร่", "ราคา", "สั่ง", "จอง"]):
                importance += 0.1  # Boost for commercial terms
            
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
    importance_scoring: ImportanceScoringConfig = Field(default_factory=ImportanceScoringConfig)