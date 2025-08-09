from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
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
        """Simplified importance scoring - clear business logic without complex penalties."""
        try:
            # Base score - everyone starts with some importance
            score = 0.2  # Default baseline
            
            # Primary score from intent confidence (main factor)
            if self.intents:
                top_intent = max(self.intents, key=lambda x: x.confidence)
                score = max(score, top_intent.confidence * 0.7)  # 70% of confidence as base score
            
            # Business keyword boost - clear commercial intent gets priority
            business_keywords = [
                "ซื้อ", "เท่าไหร่", "ราคา", "สั่ง", "จอง", "ได้ไหม", "อยาก", 
                "มีไหม", "แนะนำ", "เอา", "งบ", "บาท"
            ]
            keyword_matches = sum(1 for word in business_keywords if word in self.content.lower())
            if keyword_matches > 0:
                score += min(keyword_matches * 0.15, 0.3)  # Max 0.3 boost from keywords
            
            # Entity boost - but only if entities actually exist in the text
            if self.entities:
                valid_entities = [e for e in self.entities if e.value.lower() in self.content.lower()]
                if valid_entities:
                    # Each valid entity adds value
                    score += min(len(valid_entities) * 0.1, 0.2)  # Max 0.2 boost from entities
            
            # Sentiment boost for engaged customers
            if self.sentiment and self.sentiment.label in ["positive", "negative"]:
                score += 0.1  # Engaged customers (not neutral) are important
            
            # Length consideration (but not penalty) - very short messages get small boost if they have clear intent
            message_length = len(self.content.strip())
            if message_length <= 10 and score >= 0.6:
                score += 0.1  # Short but meaningful messages get slight boost
            
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.4  # Higher fallback to be safe
    
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