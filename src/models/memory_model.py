from datetime import datetime, timezone
from typing import Dict, List, Any
from pydantic import BaseModel, Field
from .nlu_model import NLUResult


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
                preferences["preferred_language"] = max(language_counts.keys(), key=lambda x: language_counts[x])
            
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