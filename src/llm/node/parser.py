"""
Simplified NLU parser - 80% functionality with 20% complexity
"""
import re
from typing import Dict, Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_nlu_output(raw_output: str, delimiter: str = "<||>") -> Dict[str, Any]:
    """
    Simple NLU parser using basic regex patterns.
    
    Args:
        raw_output: Raw LLM output
        delimiter: Tuple delimiter (default: "<||>")
        
    Returns:
        Dict with parsed intents, entities, languages, sentiment
    """
    result = {
        "intents": [],
        "entities": [],
        "languages": [],
        "sentiment": None,
        "success": False
    }
    
    try:
        # Clean output
        cleaned = raw_output.replace("<|COMPLETE|>", "").strip()
        delimiter_escaped = re.escape(delimiter)
        
        # Extract intents: (intent<||>name<||>confidence)
        intent_pattern = rf'\(intent{delimiter_escaped}([a-zA-Z_]+){delimiter_escaped}([0-9.]+)\)'
        for match in re.finditer(intent_pattern, cleaned, re.IGNORECASE):
            name, confidence = match.groups()
            try:
                result["intents"].append({
                    "name": name,
                    "confidence": min(1.0, float(confidence))
                })
            except ValueError:
                continue
        
        # Extract entities: (entity<||>type<||>value<||>confidence)
        entity_pattern = rf'\(entity{delimiter_escaped}([a-zA-Z_]+){delimiter_escaped}([\w\u0E00-\u0E7F\s]+?){delimiter_escaped}([0-9.]+)\)'
        for match in re.finditer(entity_pattern, cleaned, re.IGNORECASE):
            entity_type, value, confidence = match.groups()
            try:
                result["entities"].append({
                    "type": entity_type,
                    "value": value.strip(),
                    "confidence": min(1.0, float(confidence))
                })
            except ValueError:
                continue
        
        # Extract language: (language<||>code<||>confidence<||>is_primary)
        lang_pattern = rf'\(language{delimiter_escaped}([A-Z]{{3}}){delimiter_escaped}([0-9.]+){delimiter_escaped}([01]?)\)'
        for match in re.finditer(lang_pattern, cleaned, re.IGNORECASE):
            code, confidence, is_primary = match.groups()
            try:
                result["languages"].append({
                    "code": code.upper(),
                    "confidence": min(1.0, float(confidence)),
                    "is_primary": is_primary == '1'
                })
            except ValueError:
                continue
        
        # Extract sentiment: (sentiment<||>label<||>confidence)
        sentiment_pattern = rf'\(sentiment{delimiter_escaped}(positive|negative|neutral){delimiter_escaped}([0-9.]+)\)'
        sentiment_match = re.search(sentiment_pattern, cleaned, re.IGNORECASE)
        if sentiment_match:
            label, confidence = sentiment_match.groups()
            try:
                result["sentiment"] = {
                    "label": label.lower(),
                    "confidence": min(1.0, float(confidence))
                }
            except ValueError:
                pass
        
        # Fallback patterns if no structured output found
        if not result["intents"] and not result["entities"]:
            _extract_fallback_patterns(cleaned, result)
        
        # Sort by confidence
        result["intents"].sort(key=lambda x: x["confidence"], reverse=True)
        result["entities"].sort(key=lambda x: x["confidence"], reverse=True)
        
        result["success"] = bool(result["intents"] or result["entities"])
        
        return result
        
    except Exception as e:
        logger.error("NLU parsing failed", error=str(e))
        return {
            "intents": [{"name": "general_intent", "confidence": 0.5}],
            "entities": [],
            "languages": [],
            "sentiment": None,
            "success": False,
            "error": str(e)
        }


def _extract_fallback_patterns(text: str, result: Dict[str, Any]) -> None:
    """Extract using simple fallback patterns when structured parsing fails."""
    # Simple intent keywords
    intent_keywords = {
        "greet": ["สวัสดี", "ครับ", "คะ", "hello", "hi"],
        "purchase_intent": ["ซื้อ", "ขาย", "ราคา", "buy", "price"],
        "inquiry_intent": ["อยาก", "สอบถาม", "ถาม", "want", "ask"],
        "support_intent": ["ช่วย", "แก้", "ปัญหา", "help", "problem"],
        "complain_intent": ["แย่", "ไม่ดี", "บ่น", "bad", "complain"]
    }
    
    text_lower = text.lower()
    for intent_name, keywords in intent_keywords.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                result["intents"].append({
                    "name": intent_name,
                    "confidence": 0.7
                })
                break
    
    # If still no intents, add default
    if not result["intents"]:
        result["intents"].append({
            "name": "general_intent", 
            "confidence": 0.5
        })


def extract_business_insights(nlu_result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract simple business insights from NLU results."""
    insights = {
        "customer_intent": "unknown",
        "urgency_level": "low",
        "language_preference": "thai",
        "requires_human_attention": False
    }
    
    if nlu_result.get("intents"):
        top_intent = nlu_result["intents"][0]
        insights["customer_intent"] = top_intent["name"]
        
        # High confidence complaints need attention
        if (top_intent["name"] in ["complain_intent", "support_intent"] and 
            top_intent["confidence"] > 0.8):
            insights["urgency_level"] = "high"
            insights["requires_human_attention"] = True
    
    # Negative sentiment needs attention
    if (nlu_result.get("sentiment") and 
        nlu_result["sentiment"]["label"] == "negative" and 
        nlu_result["sentiment"]["confidence"] > 0.8):
        insights["requires_human_attention"] = True
    
    return insights