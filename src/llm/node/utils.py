"""
NLU utility functions and PyParsingNLUParser for enterprise-grade NLU processing.
Using pyparsing library for cleaner, more maintainable parsing.
"""

import json
import re
import time
from typing import Dict, Any
from enum import Enum

from src.utils.logging import get_logger

logger = get_logger(__name__)

class ParseStatus(Enum):
    """Parsing status enumeration."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    PARSE_ERROR = "parse_error"
    VALIDATION_ERROR = "validation_error"
    FORMAT_ERROR = "format_error"


class SimpleNLUParser:
    """
    Simplified NLU parser using direct regex patterns.
    Focuses on robustness and maintainability over complex grammar.
    """
    
    def __init__(self, 
                 tuple_delimiter: str = "<||>",
                 record_delimiter: str = "##", 
                 completion_delimiter: str = "<|COMPLETE|>"):
        self.tuple_delimiter = tuple_delimiter
        self.record_delimiter = record_delimiter
        self.completion_delimiter = completion_delimiter
        
        # Simple patterns for each NLU component
        self._build_patterns()
    
    def _build_patterns(self):
        """Build regex patterns for NLU components."""
        delimiter = re.escape(self.tuple_delimiter)
        
        # Simpler, more robust patterns
        self.patterns = {
            'intent': re.compile(
                rf'\(intent{delimiter}([a-zA-Z_]+){delimiter}([0-9.]+)(?:{delimiter}([0-9.]+))?(?:{delimiter}(\{{.*?\}}))?\)',
                re.IGNORECASE | re.DOTALL
            ),
            'entity': re.compile(
                rf'\(entity{delimiter}([a-zA-Z_]+){delimiter}([\w\u0E00-\u0E7F\s]+?){delimiter}([0-9.]+)(?:{delimiter}(\{{.*?\}}))?\)',
                re.IGNORECASE | re.DOTALL
            ),
            'language': re.compile(
                rf'\(language{delimiter}([A-Z]{{3}}){delimiter}([0-9.]+){delimiter}([01]?)(?:{delimiter}(\{{.*?\}}))?\)',
                re.IGNORECASE
            ),
            'sentiment': re.compile(
                rf'\(sentiment{delimiter}(positive|negative|neutral){delimiter}([0-9.]+)(?:{delimiter}(\{{.*?\}}))?\)',
                re.IGNORECASE
            )
        }
    
    def parse_intent_output(self, nlu_output: str) -> Dict[str, Any]:
        """
        Parse NLU output using direct regex patterns.
        
        Args:
            nlu_output (str): Raw NLU output from LLM
            
        Returns:
            Dict[str, Any]: Structured result with parsing metadata
        """
        start_time = time.time()
        result = self._init_result_structure()
        
        try:
            # Clean and normalize output
            cleaned_output = self._clean_output(nlu_output)
            
            # Parse each component type directly
            self._extract_intents(cleaned_output, result)
            self._extract_entities(cleaned_output, result)
            self._extract_languages(cleaned_output, result)
            self._extract_sentiment(cleaned_output, result)
            
            # Check if we got any results
            has_results = (result["intents"] or result["entities"] or 
                          result["languages"] or result["sentiment"])
            
            parse_time = time.time() - start_time
            
            if has_results:
                result["parsing_metadata"]["status"] = ParseStatus.SUCCESS.value
                result["parsing_metadata"]["strategy_used"] = "direct_regex"
            else:
                # Try fallback extraction for partial results
                self._extract_fallback_patterns(cleaned_output, result)
                result["parsing_metadata"]["status"] = ParseStatus.PARTIAL_SUCCESS.value
                result["parsing_metadata"]["strategy_used"] = "fallback_patterns"
            
            result["parsing_metadata"]["parse_time_ms"] = round(parse_time * 1000, 2)
            
            return self._finalize_result(result)
            
        except Exception as e:
            parse_time = time.time() - start_time
            logger.error("Parsing failed", error=str(e))
            
            result["parsing_metadata"]["status"] = ParseStatus.PARSE_ERROR.value
            result["parsing_metadata"]["error"] = str(e)
            result["parsing_metadata"]["parse_time_ms"] = round(parse_time * 1000, 2)
            
            return result
    
    def _extract_intents(self, text: str, result: Dict[str, Any]) -> None:
        """Extract intents using regex pattern."""
        matches = self.patterns['intent'].findall(text)
        for match in matches:
            name, confidence, priority, metadata = match
            try:
                result["intents"].append({
                    "name": name,
                    "confidence": min(1.0, float(confidence)),
                    "priority_score": float(priority) if priority else 0.0,
                    "metadata": self._safe_json_parse(metadata)
                })
            except ValueError:
                continue
    
    def _extract_entities(self, text: str, result: Dict[str, Any]) -> None:
        """Extract entities using regex pattern."""
        matches = self.patterns['entity'].findall(text)
        for match in matches:
            entity_type, value, confidence, metadata = match
            try:
                result["entities"].append({
                    "type": entity_type,
                    "value": value.strip(),
                    "confidence": min(1.0, float(confidence)),
                    "metadata": self._safe_json_parse(metadata)
                })
            except ValueError:
                continue
    
    def _extract_languages(self, text: str, result: Dict[str, Any]) -> None:
        """Extract languages using regex pattern."""
        matches = self.patterns['language'].findall(text)
        for match in matches:
            code, confidence, is_primary, metadata = match
            try:
                lang_data = {
                    "code": code.upper(),
                    "confidence": min(1.0, float(confidence)),
                    "is_primary": is_primary == '1',
                    "metadata": self._safe_json_parse(metadata)
                }
                result["languages"].append(lang_data)
                
                if lang_data["is_primary"]:
                    result["metadata"]["primary_language"] = code.upper()
            except ValueError:
                continue
    
    def _extract_sentiment(self, text: str, result: Dict[str, Any]) -> None:
        """Extract sentiment using regex pattern."""
        match = self.patterns['sentiment'].search(text)
        if match:
            label, confidence, metadata = match.groups()
            try:
                result["sentiment"] = {
                    "label": label.lower(),
                    "confidence": min(1.0, float(confidence)),
                    "metadata": self._safe_json_parse(metadata)
                }
                result["metadata"]["has_sentiment"] = True
            except ValueError:
                pass
    
    def _safe_json_parse(self, value: str) -> Dict[str, Any]:
        """Safely parse JSON metadata."""
        if not value or value == '{}':
            return {}
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return {"raw": value}
    
    def _extract_fallback_patterns(self, text: str, result: Dict[str, Any]) -> None:
        """Extract using simpler fallback patterns."""
        # Simple intent pattern
        intent_matches = re.findall(r'intent[:\s]*([a-zA-Z_]+)', text, re.IGNORECASE)
        for name in intent_matches[:3]:  # Limit to top 3
            result["intents"].append({
                "name": name,
                "confidence": 0.7,  # Default confidence
                "priority_score": 0.0,
                "metadata": {}
            })
        
        # Simple entity pattern
        entity_matches = re.findall(r'entity[:\s]*([a-zA-Z_]+)[:\s]+([\w\u0E00-\u0E7F\s]+)', text, re.IGNORECASE)
        for entity_type, value in entity_matches[:5]:  # Limit to top 5
            result["entities"].append({
                "type": entity_type,
                "value": value.strip(),
                "confidence": 0.7,
                "metadata": {}
            })
        
        # If still no results, add generic intent
        if not result["intents"] and not result["entities"]:
            result["intents"].append({
                "name": "general_intent",
                "confidence": 0.5,
                "priority_score": 0.0,
                "metadata": {}
            })
    
    def _init_result_structure(self) -> Dict[str, Any]:
        """Initialize the result structure with proper typing."""
        return {
            "intents": [],
            "entities": [],
            "languages": [],
            "sentiment": None,
            "metadata": {
                "total_intents": 0,
                "total_entities": 0,
                "total_languages": 0,
                "primary_language": None,
                "has_sentiment": False,
                "confidence_scores": {
                    "intent_avg": 0.0,
                    "entity_avg": 0.0,
                    "language_avg": 0.0
                }
            },
            "parsing_metadata": {
                "status": ParseStatus.SUCCESS.value,
                "strategy_used": "pyparsing_grammar",
                "warnings": [],
                "validation_errors": []
            }
        }
    
    def _clean_output(self, output: str) -> str:
        """Clean and normalize the output string."""
        # Remove completion delimiter
        cleaned = output.replace(self.completion_delimiter, "").strip()
        
        # Normalize whitespace but preserve structure
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove common artifacts
        cleaned = re.sub(r'^(Output|Result):\s*', '', cleaned, flags=re.IGNORECASE)
        
        return cleaned
    
    
    def _finalize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize the result with computed metadata."""
        # Update counts
        result["metadata"]["total_intents"] = len(result["intents"])
        result["metadata"]["total_entities"] = len(result["entities"])
        result["metadata"]["total_languages"] = len(result["languages"])
        
        # Calculate average confidence scores
        if result["intents"]:
            result["metadata"]["confidence_scores"]["intent_avg"] = sum(i["confidence"] for i in result["intents"]) / len(result["intents"])
        
        if result["entities"]:
            result["metadata"]["confidence_scores"]["entity_avg"] = sum(e["confidence"] for e in result["entities"]) / len(result["entities"])
        
        if result["languages"]:
            result["metadata"]["confidence_scores"]["language_avg"] = sum(l["confidence"] for l in result["languages"]) / len(result["languages"])
        
        # Sort results
        result["intents"].sort(key=lambda x: x["confidence"], reverse=True)
        result["languages"].sort(key=lambda x: (not x["is_primary"], -x["confidence"]))
        
        return result


def create_pyparsing_nlu_parser_from_config(config: Dict[str, Any]) -> SimpleNLUParser:
    """Create SimpleNLUParser instance from configuration."""
    return SimpleNLUParser(
        tuple_delimiter=config.get("tuple_delimiter", "<||>"),
        record_delimiter=config.get("record_delimiter", "##"),
        completion_delimiter=config.get("completion_delimiter", "<|COMPLETE|>")
    )


def extract_business_insights(nlu_result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract business-relevant insights from NLU results."""
    insights = {
        "customer_intent": None,
        "product_interest": [],
        "urgency_level": "low",
        "language_preference": "thai",
        "emotional_state": "neutral",
        "requires_human_attention": False
    }
    
    try:
        # Customer intent analysis
        if nlu_result.get("intents"):
            top_intent = nlu_result["intents"][0]
            insights["customer_intent"] = top_intent["name"]
            
            # Urgency based on intent type and confidence
            if top_intent["name"] in ["complaint", "support_intent"] and top_intent["confidence"] > 0.8:
                insights["urgency_level"] = "high"
                insights["requires_human_attention"] = True
            elif top_intent["confidence"] > 0.9:
                insights["urgency_level"] = "medium"
        
        # Product analysis
        for entity in nlu_result.get("entities", []):
            if entity["type"] in ["product", "brand", "model"]:
                insights["product_interest"].append({
                    "type": entity["type"],
                    "value": entity["value"],
                    "confidence": entity["confidence"]
                })
        
        # Language preference
        if nlu_result.get("languages"):
            primary_lang = next((l for l in nlu_result["languages"] if l["is_primary"]), None)
            if primary_lang:
                lang_map = {"THA": "thai", "USA": "english", "ENG": "english"}
                insights["language_preference"] = lang_map.get(primary_lang["code"], "thai")
        
        # Emotional state
        if nlu_result.get("sentiment"):
            insights["emotional_state"] = nlu_result["sentiment"]["label"]
            
            # High confidence negative sentiment needs attention
            if (nlu_result["sentiment"]["label"] == "negative" and 
                nlu_result["sentiment"]["confidence"] > 0.8):
                insights["requires_human_attention"] = True
        
        return insights
        
    except Exception as e:
        logger.error("Failed to extract business insights", error=str(e))
        return insights