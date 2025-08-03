import json
import re
import logging
from typing import Dict, List, Any 
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParseStatus(Enum):
    """Parsing status enumeration."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    PARSE_ERROR = "parse_error"
    VALIDATION_ERROR = "validation_error"
    FORMAT_ERROR = "format_error"

class RobustNLUParser:
    """
    Enterprise-grade robust parser for NLU output with multiple fallback strategies.
    Inspired by Microsoft GraphRAG parsing patterns.
    """
    
    def __init__(self, 
                 tuple_delimiter: str = "<||>",
                 record_delimiter: str = "##", 
                 completion_delimiter: str = "<|COMPLETE|>"):
        self.tuple_delimiter = tuple_delimiter
        self.record_delimiter = record_delimiter
        self.completion_delimiter = completion_delimiter
        self.parse_stats = {
            "total_attempts": 0,
            "successful_parses": 0,
            "fallback_used": 0,
            "errors": []
        }
    
    def parse_intent_output(self, nlu_output: str) -> Dict[str, Any]:
        """
        Main parsing function with multiple fallback strategies.
        
        Args:
            nlu_output (str): Raw NLU output from LLM
            
        Returns:
            Dict[str, Any]: Structured result with parsing metadata
        """
        self.parse_stats["total_attempts"] += 1
        
        # Initialize result structure
        result = self._init_result_structure()
        
        try:
            # Strategy 1: Primary structured parsing
            success = self._parse_structured_format(nlu_output, result)
            if success:
                result["parsing_metadata"]["status"] = ParseStatus.SUCCESS.value
                result["parsing_metadata"]["strategy_used"] = "structured"
                self.parse_stats["successful_parses"] += 1
                return self._finalize_result(result)
            
            # Strategy 2: Regex-based fallback
            logger.warning("Structured parsing failed, attempting regex fallback")
            success = self._parse_with_regex_fallback(nlu_output, result)
            if success:
                result["parsing_metadata"]["status"] = ParseStatus.PARTIAL_SUCCESS.value
                result["parsing_metadata"]["strategy_used"] = "regex_fallback"
                self.parse_stats["fallback_used"] += 1
                return self._finalize_result(result)
            
            # Strategy 3: Line-by-line parsing
            logger.warning("Regex parsing failed, attempting line-by-line parsing")
            success = self._parse_line_by_line(nlu_output, result)
            if success:
                result["parsing_metadata"]["status"] = ParseStatus.PARTIAL_SUCCESS.value
                result["parsing_metadata"]["strategy_used"] = "line_by_line"
                self.parse_stats["fallback_used"] += 1
                return self._finalize_result(result)
            
            # Strategy 4: JSON extraction attempt
            logger.warning("Line parsing failed, attempting JSON extraction")
            success = self._extract_json_content(nlu_output, result)
            if success:
                result["parsing_metadata"]["status"] = ParseStatus.PARTIAL_SUCCESS.value
                result["parsing_metadata"]["strategy_used"] = "json_extraction"
                self.parse_stats["fallback_used"] += 1
                return self._finalize_result(result)
            
        except Exception as e:
            logger.error(f"Critical parsing error: {str(e)}")
            self.parse_stats["errors"].append(str(e))
            result["parsing_metadata"]["status"] = ParseStatus.PARSE_ERROR.value
            result["parsing_metadata"]["error"] = str(e)
        
        # Final fallback: Return basic structure with error info
        result["parsing_metadata"]["status"] = ParseStatus.FORMAT_ERROR.value
        result["parsing_metadata"]["raw_output"] = nlu_output[:500]  # Store first 500 chars for debugging
        return result
    
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
                "strategy_used": "structured",
                "warnings": [],
                "validation_errors": []
            }
        }
    
    def _parse_structured_format(self, nlu_output: str, result: Dict[str, Any]) -> bool:
        """Primary structured parsing strategy."""
        try:
            # Clean the output
            cleaned_output = self._clean_output(nlu_output)
            
            # Split into records using multiple strategies
            records = self._extract_records(cleaned_output)
            
            if not records:
                logger.warning("No records found in structured format")
                return False
            
            # Parse each record
            successful_parses = 0
            for record in records:
                if self._parse_single_record(record, result):
                    successful_parses += 1
            
            # Consider successful if at least 50% of records parsed
            return successful_parses > 0 and (successful_parses / len(records)) >= 0.5
            
        except Exception as e:
            logger.error(f"Structured parsing failed: {str(e)}")
            return False
    
    def _parse_with_regex_fallback(self, nlu_output: str, result: Dict[str, Any]) -> bool:
        """Regex-based fallback parsing strategy."""
        try:
            # Intent extraction patterns
            intent_patterns = [
                r'\(intent[^)]*?([^<>|]+?)[^)]*?(\d+\.?\d*)[^)]*?\)',
                r'intent[:\s]*([a-zA-Z_]+)[^0-9]*?(\d+\.?\d*)',
                r'([a-zA-Z_]+_intent)[^0-9]*?(\d+\.?\d*)'
            ]
            
            # Entity extraction patterns  
            entity_patterns = [
                r'\(entity[^)]*?([^<>|]+?)[^)]*?([^<>|]+?)[^)]*?(\d+\.?\d*)[^)]*?\)',
                r'entity[:\s]*([^,\s]+)[:\s]*([^,\s]+)[^0-9]*?(\d+\.?\d*)'
            ]
            
            # Language extraction patterns
            language_patterns = [
                r'\(language[^)]*?([A-Z]{3})[^)]*?(\d+\.?\d*)[^)]*?\)',
                r'language[:\s]*([A-Z]{3})[^0-9]*?(\d+\.?\d*)'
            ]
            
            # Sentiment extraction patterns
            sentiment_patterns = [
                r'\(sentiment[^)]*?(positive|negative|neutral)[^)]*?(\d+\.?\d*)[^)]*?\)',
                r'sentiment[:\s]*(positive|negative|neutral)[^0-9]*?(\d+\.?\d*)'
            ]
            
            # Extract using patterns
            self._extract_with_patterns(nlu_output, intent_patterns, "intent", result)
            self._extract_with_patterns(nlu_output, entity_patterns, "entity", result)
            self._extract_with_patterns(nlu_output, language_patterns, "language", result)
            self._extract_with_patterns(nlu_output, sentiment_patterns, "sentiment", result)
            
            # Check if we got reasonable results
            has_content = (len(result["intents"]) > 0 or 
                          len(result["entities"]) > 0 or 
                          len(result["languages"]) > 0 or 
                          result["sentiment"] is not None)
            
            return has_content
            
        except Exception as e:
            logger.error(f"Regex fallback failed: {str(e)}")
            return False
    
    def _parse_line_by_line(self, nlu_output: str, result: Dict[str, Any]) -> bool:
        """Line-by-line parsing for malformed outputs."""
        try:
            lines = nlu_output.split('\n')
            successful_lines = 0
            
            for line in lines:
                line = line.strip()
                if not line or line == self.record_delimiter:
                    continue
                
                # Try to identify content type and extract
                if any(keyword in line.lower() for keyword in ['intent', 'purchase', 'ask', 'cancel']):
                    if self._extract_line_content(line, "intent", result):
                        successful_lines += 1
                
                elif any(keyword in line.lower() for keyword in ['entity', 'product', 'brand', 'color']):
                    if self._extract_line_content(line, "entity", result):
                        successful_lines += 1
                
                elif any(keyword in line.lower() for keyword in ['language', 'tha', 'usa', 'eng']):
                    if self._extract_line_content(line, "language", result):
                        successful_lines += 1
                
                elif any(keyword in line.lower() for keyword in ['sentiment', 'positive', 'negative', 'neutral']):
                    if self._extract_line_content(line, "sentiment", result):
                        successful_lines += 1
            
            return successful_lines > 0
            
        except Exception as e:
            logger.error(f"Line-by-line parsing failed: {str(e)}")
            return False
    
    def _extract_json_content(self, nlu_output: str, result: Dict[str, Any]) -> bool:
        """Attempt to extract JSON content if LLM returned JSON instead."""
        try:
            # Try to find JSON-like structures
            json_patterns = [
                r'\{[^{}]*"intent"[^{}]*\}',
                r'\{[^{}]*"entity"[^{}]*\}',
                r'\{[^{}]*"language"[^{}]*\}',
                r'\{[^{}]*"sentiment"[^{}]*\}'
            ]
            
            extracted_content = False
            
            for pattern in json_patterns:
                matches = re.findall(pattern, nlu_output, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    try:
                        json_obj = json.loads(match)
                        if self._process_json_object(json_obj, result):
                            extracted_content = True
                    except json.JSONDecodeError:
                        continue
            
            return extracted_content
            
        except Exception as e:
            logger.error(f"JSON extraction failed: {str(e)}")
            return False
    
    def _clean_output(self, output: str) -> str:
        """Clean and normalize the output string."""
        # Remove completion delimiter
        cleaned = output.replace(self.completion_delimiter, "").strip()
        
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove common artifacts
        cleaned = re.sub(r'^Output:\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'^Result:\s*', '', cleaned, flags=re.IGNORECASE)
        
        return cleaned
    
    def _extract_records(self, output: str) -> List[str]:
        """Extract individual records using multiple strategies."""
        records = []
        
        # Strategy 1: Split by record delimiter
        if self.record_delimiter in output:
            parts = output.split(self.record_delimiter)
            records.extend([part.strip() for part in parts if part.strip()])
        
        # Strategy 2: Split by newlines if no record delimiter
        if not records:
            lines = output.split('\n')
            current_record = ""
            
            for line in lines:
                line = line.strip()
                if line.startswith('(') and current_record and ')' in current_record:
                    records.append(current_record)
                    current_record = line
                else:
                    current_record += " " + line if current_record else line
            
            if current_record:
                records.append(current_record)
        
        return [r for r in records if r]  # Filter empty records
    
    def _parse_single_record(self, record: str, result: Dict[str, Any]) -> bool:
        """Parse a single record with robust error handling."""
        try:
            # Extract tuple content using flexible regex
            tuple_patterns = [
                r'\(([^)]+)\)',  # Standard parentheses
                r'\[([^\]]+)\]',  # Square brackets fallback
                r'([^()[\]]+)',   # No brackets fallback
            ]
            
            content = None
            for pattern in tuple_patterns:
                match = re.search(pattern, record)
                if match:
                    content = match.group(1)
                    break
            
            if not content:
                return False
            
            # Split by delimiter with fallback
            if self.tuple_delimiter in content:
                parts = content.split(self.tuple_delimiter)
            else:
                # Fallback: split by common delimiters
                for delimiter in ['|', ',', ';', '\t']:
                    if delimiter in content:
                        parts = content.split(delimiter)
                        break
                else:
                    parts = content.split()  # Last resort: split by whitespace
            
            parts = [p.strip() for p in parts if p.strip()]
            
            if len(parts) < 3:
                return False
            
            record_type = parts[0].lower().strip()
            
            # Parse based on type with validation
            if record_type == "intent":
                return self._add_intent(parts, result)
            elif record_type == "entity":
                return self._add_entity(parts, result)
            elif record_type == "language":
                return self._add_language(parts, result)
            elif record_type == "sentiment":
                return self._add_sentiment(parts, result)
            
            return False
            
        except Exception as e:
            logger.warning(f"Failed to parse record '{record[:50]}...': {str(e)}")
            return False
    
    def _add_intent(self, parts: List[str], result: Dict[str, Any]) -> bool:
        """Add intent with validation."""
        try:
            if len(parts) < 4:
                return False
            
            intent_data = {
                "name": parts[1].strip(),
                "confidence": self._safe_float_parse(parts[2], 0.0),
                "priority_score": self._safe_float_parse(parts[3], 0.0) if len(parts) > 3 else 0.0,
                "metadata": self._safe_metadata_parse(parts[4] if len(parts) > 4 else "{}")
            }
            
            # Validation
            if not intent_data["name"] or intent_data["confidence"] < 0 or intent_data["confidence"] > 1:
                result["parsing_metadata"]["validation_errors"].append(f"Invalid intent: {parts}")
                return False
            
            result["intents"].append(intent_data)
            return True
            
        except Exception as e:
            result["parsing_metadata"]["validation_errors"].append(f"Intent parsing error: {str(e)}")
            return False
    
    def _add_entity(self, parts: List[str], result: Dict[str, Any]) -> bool:
        """Add entity with validation."""
        try:
            if len(parts) < 4:
                return False
            
            entity_data = {
                "type": parts[1].strip(),
                "value": parts[2].strip(),
                "confidence": self._safe_float_parse(parts[3], 0.0),
                "metadata": self._safe_metadata_parse(parts[4] if len(parts) > 4 else "{}")
            }
            
            # Validation
            if not entity_data["type"] or not entity_data["value"]:
                result["parsing_metadata"]["validation_errors"].append(f"Invalid entity: {parts}")
                return False
            
            result["entities"].append(entity_data)
            return True
            
        except Exception as e:
            result["parsing_metadata"]["validation_errors"].append(f"Entity parsing error: {str(e)}")
            return False
    
    def _add_language(self, parts: List[str], result: Dict[str, Any]) -> bool:
        """Add language with validation."""
        try:
            if len(parts) < 4:
                return False
            
            language_data = {
                "code": parts[1].strip().upper(),
                "confidence": self._safe_float_parse(parts[2], 0.0),
                "is_primary": self._safe_int_parse(parts[3], 0) == 1,
                "metadata": self._safe_metadata_parse(parts[4] if len(parts) > 4 else "{}")
            }
            
            # Validation
            if len(language_data["code"]) != 3:
                result["parsing_metadata"]["validation_errors"].append(f"Invalid language code: {parts}")
                return False
            
            result["languages"].append(language_data)
            
            # Set primary language
            if language_data["is_primary"]:
                result["metadata"]["primary_language"] = language_data["code"]
            
            return True
            
        except Exception as e:
            result["parsing_metadata"]["validation_errors"].append(f"Language parsing error: {str(e)}")
            return False
    
    def _add_sentiment(self, parts: List[str], result: Dict[str, Any]) -> bool:
        """Add sentiment with validation."""
        try:
            if len(parts) < 3:
                return False
            
            sentiment_data = {
                "label": parts[1].strip().lower(),
                "confidence": self._safe_float_parse(parts[2], 0.0),
                "metadata": self._safe_metadata_parse(parts[3] if len(parts) > 3 else "{}")
            }
            
            # Validation
            valid_sentiments = ["positive", "negative", "neutral"]
            if sentiment_data["label"] not in valid_sentiments:
                result["parsing_metadata"]["validation_errors"].append(f"Invalid sentiment: {parts}")
                return False
            
            result["sentiment"] = sentiment_data
            result["metadata"]["has_sentiment"] = True
            return True
            
        except Exception as e:
            result["parsing_metadata"]["validation_errors"].append(f"Sentiment parsing error: {str(e)}")
            return False
    
    def _safe_float_parse(self, value: str, default: float = 0.0) -> float:
        """Safely parse float with fallback."""
        try:
            parsed = float(value.strip())
            return max(0.0, min(1.0, parsed))  # Clamp between 0 and 1
        except (ValueError, AttributeError):
            return default
    
    def _safe_int_parse(self, value: str, default: int = 0) -> int:
        """Safely parse int with fallback."""
        try:
            return int(float(value.strip()))  # Handle "1.0" -> 1
        except (ValueError, AttributeError):
            return default
    
    def _safe_metadata_parse(self, value: str) -> Dict[str, Any]:
        """Safely parse metadata JSON with fallback."""
        try:
            if not value or value == "{}":
                return {}
            
            # Clean up common JSON formatting issues
            cleaned = value.strip()
            if not cleaned.startswith('{'):
                cleaned = '{' + cleaned
            if not cleaned.endswith('}'):
                cleaned = cleaned + '}'
            
            # Fix common JSON issues
            cleaned = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', cleaned)  # Add quotes to keys
            cleaned = re.sub(r':\s*([^",\[\]{}\s]+)(\s*[,}])', r': "\1"\2', cleaned)  # Add quotes to string values
            
            return json.loads(cleaned)
        except (json.JSONDecodeError, AttributeError):
            return {"raw": value}
    
    def _extract_with_patterns(self, text: str, patterns: List[str], content_type: str, result: Dict[str, Any]):
        """Extract content using regex patterns."""
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    groups = match.groups()
                    if content_type == "intent" and len(groups) >= 2:
                        self._add_intent(["intent", groups[0], groups[1], "0.0", "{}"], result)
                    elif content_type == "entity" and len(groups) >= 3:
                        self._add_entity(["entity", groups[0], groups[1], groups[2], "{}"], result)
                    elif content_type == "language" and len(groups) >= 2:
                        self._add_language(["language", groups[0], groups[1], "1", "{}"], result)
                    elif content_type == "sentiment" and len(groups) >= 2:
                        self._add_sentiment(["sentiment", groups[0], groups[1], "{}"], result)
                except Exception:
                    continue
    
    def _extract_line_content(self, line: str, content_type: str, result: Dict[str, Any]) -> bool:
        """Extract content from a single line."""
        try:
            # Simple extraction based on common patterns
            if content_type == "intent":
                # Look for intent-like words with confidence scores
                intent_match = re.search(r'(purchase|ask|cancel|greet)[\w_]*[^\d]*?(\d+\.?\d*)', line, re.IGNORECASE)
                if intent_match:
                    return self._add_intent(["intent", intent_match.group(1) + "_intent", intent_match.group(2), "0.0", "{}"], result)
            
            elif content_type == "entity":
                # Look for entity values
                thai_text = re.search(r'[\u0E00-\u0E7F]+', line)
                if thai_text:
                    confidence_match = re.search(r'(\d+\.?\d*)', line)
                    confidence = confidence_match.group(1) if confidence_match else "0.8"
                    return self._add_entity(["entity", "product", thai_text.group(0), confidence, "{}"], result)
            
            elif content_type == "language":
                # Look for language codes
                lang_match = re.search(r'\b(THA|USA|ENG)\b', line, re.IGNORECASE)
                if lang_match:
                    confidence_match = re.search(r'(\d+\.?\d*)', line)
                    confidence = confidence_match.group(1) if confidence_match else "0.9"
                    is_primary = "1" if "primary" in line.lower() or "THA" in lang_match.group(1).upper() else "0"
                    return self._add_language(["language", lang_match.group(1).upper(), confidence, is_primary, "{}"], result)
            
            elif content_type == "sentiment":
                # Look for sentiment labels
                sentiment_match = re.search(r'\b(positive|negative|neutral)\b', line, re.IGNORECASE)
                if sentiment_match:
                    confidence_match = re.search(r'(\d+\.?\d*)', line)
                    confidence = confidence_match.group(1) if confidence_match else "0.7"
                    return self._add_sentiment(["sentiment", sentiment_match.group(1).lower(), confidence, "{}"], result)
            
            return False
            
        except Exception:
            return False
    
    def _process_json_object(self, json_obj: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """Process a JSON object and extract relevant content."""
        try:
            extracted = False
            
            if "intent" in json_obj:
                intent_data = json_obj["intent"]
                if isinstance(intent_data, str):
                    self._add_intent(["intent", intent_data, "0.8", "0.0", "{}"], result)
                    extracted = True
                elif isinstance(intent_data, dict):
                    name = intent_data.get("name", intent_data.get("type", "unknown"))
                    confidence = str(intent_data.get("confidence", 0.8))
                    self._add_intent(["intent", name, confidence, "0.0", "{}"], result)
                    extracted = True
            
            # Similar processing for entity, language, sentiment...
            
            return extracted
            
        except Exception:
            return False
    
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
    
    def get_parse_statistics(self) -> Dict[str, Any]:
        """Get parsing statistics for monitoring."""
        success_rate = (self.parse_stats["successful_parses"] / self.parse_stats["total_attempts"]) * 100 if self.parse_stats["total_attempts"] > 0 else 0
        
        return {
            "total_attempts": self.parse_stats["total_attempts"],
            "successful_parses": self.parse_stats["successful_parses"],
            "fallback_used": self.parse_stats["fallback_used"],
            "success_rate": round(success_rate, 2),
            "recent_errors": self.parse_stats["errors"][-10:]  # Last 10 errors
        }

 