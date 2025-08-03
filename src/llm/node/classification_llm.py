"""
NLU Analysis System - Natural Language Understanding with Robust Parsing
Replaces the old event classification system with comprehensive NLU analysis.
"""

import json
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import PromptTemplate

from src.config import config_manager
from src.models import NLUResult, NLUIntent, NLUEntity, NLULanguage, NLUSentiment
from src.llm.node.utils import create_nlu_parser_from_config, extract_business_insights
from src.utils.logging import get_logger
from src.utils.cost_calculator import format_cost_info

logger = get_logger(__name__)

# NLU Detection Prompt (using the robust format from POC)
INTENT_DETECTION_PROMPT = """
-Goal-
Given a user utterance, detect and extract the user's **intent**, **entities**, **language**, and **sentiment**. You are also provided with pre-declared lists of possible default and additional intents and entities. Only extract intents/entities that appear in either default or additional lists. Assign confidence scores for each item extracted.

-Steps-
1. Identify the **top 3 intent(s)** that match the message. Consider both `default_intent` and `additional_intent` lists with their priority scores.
Format each intent as:
(intent{tuple_delimiter}<intent_name_in_snake_case>{tuple_delimiter}<confidence>{tuple_delimiter}<priority_score>{tuple_delimiter}<metadata>)

2. Identify all **entities** present in the message, using both `default_entity` and `additional_entity` types.
Format each entity as:
(entity{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_value>{tuple_delimiter}<confidence>{tuple_delimiter}<metadata>)

3. Detect **all languages** present in the message using ISO 3166 Alpha-3 country codes. Return primary language first, followed by additional detected languages. Use 1 for primary language and 0 for contained languages.
Format each language as:
(language{tuple_delimiter}<language_code_iso_alpha3>{tuple_delimiter}<confidence>{tuple_delimiter}<primary_flag>{tuple_delimiter}<metadata>)

4. Detect the **sentiment** expressed in the message.
Format:
(sentiment{tuple_delimiter}<label>{tuple_delimiter}<confidence>{tuple_delimiter}<metadata>)

5. Return the output as a list separated by **{record_delimiter}**

6. When complete, return {completion_delimiter}

######################
-Examples-
######################

Example 1:
text: I want to book a flight to Paris next week.
default_intent: book_flight:0.9, cancel_flight:0.7
additional_intent: greet:0.3, track_flight:0.5
default_entity: location, date
additional_entity: airline, person
######################
Output:
(intent{tuple_delimiter}book_flight{tuple_delimiter}0.95{tuple_delimiter}0.9{tuple_delimiter}{{"extracted_from": "default", "context": "travel_booking"}})
{record_delimiter}
(intent{tuple_delimiter}track_flight{tuple_delimiter}0.25{tuple_delimiter}0.5{tuple_delimiter}{{"extracted_from": "additional", "context": "travel_inquiry"}})
{record_delimiter}
(intent{tuple_delimiter}cancel_flight{tuple_delimiter}0.15{tuple_delimiter}0.7{tuple_delimiter}{{"extracted_from": "default", "context": "travel_cancellation"}})
{record_delimiter}
(entity{tuple_delimiter}location{tuple_delimiter}Paris{tuple_delimiter}0.98{tuple_delimiter}{{"entity_position": [25, 30], "entity_category": "geographic"}})
{record_delimiter}
(entity{tuple_delimiter}date{tuple_delimiter}next week{tuple_delimiter}0.94{tuple_delimiter}{{"entity_position": [31, 40], "entity_category": "temporal"}})
{record_delimiter}
(language{tuple_delimiter}USA{tuple_delimiter}1.0{tuple_delimiter}1{tuple_delimiter}{{"primary_language": true, "script": "latin", "detected_tokens": 9}})
{record_delimiter}
(sentiment{tuple_delimiter}neutral{tuple_delimiter}0.80{tuple_delimiter}{{"polarity": 0.1, "subjectivity": 0.3, "emotion": "neutral"}})
{completion_delimiter}

######################

Example 2:
text: อยากซื้อรองเท้า Hello!
default_intent: purchase_intent:0.8
additional_intent: ask_product:0.6, cancel_order:0.4
default_entity: product
additional_entity: brand, color
######################
Output:
(intent{tuple_delimiter}purchase_intent{tuple_delimiter}0.95{tuple_delimiter}0.8{tuple_delimiter}{{"extracted_from": "default", "context": "shopping_intent"}})
{record_delimiter}
(intent{tuple_delimiter}ask_product{tuple_delimiter}0.30{tuple_delimiter}0.6{tuple_delimiter}{{"extracted_from": "additional", "context": "product_inquiry"}})
{record_delimiter}
(intent{tuple_delimiter}cancel_order{tuple_delimiter}0.10{tuple_delimiter}0.4{tuple_delimiter}{{"extracted_from": "additional", "context": "order_cancellation"}})
{record_delimiter}
(entity{tuple_delimiter}product{tuple_delimiter}รองเท้า{tuple_delimiter}0.97{tuple_delimiter}{{"entity_position": [6, 12], "entity_category": "product", "language": "thai"}})
{record_delimiter}
(language{tuple_delimiter}THA{tuple_delimiter}0.85{tuple_delimiter}1{tuple_delimiter}{{"primary_language": true, "script": "thai", "detected_tokens": 2}})
{record_delimiter}
(language{tuple_delimiter}USA{tuple_delimiter}0.95{tuple_delimiter}0{tuple_delimiter}{{"primary_language": false, "script": "latin", "detected_tokens": 1}})
{record_delimiter}
(sentiment{tuple_delimiter}positive{tuple_delimiter}0.75{tuple_delimiter}{{"polarity": 0.6, "subjectivity": 0.4, "emotion": "desire"}})
{completion_delimiter}

######################
-Real Data-
######################
text: {input_text}
default_intent: {default_intent}
additional_intent: {additional_intent}
default_entity: {default_entity}
additional_entity: {additional_entity}
######################
Output:
"""


def analyze_message_nlu(user_message: str, conversation_context: list = None) -> Optional[NLUResult]:
    """
    Analyze user message using NLU (Natural Language Understanding).
    
    This replaces the old classify_event function with comprehensive NLU analysis
    that extracts intents, entities, languages, and sentiment.
    
    Args:
        user_message: User's message content
        conversation_context: List of recent messages for context (default: None)
        
    Returns:
        NLUResult object or None if analysis fails
    """
    try:
        # Get configuration
        openrouter_config = config_manager.get_openrouter_config()
        main_config = config_manager.get_config()
        nlu_config = main_config.nlu
        
        # Initialize LLM client
        from langchain_core.utils import convert_to_secret_str
        
        llm = ChatOpenAI(
            model=openrouter_config.classification.model,
            api_key=convert_to_secret_str(openrouter_config.api_key),
            base_url=openrouter_config.base_url,
            temperature=openrouter_config.classification.temperature,
        )
        
        # Prepare NLU prompt with configuration parameters
        prompt_template = PromptTemplate(
            template=INTENT_DETECTION_PROMPT,
            input_variables=[
                "input_text",
                "default_intent",
                "additional_intent", 
                "default_entity",
                "additional_entity",
                "tuple_delimiter",
                "record_delimiter",
                "completion_delimiter"
            ]
        )
        
        # Create prompt with NLU config
        formatted_prompt = prompt_template.format(
            input_text=user_message,
            default_intent=nlu_config.default_intent,
            additional_intent=nlu_config.additional_intent,
            default_entity=nlu_config.default_entity,
            additional_entity=nlu_config.additional_entity,
            tuple_delimiter=nlu_config.tuple_delimiter,
            record_delimiter=nlu_config.record_delimiter,
            completion_delimiter=nlu_config.completion_delimiter
        )
        
        # Prepare messages with context
        messages = [SystemMessage(content=formatted_prompt)]
        
        # Add conversation context if provided (limit to last 4-5 messages)
        if conversation_context:
            # Get last 5 messages from conversation context (excluding current message)
            recent_messages = conversation_context[-5:] if len(conversation_context) > 5 else conversation_context
            
            if recent_messages:
                context_content = "<conversation_context>\n"
                for i, msg in enumerate(recent_messages, 1):
                    # Handle both Message objects and dict
                    if hasattr(msg, 'role') and hasattr(msg, 'content'):
                        role = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
                        content = msg.content
                    else:
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')
                    context_content += f"{i}. [{role.upper()}]: {content}\n"
                context_content += "</conversation_context>\n\n"
                context_content += f"<current_message_to_analyze>\n{user_message}\n</current_message_to_analyze>"
                
                messages.append(HumanMessage(content=context_content))
            else:
                messages.append(HumanMessage(content=user_message))
        else:
            messages.append(HumanMessage(content=user_message))
        
        logger.info("Analyzing message with NLU", 
                   message_length=len(user_message),
                   model=openrouter_config.classification.model)
        
        # Pretty print NLU Context
        print("\n" + "="*60)
        print("🧠 NLU Analysis Context")
        print("="*60)
        for i, msg in enumerate(messages, 1):
            role = type(msg).__name__.replace("Message", "").upper()
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            print(f"{i}. [{role}] {content}")
        print("="*60)
        
        # Get LLM response
        response = llm.invoke(messages)
        
        # Track token usage (response is an AIMessage)
        if isinstance(response, AIMessage) and hasattr(response, 'usage_metadata') and response.usage_metadata:
            try:
                print(f"💰 NLU Analysis Usage:")
                usage = response.usage_metadata
                input_tokens = usage.get('input_tokens', 0)
                output_tokens = usage.get('output_tokens', 0)
                total_tokens = usage.get('total_tokens', input_tokens + output_tokens)
                
                if input_tokens or output_tokens:
                    cost_info = format_cost_info(
                        openrouter_config.classification.model,
                        input_tokens,
                        output_tokens,
                        total_tokens
                    )
                    print(cost_info)
                else:
                    print("   No token usage data available")
            except Exception as e:
                print(f"   Error tracking usage: {e}")
        else:
            print("💰 NLU Analysis Usage: No usage metadata available")
        
        # Parse NLU response using RobustNLUParser
        raw_response = response.content if isinstance(response.content, str) else str(response.content)
        
        # Use RobustNLUParser
        if nlu_config.enable_robust_parsing:
            nlu_result = parse_nlu_response_robust(raw_response, nlu_config, user_message)
        else:
            # Fallback to simple parsing (if needed)
            logger.warning("Robust parsing disabled, using fallback method")
            nlu_result = parse_nlu_response_simple(raw_response, user_message)
        
        if nlu_result:
            logger.info("NLU analysis completed", 
                       intents_found=len(nlu_result.intents),
                       entities_found=len(nlu_result.entities),
                       importance_score=nlu_result.importance_score)
            
            # Print analysis summary
            print(f"\n📊 NLU Analysis Summary:")
            print(f"   Primary Intent: {nlu_result.primary_intent}")
            print(f"   Entities Found: {len(nlu_result.entities)}")
            print(f"   Language: {nlu_result.primary_language}")
            print(f"   Sentiment: {nlu_result.sentiment.label if nlu_result.sentiment else 'None'}")
            print(f"   Importance Score: {nlu_result.importance_score:.3f}")
            
            return nlu_result
        else:
            logger.error("Failed to parse NLU response")
            return None
        
    except Exception as e:
        logger.error("Failed to analyze message with NLU", error=str(e))
        return None


def parse_nlu_response_robust(raw_response: str, nlu_config, original_message: str) -> Optional[NLUResult]:
    """Parse NLU response using RobustNLUParser."""
    try:
        # Create parser from config
        parser = create_nlu_parser_from_config({
            "tuple_delimiter": nlu_config.tuple_delimiter,
            "record_delimiter": nlu_config.record_delimiter,
            "completion_delimiter": nlu_config.completion_delimiter
        })
        
        # Parse the response
        parsed_result = parser.parse_intent_output(raw_response)
        
        # Convert to NLU models
        nlu_result = convert_parsed_to_nlu_result(parsed_result, original_message)
        
        return nlu_result
        
    except Exception as e:
        logger.error("Robust NLU parsing failed", error=str(e))
        return None


def parse_nlu_response_simple(raw_response: str, original_message: str) -> Optional[NLUResult]:
    """Simple fallback parsing (for backwards compatibility)."""
    try:
        # Try to extract JSON if present
        cleaned_content = raw_response.strip()
        if cleaned_content.startswith('```json'):
            cleaned_content = cleaned_content[7:]
            if cleaned_content.endswith('```'):
                cleaned_content = cleaned_content[:-3]
            cleaned_content = cleaned_content.strip()
        elif cleaned_content.startswith('```'):
            cleaned_content = cleaned_content[3:]
            if cleaned_content.endswith('```'):
                cleaned_content = cleaned_content[:-3]
            cleaned_content = cleaned_content.strip()
        
        # Parse as JSON
        json_data = json.loads(cleaned_content)
        
        # Convert to NLUResult (basic conversion)
        nlu_result = NLUResult(
            content=original_message,
            intents=[NLUIntent(name="unknown", confidence=0.5, priority_score=0.5)],
            entities=[],
            languages=[NLULanguage(code="THA", confidence=0.8, is_primary=True)],
            sentiment=NLUSentiment(label="neutral", confidence=0.5),
            parsing_metadata={"strategy_used": "simple_json", "status": "fallback"}
        )
        
        return nlu_result
        
    except Exception as e:
        logger.error("Simple NLU parsing failed", error=str(e))
        return None


def convert_parsed_to_nlu_result(parsed_result: Dict[str, Any], original_message: str) -> NLUResult:
    """Convert parsed result from RobustNLUParser to NLUResult model."""
    try:
        # Convert intents
        intents = []
        for intent_data in parsed_result.get("intents", []):
            intent = NLUIntent(
                name=intent_data["name"],
                confidence=intent_data["confidence"],
                priority_score=intent_data["priority_score"],
                metadata=intent_data["metadata"]
            )
            intents.append(intent)
        
        # Convert entities
        entities = []
        for entity_data in parsed_result.get("entities", []):
            entity = NLUEntity(
                type=entity_data["type"],
                value=entity_data["value"],
                confidence=entity_data["confidence"],
                metadata=entity_data["metadata"]
            )
            entities.append(entity)
        
        # Convert languages
        languages = []
        for lang_data in parsed_result.get("languages", []):
            language = NLULanguage(
                code=lang_data["code"],
                confidence=lang_data["confidence"],
                is_primary=lang_data["is_primary"],
                metadata=lang_data["metadata"]
            )
            languages.append(language)
        
        # Convert sentiment
        sentiment = None
        if parsed_result.get("sentiment"):
            sentiment_data = parsed_result["sentiment"]
            sentiment = NLUSentiment(
                label=sentiment_data["label"],
                confidence=sentiment_data["confidence"],
                metadata=sentiment_data["metadata"]
            )
        
        # Create NLUResult
        nlu_result = NLUResult(
            content=original_message,
            intents=intents,
            entities=entities,
            languages=languages,
            sentiment=sentiment,
            metadata=parsed_result.get("metadata", {}),
            parsing_metadata=parsed_result.get("parsing_metadata", {})
        )
        
        return nlu_result
        
    except Exception as e:
        logger.error("Failed to convert parsed result to NLUResult", error=str(e))
        # Return basic NLUResult as fallback
        return NLUResult(
            content=original_message,
            intents=[],
            entities=[],
            languages=[],
            sentiment=None,
            parsing_metadata={"error": str(e), "status": "conversion_failed"}
        )


def should_save_to_longterm(nlu_result: NLUResult, threshold: float = None) -> bool:
    """
    Determine if NLU analysis should be saved to long-term memory.
    
    Args:
        nlu_result: NLU analysis result
        threshold: Importance threshold (uses config default if None)
        
    Returns:
        bool: True if should save to long-term memory
    """
    try:
        if threshold is None:
            config = config_manager.get_config()
            threshold = config.nlu.importance_threshold
        
        return nlu_result.importance_score >= threshold
        
    except Exception as e:
        logger.error("Failed to determine save decision", error=str(e))
        return False  # Default to not saving on error


def get_business_insights_from_nlu(nlu_result: NLUResult) -> Dict[str, Any]:
    """
    Extract business insights from NLU analysis.
    
    Args:
        nlu_result: NLU analysis result
        
    Returns:
        Dict containing business-relevant insights
    """
    return extract_business_insights({
        "intents": [{"name": i.name, "confidence": i.confidence} for i in nlu_result.intents],
        "entities": [{"type": e.type, "value": e.value, "confidence": e.confidence} for e in nlu_result.entities],
        "languages": [{"code": l.code, "confidence": l.confidence, "is_primary": l.is_primary} for l in nlu_result.languages],
        "sentiment": {"label": nlu_result.sentiment.label, "confidence": nlu_result.sentiment.confidence} if nlu_result.sentiment else None
    })


