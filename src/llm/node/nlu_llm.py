"""
NLU Analysis System - Natural Language Understanding with Robust Parsing
Replaces the old event classification system with comprehensive NLU analysis.
"""

import time
from typing import Optional, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langchain_core.prompts import PromptTemplate

from src.config import config_manager
from src.models import NLUResult, NLUIntent, NLUEntity, NLULanguage, NLUSentiment
from src.llm.node.parser import parse_nlu_output, extract_business_insights
from src.llm.factory import llm_factory
from src.utils.logging import get_logger
from src.utils.token_tracker import token_tracker

logger = get_logger(__name__)

# NLU Detection Prompt (using the robust format from POC)
INTENT_DETECTION_PROMPT = """
-Goal-
Given a user utterance, detect and extract the user's **intent**, **entities**, **language**, and **sentiment**. You are also provided with pre-declared lists of possible default and additional intents and entities. 

STRICT RULES:
1. You MUST ONLY extract intents/entities that appear in either default or additional lists
2. DO NOT create new intents or entities not in the provided lists
3. If user input doesn't match any intent, use the closest matching intent from the lists
4. Common greetings (สวัสดี, หวัดดี, hello, hi, good morning) should ALWAYS be classified as "greet"
5. Only extract entities that are EXPLICITLY mentioned in the current message being analyzed

IMPORTANT: Only extract entities that are EXPLICITLY mentioned in the current message being analyzed. Do NOT use entities from conversation context unless they appear in the current message text.

-Steps-
1. Identify the **top 3 intent(s)** that match the message. Consider both `default_intent` and `additional_intent` lists with their priority scores.
Format each intent as:
(intent{tuple_delimiter}<intent_name_in_snake_case>{tuple_delimiter}<confidence>{tuple_delimiter}<priority_score>{tuple_delimiter}<metadata>)

2. Identify all **entities** present in the message, using both `default_entity` and `additional_entity` types.
STRICT RULE: Only extract entities that are LITERALLY PRESENT in the current message text. Do not infer or assume entities from context.
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


def analyze_message_nlu(user_message: str, conversation_context: Optional[list] = None) -> Optional[NLUResult]:
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
    # Get configuration outside try block to ensure availability in exception handler
    main_config = config_manager.get_config()
    nlu_config = main_config.nlu
    
    try:
        
        # Get LLM instance from factory
        llm = llm_factory.get_classification_llm()
        
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
        messages: list[BaseMessage] = [SystemMessage(content=formatted_prompt)]
        
        # Add conversation context if provided (already limited by caller)
        if conversation_context:
            recent_messages = conversation_context
            
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
                
                # Convert to HumanMessage for context
                messages.append(HumanMessage(content=context_content))
            else:
                # Convert to HumanMessage for user message
                messages.append(HumanMessage(content=user_message))
        else:
            # Convert to HumanMessage for user message
            messages.append(HumanMessage(content=user_message))
        
        analysis_start = time.time()
        
        logger.info("Analyzing message with NLU", 
                   message_length=len(user_message))
        
        # Pretty print NLU Context
        print("\n" + "="*60)
        print("🧠 NLU Analysis Context")
        print("="*60)
        for i, msg in enumerate(messages, 1):
            role = type(msg).__name__.replace("Message", "").upper()
            content_str = str(msg.content)
            # content = content_str[:200] + "..." if len(content_str) > 200 else content_str
            # print(f"{i}. [{role}] {content}")
            print(content_str)
        print("="*60)
        
        # Get LLM response
        try:
            response = llm.invoke(messages)
                
        except Exception as llm_error:
            logger.error("LLM invoke failed", error=str(llm_error)) 
            return None
        
        # Track token usage
        openrouter_config = config_manager.get_openrouter_config()
        # Convert BaseMessage to AIMessage for token tracking
        ai_response = AIMessage(content=response.content) if isinstance(response, BaseMessage) else response
        
        usage = token_tracker.track_response(ai_response, openrouter_config.classification.model, "classification")
        if usage:
            token_tracker.print_usage(usage, "🧠")
        else:
            print("🧠 NLU Analysis Usage: No usage metadata available")
        
        # Parse NLU response using PyParsingNLUParser
        raw_response = response.content if isinstance(response.content, str) else str(response.content)
        
        # Use simplified parser
        parsed_data = parse_nlu_output(raw_response, nlu_config.tuple_delimiter)
        if parsed_data["success"]:
            nlu_result = _create_nlu_result_from_parsed_data(parsed_data, user_message)
        else:
            logger.warning("NLU parsing failed, using fallback")
            nlu_result = None
        
        if nlu_result:
            analysis_time = time.time() - analysis_start
            logger.info("NLU analysis completed", 
                       intents_found=len(nlu_result.intents),
                       entities_found=len(nlu_result.entities),
                       importance_score=nlu_result.importance_score,
                       analysis_time_ms=round(analysis_time * 1000, 2))
            
            # Warn if analysis took too long
            if analysis_time > 5.0:  # >5 seconds
                logger.warning("Slow NLU analysis detected", 
                              analysis_time_ms=round(analysis_time * 1000, 2),
                              message_length=len(user_message))
            
            # Print analysis summary
            print("\n📊 NLU Analysis Summary:")
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


def _create_nlu_result_from_parsed_data(parsed_data: Dict[str, Any], original_message: str) -> NLUResult:
    """Convert simplified parsed data to NLUResult model."""
    try:
        main_config = config_manager.get_config()
        nlu_config = main_config.nlu
        
        # Convert intents
        intents = []
        for intent_data in parsed_data.get("intents", []):
            intent = NLUIntent(
                name=intent_data["name"],
                confidence=intent_data["confidence"],
                priority_score=0.0,  # Simplified - no priority scoring
                metadata={}
            )
            intents.append(intent)
        
        # Convert entities
        entities = []
        for entity_data in parsed_data.get("entities", []):
            entity = NLUEntity(
                type=entity_data["type"],
                value=entity_data["value"],
                confidence=entity_data["confidence"],
                metadata={}
            )
            entities.append(entity)
        
        # Convert languages
        languages = []
        for lang_data in parsed_data.get("languages", []):
            language = NLULanguage(
                code=lang_data["code"],
                confidence=lang_data["confidence"],
                is_primary=lang_data["is_primary"],
                metadata={}
            )
            languages.append(language)
        
        # Convert sentiment
        sentiment = None
        if parsed_data.get("sentiment"):
            sentiment_data = parsed_data["sentiment"]
            sentiment = NLUSentiment(
                label=sentiment_data["label"],
                confidence=sentiment_data["confidence"],
                metadata={}
            )
        
        # Create simplified NLUResult
        nlu_result = NLUResult(
            content=original_message,
            intents=intents,
            entities=entities,
            languages=languages,
            sentiment=sentiment,
            metadata={},
            parsing_metadata={"status": "simplified_success"},
            config=nlu_config.importance_scoring
        )
        
        return nlu_result
        
    except Exception as e:
        logger.error("Failed to create NLUResult from parsed data", error=str(e))
        main_config = config_manager.get_config()
        return NLUResult(
            content=original_message,
            intents=[],
            entities=[],
            languages=[],
            sentiment=None,
            parsing_metadata={"error": str(e), "status": "conversion_failed"},
            config=main_config.nlu.importance_scoring
        )


def should_save_to_longterm(nlu_result: NLUResult, threshold: Optional[float] = None) -> bool:
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


