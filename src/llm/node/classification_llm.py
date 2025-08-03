"""Simple classification LLM node"""

import json
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import config_manager
from src.models import EventClassification
from src.utils.logging import get_logger
from src.utils.cost_calculator import format_cost_info

logger = get_logger(__name__)


# Event classification prompts and constants
EVENT_CLASSIFICATION_SYSTEM_PROMPT = """<system_identity>
Conversation analysis expert specializing in event classification and importance assessment.
</system_identity>

<event_types>
• INQUIRY: Questions, inquiries, asking for information
• FEEDBACK: Reviews, opinions, likes/dislikes, evaluations
• REQUEST: Requests for services, bookings, wanting something
• COMPLAINT: Problems, issues, complaints, dissatisfaction
• TRANSACTION: Buying, paying, pricing, financial matters
• SUPPORT: Help requests, guidance, how-to questions
• INFORMATION: Providing information, announcements, notifications
• GENERIC_EVENT: Greetings, thanks, social interactions
</event_types>

<importance_scale>
• 0.9-1.0: Transactions, critical issues
• 0.7-0.8: Important requests, feedback
• 0.5-0.6: Support requests
• 0.3-0.4: Simple questions
• 0.1-0.2: Greetings, social interactions
</importance_scale>

<output_format>
Respond with JSON following EventClassification schema only:
{
  "event_type": "one of the types above",
  "importance_score": 0.0-1.0,
  "intent": "text description of user intent",
  "reasoning": "brief explanation"
}
</output_format>"""

def classify_event(user_message: str) -> Optional[EventClassification]:
    """
    Classify user message into event type and importance using LLM
    
    Args:
        user_message: User's message content
        
    Returns:
        EventClassification object or None if classification fails
    """
    try:
        # Get configuration
        config = config_manager.get_openrouter_config()
        openrouter_config = config_manager.get_openrouter_config()
        
        # Initialize LLM client
        from langchain_core.utils import convert_to_secret_str
        
        llm = ChatOpenAI(
            model=config.classification.model,
            api_key=convert_to_secret_str(openrouter_config.api_key),
            base_url=openrouter_config.base_url,
            temperature=config.classification.temperature,
        ) 
        
        # Prepare messages
        messages = [
            SystemMessage(content=EVENT_CLASSIFICATION_SYSTEM_PROMPT),
            HumanMessage(content=user_message)
        ]
        
        logger.info("Classifying event", 
                   message_length=len(user_message),
                   model=config.classification.model)
        
        # Pretty print Classification LLM Context
        print("\n" + "="*60)
        print("🤖 Classification LLM Context")
        print("="*60)
        for i, msg in enumerate(messages, 1):
            role = type(msg).__name__.replace("Message", "").upper()
            print(f"{i}. [{role}] {msg.content}")
        print("="*60)
        
        # Get LLM response
        response = llm.invoke(messages)
        
        # Debug: Check what's in the response
        print(f"🔍 Debug response attributes: {dir(response)}")
        if hasattr(response, 'response_metadata'):
            print(f"🔍 Response metadata: {response.response_metadata}")
        
        # Track token usage
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            try:
                print(f"💰 Classification LLM Usage:")
                # Handle both object and dict formats
                usage = response.usage_metadata
                if hasattr(usage, 'input_tokens'):
                    # Object format
                    input_tokens = usage.input_tokens
                    output_tokens = usage.output_tokens
                    total_tokens = usage.total_tokens
                elif isinstance(usage, dict):
                    # Dict format
                    input_tokens = usage.get('input_tokens', 0)
                    output_tokens = usage.get('output_tokens', 0)
                    total_tokens = usage.get('total_tokens', input_tokens + output_tokens)
                else:
                    print("   Usage metadata format not supported")
                    input_tokens = output_tokens = total_tokens = 0
                
                if input_tokens or output_tokens:
                    cost_info = format_cost_info(
                        config.classification.model,
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
            print("💰 Classification LLM Usage: No usage metadata available")
        
        # Parse JSON response
        response_content = response.content if isinstance(response.content, str) else str(response.content)
        
        # Strip markdown code blocks if present
        cleaned_content = response_content.strip()
        if cleaned_content.startswith('```json'):
            # Remove opening ```json and closing ```
            cleaned_content = cleaned_content[7:]  # Remove ```json
            if cleaned_content.endswith('```'):
                cleaned_content = cleaned_content[:-3]  # Remove closing ```
            cleaned_content = cleaned_content.strip()
        elif cleaned_content.startswith('```'):
            # Remove opening ``` and closing ```
            cleaned_content = cleaned_content[3:]  # Remove opening ```
            if cleaned_content.endswith('```'):
                cleaned_content = cleaned_content[:-3]  # Remove closing ```
            cleaned_content = cleaned_content.strip()
        
        classification_data = json.loads(cleaned_content)
        classification = EventClassification(**classification_data)
        
        logger.info("Event classified", 
                   event_type=classification.event_type,
                   importance_score=classification.importance_score)
        
        return classification
        
    except json.JSONDecodeError as e:
        logger.error("Failed to parse LLM response as JSON", 
                    error=str(e), 
                    response=response.content,
                    cleaned_content=locals().get('cleaned_content', 'not_cleaned'))
        return None
    except Exception as e:
        logger.error("Failed to classify event", error=str(e))
        return None