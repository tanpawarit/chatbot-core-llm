"""Simple classification LLM node"""

import json
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import config_manager
from src.models import EventClassification
from src.utils.logging import get_logger

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
        
        # Get LLM response
        response = llm.invoke(messages)
        
        # Parse JSON response
        response_content = response.content if isinstance(response.content, str) else str(response.content)
        classification_data = json.loads(response_content.strip())
        classification = EventClassification(**classification_data)
        
        logger.info("Event classified", 
                   event_type=classification.event_type,
                   importance_score=classification.importance_score)
        
        return classification
        
    except json.JSONDecodeError as e:
        logger.error("Failed to parse LLM response as JSON", 
                    error=str(e), 
                    response=response.content)
        return None
    except Exception as e:
        logger.error("Failed to classify event", error=str(e))
        return None