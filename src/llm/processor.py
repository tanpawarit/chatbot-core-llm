import json
from typing import Optional
from src.models import (
    Event, EventType, EventClassification, Message, MessageRole
)
from src.llm.prompt import EVENT_CLASSIFICATION_SYSTEM_PROMPT
from src.llm.client import classification_llm, response_llm
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EventProcessor:
    """
    Handles LLM event processing logic from your diagram:
    H[LLM Event Processing] → I{Important Event?} → J[Save Event to LM] / K[Skip LM Save]
    Uses separate LLMs for classification and response generation
    """
    
    def __init__(self):
        self.classification_llm = classification_llm
        self.response_llm = response_llm
    
    def classify_event(self, user_message: str) -> Optional[EventClassification]:
        """
        Use LLM to classify user message into event type and importance
        """
        try:
            messages = [
                Message(role=MessageRole.SYSTEM, content=EVENT_CLASSIFICATION_SYSTEM_PROMPT),
                Message(role=MessageRole.USER, content=user_message)
            ]
            
            logger.info("Classifying event", message_length=len(user_message))
            
            response = self.classification_llm.classify_event(messages)
            
            # Parse JSON response
            classification_data = json.loads(response.strip())
            
            classification = EventClassification(**classification_data)
            
            logger.info("Event classified", 
                       event_type=classification.event_type,
                       importance_score=classification.importance_score)
            
            return classification
            
        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM response as JSON", error=str(e), response=response)
            return None
        except Exception as e:
            logger.error("Failed to classify event", error=str(e))
            return None
    
    def create_event_from_message(self, message: Message) -> Optional[Event]:
        """
        Create Event object from user message with LLM classification
        """
        if message.role != MessageRole.USER:
            logger.debug("Skipping event creation for non-user message", role=message.role)
            return None
        
        classification = self.classify_event(message.content)
        if not classification:
            logger.warning("Could not classify message, skipping event creation")
            return None
        
        event = Event(
            event_type=classification.event_type,
            content=message.content,
            classification=classification,
            context={
                "message_timestamp": message.timestamp.isoformat(),
                "message_metadata": message.metadata
            }
        )
        
        logger.info("Event created from message", 
                   event_type=event.event_type,
                   importance_score=event.importance_score)
        
        return event
    
    def is_important_event(self, event: Event, threshold: float = 0.7) -> bool:
        """
        Determine if event is important enough to save to LM
        Part of flow: I{Important Event?}
        """
        important = event.importance_score >= threshold
        
        logger.debug("Importance check", 
                    event_type=event.event_type,
                    importance_score=event.importance_score,
                    threshold=threshold,
                    important=important)
        
        return important
    
    def generate_chat_response(self, conversation_messages: list[Message]) -> str:
        """
        Generate chat response from conversation history
        Part of flow: M[Generate Response]
        """
        try:
            logger.info("Generating chat response", message_count=len(conversation_messages))
            
            response = self.response_llm.generate_response(conversation_messages)
            
            logger.info("Chat response generated", response_length=len(response))
            
            return response
            
        except Exception as e:
            logger.error("Failed to generate chat response", error=str(e))
            raise


# Global instance
event_processor = EventProcessor()