"""Event processing orchestrator"""

from typing import Optional, Tuple
from src.models import Message, Event
from src.llm.node.classification_llm import classify_event
from src.llm.node.response_llm import generate_response
from src.memory.long_term import long_term_memory
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EventProcessor:
    """
    Handles the event processing flow from your diagram:
    H[Classify Event] → I{Important Event?} → J[Save Event to LM] / K[Skip LM Save] 
    → L[Generate Response] → M[Add Response to SM] → N[Return Response]
    """
    
    def __init__(self):
        self.lm = long_term_memory
    
    def process_message(self, user_id: str, user_message: Message, conversation_messages: list[Message]) -> Tuple[Optional[Event], str]:
        """
        Process user message through classification and response generation
        
        Args:
            user_id: User identifier
            user_message: The user's message to process
            conversation_messages: List of conversation messages for context
            
        Returns:
            Tuple of (Event if created, generated response)
        """
        try:
            # H: Classify Event with conversation context
            # Use previous messages (last 4-5) as context, excluding current message
            previous_messages = conversation_messages[:-1] if len(conversation_messages) > 1 else []
            context_messages = previous_messages[-5:] if len(previous_messages) > 5 else previous_messages
            classification = classify_event(user_message.content, context_messages)
            
            event = None
            if classification:
                # Create event from classification
                event = Event(
                    event_type=classification.event_type,
                    content=user_message.content,
                    timestamp=user_message.timestamp,
                    classification=classification,
                    context={
                        "message_timestamp": user_message.timestamp.isoformat(),
                        "message_metadata": user_message.metadata
                    }
                )
                
                # I: Important Event? → J/K: Save or Skip LM Save
                if event.importance_score >= 0.7:
                    # J: Save Event to LM
                    self.lm.add_event(user_id, event)
                    logger.info("Important event saved to LM", 
                               user_id=user_id, 
                               event_type=event.event_type,
                               importance_score=event.importance_score)
                else:
                    # K: Skip LM Save
                    logger.debug("Event not important enough for LM", 
                                user_id=user_id,
                                importance_score=event.importance_score)
            
            # L: Generate Response with LM context
            lm_context = self.lm.load(user_id)
            response_content = generate_response(conversation_messages, lm_context)
            
            logger.info("Event processed and response generated", 
                       user_id=user_id,
                       has_event=event is not None,
                       response_length=len(response_content))
            
            return event, response_content
            
        except Exception as e:
            logger.error("Failed to process message", 
                        user_id=user_id, 
                        error=str(e))
            # Return fallback response
            return None, "ขอโทษครับ เกิดข้อผิดพลาดในการประมวลผล กรุณาลองใหม่อีกครั้งครับ"


# Global instance
event_processor = EventProcessor()