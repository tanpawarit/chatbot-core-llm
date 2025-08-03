from typing import Optional, Dict, Any
from src.models import Conversation, Message, Event
from src.memory.short_term import short_term_memory
from src.memory.long_term import long_term_memory
from src.utils.logging import get_logger

logger = get_logger(__name__)


class MemoryManager:
    """
    Orchestrates the memory flow from your diagram:
    A[User Message] → B{SM Exists & Valid?} → C[Load SM from Redis] / D[Load LM from JSON] 
    → E[Create SM from LM Context] → F[Save SM to Redis] → G[Add Message to SM]
    """
    
    def __init__(self):
        self.sm = short_term_memory
        self.lm = long_term_memory
    
    def process_user_message(self, user_id: str, user_message: Message) -> Conversation:
        """
        Main flow implementation following your diagram A→B→C...→G
        """
        logger.info("Processing user message", user_id=user_id)
        
        # B: SM Exists & Valid?
        if self.sm.exists(user_id) and self.sm.is_valid(user_id):
            # C: Load SM from Redis
            conversation = self.sm.load(user_id)
            if conversation is None:
                # Fallback: create new conversation if load failed
                conversation = Conversation(user_id=user_id)
                logger.warning("SM load failed, created new conversation", user_id=user_id)
            else:
                logger.info("Loaded existing SM", user_id=user_id)
        else:
            # D: Load LM from JSON
            lm = self.lm.load(user_id)
            
            if lm:
                # E: Create SM from LM Context
                conversation = Conversation(
                    user_id=user_id,
                    metadata=lm.context
                )
                logger.info("Created SM from existing LM context", user_id=user_id)
            else:
                # Create brand new conversation
                conversation = Conversation(user_id=user_id)
                logger.info("Created new conversation", user_id=user_id)
            
            # F: Save SM to Redis
            self.sm.save(conversation)
        
        # G: Add Message to SM
        conversation.add_message(user_message)
        self.sm.save(conversation)
        
        logger.info("User message processed and added to SM", 
                   user_id=user_id,
                   total_messages=len(conversation.messages))
        
        return conversation
    
    def add_assistant_response(self, user_id: str, response_message: Message) -> bool:
        """Add assistant response to conversation"""
        success = self.sm.add_message(user_id, response_message)
        
        if success:
            logger.info("Assistant response added to SM", user_id=user_id)
        
        return success
    
    def save_important_event(self, user_id: str, event: Event) -> bool:
        """
        Save important events to long-term memory (LM)
        Part of flow: I{Important Event?} → J[Save Event to LM]
        """
        if event.importance_score >= 0.7:  # Important event threshold
            success = self.lm.add_event(user_id, event)
            
            if success:
                logger.info("Important event saved to LM", 
                           user_id=user_id,
                           event_type=event.event_type,
                           importance_score=event.importance_score)
            
            return success
        else:
            # K: Skip LM Save (not important enough)
            logger.debug("Event not important enough for LM", 
                        user_id=user_id,
                        importance_score=event.importance_score)
            return True
    
    def get_conversation(self, user_id: str) -> Optional[Conversation]:
        """Get current conversation from SM"""
        return self.sm.load(user_id)
    
    def get_conversation_context(self, user_id: str) -> Dict[str, Any]:
        """Get conversation context from both SM and LM"""
        context = {}
        
        # Get current conversation from SM
        conversation = self.sm.load(user_id)
        if conversation:
            context['current_messages'] = len(conversation.messages)
            context['recent_messages'] = [msg.model_dump() for msg in conversation.get_recent_messages(5)]
        
        # Get historical context from LM
        lm = self.lm.load(user_id)
        if lm:
            context['important_events'] = len(lm.get_important_events())
            context['total_events'] = len(lm.events)
            context['summary'] = lm.summary
        
        return context
    
    def cleanup_conversation(self, user_id: str) -> bool:
        """Clean up both SM and LM for a user"""
        sm_deleted = self.sm.delete(user_id)
        lm_deleted = self.lm.delete(user_id)
        
        success = sm_deleted and lm_deleted
        logger.info("User data cleanup", 
                   user_id=user_id,
                   success=success)
        
        return success


# Global instance
memory_manager = MemoryManager()