from typing import Optional
from src.models import Conversation, Message
from src.utils.redis_client import redis_client
from src.config import config_manager
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ShortTermMemory:
    def __init__(self):
        self.config = config_manager.get_memory_config()
        self.redis = redis_client
    
    def _get_key(self, user_id: str) -> str:
        return f"sm:{user_id}"
    
    def exists(self, user_id: str) -> bool:
        key = self._get_key(user_id)
        exists = self.redis.exists(key)
        logger.debug("SM exists check", user_id=user_id, exists=exists)
        return exists
    
    def is_valid(self, user_id: str) -> bool:
        if not self.exists(user_id):
            return False
        
        key = self._get_key(user_id)
        ttl = self.redis.get_ttl(key)
        valid = ttl > 0
        logger.debug("SM validity check", user_id=user_id, ttl=ttl, valid=valid)
        return valid
    
    def extend_ttl(self, user_id: str) -> bool:
        """
        Extend TTL for existing conversation to keep active sessions alive
        
        Args:
            user_id: User identifier
            
        Returns:
            bool: True if TTL was extended successfully
        """
        if not self.exists(user_id):
            logger.debug("Cannot extend TTL: SM does not exist", user_id=user_id)
            return False
            
        key = self._get_key(user_id)
        success = self.redis.expire(key, self.config.sm_ttl)
        
        if success:
            logger.info("SM TTL extended", user_id=user_id, ttl=self.config.sm_ttl)
        else:
            logger.warning("Failed to extend SM TTL", user_id=user_id)
            
        return success
    
    def save(self, conversation: Conversation, extend_if_exists: bool = True) -> bool:
        """
        Save conversation to Redis with optional TTL extension for existing conversations
        
        Args:
            conversation: Conversation object to save
            extend_if_exists: If True, extend TTL for existing conversations instead of resetting
            
        Returns:
            bool: True if saved successfully
        """
        key = self._get_key(conversation.user_id)
        
        # Check if TTL extension is enabled and conversation exists
        should_extend_ttl = (extend_if_exists and 
                           self.config.extend_ttl_on_activity and 
                           self.exists(conversation.user_id))
        
        if should_extend_ttl:
            # Convert to dict for JSON serialization
            conversation_data = conversation.model_dump()
            
            # Update data without changing TTL, then extend TTL
            success = self.redis.set_json(key, conversation_data)
            if success:
                # Extend TTL to keep active conversation alive
                self.extend_ttl(conversation.user_id)
                logger.info("SM updated with TTL extension", 
                           user_id=conversation.user_id, 
                           message_count=len(conversation.messages))
            else:
                logger.error("Failed to update SM", user_id=conversation.user_id)
        else:
            # New conversation or forced TTL reset or feature disabled
            conversation_data = conversation.model_dump()
            success = self.redis.set_json(key, conversation_data, ttl=self.config.sm_ttl)
            
            action = "saved with new TTL"
            if not self.config.extend_ttl_on_activity:
                action = "saved (TTL extension disabled)"
            
            if success:
                logger.info(f"SM {action}", 
                           user_id=conversation.user_id, 
                           message_count=len(conversation.messages),
                           ttl=self.config.sm_ttl)
            else:
                logger.error("Failed to save SM", user_id=conversation.user_id)
        
        return success
    
    def load(self, user_id: str) -> Optional[Conversation]:
        if not self.is_valid(user_id):
            logger.debug("SM not valid, cannot load", user_id=user_id)
            return None
        
        key = self._get_key(user_id)
        data = self.redis.get_json(key)
        
        if data:
            try:
                conversation = Conversation(**data)
                logger.info("SM loaded", user_id=user_id, message_count=len(conversation.messages))
                return conversation
            except Exception as e:
                logger.error("Failed to parse SM data", user_id=user_id, error=str(e))
                return None
        
        logger.debug("No SM data found", user_id=user_id)
        return None
    
    def add_message(self, user_id: str, message: Message) -> bool:
        conversation = self.load(user_id)
        if not conversation:
            logger.error("Cannot add message: SM not found", user_id=user_id)
            return False
        
        conversation.add_message(message)
        success = self.save(conversation)
        
        if success:
            logger.info("Message added to SM", user_id=user_id, role=message.role)
        
        return success
    
    def delete(self, user_id: str) -> bool:
        key = self._get_key(user_id)
        success = self.redis.delete(key)
        
        if success:
            logger.info("SM deleted", user_id=user_id)
        else:
            logger.error("Failed to delete SM", user_id=user_id)
        
        return success


# Global instance
short_term_memory = ShortTermMemory()