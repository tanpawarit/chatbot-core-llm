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
    
    def save(self, conversation: Conversation) -> bool:
        key = self._get_key(conversation.user_id)
        
        # Convert to dict for JSON serialization
        conversation_data = conversation.model_dump()
        
        success = self.redis.set_json(key, conversation_data, ttl=self.config.sm_ttl)
        
        if success:
            logger.info("SM saved", user_id=conversation.user_id, message_count=len(conversation.messages))
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