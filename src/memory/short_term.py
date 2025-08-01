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
    
    def _get_key(self, conversation_id: str) -> str:
        return f"sm:{conversation_id}"
    
    def exists(self, conversation_id: str) -> bool:
        key = self._get_key(conversation_id)
        exists = self.redis.exists(key)
        logger.debug("SM exists check", conversation_id=conversation_id, exists=exists)
        return exists
    
    def is_valid(self, conversation_id: str) -> bool:
        if not self.exists(conversation_id):
            return False
        
        key = self._get_key(conversation_id)
        ttl = self.redis.get_ttl(key)
        valid = ttl > 0
        logger.debug("SM validity check", conversation_id=conversation_id, ttl=ttl, valid=valid)
        return valid
    
    def save(self, conversation: Conversation) -> bool:
        key = self._get_key(conversation.conversation_id)
        
        # Convert to dict for JSON serialization
        conversation_data = conversation.dict()
        
        success = self.redis.set_json(key, conversation_data, ttl=self.config.sm_ttl)
        
        if success:
            logger.info("SM saved", conversation_id=conversation.conversation_id, message_count=len(conversation.messages))
        else:
            logger.error("Failed to save SM", conversation_id=conversation.conversation_id)
        
        return success
    
    def load(self, conversation_id: str) -> Optional[Conversation]:
        if not self.is_valid(conversation_id):
            logger.debug("SM not valid, cannot load", conversation_id=conversation_id)
            return None
        
        key = self._get_key(conversation_id)
        data = self.redis.get_json(key)
        
        if data:
            try:
                conversation = Conversation(**data)
                logger.info("SM loaded", conversation_id=conversation_id, message_count=len(conversation.messages))
                return conversation
            except Exception as e:
                logger.error("Failed to parse SM data", conversation_id=conversation_id, error=str(e))
                return None
        
        logger.debug("No SM data found", conversation_id=conversation_id)
        return None
    
    def add_message(self, conversation_id: str, message: Message) -> bool:
        conversation = self.load(conversation_id)
        if not conversation:
            logger.error("Cannot add message: SM not found", conversation_id=conversation_id)
            return False
        
        conversation.add_message(message)
        success = self.save(conversation)
        
        if success:
            logger.info("Message added to SM", conversation_id=conversation_id, role=message.role)
        
        return success
    
    def delete(self, conversation_id: str) -> bool:
        key = self._get_key(conversation_id)
        success = self.redis.delete(key)
        
        if success:
            logger.info("SM deleted", conversation_id=conversation_id)
        else:
            logger.error("Failed to delete SM", conversation_id=conversation_id)
        
        return success


# Global instance
short_term_memory = ShortTermMemory()