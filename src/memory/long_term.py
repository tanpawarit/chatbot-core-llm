import json
import os
from pathlib import Path
from typing import Optional, Dict, Any
from src.models import LongTermMemory, Event, Conversation
from src.config import config_manager
from src.utils.logging import get_logger

logger = get_logger(__name__)


class LongTermMemoryStore:
    def __init__(self):
        self.config = config_manager.get_memory_config()
        self.base_path = Path(self.config.lm_base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _get_file_path(self, conversation_id: str) -> Path:
        return self.base_path / f"{conversation_id}.json"
    
    def exists(self, conversation_id: str) -> bool:
        file_path = self._get_file_path(conversation_id)
        exists = file_path.exists()
        logger.debug("LM exists check", conversation_id=conversation_id, exists=exists)
        return exists
    
    def save(self, lm: LongTermMemory) -> bool:
        file_path = self._get_file_path(lm.conversation_id)
        
        try:
            # Convert to dict for JSON serialization
            lm_data = lm.dict()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(lm_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info("LM saved", conversation_id=lm.conversation_id, event_count=len(lm.events))
            return True
            
        except Exception as e:
            logger.error("Failed to save LM", conversation_id=lm.conversation_id, error=str(e))
            return False
    
    def load(self, conversation_id: str) -> Optional[LongTermMemory]:
        if not self.exists(conversation_id):
            logger.debug("LM file not found", conversation_id=conversation_id)
            return None
        
        file_path = self._get_file_path(conversation_id)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            lm = LongTermMemory(**data)
            logger.info("LM loaded", conversation_id=conversation_id, event_count=len(lm.events))
            return lm
            
        except Exception as e:
            logger.error("Failed to load LM", conversation_id=conversation_id, error=str(e))
            return None
    
    def add_event(self, conversation_id: str, event: Event) -> bool:
        lm = self.load(conversation_id)
        if not lm:
            # Create new LM if it doesn't exist
            lm = LongTermMemory(conversation_id=conversation_id)
        
        lm.add_event(event)
        success = self.save(lm)
        
        if success:
            logger.info("Event added to LM", 
                       conversation_id=conversation_id, 
                       event_type=event.event_type,
                       importance_score=event.importance_score)
        
        return success
    
    def create_from_conversation(self, conversation: Conversation, context: Optional[Dict[str, Any]] = None) -> LongTermMemory:
        """Create LM from conversation context"""
        lm = LongTermMemory(
            conversation_id=conversation.conversation_id,
            context=context or {},
            summary=f"Conversation started with {len(conversation.messages)} messages"
        )
        
        logger.info("LM created from conversation", 
                   conversation_id=conversation.conversation_id,
                   message_count=len(conversation.messages))
        
        return lm
    
    def delete(self, conversation_id: str) -> bool:
        file_path = self._get_file_path(conversation_id)
        
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info("LM deleted", conversation_id=conversation_id)
                return True
            else:
                logger.debug("LM file not found for deletion", conversation_id=conversation_id)
                return False
                
        except Exception as e:
            logger.error("Failed to delete LM", conversation_id=conversation_id, error=str(e))
            return False


# Global instance
long_term_memory = LongTermMemoryStore()