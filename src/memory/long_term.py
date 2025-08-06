import json
from pathlib import Path
from typing import Optional, Dict, Any
from src.models import LongTermMemory, NLUResult, Conversation
from src.config import config_manager
from src.utils.logging import get_logger

logger = get_logger(__name__)


class LongTermMemoryStore:
    def __init__(self):
        self.config = config_manager.get_memory_config()
        self.base_path = Path(self.config.lm_base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _get_file_path(self, user_id: str) -> Path:
        return self.base_path / f"{user_id}.json"
    
    def exists(self, user_id: str) -> bool:
        file_path = self._get_file_path(user_id)
        exists = file_path.exists()
        logger.debug("LM exists check", user_id=user_id, exists=exists)
        return exists
    
    def save(self, lm: LongTermMemory) -> bool:
        file_path = self._get_file_path(lm.user_id)
        
        try:
            # Convert to dict for JSON serialization
            lm_data = lm.model_dump()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(lm_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info("LM saved", user_id=lm.user_id, analysis_count=len(lm.nlu_analyses))
            return True
            
        except Exception as e:
            logger.error("Failed to save LM", user_id=lm.user_id, error=str(e))
            return False
    
    def load(self, user_id: str) -> Optional[LongTermMemory]:
        if not self.exists(user_id):
            logger.debug("LM file not found", user_id=user_id)
            return None
        
        file_path = self._get_file_path(user_id)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            lm = LongTermMemory(**data)
            logger.info("LM loaded", user_id=user_id, analysis_count=len(lm.nlu_analyses))
            return lm
            
        except Exception as e:
            logger.error("Failed to load LM", user_id=user_id, error=str(e))
            return None
    
    def add_nlu_analysis(self, user_id: str, nlu_result: NLUResult) -> bool:
        """Add NLU analysis result to long-term memory."""
        lm = self.load(user_id)
        if not lm:
            # Create new LM if it doesn't exist
            lm = LongTermMemory(user_id=user_id)
        
        lm.add_nlu_analysis(nlu_result)
        success = self.save(lm)
        
        if success:
            logger.info("NLU analysis added to LM", 
                       user_id=user_id, 
                       primary_intent=nlu_result.primary_intent,
                       importance_score=nlu_result.importance_score)
        
        return success
    
    def create_from_conversation(self, conversation: Conversation, context: Optional[Dict[str, Any]] = None) -> LongTermMemory:
        """Create LM from conversation context"""
        lm = LongTermMemory(
            user_id=conversation.user_id,
            context=context or {},
            summary=f"Conversation started with {len(conversation.messages)} messages"
        )
        
        logger.info("LM created from conversation", 
                   user_id=conversation.user_id,
                   message_count=len(conversation.messages))
        
        return lm
    
    def delete(self, user_id: str) -> bool:
        file_path = self._get_file_path(user_id)
        
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info("LM deleted", user_id=user_id)
                return True
            else:
                logger.debug("LM file not found for deletion", user_id=user_id)
                return False
                
        except Exception as e:
            logger.error("Failed to delete LM", user_id=user_id, error=str(e))
            return False


# Global instance
long_term_memory = LongTermMemoryStore()