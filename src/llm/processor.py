"""NLU processing orchestrator - Replaces event processing with NLU analysis"""

from typing import Optional, Tuple
from src.models import Message, NLUResult
from src.llm.node.nlu_llm import analyze_message_nlu, should_save_to_longterm, get_business_insights_from_nlu
from src.llm.node.response_llm import generate_response
from src.llm.routing import context_router
from src.memory.long_term import long_term_memory
from src.config import config_manager
from src.utils.logging import get_logger
from src.utils.token_tracker import token_tracker

logger = get_logger(__name__)


class LLMProcessor:
    """
    Handles the NLU processing flow (updated from EventProcessor):
    H[Analyze NLU] → I{Important Analysis?} → J[Save Analysis to LM] / K[Skip LM Save] 
    → L[Generate Response] → M[Add Response to SM] → N[Return Response]
    """
    
    def __init__(self):
        self.lm = long_term_memory
    
    def process_message(self, user_id: str, user_message: Message, conversation_messages: list[Message]) -> Tuple[Optional[NLUResult], str]:
        """
        Process user message through NLU analysis and response generation
        
        Args:
            user_id: User identifier
            user_message: The user's message to process
            conversation_messages: List of conversation messages for context
            
        Returns:
            Tuple of (NLUResult if created, generated response)
        """
        try:
            # Load LM context BEFORE processing to avoid including current message
            lm_context = self.lm.load(user_id)
            
            # H: Analyze NLU with conversation context
            # Use previous messages (last 4-5) as context, excluding current message
            previous_messages = conversation_messages[:-1] if len(conversation_messages) > 1 else []
            context_messages = previous_messages[-5:] if len(previous_messages) > 5 else previous_messages
            
            # Perform NLU analysis - handle both text and media content
            if user_message.content.has_media:
                # For media messages, analyze the text part and include media context
                message_text = user_message.content.text or f"[{user_message.content.media_type.value} file]"
                nlu_result = analyze_message_nlu(message_text, context_messages, media_context=user_message.content)
            else:
                # Text-only message
                nlu_result = analyze_message_nlu(user_message.content.text, context_messages)
            
            if nlu_result:
                # I: Important Analysis? → J/K: Save or Skip LM Save
                importance = nlu_result.importance_score
                threshold = config_manager.get_config().nlu.importance_threshold
                message_preview = nlu_result.content[:30] + "..." if len(nlu_result.content) > 30 else nlu_result.content
                
                if should_save_to_longterm(nlu_result):
                    # J: Save Analysis to LM
                    self.lm.add_nlu_analysis(user_id, nlu_result)
                    logger.info("✅ SAVED to LM", 
                               user_id=user_id,
                               message=message_preview,
                               primary_intent=nlu_result.primary_intent,
                               importance_score=f"{importance:.3f}",
                               threshold=f"{threshold:.3f}",
                               entities_count=len(nlu_result.entities),
                               message_length=len(nlu_result.content))
                else:
                    # K: Skip LM Save
                    logger.info("🚫 FILTERED OUT", 
                               user_id=user_id,
                               message=message_preview,
                               primary_intent=nlu_result.primary_intent,
                               importance_score=f"{importance:.3f}",
                               threshold=f"{threshold:.3f}",
                               reason="Below importance threshold")
                
                # Extract business insights for logging
                insights = get_business_insights_from_nlu(nlu_result)
                logger.info("Business insights extracted", 
                           user_id=user_id,
                           customer_intent=insights.get('customer_intent'),
                           urgency_level=insights.get('urgency_level'),
                           requires_attention=insights.get('requires_human_attention'))
            
            # L: Generate Response with selective context based on NLU routing
            context_selection = context_router.determine_required_contexts(nlu_result)
            estimated_tokens = context_router.estimate_token_usage(context_selection)
            
            logger.info("Context routing completed", 
                       context_selection=context_selection,
                       estimated_tokens=estimated_tokens)
            
            response_content = generate_response(conversation_messages, lm_context, context_selection)
            
            logger.info("NLU analysis and response generated", 
                       user_id=user_id,
                       has_nlu_result=nlu_result is not None,
                       response_length=len(response_content))
            
            return nlu_result, response_content
            
        except Exception as e:
            logger.error("Failed to process message with NLU", 
                        user_id=user_id, 
                        error=str(e))
            # Return fallback response
            return None, "ขอโทษครับ เกิดข้อผิดพลาดในการประมวลผล กรุณาลองใหม่อีกครั้งครับ"
    
    def print_session_summary(self):
        """Print session token usage summary"""
        token_tracker.print_session_summary()
    

# Global instance
llm_processor = LLMProcessor()