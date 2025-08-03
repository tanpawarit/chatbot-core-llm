"""NLU processing orchestrator - Replaces event processing with NLU analysis"""

from typing import Optional, Tuple
from src.models import Message, NLUResult
from src.llm.node.classification_llm import analyze_message_nlu, should_save_to_longterm, get_business_insights_from_nlu
from src.llm.node.response_llm import generate_response
from src.memory.long_term import long_term_memory
from src.utils.logging import get_logger

logger = get_logger(__name__)


class NLUProcessor:
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
            
            # Perform NLU analysis
            nlu_result = analyze_message_nlu(user_message.content, context_messages)
            
            if nlu_result:
                # I: Important Analysis? → J/K: Save or Skip LM Save
                if should_save_to_longterm(nlu_result):
                    # J: Save Analysis to LM
                    self.lm.add_nlu_analysis(user_id, nlu_result)
                    logger.info("Important NLU analysis saved to LM", 
                               user_id=user_id, 
                               primary_intent=nlu_result.primary_intent,
                               importance_score=nlu_result.importance_score)
                else:
                    # K: Skip LM Save
                    logger.debug("NLU analysis not important enough for LM", 
                                user_id=user_id,
                                importance_score=nlu_result.importance_score)
                
                # Extract business insights for logging
                insights = get_business_insights_from_nlu(nlu_result)
                logger.info("Business insights extracted", 
                           user_id=user_id,
                           customer_intent=insights.get('customer_intent'),
                           urgency_level=insights.get('urgency_level'),
                           requires_attention=insights.get('requires_human_attention'))
            
            # L: Generate Response with LM context (loaded before processing current message)
            response_content = generate_response(conversation_messages, lm_context)
            
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
    
    def get_customer_profile(self, user_id: str) -> dict:
        """
        Get customer profile based on NLU analysis history.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict containing customer profile information
        """
        try:
            lm = self.lm.load(user_id)
            if not lm:
                return {
                    "profile_exists": False,
                    "message": "No historical data available"
                }
            
            preferences = lm.get_customer_preferences()
            
            # Calculate additional metrics
            total_analyses = len(lm.nlu_analyses)
            important_analyses = len(lm.get_important_analyses())
            
            # Get recent intent patterns
            recent_intents = []
            if lm.nlu_analyses:
                recent_analyses = lm.nlu_analyses[-10:]  # Last 10 analyses
                for analysis in recent_analyses:
                    if analysis.primary_intent:
                        recent_intents.append(analysis.primary_intent)
            
            profile = {
                "profile_exists": True,
                "user_id": user_id,
                "total_interactions": total_analyses,
                "important_interactions": important_analyses,
                "engagement_score": important_analyses / total_analyses if total_analyses > 0 else 0,
                "preferences": preferences,
                "recent_intents": recent_intents[-5:],  # Last 5 intents
                "created_at": lm.created_at.isoformat(),
                "last_updated": lm.updated_at.isoformat()
            }
            
            return profile
            
        except Exception as e:
            logger.error("Failed to get customer profile", user_id=user_id, error=str(e))
            return {
                "profile_exists": False,
                "error": str(e)
            }
    
    def get_conversation_insights(self, user_id: str) -> dict:
        """
        Get conversation insights for the current session.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict containing conversation insights
        """
        try:
            lm = self.lm.load(user_id)
            insights = {
                "has_history": lm is not None,
                "conversation_patterns": {},
                "recommendations": []
            }
            
            if lm and lm.nlu_analyses:
                # Analyze intent patterns
                intent_counts = {}
                entity_types = set()
                sentiment_distribution = {"positive": 0, "negative": 0, "neutral": 0}
                
                for analysis in lm.nlu_analyses:
                    # Count intents
                    if analysis.primary_intent:
                        intent_counts[analysis.primary_intent] = intent_counts.get(analysis.primary_intent, 0) + 1
                    
                    # Collect entity types
                    for entity in analysis.entities:
                        entity_types.add(entity.type)
                    
                    # Count sentiments
                    if analysis.sentiment:
                        sentiment_distribution[analysis.sentiment.label] += 1
                
                insights["conversation_patterns"] = {
                    "common_intents": sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)[:3],
                    "entity_types_mentioned": list(entity_types),
                    "sentiment_distribution": sentiment_distribution,
                    "total_interactions": len(lm.nlu_analyses)
                }
                
                # Generate recommendations
                if intent_counts:
                    top_intent = max(intent_counts, key=intent_counts.get)
                    if top_intent == "purchase_intent":
                        insights["recommendations"].append("Customer shows strong purchase intent - consider offering special deals")
                    elif top_intent == "inquiry_intent":
                        insights["recommendations"].append("Customer is researching - provide detailed product information")
                    elif top_intent == "support_intent":
                        insights["recommendations"].append("Customer needs support - prioritize quick resolution")
                
                if sentiment_distribution["negative"] > sentiment_distribution["positive"]:
                    insights["recommendations"].append("Customer sentiment is negative - handle with care")
            
            return insights
            
        except Exception as e:
            logger.error("Failed to get conversation insights", user_id=user_id, error=str(e))
            return {"has_history": False, "error": str(e)}


# Global instance
nlu_processor = NLUProcessor()