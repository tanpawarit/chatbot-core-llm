"""
Context Routing System - Simple and Scalable
เชื่อมโยงกับ NLU intents ใน config.yaml เพื่อกำหนด context ที่เหมาะสม
"""

from typing import Dict, List, Set, Optional
from src.models import NLUResult
from src.config import config_manager
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ContextRouter:
    """Simple context router based on NLU intents and entities"""
    
    def __init__(self):
        self.config = config_manager.get_config()
    
    def determine_required_contexts(self, nlu_result: Optional[NLUResult]) -> Dict[str, bool]:
        """
        กำหนด context ที่ต้องการจาก NLU analysis
        Routing ตาม default_intent: greet, purchase_intent, inquiry_intent, support_intent, complain_intent
        
        Args:
            nlu_result: NLU analysis result
            
        Returns:
            Dict ของ context types และสถานะการใช้งาน
        """
        if not nlu_result or not nlu_result.intents:
            logger.info("No NLU result, using full contexts")
            return self._get_full_contexts()
        
        # ดึง intent names
        intent_names = {intent.name for intent in nlu_result.intents}
        
        # Parse default intents from config
        default_intents = self._parse_default_intents()
        
        # Context routing based on default_intent
        if "greet" in intent_names and "greet" in default_intents:
            contexts = self._get_minimal_contexts()
            routing_type = "minimal"
        elif "purchase_intent" in intent_names and "purchase_intent" in default_intents:
            contexts = self._get_product_focused_contexts()
            routing_type = "product_focused"
        elif ("support_intent" in intent_names or "complain_intent" in intent_names) and \
             any(intent in default_intents for intent in ["support_intent", "complain_intent"]):
            contexts = self._get_support_focused_contexts()
            routing_type = "support_focused"
        elif "inquiry_intent" in intent_names and "inquiry_intent" in default_intents:
            contexts = self._get_full_contexts()
            routing_type = "full"
        else:
            # additional_intent หรือ intent ที่ไม่รู้จัก = full context (safe)
            contexts = self._get_full_contexts()
            routing_type = "full_default"
        
        # Log final context selection
        active_contexts = [k for k, v in contexts.items() if v]
        logger.info("Context routing completed", 
                   routing_type=routing_type,
                   intents=list(intent_names),
                   active_contexts=active_contexts,
                   total_contexts=len(active_contexts))
        
        return contexts
    
    def _parse_default_intents(self) -> set:
        """Parse default_intent string from config to get intent names"""
        try:
            default_intent_str = self.config.nlu.default_intent
            # Parse "greet:0.3, purchase_intent:0.8" -> {"greet", "purchase_intent"}
            intents = set()
            for item in default_intent_str.split(','):
                intent_name = item.strip().split(':')[0].strip()
                intents.add(intent_name)
            return intents
        except Exception as e:
            logger.error("Failed to parse default intents", error=str(e))
            return set()
    
    def _get_minimal_contexts(self) -> Dict[str, bool]:
        """Minimal contexts สำหรับ greet - ประหยัด ~64% tokens"""
        return {
            "core_behavior": True,         # Basic personality (จำเป็น)
            "interaction_guidelines": True, # Response formatting (จำเป็น)  
            "user_history": True,          # Personalize greeting (ดี)
            "product_details": False,      # ไม่จำเป็น (ประหยัด 800 tokens)
            "business_policies": False,    # ไม่จำเป็น (ประหยัด 200 tokens)
        }
    
    def _get_product_focused_contexts(self) -> Dict[str, bool]:
        """Product-focused contexts สำหรับ purchase_intent - ประหยัด ~19% tokens"""
        return {
            "core_behavior": True,         # Basic personality
            "interaction_guidelines": True, # Response formatting
            "product_details": True,       # Product catalog (จำเป็นสำหรับการซื้อ)
            "business_policies": True,     # Payment, delivery policies (จำเป็น)
            "user_history": False,         # ไม่จำเป็นสำหรับการซื้อใหม่ (ประหยัด 300 tokens)
        }
    
    def _get_support_focused_contexts(self) -> Dict[str, bool]:
        """Support-focused contexts สำหรับ support_intent, complain_intent - ประหยัด ~52% tokens"""
        return {
            "core_behavior": True,         # Basic personality
            "interaction_guidelines": True, # Response formatting
            "business_policies": True,     # Store policies (จำเป็นสำหรับ support)
            "user_history": True,          # Customer history (ช่วยแก้ปัญหา)
            "product_details": False,      # ไม่เกี่ยวกับสินค้าใหม่ (ประหยัด 800 tokens)
        }
    
    def _get_full_contexts(self) -> Dict[str, bool]:
        """Full contexts สำหรับ inquiry_intent และ additional_intent - safe default"""
        return {
            "core_behavior": True,         # Basic personality
            "interaction_guidelines": True, # Response formatting
            "user_history": True,          # Customer history
            "product_details": True,       # Product catalog (expensive but necessary)
            "business_policies": True,     # Store policies
        }
    
    def estimate_token_usage(self, contexts: Dict[str, bool]) -> int:
        """
        ประมาณการ token usage จาก context ที่เลือก
        
        Args:
            contexts: Context selection dict
            
        Returns:
            Estimated token count
        """
        # Token estimates for each context type
        token_estimates = {
            "core_behavior": 100,
            "interaction_guidelines": 150,
            "product_details": 800,     # Largest context (expensive!)
            "business_policies": 200,
            "user_history": 300,
        }
        
        total_tokens = sum(
            token_estimates.get(context_type, 0) 
            for context_type, enabled in contexts.items() 
            if enabled
        )
        
        # Calculate savings vs full context
        full_context_tokens = sum(token_estimates.values())  # 1550 tokens
        savings = full_context_tokens - total_tokens
        savings_percent = (savings / full_context_tokens) * 100 if full_context_tokens > 0 else 0
        
        # Determine routing type for logging
        routing_type = "unknown"
        if total_tokens == 550:  # minimal
            routing_type = "minimal (greet)"
        elif total_tokens == 1250:  # product focused
            routing_type = "product_focused (purchase)"
        elif total_tokens == 750:  # support focused
            routing_type = "support_focused (support/complain)"
        elif total_tokens == 1550:  # full
            routing_type = "full (inquiry/additional)"
        
        logger.info("Token usage estimated", 
                   total_estimated_tokens=total_tokens,
                   savings_tokens=savings,
                   savings_percent=f"{savings_percent:.1f}%",
                   routing_type=routing_type,
                   active_contexts=[k for k, v in contexts.items() if v])
        
        return total_tokens


# Global instance
context_router = ContextRouter()