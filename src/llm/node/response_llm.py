"""Simple response generation LLM node"""

import json
import os
from typing import List, Optional, Dict, Any 
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.config import config_manager
from src.models import Message, MessageRole, LongTermMemory
from src.llm.factory import llm_factory
from src.utils.logging import get_logger
from src.utils.token_tracker import token_tracker

logger = get_logger(__name__)


# Core System Prompt - Primary behavior definition
RESPONSE_CORE_PROMPT = """
<core_instructions>
You are a professional Thai computer store assistant providing helpful customer service.
- Respond in natural, conversational Thai without question marks (?) or exclamation marks (!)
- Only provide information based on available data - never hallucinate specifications or pricing
- Use gentle suggestions instead of direct questions for clarification
- Maintain warm, professional tone like a knowledgeable friend helping out
- Be honest about limitations and escalate when appropriate
</core_instructions>
"""

# Business Context - Store operations and policies
RESPONSE_BUSINESS_PROMPT = """
<business_context>
Store Operations:
- Payment methods: Cash, credit card, bank transfer
- Services: Warranty, returns, delivery available
- Operating hours: Check current availability
- Promotions: Inquire about current offers

Product Guidelines: 
- Verify availability before recommending
- Suggest alternatives when out of stock
- Connect with specialists for complex technical questions
- Format prices with proper currency notation (บาท)
</business_context>
"""

# Personalization & Interaction - Customer-focused approach
RESPONSE_INTERACTION_PROMPT = """
<interaction_guidelines>
Personalization Strategy:
- Adapt formality based on customer interaction history
- Reference previous purchases or interests when relevant
- Consider customer's technical knowledge level
- Tailor recommendations to stated budget and preferences

Conversation Flow:
- Acknowledge customer needs before providing solutions
- Confirm understanding before making recommendations
- Provide alternative options when primary choice isn't available
- End responses with clear, actionable next steps
- Use structured formatting with bullet points for clarity
</interaction_guidelines>
"""

# Quality Standards - Accuracy and reliability
RESPONSE_QUALITY_PROMPT = """
<quality_standards>
Content Accuracy Requirements:
- Verify all product information against available data
- Ensure current pricing and stock availability
- Double-check specifications and compatibility
- Provide factual, supportable statements only
- State clearly when information is not available

Error Handling Protocol:
- Suggest similar alternatives for out-of-stock items
- Mention checking current pricing if data may be outdated
- Escalate complex technical issues appropriately
- Maintain professional tone even when unable to help fully
</quality_standards>
"""


def _load_product_data() -> Optional[Dict[str, Any]]:
    """
    Load product data from JSON file
    
    Returns:
        Product data dictionary or None if file not found
    """
    try:
        products_path = "data/product_detail/products.json"
        if os.path.exists(products_path):
            with open(products_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.warning("Product data file not found", path=products_path)
            return None
    except Exception as e:
        logger.error("Failed to load product data", error=str(e))
        return None


def _format_product_details(product_data: Optional[Dict[str, Any]]) -> str:
    """
    Format product data for system prompt
    
    Args:
        product_data: Product data dictionary
        
    Returns:
        Formatted product details string
    """
    if not product_data or 'products' not in product_data:
        return "No product data available at this time"
    
    products = product_data['products']
    formatted_products = []
    
    for product in products:
        product_info = f"- {product['name']}: {product['price']:,} บาท (คลัง: {product['stock']} ชิ้น)"
        formatted_products.append(product_info)
    
    return "\n".join(formatted_products)


def generate_response(conversation_messages: List[Message], 
                     lm_context: Optional[LongTermMemory] = None,
                     context_selection: Optional[Dict[str, bool]] = None) -> str:
    """
    Generate chat response from conversation messages using LLM
    
    Args:
        conversation_messages: List of conversation messages
        lm_context: Optional long-term memory context for system prompt
        context_selection: Optional context selection for prompt building
        
    Returns:
        Generated response string
    """
    try:
        # Get LLM instance from factory
        llm = llm_factory.get_response_llm()
        
        # Get configuration for logging
        config = config_manager.get_openrouter_config()
        
        # Build system prompt with context selection
        system_prompt = _build_system_prompt(lm_context, context_selection)
        
        # Convert to LangChain message format
        langchain_messages = []
        
        # Add system message with context if available
        if system_prompt:
            langchain_messages.append(SystemMessage(content=system_prompt))
        
        for msg in conversation_messages:
            if msg.role == MessageRole.USER:
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.role == MessageRole.ASSISTANT:
                langchain_messages.append(AIMessage(content=msg.content))
            elif msg.role == MessageRole.SYSTEM:
                langchain_messages.append(SystemMessage(content=msg.content))
        
        logger.info("Generating chat response", 
                   message_count=len(conversation_messages))
        
        # Pretty print Response LLM Context
        print("\n" + "="*60)
        print("🤖 Response LLM Context")
        print("="*60)
        for i, msg in enumerate(langchain_messages, 1):
            role = type(msg).__name__.replace("Message", "").upper()
            print(f"{i}. [{role}] {msg.content}")
        print("="*60)
        
        # Get LLM response
        response = llm.invoke(langchain_messages)
        
        # Track token usage
        usage = token_tracker.track_response(response, config.response.model, "response")
        if usage:
            token_tracker.print_usage(usage, "🤖")
        else:
            print("🤖 Response LLM Usage: No usage metadata available")
        
        # Convert response content to string
        response_content = response.content if isinstance(response.content, str) else str(response.content)
        
        logger.info("Chat response generated", 
                   response_length=len(response_content))
        
        return response_content
        
    except Exception as e:
        logger.error("Failed to generate chat response", error=str(e))
        raise


def _build_system_prompt(lm_context: Optional[LongTermMemory] = None, 
                        context_selection: Optional[Dict[str, bool]] = None) -> str:
    """
    Build system prompt with selective context based on routing
    
    Args:
        lm_context: Optional long-term memory context
        context_selection: Dict of context types to include
        
    Returns:
        System prompt string with selected contexts
    """
    # Default to all contexts if no selection provided (backward compatibility)
    if context_selection is None:
        context_selection = {
            "core_behavior": True,
            "interaction_guidelines": True,
            "product_details": True,
            "business_policies": True,
            "user_history": True,
        }
    
    # Build prompt with selected contexts
    prompt_parts = []
    
    # Always include core behavior
    if context_selection.get("core_behavior", False):
        prompt_parts.append(RESPONSE_CORE_PROMPT)
    
    # Business context (policies, payment, services)
    if context_selection.get("business_policies", False):
        prompt_parts.append(RESPONSE_BUSINESS_PROMPT)
    
    # Interaction guidelines
    if context_selection.get("interaction_guidelines", False):
        prompt_parts.append(RESPONSE_INTERACTION_PROMPT)
    
    # Quality standards (accuracy, error handling)  
    if context_selection.get("quality_standards", False):
        prompt_parts.append(RESPONSE_QUALITY_PROMPT)
    
    # Product details (expensive context)
    if context_selection.get("product_details", False):
        product_data = _load_product_data()
        product_details = _format_product_details(product_data)
        prompt_parts.append(f"\n<product_details>\nAvailable products:\n{product_details}\n</product_details>")
    
    # User history (personalization context)
    if context_selection.get("user_history", False) and lm_context and lm_context.nlu_analyses:
        lm_content_parts = ["Important user history:"]
        
        important_analyses = lm_context.get_important_analyses(threshold=0.7)
        
        # Limit to most recent 5 analyses
        recent_important_analyses = important_analyses[-5:] if len(important_analyses) > 5 else important_analyses
        
        for analysis in recent_important_analyses:
            lm_content_parts.append(f"- User said: {analysis.content}")
            if analysis.primary_intent:
                lm_content_parts.append(f"  (Intent: {analysis.primary_intent})")
            if analysis.entities:
                entity_values = [e.value for e in analysis.entities]
                lm_content_parts.append(f"  (Mentioned: {', '.join(entity_values)})")
        
        if lm_context.summary:
            lm_content_parts.append(f"\nUser summary: {lm_context.summary}")
        
        lm_content_parts.append("\nUse the above history to provide appropriate recommendations")
        
        lm_content = "\n".join(lm_content_parts)
        prompt_parts.append(f"\n<long_term_memory>\n{lm_content}\n</long_term_memory>")
    
    return "\n".join(prompt_parts)