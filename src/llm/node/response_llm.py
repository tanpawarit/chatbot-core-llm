"""Simple response generation LLM node"""

import json
import os
from typing import List, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.config import config_manager
from src.models import Message, MessageRole, LongTermMemory
from src.utils.logging import get_logger

logger = get_logger(__name__)


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
        return "ไม่มีข้อมูลสินค้าในขณะนี้"
    
    products = product_data['products']
    formatted_products = []
    
    for product in products:
        product_info = f"- {product['name']}: {product['price']:,} บาท (คลัง: {product['stock']} ชิ้น)"
        formatted_products.append(product_info)
    
    return "\n".join(formatted_products)


def generate_response(conversation_messages: List[Message], lm_context: Optional[LongTermMemory] = None) -> str:
    """
    Generate chat response from conversation messages using LLM
    
    Args:
        conversation_messages: List of conversation messages
        lm_context: Optional long-term memory context for system prompt
        
    Returns:
        Generated response string
    """
    try:
        # Get configuration
        config = config_manager.get_openrouter_config()
        openrouter_config = config_manager.get_openrouter_config()
        
        # Initialize LLM client
        from langchain_core.utils import convert_to_secret_str
        
        llm = ChatOpenAI(
            model=config.response.model,
            api_key=convert_to_secret_str(openrouter_config.api_key),
            base_url=openrouter_config.base_url,
            temperature=config.response.temperature,
        )
        
        # Build system prompt with LM context
        system_prompt = _build_system_prompt(lm_context)
        
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
                   message_count=len(conversation_messages),
                   model=config.response.model)
        
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
        
        # Convert response content to string
        response_content = response.content if isinstance(response.content, str) else str(response.content)
        
        logger.info("Chat response generated", 
                   response_length=len(response_content))
        
        return response_content
        
    except Exception as e:
        logger.error("Failed to generate chat response", error=str(e))
        raise


def _build_system_prompt(lm_context: Optional[LongTermMemory] = None) -> str:
    """
    Build system prompt with tagged format including product details and LM context
    
    Args:
        lm_context: Optional long-term memory context
        
    Returns:
        System prompt string with structured tags
    """
    # Load product data
    product_data = _load_product_data()
    product_details = _format_product_details(product_data)
    
    # Build instructions section
    instructions = """คุณเป็นผู้ช่วยขายสินค้าคอมพิวเตอร์ที่เป็นมิตรและมีความรู้เฉพาะด้าน
            สามารถให้คำแนะนำเกี่ยวกับสินค้าคอมพิวเตอร์ อุปกรณ์ฮาร์ดแวร์ และการประกอบเครื่อง
            ตอบคำถามเกี่ยวกับราคา สเปค และความเหมาะสมของสินค้าต่างๆ
            ให้ตอบเป็นภาษาไทยที่สุภาพ เป็นมิตร และเข้าใจง่าย"""
    
    # Start building the tagged prompt
    prompt_parts = [f"<instructions>\n{instructions}\n</instructions>"]
    
    # Add product details section
    prompt_parts.append(f"\n<product_details>\nสินค้าที่มีจำหน่าย:\n{product_details}\n</product_details>")
    
    # Add long-term memory section if available
    if lm_context and lm_context.events:
        lm_content_parts = ["ข้อมูลประวัติสำคัญของผู้ใช้:"]
        
        important_events = lm_context.get_important_events(threshold=0.7)
        
        for event in important_events:
            lm_content_parts.append(f"- {event.event_type}: {event.content}")
            if event.classification.intent:
                lm_content_parts.append(f"  (ความต้องการ: {event.classification.intent})")
        
        if lm_context.summary:
            lm_content_parts.append(f"\nสรุปข้อมูลผู้ใช้: {lm_context.summary}")
        
        lm_content_parts.append("\nใช้ข้อมูลประวัติข้างต้นในการให้คำแนะนำที่เหมาะสม")
        
        lm_content = "\n".join(lm_content_parts)
        prompt_parts.append(f"\n<long_term_memory>\n{lm_content}\n</long_term_memory>")
    
    return "\n".join(prompt_parts)