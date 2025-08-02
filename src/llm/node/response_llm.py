"""Simple response generation LLM node"""

from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.config import config_manager
from src.models import Message, MessageRole, LongTermMemory
from src.utils.logging import get_logger

logger = get_logger(__name__)


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
    Build system prompt with long-term memory context
    
    Args:
        lm_context: Optional long-term memory context
        
    Returns:
        System prompt string with context
    """
    base_prompt = """คุณเป็นผู้ช่วยที่เป็นมิตรและมีประโยชน์ สามารถตอบคำถามและช่วยเหลือผู้ใช้ได้หลากหลายเรื่อง

                ให้ตอบเป็นภาษาไทยที่สุภาพและเป็นมิตร"""
    
    if not lm_context or not lm_context.events:
        return base_prompt
    
    # Build context from important events
    context_parts = [base_prompt]
    context_parts.append("\n\n=== ข้อมูลประวัติสำคัญของผู้ใช้ ===")
    
    important_events = lm_context.get_important_events(threshold=0.7)
    
    for event in important_events:
        context_parts.append(f"\n- {event.event_type}: {event.content}")
        if event.classification.intent:
            context_parts.append(f"  (ความต้องการ: {event.classification.intent})")
    
    if lm_context.summary:
        context_parts.append(f"\n\nสรุปข้อมูลผู้ใช้: {lm_context.summary}")
    
    context_parts.append("\n\nใช้ข้อมูลประวัติข้างต้นในการตอบคำถามอย่างเหมาะสม")
    
    return "".join(context_parts)