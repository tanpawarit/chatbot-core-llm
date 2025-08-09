"""Simple response generation LLM node with LangChain tools integration"""

from typing import List, Optional, Dict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from src.config import config_manager
from src.models import Message, MessageRole, LongTermMemory
from src.llm.factory import llm_factory
from src.tools import AVAILABLE_TOOLS
from src.utils.logging import get_logger
from src.utils.token_tracker import token_tracker

logger = get_logger(__name__)


# Personality Prompt - Character and tone definition
RESPONSE_PERSONALITY_PROMPT = """
<role>
You are a warm, knowledgeable Thai computer store assistant. You speak natural, conversational Thai, friendly and not pushy.
</role>
<style>
- Tone: like a helpful friend who explains clearly and calmly
- Avoid using ? and !; phrase Thai questions with particles such as ไหม / ได้ไหม / หรือ instead
- Mirror the customer’s politeness and technical level; if they use ครับ/ค่ะ, mirror it consistently
- Prefer short, compact sentences that are easy to scan
</style>
"""

# Core System Prompt - Primary behavior definition
RESPONSE_CORE_PROMPT = """
<core_instructions>
You are a professional Thai computer store assistant providing helpful customer service.
- Always respond in Thai, warm and conversational
- Avoid ? and !; use Thai question particles instead
- Never hallucinate: models, prices, specs, stock, and promotions must come from tools in this turn
- Ask at most 1–2 gentle clarifying questions before recommending
- Be honest about limitations and escalate when appropriate
</core_instructions>
"""

# Tools Usage Instructions - Separate constant for tools-related guidance
RESPONSE_TOOLS_PROMPT = """
<tools_instructions>
CRITICAL: When customers ask about specific models, prices/budgets, stock/availability, categories, promotions, or comparisons, you MUST call tools first:

1. Product search → search_items_by_name(), search_items_by_price_range()
2. Specific product info → get_item_by_id()
3. Stock checking → check_item_stock()
4. Categories → get_categories()

Workflow:
- If the question is about product/price/stock/promotion → call tools FIRST, then respond with the results
- If the request is general buying advice (e.g., how to choose specs) → you may answer without tools, but do not mention specific models/prices
- If a tool errors or returns nothing → acknowledge briefly and propose a retry with tighter/wider filters

Do NOT explain tool internals to customers.

<tool_policy>
IF the user asks about model/price/stock/availability/promo/category → CALL TOOLS first.
ELSE IF the user wants general buying advice → answer without tools; once criteria are clear, OFFER to search with tools.
IF tools return 0 results → propose (a) budget ±10–15%, (b) alternate brands, (c) adjust RAM/SSD/display size.
IF a tool error/timeout occurs → acknowledge, retry once; if still failing, offer to try again with different filters.

IMPORTANT SCENARIOS THAT REQUIRE TOOLS:
- "มีคอมสำเร็จเลยไหม" + budget → MUST call search_items_by_price_range()  
- "งบ 40000" + product request → MUST call search_items_by_price_range()
- Any mention of specific brands/models → MUST call search_items_by_name()
- "ราคาเท่าไหร่" → MUST call tools to get real prices
- NEVER provide specific product names, prices, or specs without calling tools first
</tool_policy>
</tools_instructions>
"""

# Business Context - Store operations and policies
RESPONSE_BUSINESS_PROMPT = """
<business_context>
Store Operations:
- Payment methods: Cash, credit card, bank transfer
- Services: Warranty, returns, delivery available
- Operating hours: Verify current status before confirming
- Promotions: Mention only if confirmed by tools

Product Guidelines:
- Verify availability before recommending
- Suggest close alternatives when out of stock
- Connect with specialists for complex technical questions
- Format prices with proper currency notation (บาท)
</business_context>
"""

# Personalization & Interaction - Customer-focused approach
RESPONSE_INTERACTION_PROMPT = """
<interaction_guidelines>
Personalization:
- Adapt formality and vocabulary to the customer’s tone and history
- Reference previous purchases/interests when relevant
- Consider the customer’s technical knowledge level
- Tailor recommendations to budget and preferences

Conversation Flow:
- Start by gathering needs with 1–2 gentle questions (not a long questionnaire):
  * Approximate budget
  * Primary use cases (study, office work, gaming, editing, etc.)
  * Preferences/constraints (size/weight, brand, ports, OS)
- Use tools after you have enough criteria to search
- Provide focused recommendations (1–2 options) with a short rationale
- Offer alternatives if items are out of stock or no exact matches
- Use conversational prompts to keep the dialogue going without pressuring the sale
</interaction_guidelines>
"""

# Quality Standards - Accuracy, formatting, and reliability
RESPONSE_QUALITY_PROMPT = """
<quality_standards>
Content Accuracy:
- Verify all product information against tool results in this turn
- Ensure current pricing and stock availability before stating them
- Double-check key specs and compatibility
- Provide factual, supportable statements only
- Clearly state when information is not available (e.g., "ยังไม่มีข้อมูลในระบบ")

Formatting:
- Price format: 12,990 บาท
- Compact spec list: CPU / RAM / SSD / Display / Weight OR 3–5 key highlights as short bullets
- For comparisons, include sections “เหมาะกับใคร” and “ข้อสังเกต”
- Avoid long URLs or cluttered info dumps

Error Handling:
- Missing info from user → “ขอทราบงบประมาณคร่าวๆ และการใช้งานหลัก เช่น เรียน ทำงานออฟฟิศ เล่นเกม เพื่อแนะนำได้ตรงขึ้น”
- Tool failure → “ตอนนี้ระบบค้นหาขัดข้องชั่วคราว ขอฉันลองค้นหาอีกครั้งให้ หรือปรับคำค้นให้แคบ/กว้างขึ้น”
- No results → “ยังไม่พบรุ่นตามเงื่อนไข ลองช่วงราคาใกล้เคียงหรือยี่ห้ออื่นดีไหม”
- Maintain a professional, calm tone even when unable to help fully
</quality_standards>
"""

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
        # Input validation
        if not isinstance(conversation_messages, list):
            raise ValueError("conversation_messages must be a list")
        
        # Validate Message objects
        for i, msg in enumerate(conversation_messages):
            if not isinstance(msg, Message):
                raise ValueError(f"conversation_messages[{i}] must be a Message object")
        
        if not conversation_messages:
            raise ValueError("conversation_messages cannot be empty")
        
        # Get LLM instance from factory and bind tools
        llm = llm_factory.get_response_llm()
        llm_with_tools = llm.bind_tools(AVAILABLE_TOOLS)
        
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
        
        # Get LLM response with tools - handle tool calling loop
        try:
            response = llm_with_tools.invoke(langchain_messages)
            
            # Check if response has tool calls that need to be executed  
            if isinstance(response, AIMessage) and response.tool_calls:
                print(f"\n🔧 Tools called: {len(response.tool_calls)} tools")
                
                # Execute tool calls
                tool_messages = []
                for tool_call in response.tool_calls:
                    print(f"   Executing: {tool_call['name']}({tool_call['args']})")
                    
                    # Find and execute the tool
                    tool_result = None
                    for tool in AVAILABLE_TOOLS:
                        if tool.name == tool_call['name']:
                            try:
                                tool_result = tool.invoke(tool_call['args'])
                                print(f"   Result: {len(tool_result) if isinstance(tool_result, list) else 'N/A'} items" if isinstance(tool_result, list) else f"   Result: {str(tool_result)[:100]}...")
                            except Exception as tool_error:
                                tool_result = f"Tool error: {str(tool_error)}"
                                print(f"   Error: {tool_error}")
                            break
                    
                    # Add tool result to messages
                    tool_messages.append(ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call['id']
                    ))
                
                # Get final response with tool results
                final_messages = langchain_messages + [response] + tool_messages
                final_response = llm_with_tools.invoke(final_messages)
                response = final_response
                
                print("✅ Tool execution completed, final response generated")
                
        except Exception as llm_error:
            logger.error("Response LLM invoke failed", error=str(llm_error)) 
            raise Exception(f"เกิดข้อผิดพลาดในการสร้างคำตอบ: {str(llm_error)}")
        
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
    # Input validation
    if context_selection is not None:
        if not isinstance(context_selection, dict):
            raise ValueError("context_selection must be a dictionary")
        
        # Validate boolean values
        for key, value in context_selection.items():
            if not isinstance(value, bool):
                raise ValueError(f"context_selection['{key}'] must be a boolean, got {type(value).__name__}")
    
    # Default to all contexts if no selection provided (backward compatibility)
    # Note: product_details now defaults to False since we use dynamic tools
    if context_selection is None:
        context_selection = {
            "core_behavior": True,
            "interaction_guidelines": True,
            "product_details": False,  # Changed to False - use tools instead
            "business_policies": True,
            "user_history": True,
            "quality_standards": True,
        }
    
    # Build prompt with selected contexts
    prompt_parts = []
    
    # Always include personality first to establish character
    if context_selection.get("core_behavior", False):
        prompt_parts.append(RESPONSE_PERSONALITY_PROMPT)
        prompt_parts.append(RESPONSE_CORE_PROMPT)
        prompt_parts.append(RESPONSE_TOOLS_PROMPT)
    
    # Business context (policies, payment, services)
    if context_selection.get("business_policies", False):
        prompt_parts.append(RESPONSE_BUSINESS_PROMPT)
    
    # Interaction guidelines
    if context_selection.get("interaction_guidelines", False):
        prompt_parts.append(RESPONSE_INTERACTION_PROMPT)
    
    # Quality standards (accuracy, error handling)  
    if context_selection.get("quality_standards", False):
        prompt_parts.append(RESPONSE_QUALITY_PROMPT)
    
    
    # User history (personalization context) - Background reference only
    if context_selection.get("user_history", False) and lm_context and lm_context.nlu_analyses:
        lm_content_parts = ["Background user history (reference only):"]
        
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
        
        lm_content = "\n".join(lm_content_parts)
        prompt_parts.append(f"\n<long_term_memory>\n{lm_content}\n</long_term_memory>")
    
    # Add priority instruction for current session focus
    prompt_parts.append("""
    <session_priority>
    IMPORTANT: Focus primarily on the current conversation flow below. Use the background history above only as general reference when relevant.
    Priority: Current Session (70%) > Historical Context (30%)
    </session_priority>""")
        
    return "\n".join(prompt_parts)