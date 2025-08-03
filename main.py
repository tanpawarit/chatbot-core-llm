from datetime import datetime, timezone
from src.models import Message, MessageRole
from src.llm.processor import nlu_processor
from src.memory.manager import memory_manager
from src.utils.logging import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


def get_user_id() -> str:
    """Get user ID from input"""
    while True:
        user_id = input("Enter your user ID: ").strip()
        if user_id:
            return user_id
        print("Please enter a valid user ID.")


def process_user_input_simple(user_id: str, user_input: str) -> dict:
    """
    Simplified workflow replacing LangGraph complexity
    Implements the same flow: A→B→C→D→E→F→G→H→I→J/K→M→N
    """
    logger.info("Starting simplified workflow", user_id=user_id)
    
    # A: Create user message
    user_message = Message(
        role=MessageRole.USER,
        content=user_input,
        timestamp=datetime.now(timezone.utc)
    )
    
    # B→C→D→E→F→G: Memory processing (handled by memory_manager)
    conversation = memory_manager.process_user_message(user_id, user_message)
    
    # H→I→J/K→M: Process through NLU processor (includes LM context)
    nlu_result, assistant_response = nlu_processor.process_message(
        user_id, user_message, conversation.messages
    )
     
    # Check if important NLU analysis was saved
    is_important_event = nlu_result is not None and nlu_result.importance_score >= 0.7
    
    # N: Save assistant response to SM
    assistant_message = Message(
        role=MessageRole.ASSISTANT,
        content=assistant_response,
        timestamp=datetime.now(timezone.utc)
    )
    memory_manager.add_assistant_response(user_id, assistant_message)
    
    # Get final context
    context = memory_manager.get_conversation_context(user_id)
    
    return {
        "user_message": user_message,
        "conversation": conversation,
        "nlu_result": nlu_result,
        "is_important_event": is_important_event,
        "assistant_response": assistant_response,
        "assistant_message": assistant_message,
        "context": context
    }


def main():
    """
    Simple chat interface implementing your flow diagram with LLM context printing
    """
    print("🤖 Chatbot with Dual Memory System (Debug Mode)")
    print("Type 'quit' to exit, 'new' for new user")
    print("-" * 50)
    
    user_id = get_user_id()
    logger.info("Starting session", user_id=user_id)
    print(f"Welcome {user_id}! 🎉")
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'new':
                user_id = get_user_id()
                logger.info("Starting new user session", user_id=user_id)
                print(f"🆕 New user session started for {user_id}")
                continue
            
            if not user_input:
                continue
            
            # Process user input through simplified workflow (A→B→C...→G→H→I→J/K→M→N)
            final_state = process_user_input_simple(user_id, user_input)
            
            # # Display response
            print(f"\n🤖 Bot: {final_state['assistant_response']}") 

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error("Error in chat loop", error=str(e))
            print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
