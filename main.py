import uuid
from datetime import datetime, timezone
from src.models import Message, MessageRole
from src.memory.manager import memory_manager
from src.llm.processor import event_processor
from src.utils.logging import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


def create_conversation_id() -> str:
    """Generate a unique conversation ID"""
    return str(uuid.uuid4())


def print_llm_context(title: str, messages: list, extra_info: str = ""):
    """Print the context being sent to LLM"""
    print(f"\n🧠 {title}")
    print("=" * 60)
    for i, msg in enumerate(messages, 1):
        role_emoji = {"user": "👤", "assistant": "🤖", "system": "⚙️"}.get(msg.role, "❓")
        content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
        print(f"{i}. {role_emoji} {msg.role.upper()}: {content_preview}")
    if extra_info:
        print(f"ℹ️  {extra_info}")
    print("=" * 60)


def main():
    """
    Simple chat interface implementing your flow diagram with LLM context printing
    """
    print("🤖 Chatbot with Dual Memory System (Debug Mode)")
    print("Type 'quit' to exit, 'new' for new conversation")
    print("-" * 50)
    
    conversation_id = create_conversation_id()
    logger.info("Starting new conversation", conversation_id=conversation_id)
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'new':
                conversation_id = create_conversation_id()
                logger.info("Starting new conversation", conversation_id=conversation_id)
                print(f"🆕 New conversation started")
                continue
            
            if not user_input:
                continue
            
            # Create user message
            user_message = Message(
                role=MessageRole.USER,
                content=user_input,
                timestamp=datetime.now(timezone.utc)
            )
            
            # Process user message through memory system (A→B→C...→G)
            conversation = memory_manager.process_user_message(conversation_id, user_message)
            
            # Show context for event classification
            event_classification_messages = [
                Message(role=MessageRole.SYSTEM, content="Event classification system"),
                user_message
            ]
            print_llm_context("Event Classification Context", 
                            event_classification_messages,
                            f"Classifying: '{user_input}'")
            
            # Process event with LLM (H→I→J/K)
            event = event_processor.create_event_from_message(user_message)
            if event and event_processor.is_important_event(event):
                memory_manager.save_important_event(conversation_id, event)
                print(f"💾 Important event saved: {event.event_type} (score: {event.importance_score:.2f})")
            
            # Show context for chat response generation
            print_llm_context("Chat Response Generation Context", 
                            conversation.messages,
                            f"Total messages: {len(conversation.messages)}")
            
            # Generate response (M)
            response_text = event_processor.generate_chat_response(conversation.messages)
            
            # Create assistant message and save to SM (N)
            assistant_message = Message(
                role=MessageRole.ASSISTANT,
                content=response_text,
                timestamp=datetime.now(timezone.utc)
            )
            
            memory_manager.add_assistant_response(conversation_id, assistant_message)
            
            # Display response
            print(f"\n🤖 Bot: {response_text}")
            
            # Show conversation context summary
            context = memory_manager.get_conversation_context(conversation_id)
            print(f"\n📊 Memory Summary: {context.get('current_messages', 0)} messages, {context.get('important_events', 0)} important events")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error("Error in chat loop", error=str(e))
            print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
