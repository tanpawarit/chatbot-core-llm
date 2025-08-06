import argparse
from datetime import datetime, timezone
from pathlib import Path
from src.models import Message, MessageRole
from src.llm.processor import llm_processor
from src.memory.manager import memory_manager
from src.media.processor import MediaProcessor
from src.utils.logging import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)
media_processor = MediaProcessor()


def parse_arguments():
    """Parse command line arguments for multimodal input"""
    parser = argparse.ArgumentParser(description='Multimodal Chatbot')
    parser.add_argument('--text', type=str, help='Text message')
    parser.add_argument('--image', type=str, help='Image file path')
    parser.add_argument('--audio', type=str, help='Audio file path')
    parser.add_argument('--video', type=str, help='Video file path')
    parser.add_argument('--user', type=str, help='User ID', default='default_user')
    return parser.parse_args()

def get_user_id() -> str:
    """Get user ID from input"""
    while True:
        user_id = input("Enter your user ID: ").strip()
        if user_id:
            return user_id
        print("Please enter a valid user ID.")

def create_message_from_input(text: str = None, image_path: str = None, 
                             audio_path: str = None, video_path: str = None) -> Message:
    """Create a multimodal message from input"""
    # Convert paths to absolute paths and ensure they exist
    image_abs = Path(image_path).resolve() if image_path else None
    audio_abs = Path(audio_path).resolve() if audio_path else None  
    video_abs = Path(video_path).resolve() if video_path else None
    
    # Validate media files if provided
    if image_abs:
        media_info = media_processor.validate_file(image_abs)
        if not media_info.is_valid:
            raise ValueError(f"Invalid image file: {media_info.error_message}")
        print(media_processor.get_media_summary(media_info))
    
    if audio_abs:
        media_info = media_processor.validate_file(audio_abs)
        if not media_info.is_valid:
            raise ValueError(f"Invalid audio file: {media_info.error_message}")
        print(media_processor.get_media_summary(media_info))
        
    if video_abs:
        media_info = media_processor.validate_file(video_abs)
        if not media_info.is_valid:
            raise ValueError(f"Invalid video file: {media_info.error_message}")
        print(media_processor.get_media_summary(media_info))
    
    # Create multimodal message
    return Message.media_message(
        role=MessageRole.USER,
        text=text,
        image_path=image_abs,
        audio_path=audio_abs,
        video_path=video_abs
    )


def process_multimodal_input(user_id: str, user_message: Message) -> dict:
    """
    Multimodal workflow for processing text and media content
    Implements the same flow: A→B→C→D→E→F→G→H→I→J/K→M→N but with multimodal support
    """
    logger.info("Starting multimodal workflow", user_id=user_id, 
               has_media=user_message.content.has_media, 
               media_type=user_message.content.media_type.value)
    
    # B→C→D→E→F→G: Memory processing (handled by memory_manager)
    conversation = memory_manager.process_user_message(user_id, user_message) 
    
    # H→I→J/K→M: Process through LLM processor (includes LM context)  
    nlu_result, assistant_response = llm_processor.process_message(
        user_id, user_message, conversation.messages
    )
     
    # Check if important NLU analysis was saved
    is_important_event = nlu_result is not None and nlu_result.importance_score >= 0.7
    
    # N: Save assistant response to SM
    assistant_message = Message.text_message(
        role=MessageRole.ASSISTANT,
        text=assistant_response
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
    Multimodal chat interface with CLI args and interactive mode
    """
    args = parse_arguments()
    
    # Check if CLI arguments were provided
    if args.text or args.image or args.audio or args.video:
        # CLI mode - single interaction
        try:
            print("🤖 Multimodal Chatbot - CLI Mode")
            user_id = args.user
            logger.info("Starting CLI session", user_id=user_id)
            
            # Create message from CLI arguments
            user_message = create_message_from_input(
                text=args.text,
                image_path=args.image,
                audio_path=args.audio, 
                video_path=args.video
            )
            
            print(f"\n👤 User [{user_id}]: {user_message.content.text or '[Media only]'}")
            
            # Process the multimodal input
            final_state = process_multimodal_input(user_id, user_message)
            
            # Display response
            print(f"\n🤖 Bot: {final_state['assistant_response']}")
            
        except Exception as e:
            logger.error("Error in CLI mode", error=str(e))
            print(f"❌ Error: {str(e)}")
            
    else:
        # Interactive mode
        print("🤖 Multimodal Chatbot - Interactive Mode")
        print("Commands: 'quit', 'new', 'stats'")
        print("Media: Use 'file:path/to/file.jpg Your message text'")
        print("-" * 50)
        
        user_id = get_user_id()
        logger.info("Starting interactive session", user_id=user_id)
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
                
                if user_input.lower() == 'stats':
                    # Show session statistics
                    print("\n📊 Session Statistics:")
                    llm_processor.print_session_summary()
                    continue
                
                if not user_input:
                    continue
                
                # Parse file input format: "file:path.jpg message text" or "file:"path with spaces.jpg" message text"
                if user_input.startswith('file:'):
                    remaining = user_input[5:]  # Remove 'file:' prefix
                    
                    # Handle quoted paths
                    if remaining.startswith('"'):
                        # Find closing quote
                        end_quote = remaining.find('"', 1)
                        if end_quote != -1:
                            file_path = remaining[1:end_quote]  # Extract path without quotes
                            text = remaining[end_quote+1:].strip()  # Rest as text
                        else:
                            # No closing quote, treat as unquoted
                            parts = remaining.split(' ', 1)
                            file_path = parts[0]
                            text = parts[1] if len(parts) > 1 else None
                    else:
                        # Regular path without quotes
                        parts = remaining.split(' ', 1)
                        file_path = parts[0]
                        text = parts[1] if len(parts) > 1 else None
                    
                    # Determine file type and create message
                    file_ext = Path(file_path).suffix.lower()
                    if file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                        user_message = create_message_from_input(text=text or "", image_path=file_path)
                    elif file_ext in ['.mp3', '.wav', '.ogg', '.m4a']:
                        user_message = create_message_from_input(text=text or "", audio_path=file_path)
                    elif file_ext in ['.mp4', '.webm', '.avi']:
                        user_message = create_message_from_input(text=text or "", video_path=file_path)
                    else:
                        print(f"❌ Unsupported file type: {file_ext}")
                        continue
                else:
                    # Text-only message
                    user_message = Message.text_message(MessageRole.USER, user_input)
                
                # Process the multimodal input
                final_state = process_multimodal_input(user_id, user_message)
                
                # Display response
                print(f"\n🤖 Bot: {final_state['assistant_response']}")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error("Error in interactive mode", error=str(e))
                print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
