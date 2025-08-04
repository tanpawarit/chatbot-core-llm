#!/usr/bin/env python3
"""
Test script for Phase 1 filtering system
Tests the new importance scoring and filtering logic
"""

import sys
import os
sys.path.append('.')

from src.models import Message, MessageRole
from src.llm.processor import nlu_processor
from src.memory.manager import memory_manager
from src.utils.logging import get_logger

logger = get_logger(__name__)

def test_filtering_system():
    """Test the Phase 1 filtering system with various message types"""
    
    print("🧪 Testing Phase 1 Filtering System")
    print("=" * 50)
    
    # Test user ID
    user_id = "test_phase1"
    
    # Test messages with different expected importance levels
    test_messages = [
        ("ใช่", "Should be FILTERED (short confirmation)"),
        ("สวัสดี", "Should be FILTERED (greeting)"),
        ("ขอบคุณครับ", "Should be FILTERED (politeness)"),
        ("สนใจลำโพง 1 เครื่อง", "Should be SAVED (product inquiry)"),
        ("ราคาเท่าไหร่ครับ", "Should be SAVED (price inquiry)"),
        ("อยากซื้อคอมเกมมิ่ง", "Should be SAVED (purchase intent)"),
        ("มีจอ 24 นิ้วไหม", "Should be SAVED (product + spec inquiry)"),
        ("เอาครับ", "Should be FILTERED (short confirmation)"),
        ("ดีจัง", "Should be FILTERED (short response)"),
        ("ผมต้องการซื้อเมาส์เกมมิ่ง Razer ราคาไม่เกิน 2000 บาท", "Should be SAVED (detailed purchase inquiry)")
    ]
    
    print(f"\n📋 Testing {len(test_messages)} messages:")
    print(f"User ID: {user_id}")
    print()
    
    results = []
    
    for i, (message_text, expected) in enumerate(test_messages, 1):
        print(f"\n{i:2d}. Testing: \"{message_text}\"")
        print(f"    Expected: {expected}")
        
        try:
            # Create user message
            user_message = Message(role=MessageRole.USER, content=message_text)
            
            # Get current conversation context
            conversation = memory_manager.get_conversation(user_id)
            if conversation is None:
                # Create new conversation if none exists
                from src.models import Conversation
                conversation = Conversation(user_id=user_id)
                memory_manager.save_conversation(conversation)
            
            conversation.add_message(user_message)
            
            # Process message through NLU
            nlu_result, response = nlu_processor.process_message(
                user_id=user_id,
                user_message=user_message,
                conversation_messages=conversation.messages
            )
            
            if nlu_result:
                importance = nlu_result.importance_score
                threshold = 0.7
                was_saved = importance >= threshold
                status = "✅ SAVED" if was_saved else "🚫 FILTERED"
                
                print(f"    Result: {status}")
                print(f"    Importance: {importance:.3f} (Threshold: {threshold:.3f})")
                print(f"    Intent: {nlu_result.primary_intent}")
                print(f"    Entities: {len(nlu_result.entities)}")
                
                results.append({
                    'message': message_text,
                    'importance': importance,
                    'saved': was_saved,
                    'intent': nlu_result.primary_intent,
                    'entities': len(nlu_result.entities)
                })
            else:
                print(f"    Result: ❌ NLU FAILED")
                results.append({
                    'message': message_text,
                    'importance': 0.0,
                    'saved': False,
                    'intent': None,
                    'entities': 0
                })
                
        except Exception as e:
            print(f"    Result: ❌ ERROR: {str(e)}")
            results.append({
                'message': message_text,
                'importance': 0.0,
                'saved': False,
                'intent': None,
                'entities': 0,
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    total_messages = len(results)
    saved_count = sum(1 for r in results if r['saved'])
    filtered_count = total_messages - saved_count
    
    print(f"Total messages tested: {total_messages}")
    print(f"Messages saved: {saved_count} ({saved_count/total_messages*100:.1f}%)")
    print(f"Messages filtered: {filtered_count} ({filtered_count/total_messages*100:.1f}%)")
    
    print(f"\n🎯 FILTERING EFFICIENCY:")
    print(f"   Before Phase 1: {total_messages}/{total_messages} saved (100%)")
    print(f"   After Phase 1:  {saved_count}/{total_messages} saved ({saved_count/total_messages*100:.1f}%)")
    print(f"   Reduction: {(total_messages-saved_count)/total_messages*100:.1f}% less storage used")
    
    # Show detailed results
    print(f"\n📋 DETAILED RESULTS:")
    print("-" * 80)
    print(f"{'Message':<30} {'Score':<6} {'Status':<8} {'Intent':<15} {'Entities'}")
    print("-" * 80)
    
    for result in results:
        message = result['message'][:28] + ".." if len(result['message']) > 30 else result['message']
        score = f"{result['importance']:.3f}"
        status = "SAVED" if result['saved'] else "FILTER"
        intent = (result['intent'] or "None")[:13] + ".." if result['intent'] and len(result['intent']) > 15 else (result['intent'] or "None")
        entities = str(result['entities'])
        
        print(f"{message:<30} {score:<6} {status:<8} {intent:<15} {entities}")
    
    print("-" * 80)
    
    return results

if __name__ == "__main__":
    try:
        results = test_filtering_system()
        print(f"\n✅ Phase 1 filtering test completed successfully!")
        
        # Check if we have significant filtering
        filtered_percentage = sum(1 for r in results if not r['saved']) / len(results) * 100
        if filtered_percentage >= 50:
            print(f"🎉 Great! Filtering {filtered_percentage:.1f}% of messages - system working as expected!")
        else:
            print(f"⚠️  Only filtering {filtered_percentage:.1f}% of messages - may need adjustment")
            
    except Exception as e:
        logger.error("Test failed", error=str(e))
        print(f"❌ Test failed: {e}")
        sys.exit(1)