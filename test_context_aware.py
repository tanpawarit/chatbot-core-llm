#!/usr/bin/env python3
"""Test script for context-aware classification"""

from src.models import Message, MessageRole
from src.llm.node.classification_llm import classify_event

def test_context_aware_classification():
    """Test context-aware classification with conversation history"""
    
    print("🧪 Testing Context-Aware Classification")
    print("=" * 50)
    
    # Simulate conversation history
    conversation_history = [
        Message(role=MessageRole.USER, content="สวัสดีครับ"),
        Message(role=MessageRole.ASSISTANT, content="สวัสดีครับ ยินดีต้อนรับสู่ร้านคอมพิวเตอร์ มีอะไรให้ช่วยเหลือไหมครับ"),
        Message(role=MessageRole.USER, content="ผมสนใจซื้อ CPU สำหรับเล่นเกม"),
        Message(role=MessageRole.ASSISTANT, content="เยื่ยมเลยครับ! สำหรับเล่นเกม ผมแนะนำ CPU ดังนี้ครับ..."),
        Message(role=MessageRole.USER, content="งบประมาณประมาณ 15,000 บาท"),
    ]
    
    # Test cases
    test_cases = [
        {
            "message": "งบประมาณประมาณ 15,000 บาท",
            "context": conversation_history[:-1],  # Previous messages only
            "description": "Budget inquiry with gaming context"
        },
        {
            "message": "มี i5 ไหมครับ",
            "context": conversation_history,
            "description": "Follow-up specific product question"
        },
        {
            "message": "ราคาเท่าไหร่",
            "context": conversation_history + [
                Message(role=MessageRole.USER, content="มี i5 ไหมครับ"),
                Message(role=MessageRole.ASSISTANT, content="มีครับ Intel Core i5-13600K"),
            ],
            "description": "Price inquiry following product discussion"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: {test_case['description']}")
        print(f"Message: '{test_case['message']}'")
        print(f"Context: {len(test_case['context'])} previous messages")
        
        # Test without context (old behavior)
        print("\n🔸 WITHOUT Context:")
        classification_old = classify_event(test_case['message'])
        if classification_old:
            print(f"   Type: {classification_old.event_type}")
            print(f"   Score: {classification_old.importance_score}")
            print(f"   Intent: {classification_old.intent}")
        
        # Test with context (new behavior)
        print("\n🔹 WITH Context:")
        classification_new = classify_event(test_case['message'], test_case['context'])
        if classification_new:
            print(f"   Type: {classification_new.event_type}")
            print(f"   Score: {classification_new.importance_score}")
            print(f"   Intent: {classification_new.intent}")
            print(f"   Reasoning: {classification_new.reasoning}")
        
        print("-" * 50)

if __name__ == "__main__":
    test_context_aware_classification()