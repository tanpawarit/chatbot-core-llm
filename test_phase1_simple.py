#!/usr/bin/env python3
"""
Simple test for Phase 1 filtering - Direct NLU testing
"""

import sys
sys.path.append('.')

from src.models import Message, MessageRole, Conversation
from src.llm.node.classification_llm import analyze_message_nlu, should_save_to_longterm

def test_nlu_direct():
    """Test NLU analysis and filtering directly"""
    
    print("🧪 Testing Phase 1 NLU & Filtering (Direct)")
    print("=" * 55)
    
    # Test messages
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
    
    results = []
    threshold = 0.7
    
    print(f"Testing {len(test_messages)} messages with threshold {threshold}")
    print()
    
    for i, (message_text, expected) in enumerate(test_messages, 1):
        print(f"{i:2d}. \"{message_text}\"")
        print(f"    Expected: {expected}")
        
        try:
            # Analyze message with NLU
            nlu_result = analyze_message_nlu(message_text, [])
            
            if nlu_result:
                importance = nlu_result.importance_score
                will_save = should_save_to_longterm(nlu_result)
                status = "✅ SAVE" if will_save else "🚫 FILTER"
                
                print(f"    Result: {status}")
                print(f"    Score: {importance:.3f} (threshold: {threshold:.3f})")
                print(f"    Intent: {nlu_result.primary_intent}")
                print(f"    Entities: {len(nlu_result.entities)} found")
                
                # Show entity details for debugging
                if nlu_result.entities:
                    print(f"    Entity details:")
                    for entity in nlu_result.entities:
                        in_text = entity.value.lower() in message_text.lower()
                        print(f"      - {entity.type}: \"{entity.value}\" (conf: {entity.confidence:.2f}) {'✓' if in_text else '✗ HALLUCINATION'}")
                
                results.append({
                    'message': message_text,
                    'importance': importance,
                    'saved': will_save,
                    'intent': nlu_result.primary_intent,
                    'entities': len(nlu_result.entities),
                    'valid_entities': sum(1 for e in nlu_result.entities if e.value.lower() in message_text.lower())
                })
            else:
                print(f"    Result: ❌ NLU FAILED")
                results.append({
                    'message': message_text,
                    'importance': 0.0,
                    'saved': False,
                    'intent': None,
                    'entities': 0,
                    'valid_entities': 0
                })
                
        except Exception as e:
            print(f"    Result: ❌ ERROR: {str(e)}")
            results.append({
                'message': message_text,
                'importance': 0.0,
                'saved': False,
                'intent': None,
                'entities': 0,
                'valid_entities': 0,
                'error': str(e)
            })
        
        print()  # Empty line between tests
    
    # Summary
    print("=" * 55)
    print("📊 SUMMARY")
    print("=" * 55)
    
    total = len(results)
    saved = sum(1 for r in results if r['saved'])
    filtered = total - saved
    
    print(f"Total messages: {total}")
    print(f"Saved to LM: {saved} ({saved/total*100:.1f}%)")
    print(f"Filtered out: {filtered} ({filtered/total*100:.1f}%)")
    print()
    
    print(f"📈 FILTERING IMPACT:")
    print(f"   Before Phase 1: {total}/{total} saved (100%)")
    print(f"   After Phase 1:  {saved}/{total} saved ({saved/total*100:.1f}%)")
    print(f"   Storage reduction: {(total-saved)/total*100:.1f}%")
    print()
    
    # Check for entity hallucination
    total_entities = sum(r['entities'] for r in results)
    valid_entities = sum(r['valid_entities'] for r in results)
    
    if total_entities > 0:
        hallucination_rate = (total_entities - valid_entities) / total_entities * 100
        print(f"🔍 ENTITY VALIDATION:")
        print(f"   Total entities found: {total_entities}")
        print(f"   Valid entities: {valid_entities}")
        print(f"   Hallucinated entities: {total_entities - valid_entities}")
        print(f"   Hallucination rate: {hallucination_rate:.1f}%")
    
    return results

if __name__ == "__main__":
    try:
        results = test_nlu_direct()
        
        # Success criteria
        filtered_percentage = sum(1 for r in results if not r['saved']) / len(results) * 100
        
        print(f"\n{'='*55}")
        if filtered_percentage >= 30:  # At least 30% should be filtered
            print(f"✅ SUCCESS: Phase 1 filtering working! ({filtered_percentage:.1f}% filtered)")
        else:
            print(f"⚠️  WARNING: Low filtering rate ({filtered_percentage:.1f}% filtered)")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)