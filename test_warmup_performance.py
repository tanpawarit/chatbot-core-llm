#!/usr/bin/env python3
"""
Test Warmup Performance - ทดสอบประสิทธิภาพหลัง LLM Warmup
เปรียบเทียบประสิทธิภาพก่อนและหลัง warmup
"""

import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.llm.factory import llm_factory
from src.llm.node.nlu_llm import analyze_message_nlu
from src.llm.node.response_llm import generate_response
from src.models import Message, MessageRole
from src.utils.logging import get_logger

logger = get_logger(__name__)

def test_cold_vs_warm_nlu():
    """ทดสอบ NLU ก่อนและหลัง warmup"""
    print("\n" + "="*60)
    print("🧠 NLU COLD vs WARM TEST")
    print("="*60)
    
    test_message = "อยากซื้อคอมพิวเตอร์ครับ"
    
    # Test 1: Cold start (reset factory)
    print("\n❄️ Cold Start Test:")
    llm_factory._instances.clear()
    llm_factory._warmup_status.clear()
    
    start_time = time.time()
    try:
        nlu_result = analyze_message_nlu(test_message)
        cold_time = (time.time() - start_time) * 1000
        print(f"  Cold NLU: {cold_time:.1f}ms")
        print(f"  Result: {nlu_result.primary_intent if nlu_result else 'Failed'}")
    except Exception as e:
        cold_time = (time.time() - start_time) * 1000
        print(f"  Cold NLU FAILED: {cold_time:.1f}ms - {str(e)}")
        return
    
    # Test 2: Warm start (after warmup)
    print("\n🔥 Warm Start Test:")
    warmup_start = time.time()
    warmup_success = llm_factory.warmup_classification_llm()
    warmup_time = (time.time() - warmup_start) * 1000
    print(f"  Warmup: {warmup_time:.1f}ms ({'✅' if warmup_success else '❌'})")
    
    start_time = time.time()
    try:
        nlu_result = analyze_message_nlu(test_message)
        warm_time = (time.time() - start_time) * 1000
        print(f"  Warm NLU: {warm_time:.1f}ms")
        print(f"  Result: {nlu_result.primary_intent if nlu_result else 'Failed'}")
    except Exception as e:
        warm_time = (time.time() - start_time) * 1000
        print(f"  Warm NLU FAILED: {warm_time:.1f}ms - {str(e)}")
        return
    
    # Compare results
    improvement = ((cold_time - warm_time) / cold_time) * 100
    print(f"\n📊 Performance Improvement:")
    print(f"  Cold: {cold_time:.1f}ms")  
    print(f"  Warm: {warm_time:.1f}ms")
    print(f"  Improvement: {improvement:.1f}% faster")
    
    return {
        'cold_time': cold_time,
        'warm_time': warm_time,
        'warmup_time': warmup_time,
        'improvement_percent': improvement
    }

def test_cold_vs_warm_response():
    """ทดสอบ Response LLM ก่อนและหลัง warmup"""
    print("\n" + "="*60)
    print("🤖 RESPONSE LLM COLD vs WARM TEST")
    print("="*60)
    
    test_messages = [Message(role=MessageRole.USER, content="สวัสดีครับ")]
    
    # Test 1: Cold start (reset factory)
    print("\n❄️ Cold Start Test:")
    llm_factory._instances.clear()
    llm_factory._warmup_status.clear()
    
    start_time = time.time()
    try:
        response = generate_response(test_messages)
        cold_time = (time.time() - start_time) * 1000
        print(f"  Cold Response: {cold_time:.1f}ms")
        print(f"  Result: {response[:50]}..." if len(response) > 50 else f"  Result: {response}")
    except Exception as e:
        cold_time = (time.time() - start_time) * 1000
        print(f"  Cold Response FAILED: {cold_time:.1f}ms - {str(e)}")
        return
    
    # Test 2: Warm start (after warmup)
    print("\n🔥 Warm Start Test:")
    warmup_start = time.time()
    warmup_success = llm_factory.warmup_response_llm()
    warmup_time = (time.time() - warmup_start) * 1000
    print(f"  Warmup: {warmup_time:.1f}ms ({'✅' if warmup_success else '❌'})")
    
    start_time = time.time()
    try:
        response = generate_response(test_messages)
        warm_time = (time.time() - start_time) * 1000
        print(f"  Warm Response: {warm_time:.1f}ms")
        print(f"  Result: {response[:50]}..." if len(response) > 50 else f"  Result: {response}")
    except Exception as e:
        warm_time = (time.time() - start_time) * 1000
        print(f"  Warm Response FAILED: {warm_time:.1f}ms - {str(e)}")
        return
    
    # Compare results
    improvement = ((cold_time - warm_time) / cold_time) * 100
    print(f"\n📊 Performance Improvement:")
    print(f"  Cold: {cold_time:.1f}ms")  
    print(f"  Warm: {warm_time:.1f}ms")
    print(f"  Improvement: {improvement:.1f}% faster")
    
    return {
        'cold_time': cold_time,
        'warm_time': warm_time,
        'warmup_time': warmup_time,
        'improvement_percent': improvement
    }

def test_parallel_warmup():
    """ทดสอบ parallel warmup"""
    print("\n" + "="*60)
    print("⚡ PARALLEL WARMUP TEST")
    print("="*60)
    
    # Reset factory
    llm_factory._instances.clear()
    llm_factory._warmup_status.clear()
    
    # Test parallel warmup
    start_time = time.time()
    success = llm_factory.warmup_all_llms()
    total_time = (time.time() - start_time) * 1000
    
    print(f"\n📊 Parallel Warmup Results:")
    print(f"  Total Time: {total_time:.1f}ms")
    print(f"  Success: {'✅' if success else '❌'}")
    print(f"  Status: {llm_factory.get_warmup_status()}")
    
    return {
        'total_time': total_time,
        'success': success,
        'status': llm_factory.get_warmup_status()
    }

def main():
    """ฟังก์ชันหลักสำหรับการทดสอบ warmup"""
    print("🚀 Starting LLM Warmup Performance Testing...")
    
    results = {}
    
    # ทดสอบ NLU
    try:
        results['nlu'] = test_cold_vs_warm_nlu()
    except Exception as e:
        print(f"❌ NLU test failed: {str(e)}")
        results['nlu'] = None
    
    # รอสักครู่
    time.sleep(2)
    
    # ทดสอบ Response LLM  
    try:
        results['response'] = test_cold_vs_warm_response()
    except Exception as e:
        print(f"❌ Response test failed: {str(e)}")
        results['response'] = None
    
    # รอสักครู่
    time.sleep(2)
    
    # ทดสอบ Parallel Warmup
    try:
        results['parallel'] = test_parallel_warmup()
    except Exception as e:
        print(f"❌ Parallel warmup test failed: {str(e)}")
        results['parallel'] = None
    
    # สรุปผลรวม
    print("\n" + "="*80)
    print("📈 WARMUP PERFORMANCE SUMMARY")
    print("="*80)
    
    if results['nlu']:
        print(f"\n🧠 NLU Performance:")
        print(f"   Cold Start: {results['nlu']['cold_time']:.1f}ms")
        print(f"   Warm Start: {results['nlu']['warm_time']:.1f}ms")
        print(f"   Improvement: {results['nlu']['improvement_percent']:.1f}% faster")
        print(f"   Warmup Cost: {results['nlu']['warmup_time']:.1f}ms")
    
    if results['response']:
        print(f"\n🤖 Response LLM Performance:")
        print(f"   Cold Start: {results['response']['cold_time']:.1f}ms")
        print(f"   Warm Start: {results['response']['warm_time']:.1f}ms")
        print(f"   Improvement: {results['response']['improvement_percent']:.1f}% faster")
        print(f"   Warmup Cost: {results['response']['warmup_time']:.1f}ms")
    
    if results['parallel']:
        print(f"\n⚡ Parallel Warmup:")
        print(f"   Total Time: {results['parallel']['total_time']:.1f}ms")
        print(f"   Success Rate: {'100%' if results['parallel']['success'] else 'Failed'}")
    
    # คำแนะนำ
    print(f"\n💡 Recommendations:")
    
    if results['nlu'] and results['nlu']['improvement_percent'] > 20:
        print(f"   ✅ NLU warmup provides significant improvement ({results['nlu']['improvement_percent']:.1f}%)")
    elif results['nlu']:
        print(f"   ⚠️  NLU warmup improvement is minimal ({results['nlu']['improvement_percent']:.1f}%)")
    
    if results['response'] and results['response']['improvement_percent'] > 20:
        print(f"   ✅ Response LLM warmup provides significant improvement ({results['response']['improvement_percent']:.1f}%)")
    elif results['response']:
        print(f"   ⚠️  Response LLM warmup improvement is minimal ({results['response']['improvement_percent']:.1f}%)")
    
    if results['parallel'] and results['parallel']['success']:
        print(f"   ✅ Parallel warmup works well (total: {results['parallel']['total_time']:.1f}ms)")
    elif results['parallel']:
        print(f"   ❌ Parallel warmup had issues")
    
    print("\n🎯 Recommended Implementation:")
    print("   1. Run warmup_all_llms() at application startup")
    print("   2. Expected first-response improvement: 20-80% faster")
    print("   3. Warmup cost is one-time per application restart")
    
    print("="*80)
    
    return results

if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Testing interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()