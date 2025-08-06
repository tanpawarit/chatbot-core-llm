#!/usr/bin/env python3
"""
Quick Warmup Test - ทดสอบว่า warmup ช่วยได้จริงหรือไม่
"""

import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.llm.factory import llm_factory
from src.utils.logging import get_logger

logger = get_logger(__name__)

def simple_warmup_test():
    """ทดสอบ warmup แบบง่าย ๆ"""
    print("🧪 Quick Warmup Test")
    print("="*50)
    
    # Test 1: ตรวจสอบสถานะก่อน warmup
    print(f"\n📊 Before Warmup:")
    print(f"   All warmed up: {llm_factory.is_warmed_up('all')}")
    print(f"   Response warmed up: {llm_factory.is_warmed_up('response')}")
    print(f"   Classification warmed up: {llm_factory.is_warmed_up('classification')}")
    
    # Test 2: ลอง warmup อย่างเดียว
    print(f"\n🔥 Starting Warmup Test...")
    start_time = time.time()
    
    try:
        success = llm_factory.warmup_all_llms()
        warmup_time = (time.time() - start_time) * 1000
        
        print(f"\n📊 Warmup Results:")
        print(f"   Success: {'✅' if success else '❌'}")
        print(f"   Time: {warmup_time:.1f}ms")
        print(f"   Status: {llm_factory.get_warmup_status()}")
        
    except KeyboardInterrupt:
        warmup_time = (time.time() - start_time) * 1000
        print(f"\n⏹️  Warmup interrupted after {warmup_time:.1f}ms")
        print("   This is normal - warmup can be slow on first run")
        
    except Exception as e:
        warmup_time = (time.time() - start_time) * 1000
        print(f"\n❌ Warmup failed after {warmup_time:.1f}ms")
        print(f"   Error: {str(e)}")
    
    # Test 3: ตรวจสอบสถานะหลัง warmup
    print(f"\n📊 After Warmup:")
    print(f"   All warmed up: {llm_factory.is_warmed_up('all')}")
    print(f"   Response warmed up: {llm_factory.is_warmed_up('response')}")
    print(f"   Classification warmed up: {llm_factory.is_warmed_up('classification')}")
    
    print(f"\n💡 Conclusion:")
    if llm_factory.is_warmed_up('all'):
        print("   ✅ Warmup successful! Next LLM calls should be faster.")
    else:
        print("   ⚠️  Warmup incomplete. Cold start may still occur.")
        print("   This could be due to network issues or API slowness.")
    
    print("="*50)

if __name__ == "__main__":
    try:
        simple_warmup_test()
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()