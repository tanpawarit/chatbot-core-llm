#!/usr/bin/env python3
"""
NLU Optimization Recommendations
คำแนะนำการปรับปรุงประสิทธิภาพ NLU ที่วิเคราะห์ได้
"""

OPTIMIZATION_RECOMMENDATIONS = {
    "immediate_fixes": {
        "1_reduce_prompt_size": {
            "problem": "INTENT_DETECTION_PROMPT มีขนาด 110+ บรรทัด ส่ง token มากเกินไป",
            "solution": "ลดตัวอย่างใน prompt เหลือ 1-2 ตัวอย่างสั้น ๆ",
            "impact": "ลด token 60-70%",
            "code_location": "src/llm/node/nlu_llm.py:20-110"
        },
        "2_disable_debug_output": {
            "problem": "พิมพ์ context ทั้งหมดทุกครั้ง ทำให้ช้า",
            "solution": "ปิด debug output ใน production หรือใช้ logger level",
            "impact": "ลดเวลา 100-300ms",
            "code_location": "src/llm/node/nlu_llm.py:200-209"
        },
        "3_reduce_timeout": {
            "problem": "Timeout 30 วินาทีนานเกินไป",
            "solution": "ลด timeout เป็น 10-15 วินาที",
            "impact": "ตอบสนองล้มเหลวเร็วขึ้น",
            "code_location": "src/llm/node/nlu_llm.py:220"
        }
    },
    
    "medium_term_fixes": {
        "4_optimize_model": {
            "problem": "google/gemini-2.5-flash-lite อาจช้า",
            "solution": "ทดลองใช้ model เร็วกว่า เช่น gpt-4o-mini, claude-3-haiku",
            "impact": "ลดเวลา 30-50%",
            "code_location": "config.yaml:10"
        },
        "5_simplify_parsing": {
            "problem": "PyParsing มีความซับซ้อน",
            "solution": "ใช้ JSON format แทน custom delimiter",
            "impact": "ลดเวลา parsing 50-80%",
            "code_location": "src/llm/node/nlu_llm.py:293-314"
        },
        "6_add_caching": {
            "problem": "ข้อความซ้ำต้องประมวลผลใหม่ทุกครั้ง",
            "solution": "เพิ่ม Redis cache สำหรับ NLU result",
            "impact": "ข้อความซ้ำเร็วขึ้น 95%",
            "implementation": "ใช้ message hash เป็น cache key"
        }
    },
    
    "advanced_optimizations": {
        "7_async_processing": {
            "problem": "การประมวลผลแบบ blocking",
            "solution": "ใช้ async/await สำหรับ LLM calls",
            "impact": "ปรับปรุง concurrency",
            "complexity": "สูง - ต้องแก้ไข architecture"
        },
        "8_model_quantization": {
            "problem": "ใช้ cloud LLM มี network latency",
            "solution": "ใช้ local quantized model สำหรับ classification",
            "impact": "ลดเวลา 70-90%",
            "complexity": "สูงมาก - ต้อง infrastructure เพิ่ม"
        },
        "9_batch_processing": {
            "problem": "ประมวลผลทีละข้อความ",
            "solution": "รวมหลายข้อความส่งให้ LLM พร้อมกัน",
            "impact": "ปรับปรุง throughput",
            "complexity": "กลาง - ต้องแก้ไข flow"
        }
    }
}

def print_optimization_plan():
    """พิมพ์แผนการปรับปรุงประสิทธิภาพ"""
    print("🚀 NLU OPTIMIZATION PLAN")
    print("="*80)
    
    for category, fixes in OPTIMIZATION_RECOMMENDATIONS.items():
        print(f"\n📋 {category.replace('_', ' ').title()}:")
        print("-" * 50)
        
        for fix_id, details in fixes.items():
            print(f"\n{fix_id.replace('_', '. ').title()}:")
            print(f"   Problem: {details['problem']}")
            print(f"   Solution: {details['solution']}")
            print(f"   Impact: {details['impact']}")
            if 'code_location' in details:
                print(f"   Location: {details['code_location']}")
            if 'complexity' in details:
                print(f"   Complexity: {details['complexity']}")

# Quick Fixes สำหรับแก้ไขทันที
QUICK_FIXES = {
    "optimized_prompt": '''
# แทนที่ INTENT_DETECTION_PROMPT ด้วย version สั้นนี้:

INTENT_DETECTION_PROMPT = """
Analyze this Thai message for intent, entities, language, and sentiment.

Available intents: {default_intent}, {additional_intent}
Available entities: {default_entity}, {additional_entity}

Return format:
(intent{tuple_delimiter}<name>{tuple_delimiter}<confidence>{tuple_delimiter}<priority>{tuple_delimiter}<metadata>)
{record_delimiter}
(entity{tuple_delimiter}<type>{tuple_delimiter}<value>{tuple_delimiter}<confidence>{tuple_delimiter}<metadata>)
{record_delimiter}
(language{tuple_delimiter}<iso_code>{tuple_delimiter}<confidence>{tuple_delimiter}<primary_flag>{tuple_delimiter}<metadata>)
{record_delimiter}
(sentiment{tuple_delimiter}<label>{tuple_delimiter}<confidence>{tuple_delimiter}<metadata>)
{completion_delimiter}

Text: {input_text}
Output:
"""
''',
    
    "disable_debug": '''
# ใน src/llm/node/nlu_llm.py บรรทัด 200-209
# เปลี่ยนจาก:
print("\\n" + "="*60)
print("🧠 NLU Analysis Context")
# เป็น:
if logger.level <= 10:  # DEBUG level only
    print("\\n" + "="*60)
    print("🧠 NLU Analysis Context")
''',
    
    "faster_timeout": '''
# ใน src/llm/node/nlu_llm.py บรรทัด 220
# เปลี่ยนจาก:
signal.alarm(30)  # 30 second alarm
# เป็น:
signal.alarm(10)  # 10 second alarm
''',
    
    "config_optimization": '''
# ใน config.yaml เปลี่ยน model:
classification:
  model: "openai/gpt-4o-mini"  # เร็วกว่า gemini
  temperature: 0.1
'''
}

def print_quick_fixes():
    """พิมพ์วิธีแก้ไขแบบเร่งด่วน"""
    print("\n⚡ QUICK FIXES (แก้ไขทันที)")
    print("="*80)
    
    for fix_name, code in QUICK_FIXES.items():
        print(f"\n🔧 {fix_name.replace('_', ' ').title()}:")
        print(code)

if __name__ == "__main__":
    print_optimization_plan()
    print_quick_fixes()
    
    print("\n🎯 RECOMMENDED ACTION PLAN:")
    print("="*80)
    print("1. ✅ รัน performance_test_nlu.py เพื่อวัดเวลาปัจจุบัน")
    print("2. ⚡ ใช้ Quick Fixes (ลดเวลา 60-80%)")
    print("3. 📊 รัน test อีกครั้งเพื่อเปรียบเทียบ")
    print("4. 🔄 ทำ Medium-term fixes ทีละขั้น")
    print("5. 🚀 Advanced optimizations สำหรับ production")