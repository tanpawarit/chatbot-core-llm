#!/usr/bin/env python3
"""
Performance Test for Response LLM Processing
วิเคราะห์ประสิทธิภาพการสร้างคำตอบ และหาจุดที่ทำให้ช้า
"""

import time
import asyncio
import statistics
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.llm.node.response_llm import generate_response
from src.config import config_manager
from src.models import Message, MessageRole, LongTermMemory, NLUResult, NLUEntity
from src.utils.logging import get_logger

logger = get_logger(__name__)

class ResponseLLMPerformanceTester:
    """ทดสอบประสิทธิภาพการสร้างคำตอบ Response LLM"""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        
    def test_single_response(self, 
                           conversation_messages: List[Message], 
                           lm_context: Optional[LongTermMemory] = None,
                           context_selection: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
        """ทดสอบการสร้างคำตอบเดียว"""
        last_msg = conversation_messages[-1].content if conversation_messages else "No message"
        print(f"\n🤖 Testing Response: '{last_msg[:50]}{'...' if len(last_msg) > 50 else ''}'")
        
        start_time = time.time()
        
        # วัดเวลาแต่ละขั้นตอน
        steps = {}
        
        # Step 1: Config loading
        step_start = time.time()
        try:
            config = config_manager.get_config()
            steps['config_load'] = (time.time() - step_start) * 1000
        except Exception as e:
            steps['config_load'] = (time.time() - step_start) * 1000
            steps['config_error'] = str(e)
        
        # Step 2: System prompt building
        step_start = time.time()
        try:
            # วัดเวลาการสร้าง system prompt
            from src.llm.node.response_llm import _build_system_prompt
            system_prompt = _build_system_prompt(lm_context, context_selection)
            steps['prompt_building'] = (time.time() - step_start) * 1000
            steps['prompt_length'] = len(system_prompt) if system_prompt else 0
        except Exception as e:
            steps['prompt_building'] = (time.time() - step_start) * 1000
            steps['prompt_error'] = str(e)
        
        # Step 3: Response Generation
        step_start = time.time()
        try:
            response = generate_response(conversation_messages, lm_context, context_selection)
            steps['response_generation'] = (time.time() - step_start) * 1000
            steps['response_success'] = response is not None
            steps['response_length'] = len(response) if response else 0
        except Exception as e:
            steps['response_generation'] = (time.time() - step_start) * 1000
            steps['response_error'] = str(e)
            response = None
        
        total_time = (time.time() - start_time) * 1000
        
        result = {
            'conversation_length': len(conversation_messages),
            'has_lm_context': lm_context is not None,
            'context_selection': context_selection,
            'total_time_ms': round(total_time, 2),
            'steps': steps,
            'response': response,
            'timestamp': time.time()
        }
        
        # Print immediate results
        print(f"  ⏱️  Total: {total_time:.1f}ms")
        print(f"  📊 Config: {steps.get('config_load', 0):.1f}ms")
        print(f"  📝 Prompt Building: {steps.get('prompt_building', 0):.1f}ms")
        print(f"  🤖 Response Gen: {steps.get('response_generation', 0):.1f}ms")
        if response:
            print(f"  ✅ Success: {len(response)} chars")
        else:
            print(f"  ❌ Failed: {steps.get('response_error', 'Unknown error')}")
        
        return result
    
    def test_multiple_responses(self, test_scenarios: List[Dict[str, Any]], iterations: int = 3) -> Dict[str, Any]:
        """ทดสอบหลายสถานการณ์"""
        print(f"\n🔄 Testing {len(test_scenarios)} scenarios x {iterations} iterations")
        
        all_results = []
        
        for iteration in range(iterations):
            print(f"\n--- Iteration {iteration + 1}/{iterations} ---")
            
            for i, scenario in enumerate(test_scenarios, 1):
                print(f"[{i}/{len(test_scenarios)}] {scenario['name']}", end=" ")
                result = self.test_single_response(
                    scenario['messages'], 
                    scenario.get('lm_context'),
                    scenario.get('context_selection')
                )
                result['iteration'] = iteration
                result['scenario_name'] = scenario['name']
                all_results.append(result)
                
                # เพิ่มหน่วงเวลาเล็กน้อยระหว่างการทดสอบ
                time.sleep(0.5)
        
        return self.analyze_results(all_results)
    
    def test_concurrent_processing(self, test_scenarios: List[Dict[str, Any]], max_workers: int = 3) -> Dict[str, Any]:
        """ทดสอบการประมวลผลแบบ concurrent"""
        print(f"\n⚡ Testing concurrent response processing with {max_workers} workers")
        
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self.test_single_response, 
                    scenario['messages'],
                    scenario.get('lm_context'),
                    scenario.get('context_selection')
                )
                for scenario in test_scenarios
            ]
            
            for i, future in enumerate(futures, 1):
                try:
                    result = future.result(timeout=60)  # 60 second timeout
                    result['worker_id'] = i
                    result['scenario_name'] = test_scenarios[i-1]['name']
                    results.append(result)
                except Exception as e:
                    print(f"❌ Worker {i} failed: {str(e)}")
        
        total_concurrent_time = (time.time() - start_time) * 1000
        
        print(f"\n⏱️  Concurrent total time: {total_concurrent_time:.1f}ms")
        
        analysis = self.analyze_results(results)
        analysis['concurrent_total_time_ms'] = total_concurrent_time
        
        return analysis
    
    def test_context_impact(self, base_message: List[Message]) -> Dict[str, Any]:
        """ทดสอบผลกระทบของ context ต่อประสิทธิภาพ"""
        print(f"\n📊 Testing context impact on performance")
        
        # สร้าง LM context ตัวอย่าง
        sample_lm = LongTermMemory(user_id="test_user")
        sample_analysis = NLUResult(
            content="อยากซื้อคอมพิวเตอร์ใหม่ครับ",
            entities=[NLUEntity(type="product", value="คอมพิวเตอร์", confidence=0.9)]
        )
        sample_lm.nlu_analyses.append(sample_analysis)
        
        context_scenarios = [
            {
                'name': 'No Context',
                'context_selection': None,
                'lm_context': None
            },
            {
                'name': 'Core Only',
                'context_selection': {'core_behavior': True},
                'lm_context': None
            },
            {
                'name': 'Core + Business',
                'context_selection': {
                    'core_behavior': True,
                    'business_policies': True
                },
                'lm_context': None
            },
            {
                'name': 'Full Context',
                'context_selection': {
                    'core_behavior': True,
                    'interaction_guidelines': True,
                    'product_details': True,
                    'business_policies': True,
                    'user_history': True,
                },
                'lm_context': sample_lm
            }
        ]
        
        results = []
        for scenario in context_scenarios:
            print(f"\n🔍 Testing: {scenario['name']}")
            result = self.test_single_response(
                base_message,
                scenario['lm_context'],
                scenario['context_selection']
            )
            result['context_scenario'] = scenario['name']
            results.append(result)
            time.sleep(0.5)
        
        return self.analyze_context_results(results)
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """วิเคราะห์ผลลัพธ์"""
        if not results:
            return {"error": "No results to analyze"}
        
        # รวบรวมข้อมูลเวลา
        total_times = [r['total_time_ms'] for r in results]
        config_times = [r['steps'].get('config_load', 0) for r in results]
        prompt_times = [r['steps'].get('prompt_building', 0) for r in results]
        response_times = [r['steps'].get('response_generation', 0) for r in results]
        
        # คำนวณสถิติ
        analysis = {
            'total_tests': len(results),
            'success_rate': len([r for r in results if r['steps'].get('response_success', False)]) / len(results) * 100,
            'timing_stats': {
                'total_time': {
                    'min': min(total_times),
                    'max': max(total_times),
                    'avg': statistics.mean(total_times),
                    'median': statistics.median(total_times),
                    'stdev': statistics.stdev(total_times) if len(total_times) > 1 else 0
                },
                'config_load': {
                    'min': min(config_times),
                    'max': max(config_times),
                    'avg': statistics.mean(config_times),
                },
                'prompt_building': {
                    'min': min(prompt_times),
                    'max': max(prompt_times),
                    'avg': statistics.mean(prompt_times),
                    'median': statistics.median(prompt_times),
                },
                'response_generation': {
                    'min': min(response_times),
                    'max': max(response_times),
                    'avg': statistics.mean(response_times),
                    'median': statistics.median(response_times),
                }
            },
            'performance_categories': self.categorize_performance(results),
            'bottlenecks': self.identify_bottlenecks(results),
            'recommendations': self.generate_recommendations(results)
        }
        
        return analysis
    
    def analyze_context_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """วิเคราะห์ผลลัพธ์การทดสอบ context"""
        analysis = self.analyze_results(results)
        
        # เพิ่มการวิเคราะห์เฉพาะ context
        context_comparison = {}
        for result in results:
            scenario = result['context_scenario']
            context_comparison[scenario] = {
                'total_time': result['total_time_ms'],
                'prompt_length': result['steps'].get('prompt_length', 0),
                'response_time': result['steps'].get('response_generation', 0)
            }
        
        analysis['context_comparison'] = context_comparison
        analysis['context_recommendations'] = self.generate_context_recommendations(context_comparison)
        
        return analysis
    
    def categorize_performance(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """จัดหมวดหมู่ประสิทธิภาพ"""
        fast = []      # < 3000ms
        moderate = []  # 3000-7000ms  
        slow = []      # 7000-15000ms
        very_slow = [] # > 15000ms
        
        for result in results:
            time_ms = result['total_time_ms']
            if time_ms < 3000:
                fast.append(result)
            elif time_ms < 7000:
                moderate.append(result)
            elif time_ms < 15000:
                slow.append(result)
            else:
                very_slow.append(result)
        
        return {
            'fast': {'count': len(fast), 'percentage': len(fast)/len(results)*100},
            'moderate': {'count': len(moderate), 'percentage': len(moderate)/len(results)*100},
            'slow': {'count': len(slow), 'percentage': len(slow)/len(results)*100},
            'very_slow': {'count': len(very_slow), 'percentage': len(very_slow)/len(results)*100}
        }
    
    def identify_bottlenecks(self, results: List[Dict[str, Any]]) -> List[str]:
        """ระบุจุดคอขวด"""
        bottlenecks = []
        
        # วิเคราะห์เวลาเฉลี่ย
        avg_total = statistics.mean([r['total_time_ms'] for r in results])
        avg_config = statistics.mean([r['steps'].get('config_load', 0) for r in results])
        avg_prompt = statistics.mean([r['steps'].get('prompt_building', 0) for r in results])
        avg_response = statistics.mean([r['steps'].get('response_generation', 0) for r in results])
        
        if avg_total > 10000:
            bottlenecks.append("Overall response processing is very slow (>10 seconds)")
        
        if avg_config > 100:
            bottlenecks.append("Config loading is slow")
        
        if avg_prompt > 500:
            bottlenecks.append("System prompt building is slow")
        
        if avg_response > 8000:
            bottlenecks.append("LLM response generation is the main bottleneck")
        
        # ตรวจสอบความแปรปรวน
        response_times = [r['steps'].get('response_generation', 0) for r in results]
        if len(response_times) > 1:
            stdev = statistics.stdev(response_times)
            if stdev > 3000:
                bottlenecks.append("Response generation time is highly inconsistent")
        
        # ตรวจสอบอัตราความสำเร็จ
        success_rate = len([r for r in results if r['steps'].get('response_success', False)]) / len(results) * 100
        if success_rate < 90:
            bottlenecks.append(f"Low success rate ({success_rate:.1f}%)")
        
        return bottlenecks
    
    def generate_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """สร้างคำแนะนำ"""
        recommendations = []
        
        avg_response = statistics.mean([r['steps'].get('response_generation', 0) for r in results])
        avg_prompt = statistics.mean([r['steps'].get('prompt_building', 0) for r in results])
        
        if avg_response > 6000:
            recommendations.extend([
                "Consider using a faster LLM model for response generation",
                "Implement response caching for similar conversations",
                "Optimize system prompt length and complexity",
                "Add request timeout to prevent hanging",
                "Consider using streaming responses for better UX"
            ])
        
        if avg_prompt > 300:
            recommendations.extend([
                "Optimize context selection to reduce prompt size",
                "Cache formatted product data",
                "Implement lazy loading for expensive contexts"
            ])
        
        # ตรวจสอบ error patterns
        errors = [r['steps'].get('response_error') for r in results if 'response_error' in r['steps']]
        if errors:
            recommendations.append("Fix response generation errors to improve reliability")
        
        return recommendations
    
    def generate_context_recommendations(self, context_comparison: Dict[str, Dict]) -> List[str]:
        """สร้างคำแนะนำเฉพาะ context"""
        recommendations = []
        
        # เปรียบเทียบเวลาระหว่าง context scenarios
        times = {k: v['total_time'] for k, v in context_comparison.items()}
        
        if 'Full Context' in times and 'Core Only' in times:
            time_diff = times['Full Context'] - times['Core Only']
            if time_diff > 2000:
                recommendations.append("Full context adds significant overhead - consider selective context loading")
        
        if 'No Context' in times and 'Core Only' in times:
            time_diff = times['Core Only'] - times['No Context']
            if time_diff > 1000:
                recommendations.append("Even minimal context adds noticeable latency")
        
        # ตรวจสอบขนาด prompt
        prompt_sizes = {k: v['prompt_length'] for k, v in context_comparison.items()}
        max_prompt = max(prompt_sizes.values())
        if max_prompt > 2000:
            recommendations.append("Large prompts may impact performance - optimize context selection")
        
        return recommendations
    
    def print_detailed_analysis(self, analysis: Dict[str, Any]):
        """พิมพ์การวิเคราะห์แบบละเอียด"""
        print("\n" + "="*80)
        print("🤖 DETAILED RESPONSE LLM PERFORMANCE ANALYSIS")
        print("="*80)
        
        # สถิติพื้นฐาน
        print(f"\n📈 Basic Statistics:")
        print(f"   Total Tests: {analysis['total_tests']}")
        print(f"   Success Rate: {analysis['success_rate']:.1f}%")
        
        # สถิติเวลา
        timing = analysis['timing_stats']
        print(f"\n⏱️  Timing Analysis:")
        print(f"   Total Time - Min: {timing['total_time']['min']:.1f}ms, Max: {timing['total_time']['max']:.1f}ms")
        print(f"   Total Time - Avg: {timing['total_time']['avg']:.1f}ms, Median: {timing['total_time']['median']:.1f}ms")
        print(f"   Response Gen - Avg: {timing['response_generation']['avg']:.1f}ms, Median: {timing['response_generation']['median']:.1f}ms")
        print(f"   Prompt Building - Avg: {timing['prompt_building']['avg']:.1f}ms, Median: {timing['prompt_building']['median']:.1f}ms")
        print(f"   Config Load - Avg: {timing['config_load']['avg']:.1f}ms")
        
        # หมวดหมู่ประสิทธิภาพ
        perf = analysis['performance_categories']
        print(f"\n🎯 Performance Categories:")
        print(f"   Fast (<3s): {perf['fast']['count']} tests ({perf['fast']['percentage']:.1f}%)")
        print(f"   Moderate (3-7s): {perf['moderate']['count']} tests ({perf['moderate']['percentage']:.1f}%)")
        print(f"   Slow (7-15s): {perf['slow']['count']} tests ({perf['slow']['percentage']:.1f}%)")
        print(f"   Very Slow (>15s): {perf['very_slow']['count']} tests ({perf['very_slow']['percentage']:.1f}%)")
        
        # Context comparison (ถ้ามี)
        if 'context_comparison' in analysis:
            print(f"\n📊 Context Impact Analysis:")
            for scenario, metrics in analysis['context_comparison'].items():
                print(f"   {scenario}: {metrics['total_time']:.1f}ms (prompt: {metrics['prompt_length']} chars)")
        
        # จุดคอขวด
        print(f"\n🚨 Identified Bottlenecks:")
        for bottleneck in analysis['bottlenecks']:
            print(f"   • {bottleneck}")
        
        # คำแนะนำ
        print(f"\n💡 Recommendations:")
        for rec in analysis['recommendations']:
            print(f"   • {rec}")
        
        # Context recommendations (ถ้ามี)
        if 'context_recommendations' in analysis:
            print(f"\n🎯 Context-Specific Recommendations:")
            for rec in analysis['context_recommendations']:
                print(f"   • {rec}")
        
        print("="*80)


def create_test_scenarios() -> List[Dict[str, Any]]:
    """สร้างสถานการณ์ทดสอบ"""
    
    # Base messages for different conversation types
    scenarios = [
        {
            'name': 'Simple Greeting',
            'messages': [
                Message(role=MessageRole.USER, content="สวัสดีครับ"),
            ]
        },
        {
            'name': 'Product Inquiry',
            'messages': [
                Message(role=MessageRole.USER, content="สวัสดีครับ"),
                Message(role=MessageRole.ASSISTANT, content="สวัสดีครับ มีอะไรให้ช่วยไหมครับ"),
                Message(role=MessageRole.USER, content="อยากซื้อ laptop ราคาประมาณ 30000 บาท"),
            ]
        },
        {
            'name': 'Technical Support',
            'messages': [
                Message(role=MessageRole.USER, content="เครื่องคอมพิวเตอร์เปิดไม่ติด ช่วยหน่อยครับ"),
            ]
        },
        {
            'name': 'Complex Purchase',
            'messages': [
                Message(role=MessageRole.USER, content="ต้องการ PC สำหรับเล่นเกม spec สูง ๆ งบประมาณ 80000 บาท"),
                Message(role=MessageRole.ASSISTANT, content="เข้าใจครับ กำลังหาเครื่องสำหรับเล่นเกม งบ 80,000 บาท"),
                Message(role=MessageRole.USER, content="ใช่ครับ ต้องการจอภาพดี ๆ ด้วย"),
            ]
        },
        {
            'name': 'Long Conversation',
            'messages': [
                Message(role=MessageRole.USER, content="สวัสดีครับ"),
                Message(role=MessageRole.ASSISTANT, content="สวัสดีครับ"),
                Message(role=MessageRole.USER, content="อยากซื้อคอมพิวเตอร์"),
                Message(role=MessageRole.ASSISTANT, content="ได้ครับ ต้องการใช้งานแบบไหนครับ"),
                Message(role=MessageRole.USER, content="ใช้ทำงาน และเล่นเกมบ้าง"),
                Message(role=MessageRole.ASSISTANT, content="เข้าใจครับ งบประมาณประมาณเท่าไหร่ครับ"),
                Message(role=MessageRole.USER, content="ประมาณ 50000 บาท ครับ"),
            ]
        }
    ]
    
    return scenarios


def main():
    """ฟังก์ชันหลักสำหรับการทดสอบ"""
    tester = ResponseLLMPerformanceTester()
    
    print("🚀 Starting Response LLM Performance Testing...")
    
    # สร้างสถานการณ์ทดสอบ
    test_scenarios = create_test_scenarios()
    print(f"📝 Testing {len(test_scenarios)} different conversation scenarios")
    
    # ทดสอบสถานการณ์เดียว
    print("\n" + "="*60)
    print("🤖 SINGLE RESPONSE TESTS")
    print("="*60)
    
    single_results = []
    for scenario in test_scenarios[:3]:  # ทดสอบ 3 สถานการณ์แรก
        result = tester.test_single_response(scenario['messages'])
        result['scenario_name'] = scenario['name']
        single_results.append(result)
    
    # วิเคราะห์ผลลัพธ์เบื้องต้น
    single_analysis = tester.analyze_results(single_results)
    tester.print_detailed_analysis(single_analysis)
    
    # ทดสอบผลกระทบของ context
    print("\n" + "="*60)
    print("📊 CONTEXT IMPACT TESTS")
    print("="*60)
    
    base_messages = [Message(role=MessageRole.USER, content="อยากซื้อ laptop ราคาประมาณ 30000 บาท")]
    context_analysis = tester.test_context_impact(base_messages)
    tester.print_detailed_analysis(context_analysis)
    
    # ทดสอบหลายสถานการณ์
    print("\n" + "="*60)
    print("🔄 MULTIPLE SCENARIO TESTS")
    print("="*60)
    
    multi_analysis = tester.test_multiple_responses(test_scenarios[:4], iterations=2)
    tester.print_detailed_analysis(multi_analysis)
    
    # ทดสอบ concurrent
    print("\n" + "="*60)
    print("⚡ CONCURRENT PROCESSING TESTS")
    print("="*60)
    
    concurrent_analysis = tester.test_concurrent_processing(test_scenarios[:3], max_workers=2)
    tester.print_detailed_analysis(concurrent_analysis)
    
    # สรุปผล
    print("\n" + "="*80)
    print("🏁 FINAL SUMMARY")
    print("="*80)
    
    print(f"Single Response Avg: {single_analysis['timing_stats']['total_time']['avg']:.1f}ms")
    print(f"Multiple Response Avg: {multi_analysis['timing_stats']['total_time']['avg']:.1f}ms")
    print(f"Concurrent Total: {concurrent_analysis.get('concurrent_total_time_ms', 0):.1f}ms")
    
    # หาค่าเฉลี่ยของ Response generation
    response_avg_single = single_analysis['timing_stats']['response_generation']['avg']
    response_avg_multi = multi_analysis['timing_stats']['response_generation']['avg']
    
    print(f"\n🤖 Response Generation Performance:")
    print(f"   Single: {response_avg_single:.1f}ms")
    print(f"   Multiple: {response_avg_multi:.1f}ms")
    
    if response_avg_single > 5000 or response_avg_multi > 5000:
        print("\n⚠️  WARNING: Response generation is slow (>5 seconds)")
        print("   Main causes likely:")
        print("   1. Large system prompt with full context")
        print("   2. LLM model response time (network latency)")
        print("   3. Complex context building process")
        print("   4. Product data loading overhead")
    
    # Context impact summary
    if 'context_comparison' in context_analysis:
        print(f"\n📊 Context Impact Summary:")
        for scenario, metrics in context_analysis['context_comparison'].items():
            print(f"   {scenario}: {metrics['total_time']:.1f}ms")
    
    return {
        'single': single_analysis,
        'context': context_analysis,
        'multiple': multi_analysis,
        'concurrent': concurrent_analysis
    }


if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Testing interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()