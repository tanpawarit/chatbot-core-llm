#!/usr/bin/env python3
"""
Performance Test for NLU Processing
วิเคราะห์ประสิทธิภาพการประมวลผล NLU และหาจุดที่ทำให้ช้า
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

from src.llm.node.nlu_llm import analyze_message_nlu
from src.config import config_manager
from src.models import Message, MessageRole
from src.utils.logging import get_logger

logger = get_logger(__name__)

class NLUPerformanceTester:
    """ทดสอบประสิทธิภาพการประมวลผล NLU"""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        
    def test_single_message(self, message: str, context: Optional[List[Message]] = None) -> Dict[str, Any]:
        """ทดสอบข้อความเดียว"""
        print(f"\n🧪 Testing: '{message[:50]}{'...' if len(message) > 50 else ''}'")
        
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
        
        # Step 2: NLU Analysis
        step_start = time.time()
        try:
            nlu_result = analyze_message_nlu(message, context)
            steps['nlu_analysis'] = (time.time() - step_start) * 1000
            steps['nlu_success'] = nlu_result is not None
        except Exception as e:
            steps['nlu_analysis'] = (time.time() - step_start) * 1000
            steps['nlu_error'] = str(e)
            nlu_result = None
        
        total_time = (time.time() - start_time) * 1000
        
        result = {
            'message': message,
            'message_length': len(message),
            'total_time_ms': round(total_time, 2),
            'steps': steps,
            'nlu_result': nlu_result,
            'timestamp': time.time()
        }
        
        # Print immediate results
        print(f"  ⏱️  Total: {total_time:.1f}ms")
        print(f"  📊 Config: {steps.get('config_load', 0):.1f}ms")
        print(f"  🧠 NLU: {steps.get('nlu_analysis', 0):.1f}ms")
        if nlu_result:
            print(f"  ✅ Success: {nlu_result.primary_intent} ({len(nlu_result.entities)} entities)")
        else:
            print(f"  ❌ Failed: {steps.get('nlu_error', 'Unknown error')}")
        
        return result
    
    def test_multiple_messages(self, messages: List[str], iterations: int = 3) -> Dict[str, Any]:
        """ทดสอบหลายข้อความ"""
        print(f"\n🔄 Testing {len(messages)} messages x {iterations} iterations")
        
        all_results = []
        
        for iteration in range(iterations):
            print(f"\n--- Iteration {iteration + 1}/{iterations} ---")
            
            for i, message in enumerate(messages, 1):
                print(f"[{i}/{len(messages)}]", end=" ")
                result = self.test_single_message(message)
                result['iteration'] = iteration
                all_results.append(result)
                
                # เพิ่มหน่วงเวลาเล็กน้อยระหว่างการทดสอบ
                time.sleep(0.5)
        
        return self.analyze_results(all_results)
    
    def test_concurrent_processing(self, messages: List[str], max_workers: int = 3) -> Dict[str, Any]:
        """ทดสอบการประมวลผลแบบ concurrent"""
        print(f"\n⚡ Testing concurrent processing with {max_workers} workers")
        
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.test_single_message, msg) 
                for msg in messages
            ]
            
            for i, future in enumerate(futures, 1):
                try:
                    result = future.result(timeout=60)  # 60 second timeout
                    result['worker_id'] = i
                    results.append(result)
                except Exception as e:
                    print(f"❌ Worker {i} failed: {str(e)}")
        
        total_concurrent_time = (time.time() - start_time) * 1000
        
        print(f"\n⏱️  Concurrent total time: {total_concurrent_time:.1f}ms")
        
        analysis = self.analyze_results(results)
        analysis['concurrent_total_time_ms'] = total_concurrent_time
        
        return analysis
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """วิเคราะห์ผลลัพธ์"""
        if not results:
            return {"error": "No results to analyze"}
        
        # รวบรวมข้อมูลเวลา
        total_times = [r['total_time_ms'] for r in results]
        config_times = [r['steps'].get('config_load', 0) for r in results]
        nlu_times = [r['steps'].get('nlu_analysis', 0) for r in results]
        
        # คำนวณสถิติ
        analysis = {
            'total_tests': len(results),
            'success_rate': len([r for r in results if r.get('nlu_result')]) / len(results) * 100,
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
                'nlu_analysis': {
                    'min': min(nlu_times),
                    'max': max(nlu_times),
                    'avg': statistics.mean(nlu_times),
                    'median': statistics.median(nlu_times),
                }
            },
            'performance_categories': self.categorize_performance(results),
            'bottlenecks': self.identify_bottlenecks(results),
            'recommendations': self.generate_recommendations(results)
        }
        
        return analysis
    
    def categorize_performance(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """จัดหมวดหมู่ประสิทธิภาพ"""
        fast = []      # < 2000ms
        moderate = []  # 2000-5000ms  
        slow = []      # 5000-10000ms
        very_slow = [] # > 10000ms
        
        for result in results:
            time_ms = result['total_time_ms']
            if time_ms < 2000:
                fast.append(result)
            elif time_ms < 5000:
                moderate.append(result)
            elif time_ms < 10000:
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
        avg_nlu = statistics.mean([r['steps'].get('nlu_analysis', 0) for r in results])
        
        if avg_total > 5000:
            bottlenecks.append("Overall processing is very slow (>5 seconds)")
        
        if avg_config > 100:
            bottlenecks.append("Config loading is slow")
        
        if avg_nlu > 4000:
            bottlenecks.append("NLU analysis is the main bottleneck")
        
        # ตรวจสอบความแปรปรวน
        nlu_times = [r['steps'].get('nlu_analysis', 0) for r in results]
        if len(nlu_times) > 1:
            stdev = statistics.stdev(nlu_times)
            if stdev > 2000:
                bottlenecks.append("NLU processing time is highly inconsistent")
        
        # ตรวจสอบอัตราความสำเร็จ
        success_rate = len([r for r in results if r.get('nlu_result')]) / len(results) * 100
        if success_rate < 90:
            bottlenecks.append(f"Low success rate ({success_rate:.1f}%)")
        
        return bottlenecks
    
    def generate_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """สร้างคำแนะนำ"""
        recommendations = []
        
        avg_nlu = statistics.mean([r['steps'].get('nlu_analysis', 0) for r in results])
        
        if avg_nlu > 3000:
            recommendations.extend([
                "Consider using a faster LLM model for classification",
                "Implement request timeout to prevent hanging",
                "Add caching for similar messages",
                "Optimize prompt size and complexity"
            ])
        
        # ตรวจสอบ error patterns
        errors = [r['steps'].get('nlu_error') for r in results if 'nlu_error' in r['steps']]
        if errors:
            recommendations.append("Fix NLU errors to improve reliability")
        
        return recommendations
    
    def print_detailed_analysis(self, analysis: Dict[str, Any]):
        """พิมพ์การวิเคราะห์แบบละเอียด"""
        print("\n" + "="*80)
        print("📊 DETAILED PERFORMANCE ANALYSIS")
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
        print(f"   NLU Analysis - Avg: {timing['nlu_analysis']['avg']:.1f}ms, Median: {timing['nlu_analysis']['median']:.1f}ms")
        print(f"   Config Load - Avg: {timing['config_load']['avg']:.1f}ms")
        
        # หมวดหมู่ประสิทธิภาพ
        perf = analysis['performance_categories']
        print(f"\n🎯 Performance Categories:")
        print(f"   Fast (<2s): {perf['fast']['count']} tests ({perf['fast']['percentage']:.1f}%)")
        print(f"   Moderate (2-5s): {perf['moderate']['count']} tests ({perf['moderate']['percentage']:.1f}%)")
        print(f"   Slow (5-10s): {perf['slow']['count']} tests ({perf['slow']['percentage']:.1f}%)")
        print(f"   Very Slow (>10s): {perf['very_slow']['count']} tests ({perf['very_slow']['percentage']:.1f}%)")
        
        # จุดคอขวด
        print(f"\n🚨 Identified Bottlenecks:")
        for bottleneck in analysis['bottlenecks']:
            print(f"   • {bottleneck}")
        
        # คำแนะนำ
        print(f"\n💡 Recommendations:")
        for rec in analysis['recommendations']:
            print(f"   • {rec}")
        
        print("="*80)


def main():
    """ฟังก์ชันหลักสำหรับการทดสอบ"""
    tester = NLUPerformanceTester()
    
    # ข้อความทดสอบ (Thai computer sales domain)
    test_messages = [
        "สวัสดีครับ",
        "อยากซื้อคอมพิวเตอร์ครับ",
        "มี laptop ราคาประมาณ 30000 บาท ไหมครับ",
        "ต้องการ PC สำหรับเล่นเกม spec สูง ๆ หน่อย งบประมาณ 50000 บาท",
        "MacBook Pro M3 ราคาเท่าไหร่ครับ มีสีอะไรบ้าง",
        "เครื่องเสีย อยากซ่อม",
        "ผมซื้อเมื่อวานแล้วเครื่องไม่ทำงาน ขอคืนเงินได้ไหมครับ",
        "เปรียบเทียบ Dell กับ HP หน่อยครับ",
        "ขอบคุณครับ ได้แล้ว"
    ]
    
    print("🚀 Starting NLU Performance Testing...")
    print(f"📝 Testing {len(test_messages)} different message types")
    
    # ทดสอบข้อความเดียว
    print("\n" + "="*60)
    print("🧪 SINGLE MESSAGE TESTS")
    print("="*60)
    
    single_results = []
    for msg in test_messages[:3]:  # ทดสอบ 3 ข้อความแรก
        result = tester.test_single_message(msg)
        single_results.append(result)
    
    # วิเคราะห์ผลลัพธ์เบื้องต้น
    single_analysis = tester.analyze_results(single_results)
    tester.print_detailed_analysis(single_analysis)
    
    # ทดสอบหลายข้อความ
    print("\n" + "="*60)
    print("🔄 MULTIPLE MESSAGE TESTS")
    print("="*60)
    
    multi_analysis = tester.test_multiple_messages(test_messages[:5], iterations=2)
    tester.print_detailed_analysis(multi_analysis)
    
    # ทดสอบ concurrent
    print("\n" + "="*60)
    print("⚡ CONCURRENT PROCESSING TESTS")
    print("="*60)
    
    concurrent_analysis = tester.test_concurrent_processing(test_messages[:4], max_workers=2)
    tester.print_detailed_analysis(concurrent_analysis)
    
    # สรุปผล
    print("\n" + "="*80)
    print("🏁 FINAL SUMMARY")
    print("="*80)
    
    print(f"Single Message Avg: {single_analysis['timing_stats']['total_time']['avg']:.1f}ms")
    print(f"Multiple Message Avg: {multi_analysis['timing_stats']['total_time']['avg']:.1f}ms")
    print(f"Concurrent Total: {concurrent_analysis.get('concurrent_total_time_ms', 0):.1f}ms")
    
    # หาค่าเฉลี่ยของ NLU processing
    nlu_avg_single = single_analysis['timing_stats']['nlu_analysis']['avg']
    nlu_avg_multi = multi_analysis['timing_stats']['nlu_analysis']['avg']
    
    print(f"\n🧠 NLU Analysis Performance:")
    print(f"   Single: {nlu_avg_single:.1f}ms")
    print(f"   Multiple: {nlu_avg_multi:.1f}ms")
    
    if nlu_avg_single > 3000 or nlu_avg_multi > 3000:
        print("\n⚠️  WARNING: NLU processing is slow (>3 seconds)")
        print("   Main causes likely:")
        print("   1. Large prompt size in INTENT_DETECTION_PROMPT")
        print("   2. LLM model response time")
        print("   3. Network latency to OpenRouter API")
        print("   4. Complex parsing logic")
    
    return {
        'single': single_analysis,
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