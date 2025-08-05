#!/usr/bin/env python3
"""
Test script for NLU parsing performance
Tests the PyParsingNLUParser with various inputs to measure performance
"""

import time
import json
from typing import List, Dict, Any

# Import the parser
from src.llm.node.utils import PyParsingNLUParser, create_pyparsing_nlu_parser_from_config

def test_samples() -> List[Dict[str, str]]:
    """Sample NLU outputs to test parsing performance"""
    return [
        {
            "name": "valid_complete",
            "input": """(intent|inquiry_intent|0.85|0.9|{"source": "user"})
(entity|product|พัดลม|0.8|{"category": "hardware"})
(entity|price|ราคา|0.7|{"type": "inquiry"})
(language|THA|0.9|1|{"detected": "thai"})
(sentiment|neutral|0.6|{"polarity": "neutral"})
END_ANALYSIS"""
        },
        {
            "name": "simple_intent",
            "input": """(intent|greet|0.9|0.8)
(language|THA|0.95|1)
END_ANALYSIS"""
        },
        {
            "name": "malformed_confidence",
            "input": """(intent|purchase_intent|[0.85]|0.9)
(entity|product|คอมพิวเตอร์|[0.8, 0.7]|{"type": "hardware"})
END_ANALYSIS"""
        },
        {
            "name": "missing_fields",
            "input": """(intent|support_intent||)
(entity||laptop|0.6)
END_ANALYSIS"""
        },
        {
            "name": "complex_metadata",
            "input": """(intent|purchase_intent|0.88|0.9|{"confidence_breakdown": {"keyword_match": 0.9, "context_match": 0.85}, "entities_count": 3})
(entity|product|Gaming PC|0.85|{"brand": "custom", "category": "desktop", "price_range": "high"})
(entity|budget|35000|0.7|{"currency": "THB", "type": "exact"})
END_ANALYSIS"""
        },
        {
            "name": "very_short",
            "input": "hi"
        },
        {
            "name": "empty",
            "input": ""
        },
        {
            "name": "no_end_marker",
            "input": """(intent|inquiry_intent|0.75|0.8)
(entity|product|monitor|0.6)"""
        },
        {
            "name": "multiple_same_type",
            "input": """(intent|purchase_intent|0.9|0.95)
(intent|inquiry_intent|0.7|0.8)
(entity|product|CPU|0.85)
(entity|product|RAM|0.8)
(entity|brand|Intel|0.9)
(language|THA|0.95|1)
(sentiment|positive|0.7)
END_ANALYSIS"""
        }
    ]

def run_performance_test():
    """Run parsing performance tests"""
    print("🔍 NLU Parsing Performance Test")
    print("=" * 50)
    
    # Create parser with default config
    config = {
        "tuple_delimiter": "|",
        "record_delimiter": "\n",
        "completion_delimiter": "END_ANALYSIS"
    }
    
    parser = create_pyparsing_nlu_parser_from_config(config)
    
    # Test samples
    samples = test_samples()
    results = []
    
    print(f"Testing {len(samples)} samples...\n")
    
    total_start = time.time()
    
    for i, sample in enumerate(samples, 1):
        sample_name = sample["name"]
        input_text = sample["input"]
        
        print(f"Test {i}: {sample_name}")
        print(f"Input length: {len(input_text)} characters")
        
        # Measure parsing time
        start_time = time.time()
        
        try:
            result = parser.parse_intent_output(input_text)
            parse_time = time.time() - start_time
            
            # Extract key metrics
            status = result.get("parsing_metadata", {}).get("status", "unknown")
            strategy = result.get("parsing_metadata", {}).get("strategy_used", "unknown")
            intents_count = len(result.get("intents", []))
            entities_count = len(result.get("entities", []))
            
            print(f"✅ Status: {status}")
            print(f"   Strategy: {strategy}")
            print(f"   Parse time: {parse_time*1000:.2f}ms")
            print(f"   Intents: {intents_count}, Entities: {entities_count}")
            
            # Store results
            results.append({
                "name": sample_name,
                "input_length": len(input_text),
                "parse_time_ms": round(parse_time * 1000, 2),
                "status": status,
                "strategy": strategy,
                "intents_count": intents_count,
                "entities_count": entities_count,
                "success": status in ["success", "partial_success"]
            })
            
            # Show slow parsing warning
            if parse_time > 0.2:
                print(f"⚠️  SLOW PARSING: {parse_time*1000:.2f}ms")
            
        except Exception as e:
            parse_time = time.time() - start_time
            print(f"❌ Error: {str(e)}")
            print(f"   Parse time: {parse_time*1000:.2f}ms")
            
            results.append({
                "name": sample_name,
                "input_length": len(input_text),
                "parse_time_ms": round(parse_time * 1000, 2),
                "status": "error",
                "strategy": "none",
                "intents_count": 0,
                "entities_count": 0,
                "success": False,
                "error": str(e)
            })
        
        print("-" * 30)
    
    total_time = time.time() - total_start
    
    # Summary statistics
    print("\n📊 Performance Summary")
    print("=" * 30)
    
    successful_tests = [r for r in results if r["success"]]
    failed_tests = [r for r in results if not r["success"]]
    
    print(f"Total tests: {len(results)}")
    print(f"Successful: {len(successful_tests)}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Success rate: {len(successful_tests)/len(results)*100:.1f}%")
    print(f"Total time: {total_time*1000:.2f}ms")
    print(f"Average time per test: {total_time/len(results)*1000:.2f}ms")
    
    if successful_tests:
        parse_times = [r["parse_time_ms"] for r in successful_tests]
        print(f"Fastest parse: {min(parse_times):.2f}ms")
        print(f"Slowest parse: {max(parse_times):.2f}ms")
        print(f"Average parse time: {sum(parse_times)/len(parse_times):.2f}ms")
    
    # Strategy usage
    strategies = {}
    for result in results:
        strategy = result["strategy"]
        strategies[strategy] = strategies.get(strategy, 0) + 1
    
    print(f"\nStrategy usage:")
    for strategy, count in strategies.items():
        print(f"  {strategy}: {count} times")
    
    # Show slow tests
    slow_tests = [r for r in results if r["parse_time_ms"] > 200]
    if slow_tests:
        print(f"\n⚠️  Slow tests (>200ms):")
        for test in slow_tests:
            print(f"  {test['name']}: {test['parse_time_ms']:.2f}ms")
    
    # Show failed tests
    if failed_tests:
        print(f"\n❌ Failed tests:")
        for test in failed_tests:
            error_msg = test.get("error", "Unknown error")
            print(f"  {test['name']}: {error_msg}")
    
    # Save detailed results
    with open("parsing_test_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.time(),
            "total_tests": len(results),
            "successful_tests": len(successful_tests),
            "failed_tests": len(failed_tests),
            "total_time_ms": round(total_time * 1000, 2),
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed results saved to: parsing_test_results.json")
    
    return results

def run_stress_test(num_iterations: int = 100):
    """Run stress test with repeated parsing"""
    print(f"\n🔥 Stress Test ({num_iterations} iterations)")
    print("=" * 40)
    
    config = {
        "tuple_delimiter": "|",
        "record_delimiter": "\n", 
        "completion_delimiter": "END_ANALYSIS"
    }
    
    parser = create_pyparsing_nlu_parser_from_config(config)
    
    # Use a medium complexity sample
    test_input = """(intent|purchase_intent|0.85|0.9|{"source": "user"})
(entity|product|Gaming PC|0.8|{"category": "desktop"})
(entity|budget|35000|0.7|{"currency": "THB"})
(language|THA|0.9|1|{"detected": "thai"})
(sentiment|positive|0.7|{"polarity": "positive"})
END_ANALYSIS"""
    
    parse_times = []
    errors = 0
    
    print(f"Running {num_iterations} iterations...")
    start_time = time.time()
    
    for i in range(num_iterations):
        try:
            iter_start = time.time()
            result = parser.parse_intent_output(test_input)
            iter_time = time.time() - iter_start
            parse_times.append(iter_time * 1000)
            
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{num_iterations} iterations...")
                
        except Exception as e:
            errors += 1
            print(f"Error in iteration {i + 1}: {str(e)}")
    
    total_time = time.time() - start_time
    
    print(f"\n📈 Stress Test Results:")
    print(f"Total iterations: {num_iterations}")
    print(f"Successful: {len(parse_times)}")
    print(f"Errors: {errors}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per parse: {sum(parse_times)/len(parse_times):.2f}ms")
    print(f"Fastest parse: {min(parse_times):.2f}ms")
    print(f"Slowest parse: {max(parse_times):.2f}ms")
    print(f"Throughput: {len(parse_times)/total_time:.1f} parses/second")

if __name__ == "__main__":
    try:
        # Run basic performance test
        results = run_performance_test()
        
        # Ask user if they want stress test
        response = input("\nRun stress test? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            iterations = input("Number of iterations (default 100): ").strip()
            iterations = int(iterations) if iterations.isdigit() else 100
            run_stress_test(iterations)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()