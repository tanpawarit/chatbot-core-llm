#!/usr/bin/env python3
"""
Test runner script for the routing system
Run this script to execute all routing system tests
"""

import subprocess
import sys
from pathlib import Path

def run_tests():
    """Run all routing system tests"""
    print("🧪 Running Context Routing System Tests...\n")
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    test_file = project_root / "tests" / "test_routing_system.py"
    
    # Run pytest with verbose output
    try:
        result = subprocess.run([
            "uv", "run", "python", "-m", "pytest", 
            str(test_file), "-v", "--tb=short"
        ], cwd=project_root, capture_output=False)
        
        if result.returncode == 0:
            print("\n✅ All tests passed!")
            return True
        else:
            print("\n❌ Some tests failed!")
            return False
            
    except FileNotFoundError:
        print("❌ Error: 'uv' command not found. Please install uv first.")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def run_specific_test_class(class_name):
    """Run tests for a specific test class"""
    print(f"🧪 Running {class_name} tests...\n")
    
    project_root = Path(__file__).parent.parent
    test_pattern = f"tests/test_routing_system.py::{class_name}"
    
    try:
        result = subprocess.run([
            "uv", "run", "python", "-m", "pytest", 
            test_pattern, "-v", "--tb=short"
        ], cwd=project_root, capture_output=False)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def main():
    """Main test runner"""
    if len(sys.argv) > 1:
        # Run specific test class
        class_name = sys.argv[1]
        success = run_specific_test_class(class_name)
    else:
        # Run all tests
        success = run_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    print("=" * 60)
    print("Context Routing System Test Suite")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        print(f"Test Class: {sys.argv[1]}")
        print("\nAvailable test classes:")
        print("- TestContextRouter")
        print("- TestTokenEstimation") 
        print("- TestIntegrationScenarios")
        print("- TestEdgeCases")
    else:
        print("Running all routing system tests...")
        print("\nTest Coverage:")
        print("✓ Context selection logic")
        print("✓ Token usage estimation")
        print("✓ Intent-based routing")
        print("✓ Configuration parsing")
        print("✓ Error handling")
        print("✓ Integration scenarios")
    
    print("=" * 60)
    main()