#!/usr/bin/env python3
"""
Test script for the Medical Analysis Crew System.

This script tests the core functionality of the medical analysis system
without requiring interactive input.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent  # Go up one level from tests/ to project root
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def test_disease_analysis():
    """Test disease analysis functionality."""
    print("🧪 Testing Disease Analysis...")
    
    from crew import create_medical_crew
    
    crew = create_medical_crew()
    crew.initialize()
    
    # Test symptom query
    test_query = "Hi, I have these symptoms: cough, runny nose, and headache. What could this be?"
    
    result = crew.execute_task(test_query)
    print("✅ Disease analysis test completed")
    print(f"Result length: {len(result)} characters")
    return "symptom" in result.lower() or "cough" in result.lower()

def test_breast_cancer_analysis():
    """Test breast cancer analysis functionality."""
    print("\n🧪 Testing Breast Cancer Analysis...")
    
    from crew import create_medical_crew
    
    crew = create_medical_crew()
    crew.initialize()
    
    # Test breast cancer query
    test_query = "Patient has a tumor with radius 15.2, perimeter 98.5, area 725.4"
    
    result = crew.execute_task(test_query)
    print("✅ Breast cancer analysis test completed")
    print(f"Result length: {len(result)} characters")
    return "tumor" in result.lower() or "radius" in result.lower()

def test_system_status():
    """Test system status functionality."""
    print("\n🧪 Testing System Status...")
    
    from crew import create_medical_crew
    
    crew = create_medical_crew()
    crew.initialize()
    
    status = crew.get_system_status()
    print("✅ System status test completed")
    print(f"Initialized: {status['initialized']}")
    print(f"Agents: {status['agents_count']}")
    return status['initialized'] and status['agents_count'] > 0

def test_tool_selection():
    """Test tool selection logic."""
    print("\n🧪 Testing Tool Selection...")
    
    from crew.tools import get_available_tools
    
    tools = get_available_tools()
    print(f"✅ Found {len(tools)} tools")
    
    # Test disease tool detection
    disease_tool = None
    cancer_tool = None
    
    for tool in tools:
        if "Disease" in tool.name:
            disease_tool = tool
        elif "Cancer" in tool.name:
            cancer_tool = tool
    
    # Test symptom detection
    symptom_query = "I have a cough and fever"
    disease_can_handle = disease_tool.can_handle(symptom_query) if disease_tool else False
    
    # Test cancer detection
    cancer_query = "tumor radius 12.5"
    cancer_can_handle = cancer_tool.can_handle(cancer_query) if cancer_tool else False
    
    print(f"Disease tool can handle symptoms: {disease_can_handle}")
    print(f"Cancer tool can handle tumor data: {cancer_can_handle}")
    
    return disease_can_handle and cancer_can_handle

def main():
    """Run all tests."""
    print("🔬 MEDICAL ANALYSIS CREW SYSTEM - TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Tool Selection Logic", test_tool_selection),
        ("System Status", test_system_status),
        ("Disease Analysis", test_disease_analysis),
        ("Breast Cancer Analysis", test_breast_cancer_analysis),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, None))
            print(f"✅ {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"❌ {test_name}: ERROR - {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("-" * 60)
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    
    for test_name, result, error in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
        if error:
            print(f"    Error: {error}")
    
    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
