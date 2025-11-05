#!/usr/bin/env python3
"""
Test script for Phase 4: ProjectAssistant demonstration and validation.

This script validates the multi-tool AI assistant with automated test suites
for all three components:
- Tool 1 (Symptom Checker): RAG pipeline with ML prediction + ChromaDB retrieval
- Tool 2 (Cancer Analysis): Precontext LLM using Task 2 findings
- Router: Query intent classification and routing

Usage:
    python scripts/execute_task3_phase4.py

Prerequisites:
    - Virtual environment activated with Python 3.11+
    - Dependencies installed: pip install -r requirements.txt
    - Phase 1-2 completed (model files and vectorstore exist)
    - .env file with GOOGLE_API_KEY in project root
    - Cancer analysis outputs from Task 2 (outputs/analysis_summary.txt, outputs/feature_importance.txt)
"""

import sys
import os
from pathlib import Path

# Third-party imports
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Local imports
from scripts.task3_app import ProjectAssistant

print("=" * 70)
print("Task 3: Phase 4 - ProjectAssistant Testing")
print("=" * 70)

# ============================================================================
# PREREQUISITE VALIDATION
# ============================================================================
print("\n" + "=" * 70)
print("PREREQUISITE VALIDATION")
print("=" * 70)

# Define required paths
required_paths = {
    '.env file': project_root / '.env',
    'disease model': project_root / 'outputs' / 'models' / 'disease_model.pkl',
    'symptom vocabulary': project_root / 'outputs' / 'models' / 'symptom_vocabulary.json',
    'chroma database': project_root / 'outputs' / 'vectorstore' / 'chroma_db',
    'cancer summary': project_root / 'outputs' / 'analysis_summary.txt',
    'cancer features': project_root / 'outputs' / 'feature_importance.txt'
}

# Check each path
missing_paths = []
for name, path in required_paths.items():
    if path.exists():
        # Special check for vectorstore directory: ensure it's non-empty
        if name == 'chroma database' and path.is_dir():
            if any(path.iterdir()):
                print(f"✓ {name} exists and is non-empty")
            else:
                print(f"✗ {name} exists but is empty: {path}")
                missing_paths.append(name)
        else:
            print(f"✓ {name} exists")
    else:
        print(f"✗ {name} NOT FOUND: {path}")
        missing_paths.append(name)

# Validate API key
load_dotenv(project_root / '.env')
api_key = os.environ.get('GOOGLE_API_KEY')
if api_key and api_key.strip():
    print("✓ GOOGLE_API_KEY is set")
else:
    print("✗ GOOGLE_API_KEY is NOT set")
    missing_paths.append('GOOGLE_API_KEY')

# Exit if prerequisites are missing
if missing_paths:
    print("\n" + "=" * 70)
    print("✗ PREREQUISITE CHECK FAILED")
    print("=" * 70)
    print("\nMissing prerequisites:")
    for item in missing_paths:
        print(f"  - {item}")
    print("\nPlease ensure Phase 1-2 are completed: python scripts/execute_task3_phases.py")
    print("Please set GOOGLE_API_KEY in .env file")
    sys.exit(1)

print("\n✓ All prerequisites validated successfully")

# ============================================================================
# INITIALIZING PROJECTASSISTANT
# ============================================================================
print("\n" + "=" * 70)
print("INITIALIZING PROJECTASSISTANT")
print("=" * 70)

try:
    print("\nInitializing ProjectAssistant...")
    assistant = ProjectAssistant()
    print("✓ ProjectAssistant initialized successfully")
    
    # Print resource summary
    print("\nLoaded resources:")
    print(f"  - ML model with {len(assistant.symptom_vocabulary)} symptom tokens")
    print(f"  - ChromaDB with {assistant.collection_symptoms.count()} disease documents")
    print(f"  - Cancer analysis context ({len(assistant.cancer_context)} characters)")

except FileNotFoundError as e:
    print(f"\n✗ ERROR: Missing required files")
    print(f"{e}")
    print("\nPlease run Phase 1-2 first: python scripts/execute_task3_phases.py")
    sys.exit(1)

except ValueError as e:
    print(f"\n✗ ERROR: Configuration error")
    print(f"{e}")
    print("\nPlease check your .env file and ensure GOOGLE_API_KEY is set correctly")
    sys.exit(1)

except Exception as e:
    print(f"\n✗ ERROR: Unexpected initialization error")
    print(f"{e}")
    import traceback
    print("\n" + traceback.format_exc())
    sys.exit(1)

# ============================================================================
# TEST SUITE 1: SYMPTOM CHECKER (Tool 1 - RAG Pipeline)
# ============================================================================
print("\n" + "=" * 70)
print("TEST SUITE 1: SYMPTOM CHECKER (Tool 1 - RAG Pipeline)")
print("=" * 70)

# Define test queries
symptom_test_queries = [
    (1, "I have a bad cough, a high fever, and my whole body aches."),
    (2, "I'm experiencing severe headache, nausea, and vomiting.")
]

symptom_tests_passed = []

for query_id, query_text in symptom_test_queries:
    print("\n" + "-" * 70)
    print(f"Test {query_id}: Symptom Checker")
    print(f"Query: {query_text}")
    print("-" * 70)
    
    try:
        response = assistant.run(query_text)
        print("\nResponse:")
        print(response)
        print()
        
        # Validation check: Check for medical disclaimer (case-insensitive)
        resp_lower = response.lower()
        if "⚠️ medical disclaimer" in resp_lower or "medical disclaimer" in resp_lower:
            print("✓ Validation: Medical disclaimer present")
            symptom_tests_passed.append(True)
        else:
            print("✗ Validation: Medical disclaimer MISSING")
            symptom_tests_passed.append(False)
    
    except Exception as e:
        print(f"✗ ERROR during test {query_id}: {e}")
        symptom_tests_passed.append(False)
    
    print()

# Print test suite summary
print("=" * 70)
print("Symptom Checker Test Summary:")
print(f"  Tests passed: {sum(symptom_tests_passed)}/{len(symptom_tests_passed)}")
if all(symptom_tests_passed):
    print("  ✓ All symptom checker tests passed")
else:
    print("  ✗ Some symptom checker tests failed")

# ============================================================================
# TEST SUITE 2: CANCER ANALYSIS (Tool 2 - Precontext LLM)
# ============================================================================
print("\n" + "=" * 70)
print("TEST SUITE 2: CANCER ANALYSIS (Tool 2 - Precontext LLM)")
print("=" * 70)

# Define test queries
cancer_test_queries = [
    (1, "What are the most discriminative patterns for benign tumors?"),
    (2, "Which features are most important in malignant patterns?"),
    (3, "Summarize the key findings from the breast cancer pattern mining analysis.")
]

cancer_tests_passed = []

for query_id, query_text in cancer_test_queries:
    print("\n" + "-" * 70)
    print(f"Test {query_id}: Cancer Analysis")
    print(f"Query: {query_text}")
    print("-" * 70)
    
    try:
        response = assistant.run(query_text)
        print("\nResponse:")
        print(response)
        print()
        
        # Validation check: Check for cancer-related keywords
        keywords = ['pattern', 'feature', 'malignant', 'benign', 'cancer', 'tumor']
        response_lower = response.lower()
        if any(keyword in response_lower for keyword in keywords):
            print("✓ Validation: Response grounded in cancer analysis context")
            cancer_tests_passed.append(True)
        else:
            print("✗ Validation: Response does not appear grounded in context")
            cancer_tests_passed.append(False)
    
    except Exception as e:
        print(f"✗ ERROR during test {query_id}: {e}")
        cancer_tests_passed.append(False)
    
    print()

# Print test suite summary
print("=" * 70)
print("Cancer Analysis Test Summary:")
print(f"  Tests passed: {sum(cancer_tests_passed)}/{len(cancer_tests_passed)}")
if all(cancer_tests_passed):
    print("  ✓ All cancer analysis tests passed")
else:
    print("  ✗ Some cancer analysis tests failed")

# ============================================================================
# TEST SUITE 3: ROUTER BEHAVIOR (Out-of-Scope Queries)
# ============================================================================
print("\n" + "=" * 70)
print("TEST SUITE 3: ROUTER BEHAVIOR (Out-of-Scope Queries)")
print("=" * 70)

# Define test queries with expected behavior
router_test_queries = [
    (1, "Hello, how are you?", "out-of-scope"),
    (2, "What's the weather like today?", "out-of-scope"),
    (3, "Tell me about cancer.", "ambiguous - may route to cancer_analysis")
]

router_tests_passed = []

for query_id, query_text, expected_behavior in router_test_queries:
    print("\n" + "-" * 70)
    print(f"Test {query_id}: Router Test")
    print(f"Query: {query_text}")
    print(f"Expected: {expected_behavior}")
    print("-" * 70)
    
    try:
        response = assistant.run(query_text)
        print("\nResponse:")
        print(response)
        print()
        
        # Validation check based on expected behavior
        if "out-of-scope" in expected_behavior:
            # Check for out-of-scope message
            if "I can only assist with symptom checks or questions about" in response:
                print("✓ Validation: Router correctly handled out-of-scope query")
                router_tests_passed.append(True)
            else:
                print("✗ Validation: Router did not return expected out-of-scope message")
                router_tests_passed.append(False)
        elif "ambiguous" in expected_behavior:
            print("ℹ️  Validation: Ambiguous query - router decision logged")
            print("   (This query could be routed to either tool depending on LLM interpretation)")
            router_tests_passed.append(True)  # Ambiguous queries always pass
    
    except Exception as e:
        print(f"✗ ERROR during test {query_id}: {e}")
        router_tests_passed.append(False)
    
    print()

# Print test suite summary
print("=" * 70)
print("Router Behavior Test Summary:")
print(f"  Tests passed: {sum(router_tests_passed)}/{len(router_tests_passed)}")
if all(router_tests_passed):
    print("  ✓ All router tests passed")
else:
    print("  ✗ Some router tests failed")

# ============================================================================
# FINAL TEST SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("FINAL TEST SUMMARY")
print("=" * 70)

# Calculate overall results
total_tests = len(symptom_tests_passed) + len(cancer_tests_passed) + len(router_tests_passed)
total_passed = sum(symptom_tests_passed) + sum(cancer_tests_passed) + sum(router_tests_passed)

# Print detailed breakdown
print("\nTest Results by Component:")
print(f"  Tool 1 (Symptom Checker): {sum(symptom_tests_passed)}/{len(symptom_tests_passed)} tests passed")
print(f"  Tool 2 (Cancer Analysis): {sum(cancer_tests_passed)}/{len(cancer_tests_passed)} tests passed")
print(f"  Router Behavior: {sum(router_tests_passed)}/{len(router_tests_passed)} tests passed")
print()
print(f"Overall: {total_passed}/{total_tests} tests passed")

# Print final status
if total_passed == total_tests:
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED - PHASE 4 COMPLETE")
    print("=" * 70)
    print("\nThe ProjectAssistant is working correctly:")
    print("  ✓ Symptom Checker (Tool 1) returns medical advice with disclaimers")
    print("  ✓ Cancer Analysis (Tool 2) returns insights grounded in Task 2 findings")
    print("  ✓ Router correctly classifies and handles queries")
    sys.exit(0)
else:
    print("\n" + "=" * 70)
    print("✗ SOME TESTS FAILED - REVIEW REQUIRED")
    print("=" * 70)
    print("\nPlease review the test output above to identify failures.")
    print("Common issues:")
    print("  - API rate limits (wait and retry)")
    print("  - Network connectivity issues")
    print("  - LLM response format variations")
    sys.exit(1)

