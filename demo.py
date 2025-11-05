#!/usr/bin/env python3
"""
Demo script for the Medical Analysis AI System.

This script demonstrates the key features of the system with example queries.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def demo_disease_analysis():
    """Demonstrate disease analysis functionality."""
    print("DISEASE ANALYSIS DEMO")
    print("=" * 50)
    
    from crew import create_medical_crew
    
    crew = create_medical_crew()
    crew.initialize()
    
    # Demo queries
    queries = [
        "Hi, I have these symptoms: cough, runny nose, and headache. What could this be?",
        "I'm feeling sick with fever and sore throat for 2 days",
        "Patient experiencing fatigue, nausea, and digestive issues"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\nDemo Query {i}:")
        print(f"Input: {query}")
        print("\nAnalysis Result:")
        result = crew.execute_task(query)
        print(result)
        print("\n" + "-" * 50)

def demo_breast_cancer_analysis():
    """Demonstrate breast cancer analysis functionality."""
    print("\n\nBREAST CANCER ANALYSIS DEMO")
    print("=" * 50)
    
    from crew import create_medical_crew
    
    crew = create_medical_crew()
    crew.initialize()
    
    # Demo queries
    queries = [
        "Patient has a tumor with radius 15.2, perimeter 98.5, area 725.4",
        "Breast mass detected: smoothness 0.1, compactness 0.2, concavity 0.15",
        "Mammogram shows irregular mass with microcalcifications and malignant characteristics"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\nDemo Query {i}:")
        print(f"Input: {query}")
        print("\nAnalysis Result:")
        result = crew.execute_task(query)
        print(result)
        print("\n" + "-" * 50)

def main():
    """Run the demo."""
    print("🎬 MEDICAL ANALYSIS AI SYSTEM - DEMO")
    print("=" * 60)
    print("This demo showcases the two main analysis capabilities:")
    print("1. Disease symptom analysis")
    print("2. Breast cancer data analysis")
    print("=" * 60)
    
    try:
        demo_disease_analysis()
        demo_breast_cancer_analysis()
        
        print("\n\nDEMO COMPLETE!")
        print("=" * 60)
        print("Both analysis tools demonstrated successfully")
        print("To use the interactive system, run: python app.py")
        print("Example commands: 'help', 'status', 'examples'")
        print("=" * 60)
        
    except Exception as e:
        print(f"Demo failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
