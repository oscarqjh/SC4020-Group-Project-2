#!/usr/bin/env python3
"""
Medical Analysis AI System - Main Application

This is the main entry point for the Medical Analysis AI System.
It initializes and runs the CrewAI-based medical analysis crew system
with an interactive command-line interface.

Usage:
    python app.py

The system provides:
- Disease symptom analysis with preliminary diagnosis suggestions
- Breast cancer data analysis with tumor characteristic evaluation
- Interactive CLI for user queries
- Intelligent tool selection based on input content

Example queries:
- "I have a cough, fever, and headache. What could this be?"
- "Patient has tumor radius 12.5, perimeter 85.2, area 490.1"
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def main():
    """
    Main application entry point.
    
    Initializes and runs the Medical Analysis AI System with CLI interface.
    """
    try:
        # Import the CLI after adding src to path
        from crew import MedicalAnalysisCLI

        # Create and run the CLI
        print("Starting Medical Analysis AI System...")
        cli = MedicalAnalysisCLI()
        cli.run()

    except ImportError as e:
        print(f"Import Error: {e}")
        print("Please ensure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nSystem interrupted by user. Goodbye!")
        sys.exit(0)

    except Exception as e:
        print(f"Unexpected error starting application: {e}")
        print("Please check your installation and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
