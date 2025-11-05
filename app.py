#!/usr/bin/env python3
"""
Medical Analysis AI System - Main Application

This is the main entry point for the Medical Analysis AI System.
It initializes and runs the CrewAI-based medical analysis crew system
with an interactive command-line interface.

Usage:
    python app.py                          # Interactive mode
    python app.py --prompt "your query"    # Direct prompt mode

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
import argparse
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Medical Analysis AI System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py
    Start in interactive mode

  python app.py --prompt "I have a fever and cough"
    Analyze symptoms directly

  python app.py --prompt "Patient has tumor radius 12.5, area 490.1"
    Analyze breast cancer data directly
        """
    )
    
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Direct prompt for analysis (skips interactive mode)"
    )
    
    return parser.parse_args()


def run_direct_analysis(prompt: str):
    """
    Run direct analysis with the provided prompt.
    
    Args:
        prompt: User query to analyze
    """
    try:
        # Import the crew manager after adding src to path
        from crew.crew_manager import create_medical_crew
        
        print("Medical Analysis AI System - Direct Mode")
        print("=" * 50)
        print(f"Query: {prompt}")
        print("-" * 50)
        
        # Initialize crew and process query
        print("Initializing AI system...")
        crew = create_medical_crew()
        crew.initialize()
        
        print("Processing query...")
        result = crew.execute_task(prompt)
        
        print("\nAnalysis Result:")
        print(result)
        print("\n" + "=" * 50)
        print("IMPORTANT: This analysis is for informational purposes only.")
        print("Always consult healthcare professionals for medical decisions.")
        
    except ImportError as e:
        print(f"Import Error: {e}")
        print("Please ensure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing query: {e}")
        sys.exit(1)


def main():
    """
    Main application entry point.
    
    Initializes and runs the Medical Analysis AI System with CLI interface.
    """
    args = parse_arguments()
    
    # Check if direct prompt mode is requested
    if args.prompt:
        run_direct_analysis(args.prompt)
        return
    
    # Otherwise run interactive mode
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
