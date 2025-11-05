"""
Command Line Interface for the Medical Analysis Crew System.
This module provides the interactive CLI for user interactions.
"""

import sys
from typing import Optional
from .crew_manager import create_medical_crew, MedicalAnalysisCrew


class MedicalAnalysisCLI:
    """
    Command Line Interface for the Medical Analysis System.
    
    Provides an interactive interface for users to input medical queries
    and receive analysis results from the AI crew system.
    """
    
    def __init__(self, crew: Optional[MedicalAnalysisCrew] = None):
        """
        Initialize the CLI with a medical analysis crew.
        
        Args:
            crew: Medical analysis crew instance. If None, creates default crew.
        """
        self.crew = crew or create_medical_crew()
        self.running = False
    
    def display_welcome(self) -> None:
        """Display welcome message and system information."""
        print("=" * 70)
        print("MEDICAL ANALYSIS AI SYSTEM")
        print("=" * 70)
        print()
        print("Welcome to the Medical Analysis AI System!")
        print("This system can help analyze:")
        print("• Disease symptoms and provide preliminary diagnosis suggestions")
        print("• Breast cancer patient data and tumor characteristics")
        print()
        print("Available capabilities:")
        capabilities = self.crew.get_available_capabilities()
        for capability in capabilities:
            print(f"   {capability}")
        print()
        print("IMPORTANT DISCLAIMER:")
        print("   This system provides informational analysis only.")
        print("   Always consult healthcare professionals for medical decisions.")
        print()
        print("=" * 70)
        print()
    
    def display_help(self) -> None:
        """Display help information and example queries."""
        print("\nHELP & EXAMPLES")
        print("-" * 50)
        print()
        print("Example queries for SYMPTOM ANALYSIS:")
        print("   • 'Hi, I have these symptoms: cough, runny nose, headache. What could this be?'")
        print("   • 'I'm feeling sick with fever and sore throat'")
        print("   • 'Patient experiencing fatigue, nausea, and stomach pain'")
        print()
        print("Example queries for BREAST CANCER ANALYSIS:")
        print("   • 'Patient has a tumor with radius 15.2, perimeter 85.2, area 490.1'")
        print("   • 'Breast mass detected: smoothness 0.1, compactness 0.2, concavity 0.15'")
        print("   • 'Mammogram shows irregular mass with microcalcifications'")
        print()
        print("COMMANDS:")
        print("   • 'help' or '?' - Show this help message")
        print("   • 'status' - Show system status")
        print("   • 'examples' - Show more detailed examples")
        print("   • 'exit', 'quit', or 'q' - Exit the system")
        print()
        print("-" * 50)
    
    def display_examples(self) -> None:
        """Display detailed examples with expected outputs."""
        print("\nDETAILED EXAMPLES")
        print("-" * 60)
        print()
        print("SYMPTOM ANALYSIS EXAMPLE:")
        print("   Input: 'I have a persistent cough, runny nose, and mild headache for 3 days'")
        print("   Expected: System will detect symptoms and suggest possible conditions")
        print("   Output: Analysis report with preliminary diagnosis suggestions")
        print()
        print("BREAST CANCER ANALYSIS EXAMPLE:")
        print("   Input: 'Patient data: tumor radius 15.2, perimeter 98.5, area 725.4, ")
        print("           smoothness 0.08, compactness 0.12'")
        print("   Expected: System will analyze measurements and provide insights")
        print("   Output: Detailed analysis with risk factors and recommendations")
        print()
        print("UNCLEAR INPUT EXAMPLE:")
        print("   Input: 'I don't feel good'")
        print("   Expected: System will ask for more specific information")
        print("   Output: Guidance on how to provide better input")
        print()
        print("-" * 60)
    
    def display_status(self) -> None:
        """Display current system status."""
        print("\nSYSTEM STATUS")
        print("-" * 40)
        status = self.crew.get_system_status()
        
        print(f"System Initialized: {'Yes' if status['initialized'] else 'No'}")
        print(f"Active Agents: {status['agents_count']}")
        print(f"Primary Agent: {status['primary_agent_role']}")
        print()
        
        if status['agents']:
            print("Agent Details:")
            for agent_info in status['agents']:
                print(f"   • {agent_info['role']}: {agent_info['tools_count']} tools")
                for tool in agent_info['tools']:
                    print(f"     - {tool}")
        print("-" * 40)
    
    def get_user_input(self) -> str:
        """
        Get user input with proper prompt.
        
        Returns:
            str: User input string
        """
        try:
            return input("\nEnter your medical query (or 'help' for assistance): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            return "exit"
    
    def process_command(self, user_input: str) -> bool:
        """
        Process user commands and queries.
        
        Args:
            user_input: User input string
            
        Returns:
            bool: True to continue running, False to exit
        """
        if not user_input:
            print("Please enter a query or command.")
            return True
        
        # Handle commands
        command = user_input.lower().strip()
        
        if command in ['exit', 'quit', 'q']:
            print("\nThank you for using the Medical Analysis AI System!")
            print("Remember: Always consult healthcare professionals for medical decisions.")
            return False
        
        elif command in ['help', '?']:
            self.display_help()
            return True
        
        elif command == 'status':
            self.display_status()
            return True
        
        elif command == 'examples':
            self.display_examples()
            return True
        
        elif len(command) < 5:
            print("Please provide a more detailed query or use 'help' for assistance.")
            return True
        
        # Process medical query
        else:
            try:
                print("\nAnalyzing your query...")
                result = self.crew.execute_task(user_input)
                print("\n" + result)
                
            except Exception as e:
                print(f"\nError processing your query: {str(e)}")
                print("Please try rephrasing your question or use 'help' for guidance.")
        
        return True
    
    def run(self) -> None:
        """Main CLI loop."""
        try:
            # Initialize system
            self.crew.initialize()
            
            # Display welcome
            self.display_welcome()
            
            # Main interaction loop
            self.running = True
            while self.running:
                user_input = self.get_user_input()
                self.running = self.process_command(user_input)
                
        except KeyboardInterrupt:
            print("\n\nSystem interrupted. Goodbye!")
        except Exception as e:
            print(f"\nUnexpected error: {str(e)}")
            print("Please restart the system or contact support.")
        finally:
            self.running = False


def main() -> None:
    """
    Main entry point for the CLI application.
    """
    try:
        cli = MedicalAnalysisCLI()
        cli.run()
    except Exception as e:
        print(f"Failed to start Medical Analysis CLI: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
