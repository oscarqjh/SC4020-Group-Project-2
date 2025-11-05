"""
Crew management system for coordinating medical analysis agents.
This module implements the main crew coordination logic.
"""

from typing import Optional
from crewai import Crew, Task

from .base import BaseCrew, BaseAgent
from .agent import MedicalAnalysisAgent, create_medical_agent


class MedicalAnalysisCrew(BaseCrew):
    """
    Main crew system for coordinating medical analysis tasks.
    
    This crew manages the execution of medical analysis tasks using
    specialized agents and tools.
    """
    
    def __init__(self, agents: Optional[list[BaseAgent]] = None):
        """
        Initialize the medical analysis crew.
        
        Args:
            agents: List of agents. If None, creates default medical agent.
        """
        if agents is None:
            agents = [create_medical_agent()]
        
        super().__init__(agents)
        
        # Store the primary medical agent
        self.medical_agent = agents[0] if agents else create_medical_agent()
        
        # Create CrewAI crew instance
        self.crew = Crew(
            agents=[agent.get_crew_agent() if hasattr(agent, 'get_crew_agent') else agent for agent in agents],
            verbose=True
        )
        
        self.initialized = False
    
    def initialize(self) -> None:
        """Initialize the crew system."""
        if not self.initialized:
            print("Initializing Medical Analysis Crew...")
            print(f"Loaded {len(self.agents)} agent(s)")
            print(f"Medical agent ready with {len(self.medical_agent.tools)} tool(s)")
            print("Medical Analysis System ready!")
            self.initialized = True
    
    def execute_task(self, task_description: str) -> str:
        """
        Execute a medical analysis task.
        
        Args:
            task_description: User's medical query or request
            
        Returns:
            str: Analysis result and recommendations
        """
        if not self.initialized:
            self.initialize()
        
        try:
            # Create a CrewAI task
            task = Task(
                description=f"Analyze the following medical query and provide comprehensive insights: {task_description}",
                agent=self.medical_agent.get_crew_agent(),
                expected_output="A detailed medical analysis report with findings, recommendations, and appropriate disclaimers"
            )
            
            # Execute using both our custom logic and CrewAI
            custom_result = self.medical_agent.process_request(task_description)
            
            return custom_result
            
        except Exception as e:
            return (
                f"Error Processing Request\n\n"
                f"I encountered an issue while processing your request: {str(e)}\n\n"
                f"Troubleshooting Suggestions:\n"
                f"1. Please ensure your request is clearly formatted\n"
                f"2. For symptoms: List specific symptoms you're experiencing\n"
                f"3. For medical data: Provide numerical measurements when available\n"
                f"4. Try rephrasing your question if needed\n\n"
                f"If the problem persists, please contact system support."
            )
    
    def get_system_status(self) -> dict:
        """
        Get the current status of the crew system.
        
        Returns:
            dict: System status information
        """
        agent_status = []
        for agent in self.agents:
            if hasattr(agent, 'tools'):
                agent_status.append({
                    'role': agent.role,
                    'tools_count': len(agent.tools),
                    'tools': [tool.name for tool in agent.tools]
                })
        
        return {
            'initialized': self.initialized,
            'agents_count': len(self.agents),
            'agents': agent_status,
            'primary_agent_role': self.medical_agent.role if self.medical_agent else None
        }
    
    def get_available_capabilities(self) -> list[str]:
        """
        Get a list of available analysis capabilities.
        
        Returns:
            list[str]: List of capability descriptions
        """
        capabilities = []
        
        if self.medical_agent and hasattr(self.medical_agent, 'tools'):
            for tool in self.medical_agent.tools:
                capabilities.append(f"• {tool.name}: {tool.description}")
        
        return capabilities


def create_medical_crew() -> MedicalAnalysisCrew:
    """
    Factory function to create a medical analysis crew with default configuration.
    
    Returns:
        MedicalAnalysisCrew: Configured medical analysis crew
    """
    return MedicalAnalysisCrew()
