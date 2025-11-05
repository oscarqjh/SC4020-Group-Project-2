"""
Medical analysis agents for the crew system.
This module implements specialized AI agents for medical analysis tasks.
"""

from typing import Optional
from crewai import Agent

from .base import BaseAgent, BaseAnalysisTool, AnalysisType
from .tools import get_available_tools


class MedicalAnalysisAgent(BaseAgent):
    """
    Specialized agent for medical analysis tasks.
    
    This agent can handle both disease symptom analysis and breast cancer analysis,
    automatically selecting the appropriate tool based on user input.
    """
    
    def __init__(self, tools: Optional[list[BaseAnalysisTool]] = None):
        """
        Initialize the medical analysis agent.
        
        Args:
            tools: List of analysis tools. If None, uses default tools.
        """
        if tools is None:
            tools = get_available_tools()
        
        super().__init__(
            role="Medical Analysis Specialist",
            goal="Provide accurate and helpful medical analysis based on user input, "
                 "utilizing appropriate diagnostic tools and generating comprehensive reports.",
            backstory="You are an experienced medical analysis specialist with expertise in "
                     "symptom evaluation and breast cancer analysis. You have access to advanced "
                     "analytical tools and can interpret complex medical data to provide "
                     "meaningful insights and recommendations.",
            tools=tools
        )
        
        # Create CrewAI agent instance
        self.crew_agent = Agent(
            role=self.role,
            goal=self.goal,
            backstory=self.backstory,
            tools=[tool for tool in self.tools if hasattr(tool, '_run')],  # CrewAI compatible tools
            verbose=True,
            allow_delegation=False
        )
    
    def process_request(self, input_data: str) -> str:
        """
        Process a medical analysis request.
        
        Args:
            input_data: Raw input string from user
            
        Returns:
            str: Comprehensive analysis response
        """
        try:
            # Select appropriate tool
            selected_tool = self.select_tool(input_data)
            
            if not selected_tool:
                return self._generate_fallback_response(input_data)
            
            # Perform analysis
            analysis_result = selected_tool.analyze(input_data)
            
            # Generate comprehensive response
            response = self._synthesize_response(analysis_result, input_data)
            
            return response
            
        except Exception as e:
            return f"I apologize, but I encountered an error while analyzing your request: {str(e)}. " \
                   "Please try rephrasing your question or providing more specific information."
    
    def _generate_fallback_response(self, input_data: str) -> str:
        """Generate a fallback response when no suitable tool is found."""
        return (
            "I understand you're seeking medical analysis, but I need more specific information "
            "to provide accurate insights. Please provide either:\n\n"
            "• For symptom analysis: Describe your symptoms (e.g., 'I have a cough, fever, and headache')\n"
            "• For breast cancer analysis: Provide tumor measurements or characteristics "
            "(e.g., 'Patient has a tumor with radius 12.5, perimeter 85.2')\n\n"
            "This will help me select the most appropriate analysis tool for your needs."
        )
    
    def _synthesize_response(self, analysis_result, original_input: str) -> str:
        """
        Synthesize a comprehensive response from analysis results.
        
        Args:
            analysis_result: The analysis result from the selected tool
            original_input: Original user input for context
            
        Returns:
            str: Synthesized response
        """
        # Response header based on analysis type
        if analysis_result.analysis_type == AnalysisType.DISEASE_SYMPTOMS:
            header = "**Symptom Analysis Report**"
        elif analysis_result.analysis_type == AnalysisType.BREAST_CANCER:
            header = "**Breast Cancer Analysis Report**"
        else:
            header = "**Medical Analysis Report**"
        
        # Build comprehensive response
        response_parts = [
            header,
            "=" * 50,
            "",
            f"**Analysis Summary:**",
            analysis_result.primary_findings,
            "",
            f"**Confidence Level:** {analysis_result.confidence:.1%}",
            "",
            "**Key Recommendations:**"
        ]
        
        # Add recommendations with bullet points
        for i, recommendation in enumerate(analysis_result.recommendations, 1):
            response_parts.append(f"{i}. {recommendation}")
        
        # Add important disclaimers
        response_parts.extend([
            "",
            "Important Disclaimers:",
            "• This analysis is for informational purposes only",
            "• Not a substitute for professional medical advice",
            "• Please consult a healthcare provider for proper diagnosis and treatment",
            "• Seek immediate medical attention for emergency symptoms",
            "",
            f"**Analysis Details:**",
            f"• Tool used: {analysis_result.metadata.get('tool_name', 'Unknown')}",
            f"• Processing method: {analysis_result.metadata.get('processing_method', 'Standard')}",
        ])
        
        # Add data summary if available
        if analysis_result.raw_data:
            if analysis_result.analysis_type == AnalysisType.DISEASE_SYMPTOMS:
                symptom_count = analysis_result.raw_data.get('symptom_count', 0)
                if symptom_count > 0:
                    response_parts.append(f"• Symptoms analyzed: {symptom_count}")
            elif analysis_result.analysis_type == AnalysisType.BREAST_CANCER:
                measurements = analysis_result.raw_data.get('measurements', {})
                if measurements:
                    response_parts.append(f"• Measurements processed: {len(measurements)}")
        
        response_parts.extend([
            "",
            "If you have additional questions or need clarification on any aspect of this analysis, "
            "please feel free to ask for more information."
        ])
        
        return "\n".join(response_parts)
    
    def get_crew_agent(self) -> Agent:
        """
        Get the CrewAI agent instance.
        
        Returns:
            Agent: CrewAI agent instance
        """
        return self.crew_agent


def create_medical_agent() -> MedicalAnalysisAgent:
    """
    Factory function to create a medical analysis agent with default configuration.
    
    Returns:
        MedicalAnalysisAgent: Configured medical analysis agent
    """
    return MedicalAnalysisAgent()
