"""
Base classes for the medical analysis crew system.
This module defines abstract base classes and interfaces for the AI system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class AnalysisType(Enum):
    """Enumeration of supported analysis types."""
    DISEASE_SYMPTOMS = "disease_symptoms"
    BREAST_CANCER = "breast_cancer"
    UNKNOWN = "unknown"


@dataclass
class AnalysisResult:
    """
    Standardized result structure for analysis operations.
    
    Attributes:
        analysis_type: Type of analysis performed
        confidence: Confidence score (0.0 to 1.0)
        primary_findings: Main findings from the analysis
        recommendations: List of recommendations
        raw_data: Raw data from the analysis tool
        metadata: Additional metadata about the analysis
    """
    analysis_type: AnalysisType
    confidence: float
    primary_findings: str
    recommendations: list[str]
    raw_data: Dict[str, Any]
    metadata: Dict[str, Any]


class BaseAnalysisTool(ABC):
    """
    Abstract base class for all analysis tools.
    
    This class defines the interface that all analysis tools must implement.
    It ensures consistency across different tool implementations.
    """
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def analyze(self, input_data: str) -> AnalysisResult:
        """
        Perform analysis on the input data.
        
        Args:
            input_data: Raw input string from user
            
        Returns:
            AnalysisResult: Standardized analysis result
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def can_handle(self, input_data: str) -> bool:
        """
        Determine if this tool can handle the given input.
        
        Args:
            input_data: Raw input string from user
            
        Returns:
            bool: True if this tool can handle the input, False otherwise
        """
        pass
    
    def validate_input(self, input_data: str) -> bool:
        """
        Validate input data format and content.
        
        Args:
            input_data: Raw input string from user
            
        Returns:
            bool: True if input is valid, False otherwise
        """
        return isinstance(input_data, str) and len(input_data.strip()) > 0


class BaseAgent(ABC):
    """
    Abstract base class for AI agents.
    
    This class provides the framework for creating specialized AI agents
    that can use tools and generate responses.
    """
    
    def __init__(self, role: str, goal: str, backstory: str, tools: list[BaseAnalysisTool]):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools
        self._tool_registry = {tool.name: tool for tool in tools}
    
    def select_tool(self, input_data: str) -> Optional[BaseAnalysisTool]:
        """
        Select the most appropriate tool for the given input.
        
        Args:
            input_data: Raw input string from user
            
        Returns:
            BaseAnalysisTool or None: Selected tool or None if no suitable tool found
        """
        for tool in self.tools:
            if tool.can_handle(input_data):
                return tool
        return None
    
    @abstractmethod
    def process_request(self, input_data: str) -> str:
        """
        Process a user request and generate a response.
        
        Args:
            input_data: Raw input string from user
            
        Returns:
            str: Generated response
        """
        pass


class BaseCrew(ABC):
    """
    Abstract base class for crew management.
    
    This class handles the coordination of agents and the overall
    workflow of the AI system.
    """
    
    def __init__(self, agents: list[BaseAgent]):
        self.agents = agents
    
    @abstractmethod
    def execute_task(self, task_description: str) -> str:
        """
        Execute a task using the available agents.
        
        Args:
            task_description: Description of the task to execute
            
        Returns:
            str: Result of task execution
        """
        pass
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the crew system."""
        pass
