"""
Medical Analysis Crew System

A sophisticated AI-powered medical analysis system using CrewAI framework.
This system provides intelligent analysis of disease symptoms and breast cancer data.

Main Components:
- MedicalAnalysisCLI: Interactive command-line interface
- MedicalAnalysisCrew: Main crew coordination system
- MedicalAnalysisAgent: Specialized medical analysis agent
- DiseaseAnalysisTool: Tool for symptom analysis
- BreastCancerAnalysisTool: Tool for breast cancer analysis

Usage:
    from src.crew import MedicalAnalysisCLI
    
    cli = MedicalAnalysisCLI()
    cli.run()
"""

from .base import (
    BaseAnalysisTool, 
    BaseAgent, 
    BaseCrew, 
    AnalysisResult, 
    AnalysisType
)

from .tools import (
    DiseaseAnalysisTool,
    BreastCancerAnalysisTool,
    get_available_tools
)

from .agent import (
    MedicalAnalysisAgent,
    create_medical_agent
)

from .crew_manager import (
    MedicalAnalysisCrew,
    create_medical_crew
)

from .cli import (
    MedicalAnalysisCLI,
    main
)

__all__ = [
    # Base classes
    'BaseAnalysisTool',
    'BaseAgent', 
    'BaseCrew',
    'AnalysisResult',
    'AnalysisType',
    
    # Tools
    'DiseaseAnalysisTool',
    'BreastCancerAnalysisTool',
    'get_available_tools',
    
    # Agent
    'MedicalAnalysisAgent',
    'create_medical_agent',
    
    # Crew
    'MedicalAnalysisCrew',
    'create_medical_crew',
    
    # CLI
    'MedicalAnalysisCLI',
    'main'
]

__version__ = "1.0.0"
__author__ = "Medical Analysis Crew Team"
__description__ = "AI-powered medical analysis system using CrewAI framework"
