"""
AI-powered symptom extraction agent for dynamic symptom identification.
This module implements a CrewAI agent that can intelligently extract symptoms from natural language.
"""

import re
from typing import List, Dict, Any, Type
from crewai import Agent, Task, Crew
from pydantic import BaseModel, Field
from crewai.tools.base_tool import BaseTool
from processors.symptom_data_processor import SymptomDataProcessor


class SymptomExtractionInput(BaseModel):
    """Input schema for symptom extraction."""
    text: str = Field(..., description="Text containing potential symptom descriptions")


class SymptomExtractionTool(BaseTool):
    """
    Tool for extracting symptoms from natural language text using AI analysis.
    """
    
    name: str = "Symptom Extraction Tool"
    description: str = (
        "Extracts and identifies medical symptoms from natural language text. "
        "Can identify both explicit symptoms and implied medical conditions."
    )
    args_schema: Type[BaseModel] = SymptomExtractionInput
    
    def __init__(self):
        super().__init__()
        
        # Known symptom categories for validation and enhancement
        self._symptom_categories = {
            'respiratory': [
                'cough', 'coughing', 'shortness of breath', 'breathing difficulty',
                'wheezing', 'chest pain', 'sore throat', 'throat pain',
                'runny nose', 'nasal congestion', 'stuffy nose', 'sneezing'
            ],
            'systemic': [
                'fever', 'temperature', 'hot', 'chills', 'fatigue', 'tired',
                'exhausted', 'weakness', 'malaise', 'sweating'
            ],
            'neurological': [
                'headache', 'head pain', 'dizziness', 'confusion',
                'memory loss', 'numbness', 'tingling'
            ],
            'gastrointestinal': [
                'nausea', 'vomiting', 'sick', 'diarrhea', 'constipation',
                'stomach pain', 'abdominal pain', 'heartburn', 'bloating'
            ],
            'musculoskeletal': [
                'joint pain', 'muscle pain', 'back pain', 'stiffness',
                'swelling', 'inflammation'
            ],
            'dermatological': [
                'rash', 'skin irritation', 'itching', 'redness',
                'swelling', 'bruising', 'lesions'
            ],
            'cardiovascular': [
                'chest pain', 'palpitations', 'rapid heartbeat',
                'irregular heartbeat', 'high blood pressure'
            ]
        }
        
        # Initialize the extraction agent
        self._init_extraction_agent()
        self._symptom_processor = SymptomDataProcessor(data_path="")
    
    def _init_extraction_agent(self):
        """Initialize the symptom extraction agent."""
        try:
            # Create the symptom extraction agent
            self._extraction_agent = Agent(
                role="Medical Symptom Extraction Specialist",
                goal="Extract and identify all medical symptoms from natural language text with high accuracy",
                backstory=(
                    "You are a specialized medical NLP expert with deep knowledge of symptom terminology, "
                    "medical language patterns, and the ability to identify both explicit and implicit "
                    "symptom descriptions in patient communications. You understand medical synonyms, "
                    "colloquial descriptions, and can identify symptoms even when described in non-medical terms."
                ),
                verbose=False,
                allow_delegation=False
            )
        except Exception as e:
            print(f"Warning: Could not initialize AI agent: {e}")
            self._extraction_agent = None
    
    @property
    def symptom_categories(self) -> Dict[str, List[str]]:
        """Get the symptom categories."""
        return self._symptom_categories
    
    @property
    def extraction_agent(self):
        """Get the extraction agent."""
        return getattr(self, '_extraction_agent', None)
    
    def _run(self, text: str) -> str:
        """CrewAI tool interface method."""
        try:
            result = self.extract_symptoms_with_ai(text)
            return f"Extracted symptoms: {', '.join(result)}"
        except Exception as e:
            return f"Error extracting symptoms: {str(e)}"
    
    def extract_symptoms_with_ai(self, text: str) -> List[str]:
        """
        Extract symptoms using AI analysis combined with rule-based validation.
        
        Args:
            text: Input text containing symptom descriptions
            
        Returns:
            List[str]: List of identified symptoms
        """
        if not text or len(text.strip()) == 0:
            return []
        
        # First pass: AI-based extraction
        ai_symptoms = self._ai_extract_symptoms(text)
        print("ai_symptoms: ", ai_symptoms)
        # Second pass: Rule-based validation and enhancement
        rule_symptoms = self._rule_based_extraction(text)
        
        # Combine and deduplicate
        all_symptoms = list(set(ai_symptoms + rule_symptoms))
        print("all_symptoms: ", all_symptoms)
        # Filter and standardize
        standardized_symptoms = self._standardize_symptoms(all_symptoms)
        
        return standardized_symptoms
    
    def _ai_extract_symptoms(self, text: str) -> List[str]:
        """
        Use AI agent to extract symptoms from text.
        
        Args:
            text: Input text
            
        Returns:
            List[str]: AI-extracted symptoms
        """
        # Check if AI agent is available
        if not self.extraction_agent:
            print("AI agent not available, using fallback extraction")
            return []
        
        try:
            # Create extraction task
            extraction_task = Task(
                description=(
                    f"Analyze the following text and extract all medical symptoms, conditions or problems mentioned. "
                    f"Look for both explicit symptom names and implicit descriptions of health issues. "
                    f"Return ONLY a comma-separated list of standardized symptom names, no explanations.\n\n"
                    f"Text to analyze: {text}\n\n"
                    f"Format your response as: symptom1, symptom2, symptom3"
                ),
                agent=self.extraction_agent,
                expected_output="A comma-separated list of standardized medical symptom names"
            )
            
            # Create temporary crew for execution
            crew = Crew(
                agents=[self.extraction_agent],
                tasks=[extraction_task],
                verbose=False
            )
            
            # Execute task
            result = crew.kickoff()

            # Parse the result
            if isinstance(result.raw, str):
                # Clean and split the result
                symptoms = [s.strip() for s in result.raw.split(',') if s.strip()]
                # # Remove any non-symptom text
                # symptoms = [s for s in symptoms if self._is_likely_symptom(s)]
                return symptoms
            
        except Exception as e:
            print(f"AI extraction error: {e}")
            return []
        
        return []
    
    def _rule_based_extraction(self, text: str) -> List[str]:
        """
        Extract symptoms using rule-based pattern matching.
        
        Args:
            text: Input text
            
        Returns:
            List[str]: Rule-extracted symptoms
        """
        symptoms = []
        text_lower = text.lower()
        
        # Check all symptom categories
        for category, symptom_list in self._symptom_categories.items():
            for symptom in symptom_list:
                # Create pattern for the symptom
                pattern = r'\b' + re.escape(symptom) + r'\b'
                if re.search(pattern, text_lower):
                    symptoms.append(symptom)
        
        return symptoms
    
    def _is_likely_symptom(self, text: str) -> bool:
        """
        Check if a text string is likely to be a symptom.
        
        Args:
            text: Text to check
            
        Returns:
            bool: True if likely a symptom
        """
        text_lower = text.lower().strip()
        
        # Remove common non-symptom words
        non_symptom_indicators = [
            'patient', 'doctor', 'hospital', 'medication', 'treatment',
            'diagnosis', 'test', 'exam', 'result', 'report', 'analysis',
            'and', 'or', 'the', 'a', 'an', 'is', 'are', 'was', 'were',
            'with', 'without', 'has', 'have', 'had', 'may', 'might', 'could'
        ]
        
        if text_lower in non_symptom_indicators:
            return False
        
        # Check if it matches known symptom patterns
        all_symptoms = []
        for symptom_list in self._symptom_categories.values():
            all_symptoms.extend(symptom_list)
        
        # Direct match
        if text_lower in all_symptoms:
            return True
        
        # Partial match with known symptoms
        for symptom in all_symptoms:
            if symptom in text_lower or text_lower in symptom:
                return True
        
        # Length check (symptoms are usually 1-4 words)
        word_count = len(text_lower.split())
        if word_count > 4:
            return False
        
        # Check for medical-sounding words
        medical_patterns = [
            r'pain', r'ache', r'ing$', r'ness$', r'tion$',
            r'difficulty', r'trouble', r'problem'
        ]
        
        for pattern in medical_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def _standardize_symptoms(self, symptoms: List[str]) -> List[str]:
        """
        Standardize symptom names to consistent format.
        
        Args:
            symptoms: Raw symptom list
            
        Returns:
            List[str]: Standardized symptoms
        """
        standardized = []
        
        for symptom in symptoms:
            standardized_symptom = self._symptom_processor.normalize_symptom(symptom).lower().strip()
            
            if standardized_symptom and standardized_symptom not in standardized:
                standardized.append(standardized_symptom)
        
        return standardized


class SymptomExtractor:
    """
    Main symptom extraction class that coordinates AI and rule-based extraction.
    """
    
    def __init__(self):
        """Initialize the symptom extractor with AI capabilities."""
        self.extraction_tool = SymptomExtractionTool()
    
    def extract_symptoms(self, text: str) -> List[str]:
        """
        Extract symptoms from text using combined AI and rule-based approach.
        
        Args:
            text: Input text containing symptom descriptions
            
        Returns:
            List[str]: List of extracted and standardized symptoms
        """
        return self.extraction_tool.extract_symptoms_with_ai(text)
    
    def get_symptom_categories(self) -> Dict[str, List[str]]:
        """
        Get the known symptom categories.
        
        Returns:
            Dict[str, List[str]]: Symptom categories mapping
        """
        return self.extraction_tool.symptom_categories.copy()


# Factory function
def create_symptom_extractor() -> SymptomExtractor:
    """
    Create a symptom extractor instance.
    
    Returns:
        SymptomExtractor: Configured symptom extractor
    """
    return SymptomExtractor()
