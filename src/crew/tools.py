"""
Medical analysis tools for the crew system.
This module implements specific analysis tools for disease symptoms and breast cancer analysis.
"""

import re
import random
from typing import Dict, Any, Type
from pydantic import BaseModel, Field
from crewai.tools.base_tool import BaseTool

from .base import AnalysisResult, AnalysisType


class DiseaseAnalysisInput(BaseModel):
    """Input schema for disease analysis tool."""
    symptoms_description: str = Field(..., description="Description of patient symptoms")


class DiseaseAnalysisTool(BaseTool):
    """
    Tool for analyzing disease symptoms and providing diagnosis suggestions.
    
    This tool can identify symptoms in user input and provide preliminary
    diagnosis suggestions based on common symptom patterns.
    """
    
    name: str = "Disease Analysis Tool"
    description: str = (
        "Analyzes patient symptoms to provide preliminary diagnosis suggestions. "
        "Use this tool when users describe symptoms like cough, fever, headache, etc."
    )
    args_schema: Type[BaseModel] = DiseaseAnalysisInput
    
    # Additional attributes for the tool
    symptom_patterns: list = []
    
    def __init__(self):
        super().__init__()
        
        # Common symptoms patterns for detection
        self.symptom_patterns = [
            r'\b(?:cough|coughing)\b',
            r'\b(?:fever|temperature|hot)\b',
            r'\b(?:runny nose|nasal congestion|stuffy nose)\b',
            r'\b(?:headache|head pain)\b',
            r'\b(?:sore throat|throat pain)\b',
            r'\b(?:fatigue|tired|exhausted)\b',
            r'\b(?:nausea|vomiting|sick)\b',
            r'\b(?:diarrhea|stomach pain|abdominal pain)\b',
            r'\b(?:rash|skin irritation)\b',
            r'\b(?:shortness of breath|breathing difficulty)\b'
        ]
    
    def can_handle(self, input_data: str) -> bool:
        """
        Determine if input contains symptom-related content.
        
        Args:
            input_data: Raw input string from user
            
        Returns:
            bool: True if input contains symptoms, False otherwise
        """
        input_lower = input_data.lower()
        
        # Check for symptom keywords
        symptom_indicators = [
            'symptom', 'symptoms', 'feel', 'feeling', 'pain', 'ache', 
            'sick', 'ill', 'disease', 'condition', 'have', 'experiencing'
        ]
        
        has_symptom_context = any(indicator in input_lower for indicator in symptom_indicators)
        has_specific_symptoms = any(re.search(pattern, input_lower) for pattern in self.symptom_patterns)
        
        return has_symptom_context or has_specific_symptoms
    
    def analyze(self, input_data: str) -> AnalysisResult:
        """
        Analyze symptoms and provide diagnosis suggestions.
        
        Args:
            input_data: Raw input string from user
            
        Returns:
            AnalysisResult: Analysis result with diagnosis suggestions
        """
        if not input_data or len(input_data.strip()) == 0:
            raise ValueError("Invalid input data provided")
        
        # Extract symptoms from input
        detected_symptoms = self._extract_symptoms(input_data)
        
        # Generate diagnosis suggestions (placeholder implementation)
        diagnosis_suggestions = self._generate_diagnosis_suggestions(detected_symptoms)
        
        # Calculate confidence based on number of symptoms detected
        confidence = min(0.9, len(detected_symptoms) * 0.2 + 0.3)
        
        return AnalysisResult(
            analysis_type=AnalysisType.DISEASE_SYMPTOMS,
            confidence=confidence,
            primary_findings=f"Detected symptoms: {', '.join(detected_symptoms)}",
            recommendations=diagnosis_suggestions,
            raw_data={
                "detected_symptoms": detected_symptoms,
                "symptom_count": len(detected_symptoms),
                "input_text": input_data
            },
            metadata={
                "tool_name": self.name,
                "analysis_timestamp": "placeholder_timestamp",
                "processing_method": "pattern_matching"
            }
        )
    
    def _extract_symptoms(self, text: str) -> list[str]:
        """Extract symptoms from input text using pattern matching."""
        symptoms = []
        text_lower = text.lower()
        
        symptom_mappings = {
            r'\b(?:cough|coughing)\b': 'cough',
            r'\b(?:fever|temperature|hot)\b': 'fever',
            r'\b(?:runny nose|nasal congestion|stuffy nose)\b': 'runny nose',
            r'\b(?:headache|head pain)\b': 'headache',
            r'\b(?:sore throat|throat pain)\b': 'sore throat',
            r'\b(?:fatigue|tired|exhausted)\b': 'fatigue',
            r'\b(?:nausea|vomiting|sick)\b': 'nausea',
            r'\b(?:diarrhea|stomach pain|abdominal pain)\b': 'digestive issues',
            r'\b(?:rash|skin irritation)\b': 'skin rash',
            r'\b(?:shortness of breath|breathing difficulty)\b': 'breathing difficulty'
        }
        
        for pattern, symptom_name in symptom_mappings.items():
            if re.search(pattern, text_lower):
                symptoms.append(symptom_name)
        
        return list(set(symptoms))  # Remove duplicates
    
    def _generate_diagnosis_suggestions(self, symptoms: list[str]) -> list[str]:
        """Generate diagnosis suggestions based on symptoms (placeholder)."""
        if not symptoms:
            return ["Please provide more specific symptoms for accurate analysis."]
        
        # Placeholder diagnosis logic
        common_diagnoses = [
            "Common cold - Consider rest and hydration",
            "Viral infection - Monitor symptoms and seek medical attention if worsening",
            "Seasonal allergies - Consider antihistamines if appropriate",
            "Stress-related symptoms - Consider stress management techniques"
        ]
        
        # Simple logic based on symptom combinations
        if 'fever' in symptoms and 'cough' in symptoms:
            return ["Possible respiratory infection - Consult healthcare provider"]
        elif 'runny nose' in symptoms and 'headache' in symptoms:
            return ["Possible cold or allergies - Monitor symptoms"]
        else:
            return random.sample(common_diagnoses, min(2, len(common_diagnoses)))
    
    def _run(self, symptoms_description: str) -> str:
        """CrewAI tool interface method."""
        try:
            result = self.analyze(symptoms_description)
            return f"Analysis: {result.primary_findings}. Recommendations: {'; '.join(result.recommendations)}"
        except Exception as e:
            return f"Error in disease analysis: {str(e)}"


class BreastCancerAnalysisInput(BaseModel):
    """Input schema for breast cancer analysis tool."""
    tumor_data: str = Field(..., description="Tumor measurements and characteristics data")


class BreastCancerAnalysisTool(BaseTool):
    """
    Tool for analyzing breast cancer patient data and tumor characteristics.
    
    This tool can process tumor measurements and patient data to provide
    analysis insights for breast cancer cases.
    """
    
    name: str = "Breast Cancer Analysis Tool"
    description: str = (
        "Analyzes breast cancer patient data including tumor measurements, "
        "cell characteristics, and provides risk assessment. Use this tool "
        "when users provide tumor measurements, breast cancer data, or medical imaging results."
    )
    args_schema: Type[BaseModel] = BreastCancerAnalysisInput
    
    # Additional attributes for the tool
    cancer_keywords: list = []
    
    def __init__(self):
        super().__init__()
        
        # Keywords that indicate breast cancer analysis
        self.cancer_keywords = [
            'tumor', 'mass', 'lump', 'breast', 'cancer', 'oncology',
            'malignant', 'benign', 'biopsy', 'mammogram', 'ultrasound',
            'radius', 'perimeter', 'area', 'smoothness', 'compactness',
            'concavity', 'symmetry', 'fractal', 'texture'
        ]
    
    def can_handle(self, input_data: str) -> bool:
        """
        Determine if input contains breast cancer-related content.
        
        Args:
            input_data: Raw input string from user
            
        Returns:
            bool: True if input is breast cancer related, False otherwise
        """
        input_lower = input_data.lower()
        
        # Check for breast cancer specific keywords
        has_cancer_keywords = any(keyword in input_lower for keyword in self.cancer_keywords)
        
        # Check for numeric measurements (common in breast cancer data)
        has_measurements = bool(re.search(r'\d+\.?\d*\s*(?:mm|cm|units?)', input_lower))
        
        return has_cancer_keywords or has_measurements
    
    def analyze(self, input_data: str) -> AnalysisResult:
        """
        Analyze breast cancer data and provide insights.
        
        Args:
            input_data: Raw input string from user
            
        Returns:
            AnalysisResult: Analysis result with cancer insights
        """
        if not input_data or len(input_data.strip()) == 0:
            raise ValueError("Invalid input data provided")
        
        # Extract measurements and characteristics
        measurements = self._extract_measurements(input_data)
        characteristics = self._extract_characteristics(input_data)
        
        # Generate analysis insights (placeholder implementation)
        insights = self._generate_cancer_insights(measurements, characteristics)
        
        # Calculate confidence based on amount of data provided
        data_completeness = len(measurements) + len(characteristics)
        confidence = min(0.95, data_completeness * 0.1 + 0.4)
        
        return AnalysisResult(
            analysis_type=AnalysisType.BREAST_CANCER,
            confidence=confidence,
            primary_findings=f"Analyzed {len(measurements)} measurements and {len(characteristics)} characteristics",
            recommendations=insights,
            raw_data={
                "measurements": measurements,
                "characteristics": characteristics,
                "input_text": input_data
            },
            metadata={
                "tool_name": self.name,
                "analysis_timestamp": "placeholder_timestamp",
                "processing_method": "feature_extraction"
            }
        )
    
    def _extract_measurements(self, text: str) -> Dict[str, float]:
        """Extract numeric measurements from text."""
        measurements = {}
        text_lower = text.lower()
        
        # Common measurement patterns
        measurement_patterns = {
            'radius': r'radius[:\s]*(\d+\.?\d*)',
            'perimeter': r'perimeter[:\s]*(\d+\.?\d*)',
            'area': r'area[:\s]*(\d+\.?\d*)',
            'smoothness': r'smoothness[:\s]*(\d+\.?\d*)',
            'compactness': r'compactness[:\s]*(\d+\.?\d*)',
            'concavity': r'concavity[:\s]*(\d+\.?\d*)',
            'symmetry': r'symmetry[:\s]*(\d+\.?\d*)',
        }
        
        for measurement, pattern in measurement_patterns.items():
            match = re.search(pattern, text_lower)
            if match:
                try:
                    measurements[measurement] = float(match.group(1))
                except ValueError:
                    continue
        
        return measurements
    
    def _extract_characteristics(self, text: str) -> list[str]:
        """Extract tumor characteristics from text."""
        characteristics = []
        text_lower = text.lower()
        
        characteristic_keywords = [
            'malignant', 'benign', 'irregular', 'smooth', 'rough',
            'calcifications', 'microcalcifications', 'dense', 'solid',
            'cystic', 'heterogeneous', 'homogeneous'
        ]
        
        for characteristic in characteristic_keywords:
            if characteristic in text_lower:
                characteristics.append(characteristic)
        
        return list(set(characteristics))
    
    def _generate_cancer_insights(self, measurements: Dict[str, float], characteristics: list[str]) -> list[str]:
        """Generate cancer analysis insights (placeholder)."""
        insights = []
        
        if not measurements and not characteristics:
            return ["Please provide tumor measurements or characteristics for detailed analysis."]
        
        # Placeholder analysis logic
        if measurements:
            insights.append(f"Received {len(measurements)} quantitative measurements for analysis")
            
            if 'radius' in measurements:
                radius = measurements['radius']
                if radius > 15:
                    insights.append("Large tumor size detected - requires immediate attention")
                elif radius > 10:
                    insights.append("Moderate tumor size - monitor closely")
                else:
                    insights.append("Small tumor size detected")
        
        if characteristics:
            insights.append(f"Identified {len(characteristics)} tumor characteristics")
            
            if 'malignant' in characteristics:
                insights.append("Malignant characteristics detected - urgent oncology consultation recommended")
            elif 'benign' in characteristics:
                insights.append("Benign characteristics noted - continue monitoring")
        
        if not insights:
            insights = [
                "Analysis requires more specific tumor data",
                "Recommend comprehensive imaging studies",
                "Consider multidisciplinary team consultation"
            ]
        
        return insights
    
    def _run(self, tumor_data: str) -> str:
        """CrewAI tool interface method."""
        try:
            result = self.analyze(tumor_data)
            return f"Cancer Analysis: {result.primary_findings}. Insights: {'; '.join(result.recommendations)}"
        except Exception as e:
            return f"Error in breast cancer analysis: {str(e)}"


def get_available_tools() -> list:
    """
    Factory function to get all available analysis tools.
    
    Returns:
        list: List of instantiated analysis tools
    """
    return [
        DiseaseAnalysisTool(),
        BreastCancerAnalysisTool()
    ]
