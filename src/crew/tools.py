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
from .symptom_extractor import create_symptom_extractor, SymptomExtractionTool
from .breast_cancer_extractor import create_breast_cancer_feature_extractor, BreastCancerFeatureExtractionTool


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
        
        # Initialize the AI-powered symptom extractor
        try:
            self._symptom_extractor = create_symptom_extractor()
            print("AI-powered symptom extractor initialized in DiseaseAnalysisTool")
        except Exception as e:
            print(f"Warning: Could not initialize symptom extractor: {e}")
            self._symptom_extractor = None
        
        # Get dynamic symptom patterns from the extractor
        if self._symptom_extractor:
            self._symptom_categories = self._symptom_extractor.get_symptom_categories()
        else:
            # Fallback symptom categories
            self._symptom_categories = {
                'respiratory': ['cough', 'sore throat', 'runny nose', 'shortness of breath'],
                'systemic': ['fever', 'fatigue', 'chills'],
                'neurological': ['headache', 'dizziness'],
                'gastrointestinal': ['nausea', 'diarrhea', 'abdominal pain']
            }
        
        # Create patterns from all known symptoms for fallback detection
        self.symptom_patterns = []
        for category, symptoms in self._symptom_categories.items():
            for symptom in symptoms:
                # Create regex pattern for the symptom
                pattern = r'\b(?:' + re.escape(symptom) + r')\b'
                self.symptom_patterns.append(pattern)
    
    @property
    def symptom_extractor(self):
        """Get the symptom extractor."""
        return getattr(self, '_symptom_extractor', None)
    
    @property
    def symptom_categories(self):
        """Get the symptom categories."""
        return getattr(self, '_symptom_categories', {})
    
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
        
        # Extract symptoms from input using AI-powered extraction
        detected_symptoms = self._extract_symptoms_with_ai(input_data)
        
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
    
    def _extract_symptoms_with_ai(self, text: str) -> list[str]:
        """Extract symptoms from input text using AI-powered extraction."""
        try:
            # Use the AI-powered symptom extractor if available
            if self.symptom_extractor:
                ai_symptoms = self.symptom_extractor.extract_symptoms(text)
                
                # If AI extraction returns symptoms, use them
                if ai_symptoms:
                    return ai_symptoms
            
            # Fallback to rule-based extraction if AI fails or unavailable
            return self._extract_symptoms_fallback(text)
            
        except Exception as e:
            print(f"AI symptom extraction failed: {e}")
            # Fallback to rule-based extraction
            return self._extract_symptoms_fallback(text)
    
    def _extract_symptoms_fallback(self, text: str) -> list[str]:
        """Fallback method for symptom extraction using rule-based patterns."""
        symptoms = []
        text_lower = text.lower()
        
        # Use the dynamic symptom categories for fallback
        symptom_categories = getattr(self, '_symptom_categories', {})
        for category, symptom_list in symptom_categories.items():
            for symptom in symptom_list:
                pattern = r'\b' + re.escape(symptom) + r'\b'
                if re.search(pattern, text_lower):
                    symptoms.append(symptom)
        
        return list(set(symptoms))  # Remove duplicates
    
    def _extract_symptoms(self, text: str) -> list[str]:
        """Legacy method - now redirects to AI-powered extraction."""
        return self._extract_symptoms_with_ai(text)
    
    def _generate_diagnosis_suggestions(self, symptoms: list[str]) -> list[str]:
        """Generate diagnosis suggestions based on symptoms with enhanced logic."""
        if not symptoms:
            return ["Please provide more specific symptoms for accurate analysis."]
        
        # Enhanced diagnosis logic based on symptom combinations
        suggestions = []
        
        # Respiratory symptoms
        respiratory_symptoms = {'cough', 'sore throat', 'runny nose', 'shortness of breath', 'wheezing', 'chest pain'}
        has_respiratory = any(symptom in respiratory_symptoms for symptom in symptoms)
        
        # Systemic symptoms
        systemic_symptoms = {'fever', 'fatigue', 'chills', 'sweating'}
        has_systemic = any(symptom in systemic_symptoms for symptom in symptoms)
        
        # Gastrointestinal symptoms
        gi_symptoms = {'nausea', 'vomiting', 'diarrhea', 'abdominal pain', 'stomach pain'}
        has_gi = any(symptom in gi_symptoms for symptom in symptoms)
        
        # Neurological symptoms
        neuro_symptoms = {'headache', 'dizziness', 'confusion'}
        has_neuro = any(symptom in neuro_symptoms for symptom in symptoms)
        
        # Generate specific suggestions based on symptom patterns
        if 'fever' in symptoms and 'cough' in symptoms:
            suggestions.append("Possible respiratory infection - Consult healthcare provider for proper evaluation")
        
        if has_respiratory and has_systemic:
            suggestions.append("Upper respiratory tract infection - Consider rest, fluids, and medical consultation")
        
        if 'runny nose' in symptoms and 'headache' in symptoms:
            suggestions.append("Possible cold or seasonal allergies - Monitor symptoms and consider antihistamines if appropriate")
        
        if has_gi and 'fever' in symptoms:
            suggestions.append("Possible gastrointestinal infection - Stay hydrated and seek medical attention if symptoms persist")
        
        if has_neuro and 'fever' in symptoms:
            suggestions.append("Neurological symptoms with fever require immediate medical evaluation")
        
        if 'shortness of breath' in symptoms or 'chest pain' in symptoms:
            suggestions.append("Respiratory or cardiac symptoms detected - Seek immediate medical attention")
        
        # Default suggestions if no specific patterns match
        if not suggestions:
            general_suggestions = [
                "Monitor symptoms and consult healthcare provider if they persist or worsen",
                "Consider rest, adequate hydration, and over-the-counter symptom relief as appropriate",
                "Seek medical attention if symptoms are severe or concerning",
                "Track symptom progression and report changes to healthcare provider"
            ]
            suggestions.extend(random.sample(general_suggestions, min(2, len(general_suggestions))))
        
        # Add general advice
        suggestions.append("This analysis is preliminary - professional medical evaluation is recommended")
        
        return suggestions
    
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
        
        # Initialize the AI-powered breast cancer feature extractor
        try:
            self._feature_extractor = create_breast_cancer_feature_extractor()
            print("AI-powered breast cancer feature extractor initialized in BreastCancerAnalysisTool")
        except Exception as e:
            print(f"Warning: Could not initialize feature extractor: {e}")
            self._feature_extractor = None
        
        # Get dynamic feature categories from the extractor
        if self._feature_extractor:
            self._measurement_categories = self._feature_extractor.get_measurement_categories()
            self._characteristic_categories = self._feature_extractor.get_characteristic_categories()
        else:
            # Fallback categories
            self._measurement_categories = {
                'size_measurements': ['radius', 'perimeter', 'area'],
                'texture_measurements': ['smoothness', 'compactness', 'concavity', 'symmetry']
            }
            self._characteristic_categories = {
                'malignancy': ['malignant', 'benign'],
                'texture_characteristics': ['smooth', 'rough', 'irregular']
            }
        
        # Keywords that indicate breast cancer analysis (for backward compatibility)
        self._cancer_keywords = [
            'tumor', 'mass', 'lump', 'breast', 'cancer', 'oncology',
            'malignant', 'benign', 'biopsy', 'mammogram', 'ultrasound',
            'radius', 'perimeter', 'area', 'smoothness', 'compactness',
            'concavity', 'symmetry', 'fractal', 'texture'
        ]
    
    @property
    def feature_extractor(self):
        """Get the feature extractor."""
        return getattr(self, '_feature_extractor', None)
    
    @property
    def cancer_keywords(self):
        """Get the cancer keywords."""
        return getattr(self, '_cancer_keywords', [])
    
    @property
    def measurement_categories(self):
        """Get the measurement categories."""
        return getattr(self, '_measurement_categories', {})
    
    @property
    def characteristic_categories(self):
        """Get the characteristic categories."""
        return getattr(self, '_characteristic_categories', {})
    
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
        cancer_keywords = getattr(self, '_cancer_keywords', [])
        has_cancer_keywords = any(keyword in input_lower for keyword in cancer_keywords)
        
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
        
        # Extract measurements and characteristics using AI-powered extraction
        measurements, characteristics = self._extract_features_with_ai(input_data)
        
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
    
    def _extract_features_with_ai(self, text: str) -> tuple[Dict[str, float], list[str]]:
        """Extract breast cancer features from input text using AI-powered extraction."""
        try:
            # Use the AI-powered feature extractor if available
            if self.feature_extractor:
                ai_measurements, ai_characteristics = self.feature_extractor.extract_features(text)
                
                # If AI extraction returns features, use them
                if ai_measurements or ai_characteristics:
                    return ai_measurements, ai_characteristics
            
            # Fallback to rule-based extraction if AI fails or unavailable
            return self._extract_features_fallback(text)
            
        except Exception as e:
            print(f"AI feature extraction failed: {e}")
            # Fallback to rule-based extraction
            return self._extract_features_fallback(text)
    
    def _extract_features_fallback(self, text: str) -> tuple[Dict[str, float], list[str]]:
        """Fallback method for feature extraction using rule-based patterns."""
        measurements = {}
        characteristics = []
        text_lower = text.lower()
        
        # Extract measurements using patterns
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
        
        # Extract characteristics using all categories
        characteristic_categories = getattr(self, '_characteristic_categories', {})
        for category, char_list in characteristic_categories.items():
            for characteristic in char_list:
                pattern = r'\b' + re.escape(characteristic) + r'\b'
                if re.search(pattern, text_lower):
                    characteristics.append(characteristic)
        
        return measurements, list(set(characteristics))
    
    def _extract_measurements(self, text: str) -> Dict[str, float]:
        """Legacy method - now redirects to AI-powered extraction."""
        measurements, _ = self._extract_features_with_ai(text)
        return measurements
    
    def _extract_characteristics(self, text: str) -> list[str]:
        """Legacy method - now redirects to AI-powered extraction."""
        _, characteristics = self._extract_features_with_ai(text)
        return characteristics
    
    def _generate_cancer_insights(self, measurements: Dict[str, float], characteristics: list[str]) -> list[str]:
        """Generate cancer analysis insights with enhanced AI-powered logic."""
        insights = []
        
        if not measurements and not characteristics:
            return ["Please provide tumor measurements or characteristics for detailed analysis."]
        
        # Enhanced analysis logic based on measurements
        if measurements:
            insights.append(f"Received {len(measurements)} quantitative measurements for analysis")
            
            # Analyze radius/size measurements
            if 'radius' in measurements:
                radius = measurements['radius']
                if radius > 15:
                    insights.append("Large tumor size detected - requires immediate attention")
                elif radius > 10:
                    insights.append("Moderate tumor size - monitor closely")
                else:
                    insights.append("Small tumor size detected")
            
            # Analyze texture measurements
            texture_measurements = ['smoothness', 'compactness', 'concavity', 'symmetry']
            texture_values = {k: v for k, v in measurements.items() if k in texture_measurements}
            
            if texture_values:
                insights.append(f"Texture analysis includes {len(texture_values)} parameters")
                
                # High compactness or concavity might indicate malignancy
                if 'compactness' in texture_values and texture_values['compactness'] > 0.15:
                    insights.append("High compactness value detected - may indicate irregular tumor shape")
                
                if 'concavity' in texture_values and texture_values['concavity'] > 0.1:
                    insights.append("High concavity value detected - possible malignant characteristics")
            
            # Analyze area measurements
            if 'area' in measurements:
                area = measurements['area']
                if area > 1000:
                    insights.append("Large tumor area detected - comprehensive evaluation recommended")
                elif area < 200:
                    insights.append("Small tumor area - may be early stage or benign")
        
        # Enhanced analysis logic based on characteristics
        if characteristics:
            insights.append(f"Identified {len(characteristics)} tumor characteristics")
            
            # Malignancy assessment
            if 'malignant' in characteristics:
                insights.append("Malignant characteristics detected - urgent oncology consultation recommended")
            elif 'benign' in characteristics:
                insights.append("Benign characteristics noted - continue monitoring")
            
            # Texture characteristics
            texture_chars = ['smooth', 'rough', 'irregular', 'uniform']
            found_texture = [c for c in characteristics if c in texture_chars]
            if found_texture:
                if 'irregular' in found_texture or 'rough' in found_texture:
                    insights.append("Irregular texture patterns may suggest closer evaluation")
                elif 'smooth' in found_texture or 'uniform' in found_texture:
                    insights.append("Smooth texture patterns are often associated with benign lesions")
            
            # Density characteristics
            if 'dense' in characteristics:
                insights.append("Dense tissue characteristics noted - may affect imaging sensitivity")
            elif 'cystic' in characteristics:
                insights.append("Cystic characteristics suggest fluid-filled lesion - often benign")
            
            # Border characteristics
            border_chars = ['circumscribed', 'irregular', 'spiculated', 'lobulated']
            found_borders = [c for c in characteristics if c in border_chars]
            if found_borders:
                if 'spiculated' in found_borders or 'irregular' in found_borders:
                    insights.append("Irregular border characteristics require immediate evaluation")
                elif 'circumscribed' in found_borders:
                    insights.append("Well-circumscribed borders often indicate benign lesions")
        
        # Risk assessment based on combination of factors
        if measurements and characteristics:
            risk_factors = 0
            
            # Check for high-risk measurements
            if 'radius' in measurements and measurements['radius'] > 12:
                risk_factors += 1
            if 'compactness' in measurements and measurements['compactness'] > 0.12:
                risk_factors += 1
            if 'concavity' in measurements and measurements['concavity'] > 0.08:
                risk_factors += 1
            
            # Check for high-risk characteristics
            high_risk_chars = ['malignant', 'irregular', 'spiculated', 'heterogeneous']
            if any(char in characteristics for char in high_risk_chars):
                risk_factors += 2
            
            if risk_factors >= 3:
                insights.append("Multiple concerning features identified - immediate comprehensive evaluation recommended")
            elif risk_factors >= 1:
                insights.append("Some concerning features present - follow-up evaluation advised")
        
        # Default suggestions if no specific insights generated
        if not insights:
            insights = [
                "Analysis requires more specific tumor data for detailed assessment",
                "Recommend comprehensive imaging studies with multiple modalities",
                "Consider multidisciplinary team consultation for optimal care planning",
                "Follow institutional guidelines for breast lesion evaluation"
            ]
        
        # Add general medical advice
        insights.append("All findings require professional radiological and clinical correlation")
        insights.append("This AI analysis supports but does not replace expert medical evaluation")
        
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
        BreastCancerAnalysisTool(),
        SymptomExtractionTool(),
        BreastCancerFeatureExtractionTool()
    ]
