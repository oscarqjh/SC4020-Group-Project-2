"""
AI-powered breast cancer feature extraction agent for dynamic feature identification.
This module implements a CrewAI agent that can intelligently extract breast cancer features from natural language.
"""

import re
from typing import List, Dict, Any, Type
from crewai import Agent, Task, Crew
from pydantic import BaseModel, Field
from crewai.tools.base_tool import BaseTool


class BreastCancerFeatureExtractionInput(BaseModel):
    """Input schema for breast cancer feature extraction."""
    text: str = Field(..., description="Text containing potential breast cancer measurements and characteristics")


class BreastCancerFeatureExtractionTool(BaseTool):
    """
    Tool for extracting breast cancer features from natural language text using AI analysis.
    """
    
    name: str = "Breast Cancer Feature Extraction Tool"
    description: str = (
        "Extracts and identifies breast cancer measurements and tumor characteristics from natural language text. "
        "Can identify both explicit measurements and implied tumor characteristics."
    )
    args_schema: Type[BaseModel] = BreastCancerFeatureExtractionInput
    
    def __init__(self):
        super().__init__()
        
        # Known measurement categories for validation and enhancement
        self._measurement_categories = {
            'size_measurements': [
                'radius', 'diameter', 'perimeter', 'area', 'volume',
                'length', 'width', 'height', 'circumference'
            ],
            'texture_measurements': [
                'smoothness', 'roughness', 'texture', 'compactness',
                'concavity', 'concave points', 'symmetry', 'fractal dimension'
            ],
            'shape_measurements': [
                'compactness', 'concavity', 'symmetry', 'fractal',
                'irregularity', 'roundness', 'elongation'
            ],
            'intensity_measurements': [
                'mean', 'standard error', 'worst', 'average',
                'minimum', 'maximum', 'range'
            ]
        }
        
        # Known tumor characteristics
        self._characteristic_categories = {
            'malignancy': [
                'malignant', 'benign', 'cancerous', 'non-cancerous',
                'tumor', 'mass', 'lesion', 'growth'
            ],
            'texture_characteristics': [
                'smooth', 'rough', 'irregular', 'uniform', 'heterogeneous',
                'homogeneous', 'coarse', 'fine', 'grainy'
            ],
            'density_characteristics': [
                'dense', 'solid', 'cystic', 'fluid-filled',
                'calcified', 'soft tissue', 'fibrous'
            ],
            'border_characteristics': [
                'well-defined', 'ill-defined', 'circumscribed',
                'infiltrating', 'spiculated', 'lobulated'
            ],
            'imaging_characteristics': [
                'hypoechoic', 'hyperechoic', 'isoechoic',
                'radiolucent', 'radiopaque', 'enhancement'
            ]
        }
        
        # Initialize the feature extraction agent
        self._init_extraction_agent()
    
    def _init_extraction_agent(self):
        """Initialize the breast cancer feature extraction agent."""
        try:
            # Create the feature extraction agent
            self._extraction_agent = Agent(
                role="Breast Cancer Medical Feature Extraction Specialist",
                goal="Extract and identify all breast cancer measurements and tumor characteristics from medical text with high accuracy",
                backstory=(
                    "You are a specialized medical imaging and oncology expert with deep knowledge of breast cancer "
                    "terminology, tumor measurements, and imaging characteristics. You understand medical imaging reports, "
                    "pathology descriptions, and can identify both quantitative measurements and qualitative characteristics "
                    "of breast tumors. You are familiar with mammography, ultrasound, MRI, and biopsy terminology."
                ),
                verbose=False,
                allow_delegation=False
            )
        except Exception as e:
            print(f"Warning: Could not initialize AI agent: {e}")
            self._extraction_agent = None
    
    @property
    def measurement_categories(self) -> Dict[str, List[str]]:
        """Get the measurement categories."""
        return self._measurement_categories
    
    @property
    def characteristic_categories(self) -> Dict[str, List[str]]:
        """Get the characteristic categories."""
        return self._characteristic_categories
    
    @property
    def extraction_agent(self):
        """Get the extraction agent."""
        return getattr(self, '_extraction_agent', None)
    
    def _run(self, text: str) -> str:
        """CrewAI tool interface method."""
        try:
            measurements, characteristics = self.extract_features_with_ai(text)
            return f"Extracted measurements: {measurements}, characteristics: {characteristics}"
        except Exception as e:
            return f"Error extracting features: {str(e)}"
    
    def extract_features_with_ai(self, text: str) -> tuple[Dict[str, float], List[str]]:
        """
        Extract breast cancer features using AI analysis combined with rule-based validation.
        
        Args:
            text: Input text containing feature descriptions
            
        Returns:
            tuple: (measurements_dict, characteristics_list)
        """
        if not text or len(text.strip()) == 0:
            return {}, []
        
        # First pass: AI-based extraction
        ai_measurements, ai_characteristics = self._ai_extract_features(text)
        
        # Second pass: Rule-based validation and enhancement
        rule_measurements, rule_characteristics = self._rule_based_extraction(text)
        
        # Combine and standardize
        combined_measurements = {**rule_measurements, **ai_measurements}
        combined_characteristics = list(set(ai_characteristics + rule_characteristics))
        
        # Filter and standardize
        standardized_measurements = self._standardize_measurements(combined_measurements)
        standardized_characteristics = self._standardize_characteristics(combined_characteristics)
        
        return standardized_measurements, standardized_characteristics
    
    def _ai_extract_features(self, text: str) -> tuple[Dict[str, float], List[str]]:
        """
        Use AI agent to extract features from text.
        
        Args:
            text: Input text
            
        Returns:
            tuple: (measurements_dict, characteristics_list)
        """
        # Check if AI agent is available
        if not self.extraction_agent:
            print("AI agent not available, using fallback extraction")
            return {}, []
        
        try:
            # Create extraction task for measurements
            measurement_task = Task(
                description=(
                    f"Analyze the following medical text and extract all quantitative measurements related to breast cancer/tumors. "
                    f"Look for measurements like radius, perimeter, area, smoothness, compactness, concavity, symmetry, etc. "
                    f"Return ONLY a JSON-like format: measurement_name:value, measurement_name:value\n\n"
                    f"Text to analyze: {text}\n\n"
                    f"Format your response as: radius:12.5, perimeter:85.2, area:480.1"
                ),
                agent=self.extraction_agent,
                expected_output="Comma-separated list of measurement_name:value pairs"
            )
            
            # Create extraction task for characteristics
            characteristic_task = Task(
                description=(
                    f"Analyze the following medical text and extract all qualitative characteristics related to breast cancer/tumors. "
                    f"Look for characteristics like malignant, benign, irregular, smooth, dense, calcified, etc. "
                    f"Return ONLY a comma-separated list of characteristic names, no explanations.\n\n"
                    f"Text to analyze: {text}\n\n"
                    f"Format your response as: malignant, irregular, dense"
                ),
                agent=self.extraction_agent,
                expected_output="A comma-separated list of tumor characteristics"
            )
            
            # Create temporary crew for execution
            crew = Crew(
                agents=[self.extraction_agent],
                tasks=[measurement_task, characteristic_task],
                verbose=False
            )
            
            # Execute tasks
            results = crew.kickoff()
            
            # Parse results
            measurements = {}
            characteristics = []
            
            if isinstance(results, list) and len(results) >= 2:
                # Parse measurements
                measurement_result = str(results[0]) if results[0] else ""
                measurements = self._parse_measurements(measurement_result)
                
                # Parse characteristics
                characteristic_result = str(results[1]) if results[1] else ""
                characteristics = self._parse_characteristics(characteristic_result)
            
            return measurements, characteristics
            
        except Exception as e:
            print(f"AI feature extraction error: {e}")
            return {}, []
    
    def _parse_measurements(self, result: str) -> Dict[str, float]:
        """Parse measurement results from AI response."""
        measurements = {}
        
        if not result:
            return measurements
        
        # Look for patterns like "radius:12.5" or "radius: 12.5"
        measurement_pattern = r'(\w+):\s*(\d+\.?\d*)'
        matches = re.findall(measurement_pattern, result.lower())
        
        for name, value in matches:
            try:
                measurements[name] = float(value)
            except ValueError:
                continue
        
        return measurements
    
    def _parse_characteristics(self, result: str) -> List[str]:
        """Parse characteristic results from AI response."""
        if not result:
            return []
        
        # Clean and split the result
        characteristics = [c.strip().lower() for c in result.split(',') if c.strip()]
        
        # Filter out non-characteristic text
        filtered_characteristics = [c for c in characteristics if self._is_likely_characteristic(c)]
        
        return filtered_characteristics
    
    def _rule_based_extraction(self, text: str) -> tuple[Dict[str, float], List[str]]:
        """
        Extract features using rule-based pattern matching.
        
        Args:
            text: Input text
            
        Returns:
            tuple: (measurements_dict, characteristics_list)
        """
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
            'fractal': r'fractal[:\s]*(\d+\.?\d*)',
            'diameter': r'diameter[:\s]*(\d+\.?\d*)',
            'size': r'size[:\s]*(\d+\.?\d*)'
        }
        
        for measurement, pattern in measurement_patterns.items():
            match = re.search(pattern, text_lower)
            if match:
                try:
                    measurements[measurement] = float(match.group(1))
                except ValueError:
                    continue
        
        # Extract characteristics using all categories
        for category, char_list in self._characteristic_categories.items():
            for characteristic in char_list:
                pattern = r'\b' + re.escape(characteristic) + r'\b'
                if re.search(pattern, text_lower):
                    characteristics.append(characteristic)
        
        return measurements, list(set(characteristics))
    
    def _is_likely_characteristic(self, text: str) -> bool:
        """
        Check if a text string is likely to be a breast cancer characteristic.
        
        Args:
            text: Text to check
            
        Returns:
            bool: True if likely a characteristic
        """
        text_lower = text.lower().strip()
        
        # Remove common non-characteristic words
        non_characteristic_indicators = [
            'patient', 'doctor', 'hospital', 'measurement', 'analysis',
            'report', 'study', 'examination', 'imaging', 'scan',
            'and', 'or', 'the', 'a', 'an', 'is', 'are', 'was', 'were',
            'with', 'without', 'has', 'have', 'had', 'may', 'might', 'could'
        ]
        
        if text_lower in non_characteristic_indicators:
            return False
        
        # Check if it matches known characteristics
        all_characteristics = []
        for char_list in self._characteristic_categories.values():
            all_characteristics.extend(char_list)
        
        # Direct match
        if text_lower in all_characteristics:
            return True
        
        # Partial match with known characteristics
        for characteristic in all_characteristics:
            if characteristic in text_lower or text_lower in characteristic:
                return True
        
        # Length check (characteristics are usually 1-3 words)
        word_count = len(text_lower.split())
        if word_count > 3:
            return False
        
        # Check for medical-sounding words
        medical_patterns = [
            r'ous$', r'ant$', r'ed$', r'ic$', r'al$',
            r'dense', r'solid', r'fluid', r'tissue'
        ]
        
        for pattern in medical_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def _standardize_measurements(self, measurements: Dict[str, float]) -> Dict[str, float]:
        """
        Standardize measurement names to consistent format.
        
        Args:
            measurements: Raw measurements dict
            
        Returns:
            Dict[str, float]: Standardized measurements
        """
        standardized = {}
        
        # Mapping for common variations
        standardization_map = {
            'diameter': 'radius',  # Convert diameter to radius (divide by 2)
            'size': 'radius',
            'circumference': 'perimeter',
            'fractal': 'fractal_dimension'
        }
        
        for measurement, value in measurements.items():
            measurement_lower = measurement.lower().strip()
            
            # Check for standardization mapping
            if measurement_lower in standardization_map:
                standardized_name = standardization_map[measurement_lower]
                # Special case: convert diameter to radius
                if measurement_lower == 'diameter':
                    value = value / 2.0
            else:
                standardized_name = measurement_lower
            
            standardized[standardized_name] = value
        
        return standardized
    
    def _standardize_characteristics(self, characteristics: List[str]) -> List[str]:
        """
        Standardize characteristic names to consistent format.
        
        Args:
            characteristics: Raw characteristic list
            
        Returns:
            List[str]: Standardized characteristics
        """
        standardized = []
        
        # Mapping for common variations
        standardization_map = {
            'cancerous': 'malignant',
            'non-cancerous': 'benign',
            'tumor': 'mass',
            'lesion': 'mass',
            'growth': 'mass',
            'coarse': 'rough',
            'fine': 'smooth',
            'fluid-filled': 'cystic',
            'well-defined': 'circumscribed',
            'ill-defined': 'irregular'
        }
        
        for characteristic in characteristics:
            characteristic_lower = characteristic.lower().strip()
            
            # Check for standardization mapping
            if characteristic_lower in standardization_map:
                standardized_characteristic = standardization_map[characteristic_lower]
            else:
                standardized_characteristic = characteristic_lower
            
            # Only add if not already present
            if standardized_characteristic not in standardized:
                standardized.append(standardized_characteristic)
        
        return standardized


class BreastCancerFeatureExtractor:
    """
    Main breast cancer feature extraction class that coordinates AI and rule-based extraction.
    """
    
    def __init__(self):
        """Initialize the feature extractor with AI capabilities."""
        self.extraction_tool = BreastCancerFeatureExtractionTool()
    
    def extract_features(self, text: str) -> tuple[Dict[str, float], List[str]]:
        """
        Extract features from text using combined AI and rule-based approach.
        
        Args:
            text: Input text containing feature descriptions
            
        Returns:
            tuple: (measurements_dict, characteristics_list)
        """
        return self.extraction_tool.extract_features_with_ai(text)
    
    def get_measurement_categories(self) -> Dict[str, List[str]]:
        """
        Get the known measurement categories.
        
        Returns:
            Dict[str, List[str]]: Measurement categories mapping
        """
        return self.extraction_tool.measurement_categories.copy()
    
    def get_characteristic_categories(self) -> Dict[str, List[str]]:
        """
        Get the known characteristic categories.
        
        Returns:
            Dict[str, List[str]]: Characteristic categories mapping
        """
        return self.extraction_tool.characteristic_categories.copy()


# Factory function
def create_breast_cancer_feature_extractor() -> BreastCancerFeatureExtractor:
    """
    Create a breast cancer feature extractor instance.
    
    Returns:
        BreastCancerFeatureExtractor: Configured feature extractor
    """
    return BreastCancerFeatureExtractor()
