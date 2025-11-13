"""
Medical analysis tools for the crew system.
This module implements specific analysis tools for disease symptoms and breast cancer analysis.
"""

import csv
import math
import pickle
import re
import random
from pathlib import Path
from typing import Any, Dict, Optional, Type, List
import pandas as pd
from pydantic import BaseModel, Field
from crewai.tools.base_tool import BaseTool
from crewai import LLM
from os import getenv
from processors.symptom_data_processor import SymptomDataProcessor

from .base import AnalysisResult, AnalysisType
from .symptom_extractor import create_symptom_extractor, SymptomExtractionTool
from .breast_cancer_extractor import create_breast_cancer_feature_extractor, BreastCancerFeatureExtractionTool
from .disease_model import DiseasePredictionModel
from .symptom_suggester import SymptomSuggester


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
    MIN_SYMPTOMS_FOR_CONFIDENCE: int = 5
    
    def __init__(self):
        super().__init__()
        self._symptom_extractor = self._initialize_symptom_extractor()
        self._symptom_categories = self._load_symptom_categories()
        self.symptom_patterns = self._compile_symptom_patterns(self._symptom_categories)
        self._symptom_processor = SymptomDataProcessor(data_path="")
        self._precaution_mapping = self._load_precaution_mapping()
        self._llm = self._initialize_llm()
        self._disease_model = DiseasePredictionModel()
        self._symptom_suggester = SymptomSuggester()
        self._pending_symptoms: list[str] = []
        self._awaiting_additional_input: bool = False
        self._last_prediction_context: Dict[str, Any] = {
            "predictions": [],
            "recognized_symptoms": [],
            "reason": "not_initialized"
        }
    
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
        detected_symptoms = self._extract_symptoms(input_data)
        normalized_symptoms = self._normalize_symptoms(detected_symptoms)
        
        if self._awaiting_additional_input:
            normalized_symptoms = self._merge_with_pending_symptoms(normalized_symptoms)
        
        symptom_count = len(normalized_symptoms)
        awaiting_followup = self._awaiting_additional_input
        needs_more_symptoms = symptom_count < self.MIN_SYMPTOMS_FOR_CONFIDENCE and not awaiting_followup
        followup_suggestions = self._get_followup_suggestions(
            normalized_symptoms,
            limit=max(3, self.MIN_SYMPTOMS_FOR_CONFIDENCE - symptom_count + 2),
        ) if needs_more_symptoms else []
        
        if needs_more_symptoms:
            self._store_pending_symptoms(normalized_symptoms)
            return self._build_insufficient_symptom_result(
                normalized_symptoms=normalized_symptoms,
                followup_suggestions=followup_suggestions,
                symptom_count=symptom_count,
                input_text=input_data,
            )
        
        if awaiting_followup:
            self._clear_pending_symptoms()
        else:
            self._reset_pending_state()
        
        disease_predictions = self._predict_disease(normalized_symptoms)
        prediction_context = getattr(self, "_last_prediction_context", {})
        top_prediction = disease_predictions[0] if disease_predictions else None
        top_precautions = self._get_precautions(top_prediction["disease"]) if top_prediction else []
        
        # Generate diagnosis suggestions leveraging LLM placeholder
        diagnosis_suggestions = self._generate_diagnosis_suggestions(
            top_prediction=top_prediction,
            # predictions=disease_predictions,
            symptoms=normalized_symptoms,
            precautions=top_precautions,
        )
        
        # Calculate confidence based on prediction probability (fallback to minimum confidence)
        confidence = top_prediction["probability"] if top_prediction else 0.3
        model_ready = bool(getattr(self, "_disease_model", None) and self._disease_model.is_ready)
        processing_method = "trained_classifier_inference" if model_ready else "pattern_matching_with_placeholder_prediction"
        prediction_model_name = "rf_symptom_classifier_v1" if model_ready else "placeholder_predict_disease_v1"
        
        return AnalysisResult(
            analysis_type=AnalysisType.DISEASE_SYMPTOMS,
            confidence=confidence,
            primary_findings=self._format_primary_findings(normalized_symptoms, top_prediction),
            recommendations=diagnosis_suggestions,
            raw_data={
                "detected_symptoms": normalized_symptoms,
                "symptom_count": symptom_count,
                "disease_predictions": disease_predictions,
                "top_disease_precautions": top_precautions,
                "input_text": input_data,
                "model_recognized_symptoms": prediction_context.get("recognized_symptoms", []),
                "model_reason": prediction_context.get("reason"),
                "model_available": model_ready,
                "model_metrics": getattr(self._disease_model, "metrics", {}) if model_ready else {},
                "needs_more_symptoms": needs_more_symptoms,
                "symptom_threshold": self.MIN_SYMPTOMS_FOR_CONFIDENCE,
                "followup_suggestions": followup_suggestions,
            },
            metadata={
                "tool_name": self.name,
                "analysis_timestamp": "placeholder_timestamp",
                "processing_method": processing_method,
                "prediction_model": prediction_model_name,
                "min_symptom_threshold": self.MIN_SYMPTOMS_FOR_CONFIDENCE,
            }
        )
    
    def _extract_symptoms(self, text: str) -> list[str]:
        """Extract symptoms using the shared AI-powered symptom extractor."""
        if not self.symptom_extractor:
            return []
        try:
            return self.symptom_extractor.extract_symptoms(text)
        except Exception as exc:
            print(f"Symptom extraction failed: {exc}")
            return []
    
    def _generate_diagnosis_suggestions(
        self,
        *,
        top_prediction: Optional[dict],
        symptoms: list[str],
        precautions: list[str],
    ) -> list[str]:
        """Generate diagnosis suggestions for the top predicted disease using LLM placeholder."""
        if not top_prediction:
            return [
                "Unable to determine likely conditions from the provided symptoms.",
                "Please supply more detailed symptoms for improved analysis.",
                "This analysis is preliminary - professional medical evaluation is recommended",
            ]
        
        llm = self.llm
        llm_response = None

        precautions_text = "; ".join(precautions) if precautions else "No specific precautions available."
        prompt = (
            "You are a medical triage assistant. Using the provided context, generate 2-3 actionable, concise "
            "recommendations for the patient. Always include when to seek professional care.\n\n"
            f"Patient's Condition: {top_prediction['disease']} ({top_prediction['probability']:.0%} confidence)\n"
            f"Symptoms: {', '.join(symptoms) or 'unspecified'}\n"
            f"Precautions: {precautions_text}\n"
            "Respond as bullet-like sentences separated by newline characters."
        )
        try:
            llm_response = llm.call(prompt)
            print("llm_response: ", llm_response)
        except (AttributeError, TypeError) as exc:
            print(f"LLM call failed due to configuration issue: {exc}")
        except Exception as exc:  # noqa: BLE001 - fallback logging
            print(f"LLM call failed: {exc}")
        
        llm_suggestions = []
        if llm_response:
            lines = [line.strip("•- ").strip() for line in llm_response.splitlines()]
            llm_suggestions = [line for line in lines if line]
        
        if not llm_suggestions:
            precautions_text = "; ".join(precautions) if precautions else "No specific precautions available."
            llm_suggestions = [
                f"Top predicted condition: {top_prediction['disease']} "
                f"({top_prediction['probability']:.0%} confidence).",
                f"Recommended precautions: {precautions_text}",
            ]
        
        if all("professional" not in suggestion.lower() for suggestion in llm_suggestions):
            llm_suggestions.append("This analysis is preliminary - professional medical evaluation is recommended")
        
        return llm_suggestions
    
    def _run(self, symptoms_description: str) -> str:
        """CrewAI tool interface method."""
        try:
            result = self.analyze(symptoms_description)
            top_prediction = result.raw_data.get("disease_predictions", [])
            top_disease = top_prediction[0]["disease"] if top_prediction else "undetermined condition"
            return (
                f"Analysis: {result.primary_findings}. "
                f"Top predicted disease: {top_disease}. "
                f"Recommendations: {'; '.join(result.recommendations)}"
            )
        except Exception as e:
            return f"Error in disease analysis: {str(e)}"
    
    def _initialize_symptom_extractor(self):
        try:
            extractor = create_symptom_extractor()
            print("AI-powered symptom extractor initialized in DiseaseAnalysisTool")
            return extractor
        except Exception as exc:
            print(f"Warning: Could not initialize symptom extractor: {exc}")
            return None

    def _initialize_llm(self) -> Optional[LLM]:
        try:
            llm = LLM(
                model="openai/gpt-4o",
                api_key=getenv("OPENAI_API_KEY"),
                temperature=0.7,
                max_tokens=4000,
            )
            print("CrewAI LLM initialized in DiseaseAnalysisTool")
            return llm
        except Exception as exc:  # noqa: BLE001 - logging for initialization issues
            print(f"Warning: Could not initialize CrewAI LLM: {exc}")
            return None

    def _load_symptom_categories(self) -> Dict[str, List[str]]:
        if self.symptom_extractor:
            return self.symptom_extractor.get_symptom_categories()
        return {
            'respiratory': ['cough', 'sore throat', 'runny nose', 'shortness of breath'],
            'systemic': ['fever', 'fatigue', 'chills'],
            'neurological': ['headache', 'dizziness'],
            'gastrointestinal': ['nausea', 'diarrhea', 'abdominal pain'],
        }

    def _compile_symptom_patterns(self, categories: Dict[str, List[str]]) -> List[str]:
        patterns: List[str] = []
        for symptom_list in categories.values():
            for symptom in symptom_list:
                patterns.append(r'\b(?:' + re.escape(symptom) + r')\b')
        return patterns
    
    def _normalize_symptoms(self, symptoms: List[str]) -> List[str]:
        normalized = []
        seen = set()
        processor = getattr(self, "_symptom_processor", None)
        for symptom in symptoms:
            if not symptom:
                continue
            try:
                cleaned = processor.normalize_symptom(symptom) if processor else symptom
            except Exception:
                cleaned = symptom
            cleaned = cleaned.lower().strip()
            cleaned = cleaned.replace('-', ' ')
            cleaned = re.sub(r'[^a-z0-9\s]', ' ', cleaned)
            cleaned = re.sub(r'\s+', '_', cleaned).strip('_')
            if cleaned and cleaned not in seen:
                normalized.append(cleaned)
                seen.add(cleaned)
        return normalized
    
    def _get_followup_suggestions(self, symptoms: List[str], limit: int = 5) -> List[dict]:
        """Suggest additional symptoms using frequent itemsets."""
        suggester = getattr(self, "_symptom_suggester", None)
        if not suggester or not symptoms:
            return []
        
        suggestions = suggester.suggest(symptoms, limit=limit)
        formatted = []
        for suggestion in suggestions:
            formatted.append(
                {
                    "symptom": suggestion.symptom,
                    "display_name": suggestion.display_name,
                    "support": suggestion.support,
                    "disease": suggestion.disease,
                    "trigger_symptoms": suggestion.trigger_symptoms,
                }
            )
        return formatted
    
    def _merge_with_pending_symptoms(self, new_symptoms: List[str]) -> List[str]:
        """Merge newly detected symptoms with cached ones, preserving order."""
        combined: list[str] = []
        seen = set()
        for symptom in list(self._pending_symptoms) + list(new_symptoms):
            if symptom and symptom not in seen:
                combined.append(symptom)
                seen.add(symptom)
        return combined
    
    def _store_pending_symptoms(self, symptoms: List[str]) -> None:
        """Store symptoms while waiting for user follow-up."""
        self._pending_symptoms = list(symptoms)
        self._awaiting_additional_input = True
    
    def _clear_pending_symptoms(self) -> None:
        """Clear pending symptoms after follow-up is processed."""
        self._pending_symptoms = []
        self._awaiting_additional_input = False
    
    def _reset_pending_state(self) -> None:
        """Ensure pending state is cleared when not awaiting follow-up."""
        if not self._awaiting_additional_input:
            self._pending_symptoms = []
    
    def _build_insufficient_symptom_result(
        self,
        *,
        normalized_symptoms: List[str],
        followup_suggestions: List[dict],
        symptom_count: int,
        input_text: str,
    ) -> AnalysisResult:
        """Create an analysis result that prioritizes gathering more symptoms."""
        self._last_prediction_context = {
            "predictions": [],
            "recognized_symptoms": normalized_symptoms,
            "reason": "needs_more_symptoms",
        }
        recommendations = [
            "Please review the follow-up symptoms listed below and let me know which ones you also experience.",
            "You can also describe any other sensations (duration, severity, or related issues) to enrich the analysis."
        ]
        
        return AnalysisResult(
            analysis_type=AnalysisType.DISEASE_SYMPTOMS,
            confidence=0.0,
            primary_findings=self._format_primary_findings(normalized_symptoms, None),
            recommendations=recommendations,
            raw_data={
                "detected_symptoms": normalized_symptoms,
                "symptom_count": symptom_count,
                "disease_predictions": [],
                "top_disease_precautions": [],
                "input_text": input_text,
                "model_recognized_symptoms": [],
                "model_reason": "not_enough_symptoms",
                "model_available": bool(self._disease_model and self._disease_model.is_ready),
                "model_metrics": getattr(self._disease_model, "metrics", {}) if self._disease_model and self._disease_model.is_ready else {},
                "needs_more_symptoms": True,
                "symptom_threshold": self.MIN_SYMPTOMS_FOR_CONFIDENCE,
                "followup_suggestions": followup_suggestions,
                "pending_symptoms": list(self._pending_symptoms),
            },
            metadata={
                "tool_name": self.name,
                "analysis_timestamp": "placeholder_timestamp",
                "processing_method": "symptom_collection_phase",
                "prediction_model": "not_run_insufficient_symptoms",
                "min_symptom_threshold": self.MIN_SYMPTOMS_FOR_CONFIDENCE,
            }
        )
    
    def _predict_disease(self, symptoms: List[str]) -> List[dict]:
        """
        Predict diseases using the trained model with a graceful fallback.
        """
        self._last_prediction_context = {
            "predictions": [],
            "recognized_symptoms": [],
            "reason": "not_evaluated"
        }

        if not symptoms:
            return []

        model = getattr(self, "_disease_model", None)
        if model and model.is_ready:
            result = model.predict(symptoms, top_k=5)
            self._last_prediction_context = {
                "predictions": result.predictions,
                "recognized_symptoms": result.recognized_symptoms,
                "reason": result.reason,
            }
            if result.predictions:
                return result.predictions

        placeholder = self._generate_placeholder_predictions(symptoms)
        self._last_prediction_context = {
            "predictions": placeholder,
            "recognized_symptoms": symptoms,
            "reason": "placeholder_fallback"
        }
        return placeholder

    def _generate_placeholder_predictions(self, symptoms: List[str]) -> List[dict]:
        """
        Deterministic placeholder predictions used when the trained model is unavailable.
        """
        if not symptoms:
            return []

        candidate_diseases = [
            "Common Cold",
            "Influenza",
            "Migraine",
            "Gastroenteritis",
            "Bronchial Asthma",
            "Hypertension",
            "Diabetes",
            "Pneumonia",
            "Heart attack",
        ]

        randomizer = random.Random("|".join(symptoms))
        randomizer.shuffle(candidate_diseases)
        selected = candidate_diseases[:5]

        raw_scores = [randomizer.uniform(0.2, 1.0) for _ in selected]
        score_sum = sum(raw_scores) or 1.0

        predictions = []
        for disease, score in zip(selected, raw_scores):
            predictions.append(
                {
                    "disease": disease,
                    "probability": round(score / score_sum, 4),
                }
            )

        predictions.sort(key=lambda item: item["probability"], reverse=True)
        return predictions
    
    def _normalize_disease_name(self, disease: str) -> str:
        return re.sub(r'\s+', ' ', disease.strip().lower())
    
    def _load_precaution_mapping(self) -> Dict[str, List[str]]:
        csv_path = Path(__file__).resolve().parents[2] / "data" / "symptom_precaution.csv"
        precaution_mapping: Dict[str, List[str]] = {}
        
        if not csv_path.exists():
            print(f"Warning: Precaution data file not found at {csv_path}")
            return precaution_mapping
        
        try:
            with csv_path.open(mode="r", encoding="utf-8", newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    disease_name = row.get("Disease")
                    if not disease_name:
                        continue
                    normalized_name = self._normalize_disease_name(disease_name)
                    precautions = [
                        (row.get("Precaution_1") or "").strip(),
                        (row.get("Precaution_2") or "").strip(),
                        (row.get("Precaution_3") or "").strip(),
                        (row.get("Precaution_4") or "").strip(),
                    ]
                    precaution_mapping[normalized_name] = [p for p in precautions if p]
        except (OSError, csv.Error) as exc:
            print(f"Warning: Failed to load precautions: {exc}")
        
        return precaution_mapping
    
    def _get_precautions(self, disease: Optional[str]) -> List[str]:
        if not disease:
            return []
        normalized_name = self._normalize_disease_name(disease)
        return self._precaution_mapping.get(normalized_name, [])
    
    def _format_primary_findings(self, symptoms: List[str], top_prediction: Optional[dict]) -> str:
        if not symptoms:
            base = "No symptoms detected."
        else:
            base = f"Detected symptoms: {', '.join(symptoms)}."
        
        if top_prediction:
            return f"{base} \nTop predicted disease: {top_prediction['disease']} ({top_prediction['probability']:.0%} confidence)."
        return base

    @property
    def llm(self) -> Optional[LLM]:
        return getattr(self, "_llm", None)


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
        
        # Initialize LLM context and supporting artifacts
        self._llm = self._initialize_llm()
        self._feature_importance_text = self._load_context_file("outputs/feature_importance.txt")
        self._analysis_summary_text = self._load_context_file("outputs/analysis_summary.txt")
        self._cancer_prediction_model = None

        model_relative_path = Path("outputs/task3_tool2/random_forest_model_20251106_001604.pkl")
        project_root = Path(__file__).resolve().parents[2]
        model_path = project_root / model_relative_path

        try:
            with model_path.open("rb") as model_file:
                self._cancer_prediction_model = pickle.load(model_file)
            print(f"Cancer prediction model loaded from {model_path}")
        except FileNotFoundError:
            print(f"Warning: Cancer prediction model file not found at {model_path}")
        except Exception as exc:
            print(f"Warning: Failed to load cancer prediction model: {exc}")
            self._cancer_prediction_model = None
    
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
    
    @property
    def cancer_prediction_model(self):
        """Get the trained cancer prediction model."""
        return getattr(self, "_cancer_prediction_model", None)
    
    @property
    def llm(self) -> Optional[LLM]:
        """Get the CrewAI LLM instance."""
        return getattr(self, "_llm", None)
    
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
        
        predicted_class, probability = self._predict_cancer(measurements)
        
        # Generate analysis insights
        insights = self._generate_cancer_insights(measurements, characteristics)
        
        explanation = self._generate_prediction_explanation(
            predicted_class=predicted_class,
            probability=probability,
            measurements=measurements,
            characteristics=characteristics,
        )
        if explanation:
            insights.insert(0, explanation)
        
        # Calculate confidence based on amount of data provided
        data_completeness = len(measurements) + len(characteristics)
        baseline_confidence = min(0.95, data_completeness * 0.1 + 0.4)
        confidence = max(baseline_confidence, probability)
        
        return AnalysisResult(
            analysis_type=AnalysisType.BREAST_CANCER,
            confidence=confidence,
            primary_findings=(
                f"Analyzed {len(measurements)} measurements and {len(characteristics)} characteristics "
                f"→ predicted class {predicted_class} ({probability:.0%} probability)"
            ),
            recommendations=insights,
            raw_data={
                "measurements": measurements,
                "characteristics": characteristics,
                "prediction": {
                    "predicted_class": predicted_class,
                    "probability": probability,
                },
                "input_text": input_data
            },
            metadata={
                "tool_name": self.name,
                "analysis_timestamp": "placeholder_timestamp",
                "processing_method": "feature_extraction_with_placeholder_prediction"
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
        for _, char_list in characteristic_categories.items():
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
    
    def _predict_cancer(self, measurements: Dict[str, Any]) -> tuple[str, float]:
        """
        Predict breast cancer classification using the trained model with fallbacks.
        
        Args:
            measurements: Dictionary of tumor measurement values keyed by feature name.
        
        Returns:
            Tuple containing predicted class label ("Benign" or "Malignant")
            and the associated probability score for the predicted class.
        """
        print("measurements: ", measurements)

        def random_prediction() -> tuple[str, float]:
            random_prob_malignant = random.uniform(0.05, 0.95)
            predicted_label = "Malignant" if random_prob_malignant >= 0.5 else "Benign"
            probability_for_label = (
                random_prob_malignant if predicted_label == "Malignant"
                else 1 - random_prob_malignant
            )
            print("Using random probability fallback for cancer prediction.")
            return predicted_label, probability_for_label

        if not measurements:
            print("No measurements provided; using random probability fallback.")
            return random_prediction()

        def heuristic_prediction() -> tuple[str, float]:
            numeric_values = [
                value for value in (
                    self._coerce_measurement_value(val) for val in measurements.values()
                ) if value is not None
            ]

            if not numeric_values:
                print("Measurements missing numeric values; using random fallback.")
                return random_prediction()

            normalized_sum = sum(numeric_values)
            malignant_probability = 1 / (1 + math.e ** (-0.02 * (normalized_sum - 15)))
            malignant_probability = max(0.05, min(0.95, float(malignant_probability)))
            predicted_label = "Malignant" if malignant_probability >= 0.5 else "Benign"
            probability_for_label = (
                malignant_probability if predicted_label == "Malignant"
                else 1 - malignant_probability
            )
            print("Using heuristic fallback for cancer prediction.")
            return predicted_label, probability_for_label

        model = self.cancer_prediction_model
        if model is None:
            print("Cancer prediction model unavailable; using heuristic fallback.")
            return heuristic_prediction()

        feature_names = getattr(model, "feature_names", None)
        if not feature_names:
            print("Model feature names unavailable; using heuristic fallback.")
            return heuristic_prediction()

        canonical_measurements: Dict[str, float] = {}
        for raw_key, raw_value in measurements.items():
            canonical_key = self._canonicalize_measurement_key(raw_key)
            numeric_value = self._coerce_measurement_value(raw_value)
            if not canonical_key or numeric_value is None:
                continue
            if canonical_key in canonical_measurements:
                print(f"Duplicate measurement key detected after normalization: {raw_key}")
            canonical_measurements[canonical_key] = numeric_value

        feature_values: list[float] = []
        missing_features: list[str] = []
        for feature_name in feature_names:
            canonical_feature = self._canonicalize_measurement_key(feature_name)
            if canonical_feature in canonical_measurements:
                feature_values.append(canonical_measurements[canonical_feature])
            else:
                missing_features.append(feature_name)

        if missing_features:
            print(
                "Missing required features for model prediction: "
                f"{missing_features}. Using heuristic fallback."
            )
            return heuristic_prediction()

        try:
            feature_frame = pd.DataFrame(
                [dict(zip(feature_names, feature_values))],
                columns=feature_names,
            )
            print("Using trained Random Forest model for cancer prediction.")
            predicted_label_array = model.predict(feature_frame)
            probability_matrix = model.predict_proba(feature_frame)

            predicted_label = predicted_label_array[0] if len(predicted_label_array) > 0 else "B"
            mapped_label = "Malignant" if predicted_label == "M" else "Benign"

            class_labels: list[str] = []
            if hasattr(model, "model") and hasattr(model.model, "classes_"):
                class_labels = list(model.model.classes_)
            elif hasattr(model, "classes_"):
                class_labels = list(model.classes_)

            probability_for_label = self._extract_class_probability(
                probability_matrix,
                class_labels,
                mapped_label,
                fallback=0.5,
            )

            return mapped_label, probability_for_label
        except Exception as exc:
            print(f"Model-based cancer prediction failed: {exc}")
            return heuristic_prediction()

    @staticmethod
    def _canonicalize_measurement_key(key: Any) -> str:
        if key is None:
            return ""
        text = str(key).lower()
        text = text.replace("-", " ")
        text = text.replace("/", " ")
        text = text.replace("(", " ").replace(")", " ")
        text = re.sub(r"\s+", " ", text).strip()
        text = text.replace("_", "")
        text = text.replace(" ", "")
        return re.sub(r"[^a-z0-9]", "", text)

    @staticmethod
    def _coerce_measurement_value(value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            match = re.search(r"-?\d+(?:\.\d+)?", value)
            if match:
                try:
                    return float(match.group())
                except ValueError:
                    return None
        return None

    @staticmethod
    def _extract_class_probability(
        probability_matrix: Any,
        class_labels: list[str],
        target_label: str,
        *,
        fallback: Optional[float] = None,
    ) -> float:
        if probability_matrix.size == 0:
            return fallback if fallback is not None else 0.5

        target_symbol = "M" if target_label == "Malignant" else "B"
        if class_labels:
            try:
                index = class_labels.index(target_symbol)
                return float(probability_matrix[0][index])
            except (ValueError, IndexError, TypeError):
                pass

        if target_symbol == "M" and probability_matrix.shape[1] > 1:
            try:
                return float(probability_matrix[0][1])
            except (IndexError, TypeError):
                pass
        if target_symbol == "B":
            try:
                return float(probability_matrix[0][0])
            except (IndexError, TypeError):
                pass

        return fallback if fallback is not None else 0.5
    
    def _generate_prediction_explanation(
        self,
        *,
        predicted_class: str,
        probability: float,
        measurements: Dict[str, float],
        characteristics: list[str],
    ) -> Optional[str]:
        """
        Generate an LLM-based explanation for the cancer prediction.
        """
        if not self.llm:
            return (
                f"Prediction summary: {predicted_class} ({probability:.0%} probability). "
                "LLM explanation unavailable because the language model is not configured."
            )
        
        feature_info = self._feature_importance_text or "Feature importance data unavailable."
        analysis_summary = self._analysis_summary_text or "No analysis summary available."
        measurement_text = ", ".join(f"{k}={v}" for k, v in measurements.items()) or "No measurements provided."
        characteristic_text = ", ".join(characteristics) or "No qualitative characteristics provided."
        
        prompt = (
            "You are an oncology decision-support assistant. Explain the breast cancer prediction.\n\n"
            f"Predicted class: {predicted_class}\n"
            f"Probability: {probability:.2f}\n"
            f"Measurements: {measurement_text}\n"
            f"Characteristics: {characteristic_text}\n\n"
            "Feature importance report:\n"
            f"{feature_info}\n\n"
            "Sequential pattern mining summary:\n"
            f"{analysis_summary}\n\n"
            "Provide a concise explanation (2-3 sentences) describing why the prediction aligns "
            "with the feature importance and discriminative patterns when relevant. Mention specific "
            "measurements that influenced the decision. Conclude with a reminder to seek professional evaluation."
        )
        
        try:
            response = self.llm.call(prompt)
            if response:
                return response.strip()
        except Exception as exc:  # noqa: BLE001 - logging placeholder
            print(f"LLM explanation generation failed: {exc}")
        
        return (
            f"Prediction summary: {predicted_class} ({probability:.0%} probability). "
            "Failed to generate LLM explanation."
        )
    
    def _initialize_llm(self) -> Optional[LLM]:
        """Initialize LLM for prediction explanations."""
        try:
            llm = LLM(
                model="openai/gpt-4o",
                api_key=getenv("OPENAI_API_KEY"),
                temperature=0.3,
                max_tokens=4000,
            )
            print("CrewAI LLM initialized in BreastCancerAnalysisTool")
            return llm
        except Exception as exc:
            print(f"Warning: Could not initialize CrewAI LLM for BreastCancerAnalysisTool: {exc}")
            return None
    
    def _load_context_file(self, relative_path: str) -> Optional[str]:
        """Load contextual text files for LLM prompts."""
        project_root = Path(__file__).resolve().parents[2]
        full_path = project_root / relative_path
        try:
            with full_path.open("r", encoding="utf-8") as file:
                return file.read().strip()
        except FileNotFoundError:
            print(f"Context file not found: {full_path}")
        except OSError as exc:
            print(f"Failed to read context file {full_path}: {exc}")
        return None
    
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
