"""
Model loading and inference helpers for symptom -> disease classification.

This module centralizes the logic required by Crew AI tools to:
1. Load the persisted scikit-learn model artifact created in the notebooks.
2. Normalize free-form symptom strings (including synonyms and simple slang).
3. Build binary feature vectors that match the training feature order.
4. Produce ranked disease predictions with probabilities for downstream use.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np

from processors.symptom_data_processor import SymptomDataProcessor


@dataclass
class ModelOutput:
    """Container for model inference results."""

    predictions: list[dict]
    recognized_symptoms: list[str]
    reason: str


class SymptomFeatureBuilder:
    """
    Converts normalized symptom strings into binary model features.

    The builder reuses the existing SymptomDataProcessor to handle spelling
    corrections, then applies lightweight heuristics to match the feature names
    used when training the classifier.
    """

    def __init__(self, symptom_names: Sequence[str]):
        if not symptom_names:
            raise ValueError("symptom_names cannot be empty")

        self.symptom_names = list(symptom_names)
        self.symptom_to_index = {name: idx for idx, name in enumerate(self.symptom_names)}
        self._symptom_processor = SymptomDataProcessor(data_path="")
        self._aliases = self._build_alias_map()

    def _build_alias_map(self) -> dict[str, str]:
        """Map common synonyms/slang to canonical feature names."""
        return {
            "running_nose": "runny_nose",
            "runny_nose": "runny_nose",
            "runny_noses": "runny_nose",
            "runningnose": "runny_nose",
            "coughing": "cough",
            "yellow_pee": "yellow_urine",
            "yellowpee": "yellow_urine",
            "yellow_peeing": "yellow_urine",
            "yellow_peepee": "yellow_urine",
            "high_temperature": "high_fever",
            "high_temp": "high_fever",
            "feverish": "high_fever",
            "pyrexia": "high_fever",
            "body_ache": "muscle_pain",
            "body_pain": "muscle_pain",
            "body_pains": "muscle_pain",
            "loose_motion": "diarrhoea",
            "loose_motions": "diarrhoea",
            "loose_stools": "diarrhoea",
            "loose_stool": "diarrhoea",
            "stomach_ache": "stomach_pain",
            "tummy_ache": "stomach_pain",
            "sore_throat": "throat_irritation",
            "throat_pain": "throat_irritation",
        }

    def normalize(self, text: str) -> str:
        """Normalize arbitrary symptom text to the model's feature name."""
        if not isinstance(text, str):
            return ""

        normalized = self._symptom_processor.normalize_symptom(text)
        normalized = normalized.lower().strip()
        normalized = normalized.replace("-", " ")
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        normalized = re.sub(r"\s+", "_", normalized).strip("_")
        normalized = self._aliases.get(normalized, normalized)

        return normalized

    def vectorize(self, symptoms: Iterable[str]) -> tuple[np.ndarray, list[str]]:
        """
        Convert a collection of symptoms into the binary feature vector.

        Returns:
            vector: np.ndarray with shape (num_symptoms,)
            recognized: ordered list of canonical symptoms that were mapped
        """
        vector = np.zeros(len(self.symptom_names), dtype=np.float32)
        recognized: list[str] = []

        for symptom in symptoms:
            canonical = self.normalize(symptom)
            if canonical and canonical in self.symptom_to_index:
                idx = self.symptom_to_index[canonical]
                vector[idx] = 1.0
                if canonical not in recognized:
                    recognized.append(canonical)

        return vector, recognized


class DiseasePredictionModel:
    """
    Lightweight wrapper around the persisted scikit-learn classifier.

    This class hides the artifact loading details and exposes a simple
    `predict` method that the Crew AI disease analysis tool can call.
    """

    DEFAULT_ARTIFACT = Path(__file__).resolve().parents[2] / "outputs" / "task3_tool1" / "disease_prediction_model.pkl"

    def __init__(self, artifact_path: Path | None = None):
        self.artifact_path = Path(artifact_path) if artifact_path else self.DEFAULT_ARTIFACT
        self._model = None
        self._label_classes = None
        self._feature_builder: SymptomFeatureBuilder | None = None
        self._metrics = {}
        self._metadata = {}
        self._is_ready = False
        self._load_artifact()

    @property
    def is_ready(self) -> bool:
        """Return True when the model artifact was loaded successfully."""
        return self._is_ready

    @property
    def metrics(self) -> dict:
        """Return stored training metrics if available."""
        return self._metrics

    def _load_artifact(self) -> None:
        """Load the persisted model artifact from disk."""
        if not self.artifact_path.exists():
            print(f"[DiseasePredictionModel] Artifact not found at {self.artifact_path}")
            self._is_ready = False
            return

        try:
            artifact = joblib.load(self.artifact_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[DiseasePredictionModel] Failed to load artifact: {exc}")
            self._is_ready = False
            return

        model = artifact.get("model")
        symptom_names = artifact.get("symptom_names")
        label_classes = artifact.get("label_encoder_classes")

        if model is None or not symptom_names or label_classes is None:
            print("[DiseasePredictionModel] Artifact missing required keys (model/symptom_names/label_encoder_classes)")
            self._is_ready = False
            return

        self._model = model
        self._label_classes = np.array(label_classes)
        self._feature_builder = SymptomFeatureBuilder(symptom_names)
        self._metrics = artifact.get("metrics", {})
        self._metadata = artifact.get("metadata", {})
        self._is_ready = True
        print(f"[DiseasePredictionModel] Loaded artifact from {self.artifact_path}")

    def predict(self, symptoms: Iterable[str], top_k: int = 5) -> ModelOutput:
        """
        Run inference on a list of symptom phrases.

        Args:
            symptoms: Iterable of natural-language symptom fragments.
            top_k: Number of ranked diseases to return.

        Returns:
            ModelOutput containing predictions and mapping diagnostics.
        """
        if not self._is_ready or not self._model or not self._feature_builder:
            return ModelOutput(predictions=[], recognized_symptoms=[], reason="model_not_ready")

        vector, recognized = self._feature_builder.vectorize(symptoms)
        if not recognized:
            return ModelOutput(predictions=[], recognized_symptoms=[], reason="no_recognized_symptoms")

        safe_top_k = max(1, min(top_k, len(self._label_classes)))

        probabilities = self._model.predict_proba(vector.reshape(1, -1))[0]
        ranked_indices = np.argsort(probabilities)[::-1][:safe_top_k]

        predictions = [
            {
                "disease": str(self._label_classes[idx]),
                "probability": float(probabilities[idx]),
            }
            for idx in ranked_indices
        ]

        return ModelOutput(predictions=predictions, recognized_symptoms=recognized, reason="ok")
