"""
Frequent itemset-based symptom suggestion helper.

This module loads pre-mined frequent symptom combinations and uses them
to recommend additional symptoms to ask about when user input is sparse.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from processors.symptom_data_processor import SymptomDataProcessor


@dataclass
class SymptomSuggestion:
    """Structured representation of a suggested follow-up symptom."""

    symptom: str
    display_name: str
    support: float
    disease: str
    trigger_symptoms: list[str]


class SymptomSuggester:
    """
    Provides follow-up symptom recommendations using frequent itemsets.

    The suggester expects the pickle file created for task 3
    (`outputs/disease_frequent_itemsets.pkl`). Each record should
    contain an `itemset`, its `support`, and the associated `disease`.
    """

    def __init__(
        self,
        itemsets_path: Path | None = None,
        max_cache_size: int = 5000,
    ) -> None:
        project_root = Path(__file__).resolve().parents[2]
        preferred_path = project_root / "outputs" / "task3_tool1" / "disease_frequent_itemsets.pkl"
        self.itemsets_path = Path(itemsets_path) if itemsets_path else preferred_path
        self.max_cache_size = max_cache_size
        self._symptom_processor = SymptomDataProcessor(data_path="")
        self._itemsets: list[dict] = []
        self._load_itemsets()

    def _normalize_symptom(self, value: str) -> str:
        """Normalize symptom terms to match the classifier feature format."""
        if not value:
            return ""
        normalized = self._symptom_processor.normalize_symptom(value)
        normalized = normalized.lower().strip()
        normalized = normalized.replace("-", " ")
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        normalized = re.sub(r"\s+", "_", normalized).strip("_")
        return normalized

    def _load_itemsets(self) -> None:
        """Load and normalize frequent itemsets from disk."""
        if not self.itemsets_path.exists():
            print(f"[SymptomSuggester] Itemsets file not found: {self.itemsets_path}")
            return

        loaded = None
        try:
            loaded = pd.read_pickle(self.itemsets_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[SymptomSuggester] pandas could not read itemsets ({exc}); falling back to pickle.")
            try:
                with self.itemsets_path.open("rb") as fh:
                    loaded = pickle.load(fh)
            except Exception as inner_exc:  # noqa: BLE001
                print(f"[SymptomSuggester] Failed to load itemsets via pickle: {inner_exc}")
                return

        if isinstance(loaded, pd.DataFrame):
            records = loaded.to_dict("records")
        elif isinstance(loaded, list):
            records = loaded
        elif isinstance(loaded, tuple):
            records = list(loaded)
        elif isinstance(loaded, dict):
            records = []
            for value in loaded.values():
                if isinstance(value, list):
                    records.extend(value)
                else:
                    records.append(value)
        else:
            records = []

        normalized_records: list[dict] = []
        for record in records[: self.max_cache_size]:
            itemset = record.get("itemset")
            support = float(record.get("support", 0.0) or 0.0)
            disease = str(record.get("disease") or "").strip()

            if not itemset or not isinstance(itemset, (list, tuple, set, frozenset)):
                continue

            normalized_itemset = [
                self._normalize_symptom(symptom) for symptom in itemset if symptom
            ]
            normalized_itemset = [sym for sym in normalized_itemset if sym]

            if len(normalized_itemset) < 2:
                continue

            normalized_records.append(
                {
                    "itemset": tuple(normalized_itemset),
                    "support": support,
                    "disease": disease,
                }
            )

        self._itemsets = normalized_records
        print(f"[SymptomSuggester] Loaded {len(self._itemsets)} frequent itemsets")

    def suggest(
        self,
        known_symptoms: Iterable[str],
        limit: int = 5,
    ) -> List[SymptomSuggestion]:
        """
        Recommend additional symptoms related to the provided ones.

        Args:
            known_symptoms: Symptoms already detected in the user's input.
            limit: Maximum number of suggestions to return.

        Returns:
            List of SymptomSuggestion objects sorted by descending support.
        """
        known_set = {sym.strip().lower() for sym in known_symptoms if sym}
        if not known_set or not self._itemsets:
            return []

        suggestion_map: dict[str, SymptomSuggestion] = {}

        for record in self._itemsets:
            itemset = record["itemset"]
            if not any(symptom in known_set for symptom in itemset):
                continue

            for candidate in itemset:
                if candidate in known_set:
                    continue

                suggestion = suggestion_map.get(candidate)
                trigger = [sym for sym in itemset if sym in known_set]
                display_name = candidate.replace("_", " ")

                if (
                    suggestion is None
                    or record["support"] > suggestion.support
                ):
                    suggestion_map[candidate] = SymptomSuggestion(
                        symptom=candidate,
                        display_name=display_name,
                        support=record["support"],
                        disease=record["disease"],
                        trigger_symptoms=trigger,
                    )

        suggestions = sorted(
            suggestion_map.values(),
            key=lambda s: s.support,
            reverse=True,
        )

        return suggestions[:limit]
