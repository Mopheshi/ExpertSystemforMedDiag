"""
Knowledge Base
==============
Loads and exposes the decoupled knowledge_base.json containing disease
rules, symptoms, and certainty factors.
"""

import json
from pathlib import Path

from .config import KB_PATH


class KnowledgeBase:
    """
    Loads and exposes the decoupled knowledge_base.json.

    Attributes
    ----------
    diseases : list[str]
        The four target diseases.
    rules : list[dict]
        All inference rules with conditions and certainty factors.
    vocabulary : set[str]
        The complete set of recognised symptom keys drawn from every rule.
    """

    def __init__(self, kb_path: Path = KB_PATH) -> None:
        """Load and validate the knowledge base from disk."""
        if not kb_path.exists():
            raise FileNotFoundError(
                f"Knowledge base not found at {kb_path}. "
                "Please ensure knowledge_base.json is in the project root."
            )
        with open(kb_path, "r", encoding="utf-8") as fh:
            data: dict = json.load(fh)

        self.diseases: list[str] = data["diseases"]
        self.rules: list[dict] = data["rules"]

        # Build the full symptom vocabulary from every rule's conditions
        self.vocabulary: set[str] = set()
        for rule in self.rules:
            for condition in rule["conditions"]:
                self.vocabulary.add(condition["symptom"])

    def get_rules_for_disease(self, disease: str) -> list[dict]:
        """Return all rules whose hypothesis matches *disease*."""
        return [r for r in self.rules if r["hypothesis"] == disease]

    def get_all_symptoms(self) -> set[str]:
        """Return the full symptom vocabulary set."""
        return self.vocabulary.copy()

    def __repr__(self) -> str:
        return (
            f"KnowledgeBase(diseases={self.diseases}, "
            f"rules={len(self.rules)}, symptoms={len(self.vocabulary)})"
        )

