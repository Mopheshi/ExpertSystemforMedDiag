"""
Unification Layer
=================
Maps the neural layer's raw symptom output onto the symbolic knowledge
base vocabulary.  Unrecognised symptoms are logged for future KB expansion.
"""

import logging

from .config import UNMAPPED_LOG_PATH
from .knowledge_base import KnowledgeBase


class UnificationLayer:
    """
    Maps the neural layer's raw symptom output onto the symbolic knowledge
    base vocabulary.  Unrecognised symptoms are logged for future KB expansion.
    """

    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb
        self._logger = self._setup_unmapped_logger()

    @staticmethod
    def _setup_unmapped_logger() -> logging.Logger:
        """Configure a file logger for unmapped symptoms."""
        logger = logging.getLogger("unmapped_symptoms")
        logger.setLevel(logging.INFO)
        # Avoid adding duplicate handlers on repeated instantiation
        if not logger.handlers:
            handler = logging.FileHandler(UNMAPPED_LOG_PATH, encoding="utf-8")
            formatter = logging.Formatter("%(asctime)s | %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def unify(
        self, raw_symptoms: dict[str, float]
    ) -> tuple[dict[str, float], dict[str, float]]:
        """
        Split raw symptoms into mapped (in KB vocabulary) and unmapped.

        Parameters
        ----------
        raw_symptoms : dict[str, float]
            Output from the neural layer.

        Returns
        -------
        tuple[dict[str, float], dict[str, float]]
            (mapped_facts, unmapped_symptoms)
        """
        vocabulary = self.kb.get_all_symptoms()
        mapped: dict[str, float] = {}
        unmapped: dict[str, float] = {}

        for symptom, cf in raw_symptoms.items():
            if symptom in vocabulary:
                mapped[symptom] = cf
            else:
                unmapped[symptom] = cf
                self._logger.info(
                    "UNMAPPED symptom detected: '%s' (CF=%.2f)", symptom, cf
                )

        return mapped, unmapped

