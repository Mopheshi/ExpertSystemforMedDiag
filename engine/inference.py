"""
Inference Engine – Forward & Backward Chaining with MYCIN CF Maths
===================================================================
Evidence-Weighted DAG inference engine implementing MYCIN-based
certainty factor combination for disease hypothesis ranking.
"""

from typing import Any

from .config import BACKWARD_CHAIN_THRESHOLD
from .knowledge_base import KnowledgeBase


class InferenceEngine:
    """
    Evidence-Weighted DAG inference engine implementing MYCIN-based
    certainty factor combination.

    CF Mathematics
    ──────────────
    Series Combination (AND logic within a single rule):
        CF_rule = min(CF_symptom1, CF_symptom2, …) × Rule_Base_CF

    Parallel Combination (multiple rules supporting the same hypothesis):
        CF_combine = CF_old + CF_new × (1 − CF_old)
    """

    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb
        self.audit_trail: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # CF Mathematics
    # ------------------------------------------------------------------

    @staticmethod
    def series_combination(symptom_cfs: list[float], rule_cf: float) -> float:
        """
        Series (AND) combination for a single rule.

        CF_rule = min(symptom CFs) × rule_base_cf
        """
        return min(symptom_cfs) * rule_cf

    @staticmethod
    def parallel_combination(cf_old: float, cf_new: float) -> float:
        """
        Parallel (incremental evidence) combination for a disease hypothesis.

        CF_combine = CF_old + CF_new × (1 − CF_old)
        """
        return cf_old + cf_new * (1.0 - cf_old)

    # ------------------------------------------------------------------
    # Forward Chaining
    # ------------------------------------------------------------------

    def forward_chain(
            self, facts: dict[str, float]
    ) -> dict[str, float]:
        """
        Iterate every rule in the KB.  If ALL conditions of a rule are
        present in *facts*, fire the rule and accumulate its CF into the
        disease hypothesis using parallel combination.

        Parameters
        ----------
        facts : dict[str, float]
            Symptom → CF mapping (the validated evidence set).

        Returns
        -------
        dict[str, float]
            Disease → combined CF mapping.
        """
        disease_cfs: dict[str, float] = {d: 0.0 for d in self.kb.diseases}
        self.audit_trail = []  # Reset for this inference cycle

        for rule in self.kb.rules:
            rule_id: str = rule["id"]
            hypothesis: str = rule["hypothesis"]
            conditions: list[dict] = rule["conditions"]
            rule_base_cf: float = rule["rule_cf"]

            # Check whether ALL conditions are satisfied
            required_symptoms = [c["symptom"] for c in conditions]
            present_cfs: list[float] = []
            all_present = True

            for symptom in required_symptoms:
                if symptom in facts:
                    present_cfs.append(facts[symptom])
                else:
                    all_present = False
                    break

            if not all_present:
                continue  # Rule does not fire

            # --- Series combination ---
            rule_cf_result = self.series_combination(present_cfs, rule_base_cf)

            # --- Parallel combination with existing disease CF ---
            old_cf = disease_cfs[hypothesis]
            new_cf = self.parallel_combination(old_cf, rule_cf_result)
            disease_cfs[hypothesis] = round(new_cf, 4)

            # --- Record in audit trail ---
            audit_entry = {
                "rule_id": rule_id,
                "hypothesis": hypothesis,
                "matched_symptoms": {
                    s: facts[s] for s in required_symptoms
                },
                "min_symptom_cf": round(min(present_cfs), 2),
                "rule_base_cf": rule_base_cf,
                "series_cf": round(rule_cf_result, 4),
                "previous_disease_cf": round(old_cf, 4),
                "new_disease_cf": round(new_cf, 4),
                "formula_series": (
                    f"min({', '.join(f'{facts[s]:.2f}' for s in required_symptoms)}) "
                    f"× {rule_base_cf} = {rule_cf_result:.4f}"
                ),
                "formula_parallel": (
                    f"{old_cf:.4f} + {rule_cf_result:.4f} × "
                    f"(1 − {old_cf:.4f}) = {new_cf:.4f}"
                ),
            }
            self.audit_trail.append(audit_entry)

        return disease_cfs

    # ------------------------------------------------------------------
    # Backward Chaining
    # ------------------------------------------------------------------

    def backward_chain(
            self,
            facts: dict[str, float],
            disease_cfs: dict[str, float],
    ) -> tuple[dict[str, float], dict[str, float]]:
        """
        If the top disease hypothesis is below BACKWARD_CHAIN_THRESHOLD,
        identify its rules with missing symptoms and ask the user to
        supply CFs for those symptoms.

        Parameters
        ----------
        facts : dict[str, float]
            Current evidence set (will be augmented in-place).
        disease_cfs : dict[str, float]
            Disease CFs from the forward pass.

        Returns
        -------
        tuple[dict[str, float], dict[str, float]]
            Updated (facts, disease_cfs) after the backward pass.
        """
        # Find the top hypothesis
        top_disease = max(disease_cfs, key=disease_cfs.get)  # type: ignore[arg-type]
        top_cf = disease_cfs[top_disease]

        if top_cf >= BACKWARD_CHAIN_THRESHOLD:
            return facts, disease_cfs  # No backward chaining needed

        print(
            f"\n⚠  Forward chaining yielded a low-confidence result "
            f"({top_disease}: {top_cf:.2f})."
        )
        print("   Initiating backward chaining to gather more evidence…\n")

        # Get rules for the top hypothesis and find missing symptoms
        rules_for_top = self.kb.get_rules_for_disease(top_disease)
        missing_symptoms: set[str] = set()

        for rule in rules_for_top:
            for condition in rule["conditions"]:
                symptom = condition["symptom"]
                if symptom not in facts:
                    missing_symptoms.add(symptom)

        if not missing_symptoms:
            print("   No additional symptoms to query. Returning current results.\n")
            return facts, disease_cfs

        # Ask the user about each missing symptom
        print(
            f"   To refine the diagnosis for '{top_disease.replace('_', ' ').title()}', "
            "please answer the following:\n"
        )
        for symptom in sorted(missing_symptoms):
            readable = symptom.replace("_", " ").title()
            while True:
                user_input = input(
                    f"   ➤ Do you have '{readable}'? "
                    "(Enter CF 0.0–1.0, or 0 if absent): "
                ).strip()
                try:
                    cf = float(user_input)
                    cf = max(0.0, min(1.0, cf))
                    break
                except ValueError:
                    print("     Please enter a valid number between 0.0 and 1.0.")

            if cf > 0.0:
                facts[symptom] = round(cf, 2)

        # Re-run forward chaining with augmented facts
        print("\n   Re-running inference with augmented evidence…\n")
        disease_cfs = self.forward_chain(facts)
        return facts, disease_cfs
