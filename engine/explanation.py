"""
Explanation Facility – Audit Trail
===================================
Formats and presents the inference engine's audit trail in both
machine-readable JSON and a human-readable terminal summary.
"""

import json


class ExplanationFacility:
    """
    Formats and presents the inference engine's audit trail in both
    machine-readable JSON and a human-readable terminal summary.
    """

    @staticmethod
    def get_audit_json(audit_trail: list[dict]) -> str:
        """Return the audit trail as a pretty-printed JSON string."""
        return json.dumps(audit_trail, indent=2, ensure_ascii=False)

    @staticmethod
    def print_summary(
        disease_cfs: dict[str, float],
        audit_trail: list[dict],
        mapped_facts: dict[str, float],
        unmapped: dict[str, float],
    ) -> None:
        """Print a clean, readable diagnostic summary to the terminal."""
        divider = "═" * 72

        print(f"\n{divider}")
        print("  DIAGNOSTIC REPORT – Neuro-Symbolic Expert System")
        print(f"{divider}\n")

        # ------ Extracted Evidence ------
        print("  ▸ EXTRACTED SYMPTOMS (Mapped to Knowledge Base):")
        if mapped_facts:
            for symptom, cf in sorted(mapped_facts.items(), key=lambda x: -x[1]):
                readable = symptom.replace("_", " ").title()
                print(f"      • {readable:<45} CF = {cf:.2f}")
        else:
            print("      (none)")

        if unmapped:
            print("\n  ▸ UNMAPPED SYMPTOMS (Logged for KB expansion):")
            for symptom, cf in sorted(unmapped.items(), key=lambda x: -x[1]):
                readable = symptom.replace("_", " ").title()
                print(f"      • {readable:<45} CF = {cf:.2f}")

        # ------ Rule Firing Trace ------
        print(f"\n{'─' * 72}")
        print("  ▸ INFERENCE TRACE (Rules Fired):\n")
        if audit_trail:
            for entry in audit_trail:
                print(f"    Rule {entry['rule_id']}  →  {entry['hypothesis'].replace('_', ' ').title()}")
                print(f"      Matched symptoms : {entry['matched_symptoms']}")
                print(f"      Series (AND)     : {entry['formula_series']}")
                print(f"      Parallel (OR)    : {entry['formula_parallel']}")
                print()
        else:
            print("    No rules were triggered by the available evidence.\n")

        # ------ Final Diagnosis ------
        print(f"{'─' * 72}")
        print("  ▸ FINAL DISEASE CERTAINTY FACTORS:\n")

        # Sort diseases by CF descending
        sorted_diseases = sorted(disease_cfs.items(), key=lambda x: -x[1])
        for disease, cf in sorted_diseases:
            bar_len = int(cf * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            readable = disease.replace("_", " ").title()
            print(f"    {readable:<20} [{bar}] {cf:.4f}")

        # Top diagnosis
        top_disease, top_cf = sorted_diseases[0]
        print(f"\n{'─' * 72}")
        if top_cf > 0.0:
            print(
                f"  ✦ MOST LIKELY DIAGNOSIS: "
                f"{top_disease.replace('_', ' ').upper()}  "
                f"(CF = {top_cf:.4f})"
            )
        else:
            print(
                "  ✦ INCONCLUSIVE: No disease hypothesis reached a positive "
                "certainty factor with the available evidence."
            )

        print(f"\n  ⚕  DISCLAIMER: This is a decision-support tool only.")
        print(f"     A qualified clinician must confirm any diagnosis.")
        print(f"{divider}\n")

        # ------ Full Audit Trail (JSON) ------
        print("  ▸ FULL AUDIT TRAIL (JSON):\n")
        print(ExplanationFacility.get_audit_json(audit_trail))
        print()

