"""
Main Orchestration
==================
Orchestrates the full neuro-symbolic diagnostic pipeline:
    1. Load the Knowledge Base.
    2. Accept a patient complaint from the terminal.
    3. Send it to Gemini for symptom extraction.
    4. Unify the output with the KB vocabulary.
    5. Run forward chaining.
    6. Optionally run backward chaining if confidence is low.
    7. Present the diagnosis and full audit trail.
"""

import json
import sys

from .config import GEMINI_MODEL
from .knowledge_base import KnowledgeBase
from .neural_layer import GeminiNeuralLayer
from .unification import UnificationLayer
from .inference import InferenceEngine
from .explanation import ExplanationFacility


def run() -> None:
    """
    Main entry point: orchestrates the full neuro-symbolic diagnostic pipeline.
    """

    print("═" * 72)
    print("  NEURO-SYMBOLIC MEDICAL EXPERT SYSTEM")
    print("  Endemic West African Febrile Illness Differentiator")
    print("  (Malaria · Typhoid · Dengue · Lassa Fever)")
    print("═" * 72)

    # --- Step 1: Load the Knowledge Base ---
    try:
        kb = KnowledgeBase()
        print(f"\n  ✔ Knowledge Base loaded: {kb}")
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
        print(f"\n  ✘ Failed to load Knowledge Base: {exc}")
        sys.exit(1)

    # --- Step 2: Initialise the Neural Layer ---
    try:
        neural_layer = GeminiNeuralLayer()
        print(f"  ✔ Gemini Neural Layer initialised (model: {GEMINI_MODEL})")
    except (EnvironmentError, ImportError) as exc:
        print(f"\n  ✘ Neural Layer initialisation failed: {exc}")
        sys.exit(1)

    # --- Initialise supporting components ---
    unifier = UnificationLayer(kb)
    engine = InferenceEngine(kb)
    explainer = ExplanationFacility()

    # --- Consultation Loop ---
    while True:
        print("\n" + "─" * 72)
        complaint = input(
            "  Enter patient complaint (or 'quit' to exit):\n  ➤ "
        ).strip()

        if complaint.lower() in ("quit", "exit", "q"):
            print("\n  Goodbye. Stay safe.\n")
            break

        if not complaint:
            print("  Please enter a valid complaint.")
            continue

        # --- Step 3: Neural Symptom Extraction ---
        print("\n  ⏳ Sending complaint to Gemini for symptom extraction…")
        try:
            raw_symptoms = neural_layer.extract_symptoms(complaint)
        except RuntimeError as exc:
            print(f"\n  ✘ Symptom extraction failed: {exc}")
            continue

        if not raw_symptoms:
            print("  ⚠ Gemini did not extract any symptoms. Please try rephrasing.")
            continue

        print(f"  ✔ Gemini extracted {len(raw_symptoms)} symptom(s): {raw_symptoms}")

        # --- Step 4: Unification ---
        mapped_facts, unmapped = unifier.unify(raw_symptoms)
        print(f"  ✔ Mapped to KB: {len(mapped_facts)} symptom(s)")
        if unmapped:
            print(
                f"  ⚠ {len(unmapped)} unmapped symptom(s) logged: "
                f"{list(unmapped.keys())}"
            )

        if not mapped_facts:
            print(
                "  ⚠ No extracted symptoms matched the Knowledge Base vocabulary.\n"
                "     The unmapped symptoms have been logged for future KB expansion."
            )
            continue

        # --- Step 5: Forward Chaining ---
        print("\n  ⏳ Running forward chaining…")
        disease_cfs = engine.forward_chain(mapped_facts)

        # --- Step 6: Backward Chaining (if needed) ---
        mapped_facts, disease_cfs = engine.backward_chain(
            mapped_facts, disease_cfs
        )

        # --- Step 7: Explanation & Diagnosis ---
        explainer.print_summary(
            disease_cfs=disease_cfs,
            audit_trail=engine.audit_trail,
            mapped_facts=mapped_facts,
            unmapped=unmapped,
        )

