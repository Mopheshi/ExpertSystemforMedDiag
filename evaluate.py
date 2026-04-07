"""
Comparative Empirical Benchmarking – ICHORA 2026 Revision
==========================================================
Evaluates three diagnostic systems against the same 100-vignette dataset:

    1. NEURO-SYMBOLIC PIPELINE  – Full three-tier architecture (this paper)
    2. PURE LLM BASELINE        – Direct Gemini diagnosis with no symbolic layer
    3. CLASSICAL RULE-BASED     – Keyword matching against KB vocabulary; no LLM

Outputs per-system classification reports and a single comparison table
formatted for direct inclusion in the paper revision.

Usage
-----
    python evaluate.py [--dataset vignettes_dataset.csv] [--delay 2.0]

The --delay flag (default 2.0 s) controls the pause between Gemini calls
to respect free-tier rate limits.  Reduce it on a paid plan.
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)

from engine.knowledge_base import KnowledgeBase
from engine.neural_layer import GeminiNeuralLayer
from engine.unification import UnificationLayer
from engine.inference import InferenceEngine
from engine.config import GEMINI_MODEL, GEMINI_SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label normalisation helpers
# ---------------------------------------------------------------------------
LABEL_ALIASES: dict[str, str] = {
    "malaria": "malaria",
    "typhoid": "typhoid",
    "typhoid_fever": "typhoid",
    "dengue": "dengue",
    "dengue_fever": "dengue",
    "lassa": "lassa_fever",
    "lassa fever": "lassa_fever",
    "lassa_fever": "lassa_fever",
}

VALID_LABELS = {"malaria", "typhoid", "dengue", "lassa_fever"}


def normalise_label(raw: str) -> str:
    """
    Lowercase and strip a disease label, mapping aliases to canonical forms.
    Anything unrecognised is returned as 'inconclusive'.
    """
    cleaned = raw.strip().lower().replace(" ", "_")
    return LABEL_ALIASES.get(cleaned, "inconclusive")


# ===========================================================================
# BASELINE 1 – NEURO-SYMBOLIC PIPELINE
# ===========================================================================

def run_neuro_symbolic(
    df: pd.DataFrame,
    kb: KnowledgeBase,
    neural_layer: GeminiNeuralLayer,
    delay: float,
) -> list[str]:
    """
    Full three-tier pipeline:
        Neural Layer → Unification Layer → DAG Inference Engine

    A prediction is 'inconclusive' if all disease CFs are 0.0 after
    forward chaining (no rule fired).  Backward chaining is deliberately
    suppressed here to keep the benchmark deterministic and automated.
    """
    log.info("─" * 60)
    log.info("BASELINE 1 – NEURO-SYMBOLIC PIPELINE")
    log.info("─" * 60)

    unifier = UnificationLayer(kb)
    y_pred: list[str] = []
    total = len(df)

    for i, row in df.iterrows():
        vignette_id = row["vignette_id"]
        text = row["vignette_text"]
        log.info("[%d/%d] %s", i + 1, total, vignette_id)

        try:
            # Step 1: LLM extraction
            raw_symptoms = neural_layer.extract_symptoms(text)

            # Step 2: Ontological unification
            mapped_facts, unmapped = unifier.unify(raw_symptoms)

            if unmapped:
                log.debug("  Unmapped symptoms logged: %s", list(unmapped.keys()))

            # Step 3: Forward chaining (no backward chaining in batch mode)
            engine = InferenceEngine(kb)
            disease_cfs = engine.forward_chain(mapped_facts)

            # Decision rule: highest CF; 'inconclusive' if all are 0.0
            top_disease = max(disease_cfs, key=disease_cfs.get)  # type: ignore[arg-type]
            prediction = top_disease if disease_cfs[top_disease] > 0.0 else "inconclusive"

        except Exception as exc:
            log.warning("  Error on %s: %s", vignette_id, exc)
            prediction = "error"

        y_pred.append(normalise_label(prediction))
        time.sleep(delay)

    return y_pred


# ===========================================================================
# BASELINE 2 – PURE LLM (Direct Gemini Diagnosis)
# ===========================================================================

_LLM_DIAGNOSIS_SYSTEM_PROMPT = """You are a medical diagnosis AI. Based on the patient complaint, output ONLY a JSON object with a single key "diagnosis" whose value is one of exactly these four strings:
  "malaria", "typhoid", "dengue", "lassa_fever"

If you cannot determine a diagnosis, use "inconclusive".

Do not include any explanation, markdown, or text outside the JSON object.

EXAMPLE OUTPUT:
{"diagnosis": "dengue"}"""


def _call_gemini_raw(
    neural_layer: GeminiNeuralLayer,
    complaint: str,
    system_prompt: str,
) -> str:
    """
    Reuse the GeminiNeuralLayer client to make a raw generation call
    with an arbitrary system prompt, returning the response text.
    """
    from google.genai import types  # type: ignore[import-untyped]

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.0,  # Zero temperature – maximally deterministic
    )
    response = neural_layer._client.models.generate_content(
        model=GEMINI_MODEL,
        contents=complaint,
        config=config,
    )
    return response.text.strip()


def _parse_llm_diagnosis(raw_text: str) -> str:
    """
    Parse the LLM's raw JSON response into a canonical disease label.
    Falls back to 'inconclusive' on any parse failure.
    """
    # Strip Markdown fences if present
    text = re.sub(r"```(?:json)?", "", raw_text).strip().rstrip("`").strip()

    try:
        payload: dict = json.loads(text)
        diagnosis = payload.get("diagnosis", "inconclusive")
        return normalise_label(str(diagnosis))
    except json.JSONDecodeError:
        # Last-resort regex extraction
        match = re.search(
            r'"diagnosis"\s*:\s*"([^"]+)"', raw_text, re.IGNORECASE
        )
        if match:
            return normalise_label(match.group(1))
        return "inconclusive"


def run_pure_llm(
    df: pd.DataFrame,
    neural_layer: GeminiNeuralLayer,
    delay: float,
) -> list[str]:
    """
    Pure LLM baseline: the complaint is sent directly to Gemini and it is
    asked to produce a diagnosis label with zero symbolic grounding.

    This intentionally mirrors how a naive 'LLM-as-doctor' deployment would
    behave, exposing its tendency to hallucinate confident but unverified
    classifications.
    """
    log.info("─" * 60)
    log.info("BASELINE 2 – PURE LLM (DIRECT GEMINI DIAGNOSIS)")
    log.info("─" * 60)

    y_pred: list[str] = []
    total = len(df)

    for i, row in df.iterrows():
        vignette_id = row["vignette_id"]
        text = row["vignette_text"]
        log.info("[%d/%d] %s", i + 1, total, vignette_id)

        try:
            raw_text = _call_gemini_raw(
                neural_layer, text, _LLM_DIAGNOSIS_SYSTEM_PROMPT
            )
            prediction = _parse_llm_diagnosis(raw_text)
            log.debug("  Raw LLM response: %s  →  %s", raw_text[:80], prediction)

        except Exception as exc:
            log.warning("  Error on %s: %s", vignette_id, exc)
            prediction = "error"

        y_pred.append(prediction)
        time.sleep(delay)

    return y_pred


# ===========================================================================
# BASELINE 3 – CLASSICAL RULE-BASED (Keyword Matching, No LLM)
# ===========================================================================

def _build_keyword_map(kb: KnowledgeBase) -> dict[str, list[str]]:
    """
    Build a mapping of KB symptom key → list of natural-language surface
    forms by splitting the snake_case key and including common synonyms.

    This is intentionally naive — it replicates the rigid, brittle nature
    of pre-neural rule-based systems that motivated this research.
    """
    synonym_map: dict[str, list[str]] = {
        "high_temperature":            ["high temperature", "high fever", "fever", "febrile"],
        "chills_and_rigors":           ["chills", "rigors", "shivering", "shaking"],
        "heavy_sweating":              ["sweating", "perspiring", "drenched", "sweat"],
        "cyclical_fever_48h":          ["every 48", "cyclical fever", "fever every two days"],
        "headache":                    ["headache", "head pain", "head hurts", "head is pounding"],
        "jaundice_yellow_eyes":        ["yellow eyes", "jaundice", "yellowing", "yellow skin"],
        "gradually_increasing_high_fever": ["gradually", "rising fever", "increasing fever"],
        "abdominal_pain":              ["abdominal pain", "stomach pain", "belly pain", "stomach ache"],
        "constipation":                ["constipation", "cannot pass stool", "no bowel"],
        "persistent_fever":            ["persistent fever", "fever that won't go", "prolonged fever"],
        "rose_spots_rash_on_trunk":    ["rose spots", "rash on trunk", "pink spots", "skin spots"],
        "extreme_tiredness":           ["extreme tiredness", "exhausted", "fatigue", "very tired"],
        "sudden_high_fever_40C":       ["sudden fever", "40", "temperature of 40", "spiked"],
        "retro_orbital_pain":          ["behind the eyes", "retro orbital", "eye pain", "orbital pain"],
        "severe_bone_joint_pain":      ["joint pain", "bone pain", "body aches", "severe aches"],
        "slight_fever":                ["slight fever", "low grade fever", "mild fever"],
        "facial_and_neck_swelling":    ["facial swelling", "face swollen", "neck swelling", "puffy face"],
        "general_weakness":            ["weakness", "weak", "fatigue", "lethargic", "no energy"],
        "fever":                       ["fever", "temperature", "hot"],
        "deafness_or_hearing_loss":    ["deaf", "hearing loss", "can't hear", "lost hearing"],
        "chest_pain":                  ["chest pain", "chest hurts", "chest tightness"],
        "mucosal_bleeding_eyes_gums":  ["bleeding gums", "gum bleed", "blood from eyes", "nose bleed",
                                        "nosebleed", "mucosal bleeding", "bleeding"],
        "difficulty_breathing":        ["difficulty breathing", "breathless", "short of breath",
                                        "can't breathe", "dyspnea"],
    }
    return synonym_map


def _extract_symptoms_classical(
    text: str,
    keyword_map: dict[str, list[str]],
) -> dict[str, float]:
    """
    Detect KB symptom keys in the complaint using simple case-insensitive
    substring matching.  All detected symptoms receive CF = 1.0 (binary).
    This mirrors the rigid true/false logic of legacy rule-based systems.
    """
    text_lower = text.lower()
    detected: dict[str, float] = {}

    for symptom_key, surface_forms in keyword_map.items():
        for form in surface_forms:
            if form in text_lower:
                detected[symptom_key] = 1.0
                break  # One match per symptom is sufficient

    return detected


def run_classical_rule_based(
    df: pd.DataFrame,
    kb: KnowledgeBase,
) -> list[str]:
    """
    Classical rule-based baseline: symptom detection via exact keyword
    matching followed by the same DAG inference engine.

    No LLM is used.  The system can only recognise symptoms if their
    surface form literally appears in the text.  This directly demonstrates
    the brittleness problem that motivated the neural extraction layer.
    """
    log.info("─" * 60)
    log.info("BASELINE 3 – CLASSICAL RULE-BASED (KEYWORD MATCHING)")
    log.info("─" * 60)

    keyword_map = _build_keyword_map(kb)
    y_pred: list[str] = []
    total = len(df)

    for i, row in df.iterrows():
        vignette_id = row["vignette_id"]
        text = row["vignette_text"]
        log.info("[%d/%d] %s", i + 1, total, vignette_id)

        try:
            # Keyword extraction (no LLM)
            detected_symptoms = _extract_symptoms_classical(text, keyword_map)

            # Same inference engine as the full pipeline
            engine = InferenceEngine(kb)
            disease_cfs = engine.forward_chain(detected_symptoms)

            top_disease = max(disease_cfs, key=disease_cfs.get)  # type: ignore[arg-type]
            prediction = (
                top_disease if disease_cfs[top_disease] > 0.0 else "inconclusive"
            )

        except Exception as exc:
            log.warning("  Error on %s: %s", vignette_id, exc)
            prediction = "error"

        y_pred.append(normalise_label(prediction))
        # No API calls here, so no delay needed

    return y_pred


# ===========================================================================
# REPORTING
# ===========================================================================

def _build_per_class_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> dict[str, dict[str, float]]:
    """Return per-class precision, recall, F1 for the given label set."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        label: {"precision": p, "recall": r, "f1": f}
        for label, p, r, f in zip(labels, precision, recall, f1)
    }


def print_comparison_table(
    y_true: list[str],
    results: dict[str, list[str]],
) -> None:
    """
    Print a LaTeX-ready comparison table suitable for direct inclusion
    in the paper revision, plus a human-readable terminal summary.
    """
    labels = sorted(VALID_LABELS)

    # ------------------------------------------------------------------
    # Terminal summary
    # ------------------------------------------------------------------
    col_w = 22
    print("\n" + "═" * 90)
    print("  COMPARATIVE EVALUATION RESULTS")
    print("═" * 90)

    for system_name, y_pred in results.items():
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\n  ── {system_name} ──")
        print(f"  Overall Accuracy: {accuracy * 100:.2f}%\n")
        print(
            classification_report(
                y_true,
                y_pred,
                labels=labels + ["inconclusive"],
                zero_division=0,
            )
        )

    # ------------------------------------------------------------------
    # Cross-system per-class summary
    # ------------------------------------------------------------------
    print("═" * 90)
    print("  SIDE-BY-SIDE PRECISION / RECALL / F1 (per disease class)")
    print("═" * 90)

    header_cells = ["Disease", "Metric"] + list(results.keys())
    header = "  " + " | ".join(f"{c:<{col_w}}" for c in header_cells)
    print(header)
    print("  " + "─" * (len(header) - 2))

    all_metrics = {
        name: _build_per_class_metrics(y_true, y_pred, labels)
        for name, y_pred in results.items()
    }

    for label in labels:
        readable = label.replace("_", " ").title()
        for metric in ("precision", "recall", "f1"):
            row_cells = [readable if metric == "precision" else "", metric.upper()]
            for name in results:
                val = all_metrics[name][label][metric]
                row_cells.append(f"{val:.4f}")
            print("  " + " | ".join(f"{c:<{col_w}}" for c in row_cells))
        print("  " + "·" * (len(header) - 2))

    # Overall accuracy row
    acc_cells = ["OVERALL", "ACCURACY"]
    for name, y_pred in results.items():
        acc_cells.append(f"{accuracy_score(y_true, y_pred) * 100:.2f}%")
    print("  " + " | ".join(f"{c:<{col_w}}" for c in acc_cells))
    print("═" * 90)

    # ------------------------------------------------------------------
    # LaTeX table output (ready to paste into the paper)
    # ------------------------------------------------------------------
    print("\n  ── LaTeX Table (copy-paste into paper) ──\n")
    system_names = list(results.keys())

    latex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Comparative Evaluation Results (100 Vignettes)}",
        r"\label{tab:comparative_results}",
        r"\begin{tabular}{ll" + "c" * len(system_names) + "}",
        r"\toprule",
        r"\textbf{Disease} & \textbf{Metric} & "
        + " & ".join(f"\\textbf{{{n}}}" for n in system_names)
        + r" \\",
        r"\midrule",
    ]

    for label in labels:
        readable = label.replace("_", " ").title()
        for i, metric in enumerate(("Precision", "Recall", "F1")):
            metric_key = metric.lower()
            vals = " & ".join(
                f"{all_metrics[name][label][metric_key]:.2f}"
                for name in system_names
            )
            row_label = f"\\multirow{{3}}{{*}}{{{readable}}}" if i == 0 else ""
            latex_lines.append(
                f"{row_label} & {metric} & {vals} \\\\"
            )
        latex_lines.append(r"\midrule")

    acc_vals = " & ".join(
        f"{accuracy_score(y_true, results[n]) * 100:.1f}\\%"
        for n in system_names
    )
    latex_lines += [
        f"\\multicolumn{{2}}{{l}}{{\\textbf{{Overall Accuracy}}}} & {acc_vals} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    print("\n".join(latex_lines))
    print()

    # ------------------------------------------------------------------
    # Save full results to JSON for reproducibility
    # ------------------------------------------------------------------
    output: dict[str, Any] = {
        "y_true": y_true,
        "predictions": {name: y_pred for name, y_pred in results.items()},
        "metrics": {
            name: {
                "accuracy": accuracy_score(y_true, results[name]),
                "per_class": all_metrics[name],
            }
            for name in results
        },
    }
    out_path = Path("evaluation_results.json")
    out_path.write_text(json.dumps(output, indent=2))
    log.info("Full results saved to %s", out_path)


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comparative benchmark: Neuro-Symbolic vs. Pure LLM vs. Classical"
    )
    parser.add_argument(
        "--dataset",
        default="vignettes_dataset.csv",
        help="Path to the vignettes CSV (default: vignettes_dataset.csv)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to sleep between Gemini API calls (default: 2.0)",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip the Pure LLM baseline (useful for offline testing)",
    )
    args = parser.parse_args()

    # --- Load dataset ---
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            "Ensure 'vignettes_dataset.csv' is in the working directory.\n"
            "Expected columns: vignette_id, true_label, vignette_text"
        )

    df = pd.read_csv(dataset_path)
    required_cols = {"vignette_id", "true_label", "vignette_text"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV must have columns: {required_cols}. Found: {set(df.columns)}"
        )

    y_true = [normalise_label(label) for label in df["true_label"]]
    log.info("Dataset loaded: %d vignettes.", len(df))

    # --- Initialise shared components ---
    log.info("Initialising Knowledge Base…")
    kb = KnowledgeBase()
    log.info("KB: %s", kb)

    log.info("Initialising Gemini Neural Layer…")
    neural_layer = GeminiNeuralLayer()
    log.info("Gemini model: %s", GEMINI_MODEL)

    # --- Run all three baselines ---
    results: dict[str, list[str]] = {}

    # 1. Neuro-Symbolic (full pipeline)
    results["Neuro-Symbolic"] = run_neuro_symbolic(
        df, kb, neural_layer, args.delay
    )

    # 2. Pure LLM baseline
    if not args.skip_llm:
        results["Pure LLM"] = run_pure_llm(df, neural_layer, args.delay)
    else:
        log.info("Pure LLM baseline skipped (--skip-llm flag set).")

    # 3. Classical rule-based baseline (no API calls)
    results["Classical Rule-Based"] = run_classical_rule_based(df, kb)

    # --- Report ---
    print_comparison_table(y_true, results)


if __name__ == "__main__":
    main()