"""
app.py – Flask Backend for Neuro-Symbolic Medical Expert System
==============================================================
Server connecting the Python inference engine to the web frontend.
"""

import logging
import os

from flask import Flask, request, jsonify, render_template

# --- Import from the Engine Package ---
from engine.config import GEMINI_MODEL
from engine.explanation import ExplanationFacility
from engine.inference import InferenceEngine
from engine.knowledge_base import KnowledgeBase
from engine.neural_layer import GeminiNeuralLayer
from engine.unification import UnificationLayer

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- Global Component Initialization ---
# We initialise these once to avoid reloading overhead on every request.
try:
    kb = KnowledgeBase()
    logger.info(f"✔ Knowledge Base loaded: {len(kb.diseases)} diseases, {len(kb.rules)} rules.")
except Exception as e:
    logger.error(f"✘ Failed to load Knowledge Base: {e}")
    kb = None

try:
    # Ensure GEMINI_API_KEY is set in environment, or the engine will raise an error.
    # We let the engine handle the check, catching the exception if it fails.
    if not os.environ.get("GEMINI_API_KEY"):
        logger.warning("⚠ GEMINI_API_KEY not found in environment variables.")

    neural_layer = GeminiNeuralLayer()
    logger.info(f"✔ Gemini Neural Layer initialised (model: {GEMINI_MODEL})")
except Exception as e:
    logger.error(f"✘ Neural Layer initialisation failed: {e}")
    neural_layer = None

# We can instantiate these per request or globally. Globally is fine as they are stateless or reset state.
# But InferenceEngine holds 'audit_trail' state, so it MUST be instantiated per request.
# ExplanationFacility is stateless (static methods).
explainer = ExplanationFacility()


# --- Helper: Graph Structure Extraction ---
def get_kb_graph_structure():
    """
    Returns the static structure of the Knowledge Base (Rules, Symptoms, Diseases)
    for visualisation purposes.
    """
    if not kb:
        return {"nodes": [], "edges": []}

    structure = {
        "rules": kb.rules,
        "diseases": kb.diseases
    }
    return structure


@app.route('/')
def index():
    """
    Serve the single-page application.

    This function renders the main HTML template which acts as the
    user interface for the expert system.
    """
    return render_template('index.html')


@app.route('/api/diagnose', methods=['POST'])
def diagnose():
    """
    API Endpoint for Diagnosis.

    Expects JSON: { "complaint": "patient details..." }
    Returns JSON: { "diagnosis": { "disease": cf, ... }, "audit_trail": [ ... ] }

    This function orchestrates the diagnosis pipeline:
    1. Extracts symptoms using the neural layer.
    2. Unifies symptoms with the knowledge base.
    3. Triggers the inference engine to perform forward chaining.
    """
    if not kb or not neural_layer:
        return jsonify({"error": "Server is not fully initialised (KB or Neural Layer missing)."}), 503

    data = request.get_json()
    if not data or 'complaint' not in data:
        return jsonify({"error": "No complaint provided."}), 400

    complaint = data['complaint'].strip()
    if not complaint:
        return jsonify({"error": "Empty complaint."}), 400

    logger.info(f"Received complaint: {complaint[:50]}...")

    try:
        # --- Step 1: Neural Symptom Extraction ---
        raw_symptoms = neural_layer.extract_symptoms(complaint)
        logger.info(f"Gemini extracted: {raw_symptoms}")

        # --- Step 2: Unification ---
        unifier = UnificationLayer(kb)
        mapped_facts, unmapped = unifier.unify(raw_symptoms)
        logger.info(f"Mapped facts: {mapped_facts}")

        # --- Step 3: Forward Chaining ---
        engine = InferenceEngine(kb)
        disease_cfs = engine.forward_chain(mapped_facts)

        # --- Console Output (for debugging mismatch) ---
        explainer.print_summary(
            disease_cfs=disease_cfs,
            audit_trail=engine.audit_trail,
            mapped_facts=mapped_facts,
            unmapped=unmapped
        )

        # --- Prepare Response ---
        # Sort diagnosis by CF descending
        sorted_diagnosis = dict(sorted(disease_cfs.items(), key=lambda item: item[1], reverse=True))

        response = {
            "diagnosis": sorted_diagnosis,
            "audit_trail": engine.audit_trail,
            "mapped_symptoms": mapped_facts,
            "unmapped_symptoms": list(unmapped.keys()),
            "kb_structure": get_kb_graph_structure()  # Send full graph structure for visualisation
        }

        return jsonify(response)

    except Exception as e:
        logger.exception("Error during diagnosis")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, port=5000)
