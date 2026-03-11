"""
Configuration & Constants
=========================
Central configuration for the Neuro-Symbolic Medical Expert System.
Loads environment variables and defines all shared constants, paths,
and the Gemini system prompt.
"""

from pathlib import Path

# ── Project root (one level above the engine/ package) ──────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Load .env file from project root ────────────────────────────────────
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass  # dotenv not installed; fall back to system environment variables

# ── Paths ───────────────────────────────────────────────────────────────
KB_PATH = PROJECT_ROOT / "knowledge_base.json"
UNMAPPED_LOG_PATH = PROJECT_ROOT / "unmapped_symptoms.log"

# ── Inference thresholds ────────────────────────────────────────────────
# Confidence threshold below which backward chaining is triggered
BACKWARD_CHAIN_THRESHOLD: float = 0.4

# ── Gemini API settings ────────────────────────────────────────────────
GEMINI_MODEL: str = "gemini-2.5-flash"
GEMINI_MAX_RETRIES: int = 3
GEMINI_INITIAL_BACKOFF_SECS: float = 30.0  # first wait (seconds)

# ── Gemini system prompt ───────────────────────────────────────────────
GEMINI_SYSTEM_PROMPT: str = """You are a highly precise clinical data extraction NLP module. You do not diagnose. Your ONLY function is to read an unstructured patient complaint and extract recognised symptoms into a strict, flat JSON dictionary.

Your output MUST ONLY be a valid JSON object. Do not include conversational text, greetings, or Markdown formatting outside of the JSON block.

RULE 1: VOCABULARY MATCHING
You must map the patient's natural language to the following exact symptom keys whenever possible:
["high_temperature", "chills_and_rigors", "heavy_sweating", "cyclical_fever_48h", "headache", "jaundice_yellow_eyes", "gradually_increasing_high_fever", "abdominal_pain", "constipation", "persistent_fever", "rose_spots_rash_on_trunk", "extreme_tiredness", "sudden_high_fever_40C", "retro_orbital_pain", "severe_bone_joint_pain", "slight_fever", "facial_and_neck_swelling", "general_weakness", "fever", "deafness_or_hearing_loss", "chest_pain", "mucosal_bleeding_eyes_gums", "difficulty_breathing"]

RULE 2: KNOWLEDGE EXPANSION
If a clinically relevant symptom is described that does not match the vocabulary above, extract it anyway using snake_case (e.g., "vomiting_blood"). The downstream system will log this for knowledge expansion.

RULE 3: CERTAINTY FACTOR (CF) CALCULATION
Assign a float value between 0.1 and 1.0 to each extracted symptom based on the intensity described:
* 0.9 to 1.0: Severe, extreme, unbearable, or explicitly stated as definite (e.g., "pounding headache", "fever of 40C").
* 0.6 to 0.8: Moderate, standard presence (e.g., "I have a headache", "I feel hot").
* 0.2 to 0.5: Mild, slight, or uncertain (e.g., "a little bit of a headache", "maybe a slight fever").
* Do NOT include symptoms the patient explicitly denies.

EXAMPLE INPUT:
"I've been feeling terrible since yesterday. My head is absolutely pounding, I feel really weak, and my eyes hurt a lot right behind them. No stomach pain though."

EXAMPLE OUTPUT:
{
  "headache": 0.95,
  "general_weakness": 0.85,
  "retro_orbital_pain": 0.90
}"""
