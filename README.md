<p align="center">
  <img src="https://img.shields.io/badge/python-3.13%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12+"/>
  <img src="https://img.shields.io/badge/Gemini-2.5--flash-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="Gemini 2.5 Flash"/>
  <img src="https://img.shields.io/badge/architecture-Neuro--Symbolic-blueviolet?style=for-the-badge" alt="Neuro-Symbolic"/>
  <img src="https://img.shields.io/badge/licence-MIT-green?style=for-the-badge" alt="MIT Licence"/>
</p>

# 🧠 Neuro-Symbolic Medical Expert System

### Endemic West African Febrile Illness Differentiator

> A GUI-based clinical decision-support tool that combines **Google Gemini LLM** intelligence with a **MYCIN-inspired symbolic inference engine** to differentiate four endemic West African febrile illnesses: **Malaria**, **Typhoid**, **Dengue**, and **Lassa Fever**.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Usage](#-usage)
- [Knowledge Base](#-knowledge-base)
- [CF Mathematics](#-cf-mathematics-mycin)
- [Audit Trail & Explainability](#-audit-trail--explainability)
- [Configuration](#%EF%B8%8F-configuration)
- [Troubleshooting](#-troubleshooting)
- [Disclaimer](#%EF%B8%8F-disclaimer)

---

## 🔬 Overview

This system implements a **neuro-symbolic** approach to medical diagnosis:

| Layer | Role | Technology |
|-------|------|------------|
| **Neural** | Understands free-text patient complaints and extracts structured symptoms | Google Gemini 2.5 Flash |
| **Unification** | Maps neural output onto the symbolic knowledge base vocabulary | Python rule-based mapping |
| **Symbolic** | Performs evidence-weighted inference using certainty factor mathematics | MYCIN-based forward & backward chaining |
| **Explanation** | Produces a full audit trail of every rule fired and calculation made | JSON audit trail + formatted output |

The system is **not a black box** — every diagnostic conclusion is fully traceable through its audit trail.

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Patient Complaint                          │
│              "I have a pounding headache and pain               │
│               behind my eyes, and I feel very weak"             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   🧠 NEURAL LAYER      │
              │   (Gemini 2.5 Flash)   │
              │                        │
              │  Extracts symptoms as  │
              │  structured JSON with  │
              │  certainty factors     │
              └───────────┬────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │  🔗 UNIFICATION LAYER  │
              │                        │
              │  Maps neural output    │
              │  → KB vocabulary       │
              │  Logs unmapped to file │
              └───────────┬────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │  ⚙️ INFERENCE ENGINE   │
              │  (Evidence-Weighted    │
              │       DAG)             │
              │                        │
              │  Forward Chaining ──►  │
              │  Backward Chaining ──► │
              │  MYCIN CF Maths        │
              └───────────┬────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │  📊 EXPLANATION        │
              │     FACILITY           │
              │                        │
              │  Audit trail (JSON)    │
              │  Diagnostic report     │
              │  CF bar charts         │
              └────────────────────────┘
```

---

## ⚙ How It Works

### 1. Symptom Extraction (Neural Layer)

The patient types a free-text complaint. Gemini parses it into a strict JSON dictionary:

```
Patient: "I've had a terrible headache and my eyes hurt behind them, and I feel really weak"
```

```json
{
  "headache": 0.95,
  "retro_orbital_pain": 0.90,
  "general_weakness": 0.85
}
```

### 2. Vocabulary Unification

The extracted symptoms are matched against the 23 recognised symptom keys in `knowledge_base.json`. Any valid but unrecognised symptoms are logged to `unmapped_symptoms.log` for future KB expansion.

### 3. Forward Chaining

The engine iterates every rule. When **all** conditions of a rule are present in the evidence, the rule fires. Multiple rules supporting the same disease are combined using **parallel (incremental evidence) combination**.

### 4. Backward Chaining

If the highest-confidence diagnosis is below the threshold (default: **0.40**), the engine identifies the top hypothesis, finds which symptoms are missing from its rules, and **asks the user directly**:

```
➤ Do you have 'Severe Bone Joint Pain'? (Enter CF 0.0–1.0, or 0 if absent):
```

It then re-runs forward chaining with the augmented evidence set.

### 5. Explanation & Audit Trail

Every rule that fired, the symptoms that triggered it, and the exact mathematical calculation are recorded and printed.

---

## 📁 Project Structure

```
ExpertSystemforMedDiag/
│
├── app.py                      # Flask Web Server — API & Frontend Handler
├── main.py                     # CLI Entry point — thin launcher
├── knowledge_base.json         # Decoupled disease rules & symptom vocabulary
├── requirements.txt            # Python dependencies
├── .env                        # Your Gemini API key (create from .env.example)
├── .env.example                # Template for the .env file
├── unmapped_symptoms.log       # Auto-generated log of unrecognised symptoms
├── README.md                   # This file
│
├── templates/                  # HTML Templates (Single Page App)
│   └── index.html              # The Web UI
│
├── static/                     # Frontend Assets
│   ├── app.js                  # Frontend Logic (API calls, DAG rendering)
│   └── style.css               # Custom Styles
│
└── engine/                     # Core package (clean architecture)
    ├── __init__.py             # Public API exports
    ├── __main__.py             # Allows: python -m engine
    ├── config.py               # Constants, paths, .env loading, Gemini prompt
    ├── knowledge_base.py       # KnowledgeBase class (loads & queries the KB)
    ├── neural_layer.py         # GeminiNeuralLayer (Gemini API integration)
    ├── unification.py          # UnificationLayer (neural → symbolic mapping)
    ├── inference.py            # InferenceEngine (forward/backward chaining)
    ├── explanation.py          # ExplanationFacility (audit trail & reporting)
    └── orchestrator.py         # run() — main consultation loop
```

### Module Responsibilities

| Module | Class / Function | Responsibility |
|--------|-----------------|----------------|
| `config.py` | — | Loads `.env`, defines all constants, paths, thresholds, and the Gemini system prompt |
| `knowledge_base.py` | `KnowledgeBase` | Loads `knowledge_base.json`, exposes disease rules and symptom vocabulary |
| `neural_layer.py` | `GeminiNeuralLayer` | Sends complaints to Gemini, parses the structured JSON response, handles retries |
| `unification.py` | `UnificationLayer` | Splits neural output into mapped (in KB) and unmapped symptoms; logs unmapped |
| `inference.py` | `InferenceEngine` | Forward & backward chaining with MYCIN CF series/parallel combination |
| `explanation.py` | `ExplanationFacility` | Formats the audit trail as JSON and a human-readable report |
| `orchestrator.py` | `run()` | Wires all components together in the interactive consultation loop |

---

## 🚀 Setup & Installation

### Prerequisites

- **Python 3.13+**
- A **Google Gemini API key** — get one free at [Google AI Studio](https://aistudio.google.com/apikey)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/ExpertSystemforMedDiag.git
cd ExpertSystemforMedDiag

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your API key (ONE-TIME setup)
copy .env.example .env
# Then edit .env and replace 'your-api-key-here' with your actual key
```

> **💡 You only set the API key once.** It's stored in `.env` and loaded automatically every time you run the program.

---

## 💻 Usage

### Run the system

**Option 1: Web Interface (Recommended)**
```bash
python app.py
# Open http://127.0.0.1:5000 in your browser
```

**Option 2: CLI Mode**
```bash
python main.py
# or
python -m engine
```

### Example Session

```
════════════════════════════════════════════════════════════════════════
  NEURO-SYMBOLIC MEDICAL EXPERT SYSTEM
  Endemic West African Febrile Illness Differentiator
  (Malaria · Typhoid · Dengue · Lassa Fever)
════════════════════════════════════════════════════════════════════════

  ✔ Knowledge Base loaded: KnowledgeBase(diseases=['malaria', ...], rules=8, symptoms=23)
  ✔ Gemini Neural Layer initialised (model: gemini-2.5-flash)

────────────────────────────────────────────────────────────────────────
  Enter patient complaint (or 'quit' to exit):
  ➤ I have a pounding headache and pain behind my eyes, with severe joint pain

  ⏳ Sending complaint to Gemini for symptom extraction…
  ✔ Gemini extracted 3 symptom(s): {'headache': 0.95, 'retro_orbital_pain': 0.90, ...}
  ✔ Mapped to KB: 3 symptom(s)

  ⏳ Running forward chaining…

════════════════════════════════════════════════════════════════════════
  DIAGNOSTIC REPORT – Neuro-Symbolic Expert System
════════════════════════════════════════════════════════════════════════

  ▸ FINAL DISEASE CERTAINTY FACTORS:

    Dengue               [████████████████████████░░░░░░] 0.8100
    Malaria              [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 0.0000
    Typhoid              [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 0.0000
    Lassa Fever          [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 0.0000

  ✦ MOST LIKELY DIAGNOSIS: DENGUE  (CF = 0.8100)
```

Type `quit`, `exit`, or `q` to end the session.

---

## 📚 Knowledge Base

The knowledge base (`knowledge_base.json`) is a **decoupled, editable JSON file** containing:

- **4 diseases**: Malaria, Typhoid, Dengue, Lassa Fever
- **8 evidence-based rules** (2–3 per disease) with pathognomonic symptom sets
- **23 unique symptom keys** in a controlled vocabulary

### Recognised Symptom Vocabulary

| Malaria | Typhoid | Dengue | Lassa Fever |
|---------|---------|--------|-------------|
| `high_temperature` | `gradually_increasing_high_fever` | `sudden_high_fever_40C` | `slight_fever` |
| `chills_and_rigors` | `abdominal_pain` | `retro_orbital_pain` | `facial_and_neck_swelling` |
| `heavy_sweating` | `constipation` | `severe_bone_joint_pain` | `general_weakness` |
| `cyclical_fever_48h` | `persistent_fever` | | `fever` |
| `headache` | `rose_spots_rash_on_trunk` | | `deafness_or_hearing_loss` |
| `jaundice_yellow_eyes` | `extreme_tiredness` | | `chest_pain` |
| | | | `mucosal_bleeding_eyes_gums` |
| | | | `difficulty_breathing` |

### Adding New Rules

Edit `knowledge_base.json` directly. Each rule follows this schema:

```json
{
  "id": "R10",
  "hypothesis": "disease_name",
  "conditions": [
    {"symptom": "symptom_key_1"},
    {"symptom": "symptom_key_2"}
  ],
  "rule_cf": 0.85
}
```

---

## 📐 CF Mathematics (MYCIN)

The inference engine implements the classic **MYCIN certainty factor** formulas:

### Series Combination (AND Logic)

When a rule requires multiple symptoms, the **minimum** symptom CF is used (weakest-link principle), then multiplied by the rule's base confidence:

$$CF_{rule} = \min(CF_{symptom_1},\ CF_{symptom_2},\ \ldots) \times Rule_{Base\_CF}$$

### Parallel Combination (Incremental Evidence)

When multiple rules support the **same disease**, their CFs are combined using the remaining uncertainty margin — ensuring the combined CF never exceeds 1.0:

$$CF_{combine} = CF_{old} + CF_{new} \times (1 - CF_{old})$$

### Example Calculation

```
Rule R05 fires for Dengue:
  Symptoms: sudden_high_fever_40C=0.90, retro_orbital_pain=0.85, severe_bone_joint_pain=0.80
  Series:   min(0.90, 0.85, 0.80) × 0.90 = 0.7200
  Parallel: 0.0000 + 0.7200 × (1 − 0.0000) = 0.7200
```

---

## 📊 Audit Trail & Explainability

The system is **not a black box**. Every consultation produces:

1. **Extracted symptoms** with their certainty factors
2. **Unmapped symptoms** logged for KB expansion
3. **Rule firing trace** — which rules fired, which symptoms triggered them, and the exact maths
4. **CF bar chart** — visual comparison of all disease hypotheses
5. **Full JSON audit trail** — machine-readable record of the entire inference

```json
[
  {
    "rule_id": "R05",
    "hypothesis": "dengue",
    "matched_symptoms": {
      "sudden_high_fever_40C": 0.90,
      "retro_orbital_pain": 0.85,
      "severe_bone_joint_pain": 0.80
    },
    "min_symptom_cf": 0.80,
    "rule_base_cf": 0.90,
    "series_cf": 0.7200,
    "previous_disease_cf": 0.0000,
    "new_disease_cf": 0.7200,
    "formula_series": "min(0.90, 0.85, 0.80) × 0.90 = 0.7200",
    "formula_parallel": "0.0000 + 0.7200 × (1 − 0.0000) = 0.7200"
  }
]
```

---

## ⚙️ Configuration

All tuneable parameters live in `engine/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BACKWARD_CHAIN_THRESHOLD` | `0.4` | CF below which backward chaining is triggered |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model to use for symptom extraction |
| `GEMINI_MAX_RETRIES` | `3` | Max retry attempts on rate-limit (429) errors |
| `GEMINI_INITIAL_BACKOFF_SECS` | `30.0` | Initial wait time before first retry (doubles each attempt) |

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | ✅ | Your Google Gemini API key. Set in `.env` file. |

---

## 🔧 Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `GEMINI_API_KEY environment variable is not set` | Missing `.env` file or key | Copy `.env.example` to `.env` and add your key |
| `429 RESOURCE_EXHAUSTED` | Free-tier Gemini quota used up | Wait for daily reset or upgrade at [ai.dev/rate-limit](https://ai.dev/rate-limit) |
| `Knowledge base not found` | `knowledge_base.json` missing or moved | Ensure it's in the project root directory |
| `No rules were triggered` | Extracted symptoms don't match any full rule set | The system will attempt backward chaining to gather more evidence |

---

## ⚕️ Disclaimer

> **This is a decision-support tool for educational and research purposes only.**
> It is **not** a substitute for professional medical advice, diagnosis, or treatment.
> A qualified clinician must confirm any diagnosis. Always seek the advice of your
> physician or other qualified health provider with any questions you may have
> regarding a medical condition.

---

<p align="center">
  <em>Built with 🧠 Neuro-Symbolic AI — where neural understanding meets symbolic reasoning.</em>
</p>

