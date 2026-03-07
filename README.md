# A Neuro-Symbolic Approach to Medical Diagnostic Reasoning

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Architecture](https://img.shields.io/badge/Architecture-Neuro--Symbolic-success.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Project Overview
This repository contains the proof-of-concept for a novel Neuro-Symbolic Medical Expert System. It is designed to bridge the gap between the natural language perception of modern Large Language Models (LLMs) and the strict, mathematically verifiable logic of traditional expert systems. 

The current landscape of AI in healthcare suffers from the "black box" problem, where neural networks cannot explain their diagnostic reasoning. This project solves that by restricting the neural network entirely to data extraction, passing structured clinical facts to a deterministic, rule-based inference engine.

This system is explicitly engineered for under-resourced clinics, where advanced laboratory infrastructure is limited, and complex diagnostic reasoning is critical for patient survival.

## Clinical Focus: Endemic Febrile Illnesses
To rigorously test the system's mathematical logic, the knowledge base focuses on differentiating four overlapping tropical diseases that share nearly identical initial clinical presentations. The diagnostic rules are strictly grounded in peer-reviewed clinical guidelines from the WHO, CDC, and NHS:
* **Malaria**
* **Typhoid Fever**
* **Dengue**
* **Lassa Fever**

## System Architecture

The architecture is entirely decoupled into three distinct layers to ensure scalability, clinical safety, and absolute explainability.



### 1. The Neural Layer (Perception)
Powered by the Gemini API, this layer acts as the user interface. It takes unstructured, messy natural language inputs from a patient or physician (e.g., "The patient has a pounding headache and pain behind the eyes") and performs strict information extraction. It does not diagnose. It simply outputs a structured JSON payload of recognised symptoms and their certainty factors.

### 2. The Integration Layer (The Bridge)
A Python middleware script that catches the LLM's JSON output, validating it against the system's hardcoded medical vocabulary. This entirely eliminates AI hallucinations by dropping any symptom not officially recognised by the Knowledge Base.

### 3. The Symbolic Layer (The Inference Engine)
A custom-built Python inference engine using an **Evidence-Weighted Directed Acyclic Graph (DAG)**. 
* It processes the validated facts against a machine-readable JSON Knowledge Base.
* It uses MYCIN-style probabilistic mathematics (Series and Parallel combinations) to calculate the final diagnostic certainty.
* **Explainable AI (XAI):** The engine generates a real-time `audit_trail`, plotting the exact logical path from symptom (Root) to rule (Edge) to diagnosis (Sink), guaranteeing 100% transparency.

## Repository Structure
* `/data`: Contains `knowledge_base.json` (the decoupled clinical rules).
* `/engine`: Contains the Python OOP implementation of the DAG inference engine.
* `/integration`: Contains the LLM prompting and JSON validation scripts.
* `/docs`: System architecture diagrams and mathematical proofs for the certainty factor calculations.

## Setup and Execution
*(Instructions for environment setup, API keys, and execution will be added here as development progresses).*
