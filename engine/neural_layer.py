"""
Neural Layer – Gemini API Integration
======================================
Sends the patient's unstructured complaint to Google Gemini and
receives a structured JSON dictionary of extracted symptoms with
certainty factors.
"""

import json
import os
import time

from .config import (
    GEMINI_INITIAL_BACKOFF_SECS,
    GEMINI_MAX_RETRIES,
    GEMINI_MODEL,
    GEMINI_SYSTEM_PROMPT,
)


class GeminiNeuralLayer:
    """
    Sends the patient's unstructured complaint to Google Gemini and
    receives a structured JSON dictionary of extracted symptoms with
    certainty factors.
    """

    def __init__(self) -> None:
        """
        Initialise the Gemini client.

        The API key is read from the GEMINI_API_KEY environment variable.
        """
        self.api_key: str | None = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY environment variable is not set. "
                "Please export your Google Gemini API key before running."
            )

        # Import google-genai at runtime so the rest of the module can be
        # tested / imported without the package installed.
        try:
            from google import genai  # type: ignore[import-untyped]

            self._genai = genai
        except ImportError as exc:
            raise ImportError(
                "The 'google-genai' package is required. "
                "Install it with: pip install google-genai==1.66.0"
            ) from exc

        # Create the Gemini client
        self._client = self._genai.Client(api_key=self.api_key)

    def extract_symptoms(self, complaint: str) -> dict[str, float]:
        """
        Send the patient complaint to Gemini and return a symptom → CF dict.

        Parameters
        ----------
        complaint : str
            The patient's free-text description of their symptoms.

        Returns
        -------
        dict[str, float]
            Mapping of symptom keys to certainty factors (0.0–1.0).

        Raises
        ------
        RuntimeError
            If the API call fails or the response cannot be parsed.
        """
        from google.genai import types  # type: ignore[import-untyped]

        config = types.GenerateContentConfig(
            system_instruction=GEMINI_SYSTEM_PROMPT,
            temperature=0.1,  # Low temperature for deterministic extraction
        )

        # Retry loop with exponential backoff for rate-limit (429) errors
        last_exc: Exception | None = None
        backoff = GEMINI_INITIAL_BACKOFF_SECS

        for attempt in range(1, GEMINI_MAX_RETRIES + 1):
            try:
                response = self._client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=complaint,
                    config=config,
                )
                break  # Success – exit the retry loop
            except Exception as exc:
                last_exc = exc
                err_msg = str(exc)

                # Only retry on 429 RESOURCE_EXHAUSTED; all other errors fail immediately
                if "429" not in err_msg and "RESOURCE_EXHAUSTED" not in err_msg:
                    raise RuntimeError(
                        f"Gemini API call failed: {exc}"
                    ) from exc

                if attempt < GEMINI_MAX_RETRIES:
                    print(
                        f"\n  ⏳ Rate-limited by Gemini (attempt {attempt}/{GEMINI_MAX_RETRIES}). "
                        f"Retrying in {backoff:.0f}s…"
                    )
                    time.sleep(backoff)
                    backoff *= 2  # Exponential backoff
        else:
            # All retries exhausted
            raise RuntimeError(
                f"Gemini API rate limit exceeded after {GEMINI_MAX_RETRIES} retries. "
                f"Please wait a few minutes or check your quota at "
                f"https://ai.dev/rate-limit\nLast error: {last_exc}"
            ) from last_exc

        # Extract the text payload from the response
        raw_text: str = response.text.strip()

        # Strip Markdown code fences if present (```json ... ```)
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            # Remove opening and closing fence lines
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            raw_text = "\n".join(lines).strip()

        # Parse the JSON
        try:
            symptoms: dict = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Failed to parse Gemini response as JSON.\n"
                f"Raw response:\n{raw_text}\nError: {exc}"
            ) from exc

        # Validate that every value is a float between 0.0 and 1.0
        validated: dict[str, float] = {}
        for key, value in symptoms.items():
            try:
                cf = float(value)
                cf = max(0.0, min(1.0, cf))  # Clamp to [0, 1]
                validated[key] = round(cf, 2)
            except (TypeError, ValueError):
                # Skip entries that aren't valid floats
                continue

        return validated

