"""
main.py – Entry Point
=====================
Thin launcher for the Neuro-Symbolic Medical Expert System.
Run with:  python main.py
"""

import sys

from engine import run


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\n\n  Session interrupted by user. Goodbye.\n")
        sys.exit(0)
    except Exception as exc:
        print(f"\n  ✘ An unexpected error occurred: {exc}")
        sys.exit(1)
