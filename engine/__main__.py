"""
Allow running the engine package directly:  python -m engine
"""

import sys

from .orchestrator import run

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\n\n  Session interrupted by user. Goodbye.\n")
        sys.exit(0)
    except Exception as exc:
        print(f"\n  ✘ An unexpected error occurred: {exc}")
        sys.exit(1)

