#!/usr/bin/env python
"""
PersonaLayer WebEval Runner
Quick script to run web evaluation for the PersonaLayer research project
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pl_webeval.cli import run

if __name__ == "__main__":
    run()