"""Ensures the repo root (where train.py/evaluate.py/predict.py live) is importable in tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
