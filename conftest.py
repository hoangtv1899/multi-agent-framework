# conftest.py  ← place at project ROOT (same level as src/)
"""
Adds project root to Python path so pytest can find src/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))