import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if project_root not in sys.path:
    sys.path.append(project_root)
