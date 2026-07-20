"""Canonical paths for the task_planner package.

Keeping these paths here avoids coupling Python modules to their own directory
depth and lets scripts be moved without changing runtime data locations.
"""
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PACKAGE_DIR / "scripts"
CONFIG_DIR = PACKAGE_DIR / "config"
OUTPUT_DIR = PACKAGE_DIR / "output"
WORKSPACE_DIR = PACKAGE_DIR.parent.parent
DATA_DIR = WORKSPACE_DIR / "data"
