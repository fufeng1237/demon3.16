"""Task planner Python modules.

Subpackages are grouped by responsibility: core data/model code, optimisation
algorithms, runtime services, experiments, visualisation, tests and CLI apps.

The source tree retains a temporary import bridge for legacy modules that use
flat imports (for example ``from road_network import ...``).  Invoke source
entry points as ``python -m scripts.<group>.<module>`` from ``task_planner``.
Installed catkin scripts continue to use their flat install directory.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _name in ("core", "algorithms", "runtime", "visualization", "learning"):
    _path = str(_ROOT / _name)
    if _path not in sys.path:
        sys.path.insert(0, _path)
