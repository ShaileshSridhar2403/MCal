"""Where MCal looks for datasets and trained weights.

Set ``MCAL_DATA_ROOT`` and ``MCAL_MODEL_ROOT`` to point elsewhere. The
defaults are the ``data/`` and ``saved_models/`` folders at the root of the
source checkout, which is where they resolve with an editable install
(``pip install -e .``).
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("MCAL_DATA_ROOT", REPO_ROOT / "data"))
MODEL_ROOT = Path(os.environ.get("MCAL_MODEL_ROOT", REPO_ROOT / "saved_models"))
