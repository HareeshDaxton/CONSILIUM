"""CONSILIUM training package (P2).

NEVER imported by serving code (AGENTS.md §5). Serving consumes remote
inference through VisionClient; this package produces the weights and specs
that get deployed to the endpoint.
"""

from training.common.compat import ensure_scipy_optimize_importable

# Must run before any transformers import (see training/common/compat.py).
ensure_scipy_optimize_importable()
