"""Environment compatibility shim for the training stack.

Some Windows hosts enforce an Application Control policy (WDAC/AppLocker)
that blocks specific compiled extensions. One observed casualty is scipy's
``_direct`` DLL, which makes ALL of ``scipy.optimize`` unimportable — and
transformers imports ``scipy.optimize.linear_sum_assignment`` at modeling
import time, so the whole training stack dies with it.

``ensure_scipy_optimize_importable()`` pre-stubs ONLY the blocked symbol so
the rest of scipy.optimize imports normally. It is a no-op on healthy
systems (CI, Linux, unblocked Windows). The stubbed ``direct`` raises if it
is ever actually called — nothing in transformers or this repo calls it.
"""

from __future__ import annotations

import sys
import types


def ensure_scipy_optimize_importable() -> None:
    try:
        import scipy.optimize  # noqa: F401

        return  # healthy system — nothing to do
    except ImportError:
        pass
    stub = types.ModuleType("scipy.optimize._direct")

    def _blocked_direct(*args: object, **kwargs: object) -> object:
        raise NotImplementedError(
            "scipy.optimize.direct is unavailable: its DLL is blocked by this "
            "host's Application Control policy. Nothing in CONSILIUM uses it."
        )

    stub.direct = _blocked_direct  # type: ignore[attr-defined]
    sys.modules["scipy.optimize._direct"] = stub
