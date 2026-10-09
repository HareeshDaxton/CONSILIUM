"""Base model for every CONSILIUM contract (AGENTS.md §7, I-8).

All cross-component Pydantic models inherit ``Strict``:

- ``strict=True`` — no coercion (``"1"`` is not an ``int``); LLM output must
  match the declared types exactly.
- ``extra="forbid"`` — unknown fields are rejected; a model version mismatch
  surfaces as an error, never as silently dropped data.
- ``frozen=True`` — models are immutable value objects; facts and CAD numbers
  cannot be mutated in flight (supports I-1).
"""

from pydantic import BaseModel, ConfigDict


class Strict(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
