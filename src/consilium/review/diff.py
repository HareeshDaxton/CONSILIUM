"""Field-level diff between the AI draft revision and the clinician's edit
(AGENTS.md §15, clinician-review skill step 2).

Paths are stable and human-readable: list items that carry an ``id`` (claims)
are addressed by it — ``claims[id=c2].text`` — so a diff survives claim
reordering; other list items by index — ``limitations[1]``. Both sides are
stored in ``reviews`` (``edited_draft_json`` is the source of truth; the diff
is the analysable view), because a diff alone is not recoverable if the
schema evolves.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

__all__ = ["Change", "apply_diff", "diff_drafts", "flatten"]

_TOKEN = re.compile(r"([^.\[\]]+)|\[(id=[^\]]+|\d+)\]")


@dataclass(frozen=True, slots=True)
class Change:
    path: str
    op: Literal["changed", "added", "removed"]
    before: Any = None
    after: Any = None

    def to_json(self) -> dict[str, Any]:
        return {"path": self.path, "op": self.op, "before": self.before, "after": self.after}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Change:
        return cls(data["path"], data["op"], data.get("before"), data.get("after"))


def _list_key(item: Any, index: int) -> str:
    if isinstance(item, dict) and isinstance(item.get("id"), str):
        return f"[id={item['id']}]"
    return f"[{index}]"


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """JSON value → ordered {path: leaf}. Empty containers are leaves."""
    out: dict[str, Any] = {}
    if isinstance(obj, dict) and obj:
        for key, value in obj.items():
            out.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(obj, list) and obj:
        for i, value in enumerate(obj):
            out.update(flatten(value, f"{prefix}{_list_key(value, i)}"))
    else:
        out[prefix] = obj
    return out


def diff_drafts(before: dict[str, Any], after: dict[str, Any]) -> list[Change]:
    a, b = flatten(before), flatten(after)
    changes: list[Change] = []
    for path, value in a.items():
        if path not in b:
            changes.append(Change(path, "removed", before=value))
        elif b[path] != value:
            changes.append(Change(path, "changed", before=value, after=b[path]))
    changes.extend(Change(path, "added", after=value) for path, value in b.items() if path not in a)
    return changes


def _tokens(path: str) -> list[str]:
    return [m.group(1) or f"[{m.group(2)}]" for m in _TOKEN.finditer(path)]


def _unflatten(flat: dict[str, Any]) -> Any:
    root: dict[str, Any] = {}
    for path, leaf in flat.items():
        node: Any = root
        toks = _tokens(path)
        for i, tok in enumerate(toks):
            last = i == len(toks) - 1
            if last:
                node[tok] = leaf
            else:
                node = node.setdefault(tok, {})
    return _materialise(root)


def _materialise(node: Any) -> Any:
    if not isinstance(node, dict) or not node:
        return node
    keys = list(node)
    if all(k.startswith("[") for k in keys):  # an ordered list encoded by keys
        return [_materialise(node[k]) for k in keys]
    return {k: _materialise(v) for k, v in node.items()}


def apply_diff(before: dict[str, Any], changes: list[Change]) -> dict[str, Any]:
    """Rebuild the edited document from ``before`` + ``changes`` (round-trip)."""
    flat = flatten(before)
    for ch in changes:
        if ch.op == "removed":
            flat.pop(ch.path, None)
        else:
            flat[ch.path] = ch.after
    result = _unflatten(flat)
    return result if isinstance(result, dict) else {}
