"""Reasoning-diversity metrics (plan Section 13.2; AGENTS.md §19.2, §24.3).

``S_inter`` — chosen ONCE for the whole study: the mean pairwise cosine
**similarity** of the specialists' sub-report embeddings (higher = more
convergent; 1.0 = identical direction). It is never re-expressed as a
"distance" anywhere in the harness or the report, which resolves the paper's
similarity-vs-distance inconsistency (§24.3).

Embedding model — eval-only, fixed and versioned (I-10): ``hashed-ngram-tf``
v1, a deterministic, dependency-free lexical embedding (lower-cased word
unigrams + bigrams, sha256 feature hashing into 4096 dims, sublinear term
frequency, L2-normalised). It is chosen because it is offline, reproducible
byte-for-byte across machines and never imported by serving code. Known
limit, reported with every number: it measures LEXICAL overlap, not semantic
paraphrase — two sub-reports saying the same thing in different words score
as diverse. The value is embedding-dependent, so every stored metric carries
the embedder's name and version. A neural embedder can be added behind the
``Embedder`` protocol as a new, separately named instrument — never swapped
silently mid-study.

Model-free complement: unique-fact coverage — for each role, the fraction of
the facts ONLY that role is routed (its role-specific evidence under
``configs/routing.yaml``) that its sub-report actually cites.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations, pairwise
from typing import Protocol

import numpy as np
import numpy.typing as npt

from consilium.context.router import RoutingConfig, route
from consilium.schemas.agents import SubReport
from consilium.schemas.context import GlobalContext
from consilium.schemas.enums import Role

S_INTER_DEFINITION = (
    "mean pairwise cosine SIMILARITY of specialist sub-report embeddings (higher = more convergent)"
)

_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


class Embedder(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def embed(self, texts: Sequence[str]) -> npt.NDArray[np.float64]:
        """One L2-normalised row per text."""
        ...


@dataclass(frozen=True)
class HashedNgramEmbedder:
    """Deterministic lexical embedding (see module docstring)."""

    name: str = "hashed-ngram-tf"
    version: str = "v1"
    dims: int = 4096

    def _features(self, text: str) -> list[str]:
        words = _TOKEN.findall(text.lower())
        return words + [f"{a} {b}" for a, b in pairwise(words)]

    def embed(self, texts: Sequence[str]) -> npt.NDArray[np.float64]:
        out = np.zeros((len(texts), self.dims), dtype=np.float64)
        for row, text in enumerate(texts):
            for feature in self._features(text):
                h = int.from_bytes(hashlib.sha256(feature.encode()).digest()[:8], "big")
                out[row, h % self.dims] += 1.0
            nz = out[row] > 0
            out[row, nz] = 1.0 + np.log(out[row, nz])
            norm = float(np.linalg.norm(out[row]))
            if norm > 0:
                out[row] /= norm
        return out


def subreport_text(report: SubReport) -> str:
    """The reasoning content compared for diversity: findings + recommendations."""
    return " ".join([f.text for f in report.findings] + [r.text for r in report.recommendations])


def s_inter(reports: Sequence[SubReport], embedder: Embedder) -> float | None:
    """Mean pairwise cosine similarity; None when fewer than two reports."""
    if len(reports) < 2:
        return None
    vectors = embedder.embed([subreport_text(r) for r in reports])
    sims = [float(vectors[i] @ vectors[j]) for i, j in combinations(range(len(reports)), 2)]
    return float(np.mean(sims))


def role_specific_facts(
    global_ctx: GlobalContext, routing: RoutingConfig
) -> dict[Role, frozenset[str]]:
    """Facts present in this case that exactly ONE role is routed."""
    allowed = {role: ctx.allowed_ids() for role, ctx in route(global_ctx, routing).items()}
    return {
        role: frozenset(
            fid
            for fid in ids
            if not any(fid in other for r, other in allowed.items() if r is not role)
        )
        for role, ids in allowed.items()
    }


def unique_fact_coverage(
    reports: Sequence[SubReport], global_ctx: GlobalContext, routing: RoutingConfig
) -> dict[Role, float | None]:
    """Per role: share of its role-specific facts its sub-report cites
    (None when the role has no role-specific fact in this case)."""
    specific = role_specific_facts(global_ctx, routing)
    out: dict[Role, float | None] = {}
    for report in reports:
        expected = specific.get(report.role, frozenset())
        if not expected:
            out[report.role] = None
            continue
        cited = {ref for f in report.findings for ref in f.evidence_refs} | {
            ref for r in report.recommendations for ref in r.triggered_by
        }
        out[report.role] = len(expected & cited) / len(expected)
    return out


def mean_defined(values: Sequence[float | None]) -> float | None:
    defined = [v for v in values if v is not None and not math.isnan(v)]
    return float(np.mean(defined)) if defined else None
