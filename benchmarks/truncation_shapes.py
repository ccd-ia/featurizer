"""DB-free comparison of identifier-truncation shapes (63-byte cap).

Generated feature names routinely exceed PostgreSQL's 63-byte identifier
limit, so ``_truncate_identifier`` caps them as ``raw[:54] + "~" + md5(raw)[:8]``
— it keeps the *head*. That erases the innermost ``alias.variable`` fragment on
deep compositions, which periodically prompts the suggestion to truncate
*tail*-preservingly instead so glob patterns like ``*(inspections.kw_*`` match.

This module measures that suggestion instead of arguing about it. It renders a
committed config, takes every name that actually truncates, and scores three
shapes on what each one blinds you to:

``var lost``
    the innermost ``alias.variable`` is not recoverable from the physical name
``op lost``
    the outermost transformer/aggregation is not recoverable
``ambiguous``
    columns whose visible (hash-elided) text duplicates another column's — the
    glob-friendliness that motivates the change in the first place
``worst``
    size of the largest visually-identical group

No scheme has a *correctness* bug: the hash suffix keeps every column unique
under all three. This is purely legibility.

The measured result (see ``.out-of-scope/tail-preserving-truncation.md``):
tail-preserving is a regression — it trades ~200 lost variable names for ~1,200
lost operator names and makes ambiguity *worse*. A both-ends shape does
dominate the status quo, but still leaves hundreds of ambiguous columns, so it
cannot retire the manifest and does not justify a universal column rename under
the ADR-0015 naming freeze.

Run: ``python -m benchmarks.truncation_shapes`` (or ``just bench-truncation``).
``tests/test_truncation_shapes.py`` pins the ordering so the conclusion cannot
silently invert.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List

from featurizer import Featurizer

#: The one committed config wide enough to actually truncate. The example
#: configs use focused primitive sets and never hit the 63-byte cap — this is a
#: wide-config phenomenon (which is how triage-pg met it, at ~2,500 columns).
DEFAULT_CONFIG = (
    Path(__file__).resolve().parent.parent / "featurizer" / "featurizer.yaml"
)

_INNER = re.compile(r"([A-Za-z_]\w*)\.([A-Za-z_]\w*)\)*$")
_LEAD = re.compile(r"^([A-Z][A-Z0-9_]*)\(")


def _h8(raw: str) -> str:
    return hashlib.md5(raw.encode()).hexdigest()[:8]


def head_keep(raw: str) -> str:
    """Today's shape: first 54 chars, then the hash."""
    return f"{raw[:54]}~{_h8(raw)}"


def tail_keep(raw: str) -> str:
    """The proposed shape: hash, then the LAST 54 chars."""
    return f"{_h8(raw)}~{raw[-54:]}"


def both_ends(raw: str) -> str:
    """The steelman: 26 head + hash + 26 tail (62 bytes)."""
    return f"{raw[:26]}~{_h8(raw)}~{raw[-26:]}"


SHAPES: Dict[str, Callable[[str], str]] = {
    "head (today)": head_keep,
    "tail (proposed)": tail_keep,
    "both ends": both_ends,
}


def truncating_labels(config: Path = DEFAULT_CONFIG) -> List[str]:
    """Every generated feature label that exceeds the 63-byte cap."""
    featurizer = Featurizer(str(config), validate=False)
    labels = []
    for row in featurizer.feature_manifest:
        label = (
            row.get("label") if isinstance(row, dict) else getattr(row, "label", "")
        ) or ""
        if len(label.encode()) > 63:
            labels.append(label)
    return labels


def score(labels: List[str], shape: Callable[[str], str]) -> Dict[str, int]:
    """What `shape` blinds you to, over `labels`."""
    var_lost = op_lost = 0
    visible: Counter[str] = Counter()
    for raw in labels:
        out = shape(raw)
        inner = _INNER.search(raw.rstrip(")"))
        if inner and inner.group(2) not in out:
            var_lost += 1
        lead = _LEAD.match(raw)
        if lead and lead.group(1) not in out:
            op_lost += 1
        visible[out.replace(_h8(raw), "#")] += 1
    return {
        "var_lost": var_lost,
        "op_lost": op_lost,
        "ambiguous": sum(c - 1 for c in visible.values() if c > 1),
        "worst": max(visible.values()) if visible else 0,
    }


def report(config: Path = DEFAULT_CONFIG) -> Dict[str, Dict[str, int]]:
    return {name: score(truncating_labels(config), fn) for name, fn in SHAPES.items()}


def main() -> None:
    labels = truncating_labels()
    print(
        f"truncation-shape comparison — {DEFAULT_CONFIG.name}, {len(labels)} truncating names"
    )
    print("-" * 68)
    print(f"{'scheme':<17}{'var lost':>10}{'op lost':>10}{'ambiguous':>12}{'worst':>8}")
    for name, fn in SHAPES.items():
        s = score(labels, fn)
        print(
            f"{name:<17}{s['var_lost']:>10}{s['op_lost']:>10}"
            f"{s['ambiguous']:>12}{s['worst']:>8}"
        )
    print("-" * 68)
    print("tail-preserving trades variable names for operator names and worsens")
    print("ambiguity; both-ends dominates but still cannot retire the manifest.")
    print("Decision: .out-of-scope/tail-preserving-truncation.md")


if __name__ == "__main__":
    main()
