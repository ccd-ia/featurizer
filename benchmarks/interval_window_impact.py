"""What ADR-0021 (an interval window cuts at the as-of instant, #87 #90) moves
on the live databases.

The triage experiments use ``date`` as-of dates and whole-day intervals, the
shape on which the new window selects the rows the ``daterange`` one did. What
moves there is #90: ``skewness``, ``kurtosis``, ``harmonic_mean`` and
``geometric_mean`` applied no window, so their ``|interval=W`` columns held the
whole history. ``mean`` rides along as the control: its windows changed
spelling only, so none of its cells may move.

Same method as ``benchmarks.null_state_impact``, whose ``run`` and ``diff``
this reuses: canonical rows at the dataset's latest knowledge date, one file
per engine, a diff of two files.

Usage::

    PYTHONPATH=<dir holding the old featurizer package> \\
        uv run python -m benchmarks.interval_window_impact --db dirtyduck --out before.json
    uv run python -m benchmarks.interval_window_impact --db dirtyduck --out after.json
    uv run python -m benchmarks.interval_window_impact --diff before.json after.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks import final_matrix
from benchmarks.null_state_impact import diff, run

#: The four aggregators #90 fixed, and the control.
PRIMITIVES = ["geometric_mean", "harmonic_mean", "kurtosis", "skewness", "mean"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", choices=sorted(final_matrix.DATASETS))
    parser.add_argument(
        "--triage-dir", type=Path, default=final_matrix.DEFAULT_TRIAGE_DIR
    )
    parser.add_argument("--out", type=Path, help="write canonical rows to FILE")
    parser.add_argument("--diff", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))
    args = parser.parse_args()

    if args.diff:
        before, after = (json.loads(p.read_text()) for p in args.diff)
        print(json.dumps(diff(before, after), indent=2))
        return
    if not (args.db and args.out):
        parser.error("--db and --out go together, or --diff alone")
    result = run(args.triage_dir, args.db, PRIMITIVES)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result) + "\n")
    print(
        f"{args.db}: {len(result['rows'])} rows x {len(result['features'])} "
        f"features at {result['as_of_date']} from {result['featurizer']} "
        f"-> {args.out}"
    )


if __name__ == "__main__":
    main()
