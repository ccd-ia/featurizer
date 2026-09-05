"""The cockpit's CLI surface that needs no lynkeus: the verbs and the exit-2 path.

These run on every Python in the matrix — including 3.10 and 3.11, where the
``tui`` extra is empty by its version marker — so the compatibility claim
("the verbs say what to install and exit 2") is tested where it matters.
The lynkeus-dependent half lives in ``test_tui_models.py`` and skips itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

from featurizer.cli import build_parser, main

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "examples" / "01-basic-aggregations" / "config.yaml"


def test_a_missing_extra_says_what_to_install_and_exits_2(monkeypatch, capsys):
    """``featurizer.tui`` absent → one line on stderr, exit 2, no traceback.

    Both halves of the patch matter: ``from . import tui`` returns the package
    attribute when an earlier test already imported the submodule, and only
    consults ``sys.modules`` (where ``None`` means "import fails", which is
    what a 3.10 interpreter sees) once the attribute is gone.
    """
    import featurizer

    monkeypatch.delattr(featurizer, "tui", raising=False)
    monkeypatch.setitem(sys.modules, "featurizer.tui", None)
    for argv in (
        ["tui", "--config", str(CONFIG)],
        ["status", "--config", str(CONFIG)],
        ["runs", "list"],
        ["query", "select 1"],
        ["actions", "list"],
    ):
        assert main(argv) == 2, argv
    err = capsys.readouterr().err
    assert "featurizer[tui]" in err
    assert "3.12" in err


def test_the_parser_carries_every_verb():
    parser = build_parser()
    choices = next(
        a.choices for a in parser._actions if hasattr(a, "choices") and a.choices
    )
    assert {
        "list-primitives",
        "validate",
        "render",
        "materialize",
        "tui",
        "status",
        "runs",
        "query",
        "actions",
    } <= set(choices)


def test_render_prints_the_query_and_names_the_groups(capsys):
    assert main(["render", "--config", str(CONFIG)]) == 0
    out = capsys.readouterr().out
    assert "as_of_dates" in out
    assert "cross join lateral" in out

    assert main(["render", "--config", str(CONFIG), "--group", "group_000"]) == 0
    assert "as_of_dates" in capsys.readouterr().out

    assert main(["render", "--config", str(CONFIG), "--group", "group_999"]) == 1
    err = capsys.readouterr().err
    assert "group_999" in err and "group_000" in err


def test_materialize_without_a_database_is_an_error_not_a_traceback(
    monkeypatch, capsys
):
    for name in ("DATABASE_URL", "PGDATABASE", "PGHOST"):
        monkeypatch.delenv(name, raising=False)
    code = main(["materialize", "--config", str(CONFIG), "--schema", "nope"])
    assert code == 1
    assert "No PostgreSQL configured" in capsys.readouterr().err
