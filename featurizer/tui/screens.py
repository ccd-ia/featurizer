"""featurizer's own screens: Config (tab 6), Manifest (tab 7), SQL (tab 8).

None of them holds business logic. They render what ``Featurizer``'s public
methods return (``entities``, ``relationships``, ``feature_manifest``,
``manifest_matching``, ``query_groups``) and what the ``validate`` verb's own
function finds; the only thing that starts work is the Actions screen,
through a subprocess. Every colour is a lynkeus theme variable.

Python 3.10 syntax throughout — see the package docstring.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from lynkeus.models import QueryResult, TableInfo
from lynkeus.screens import DataScreen, ShellScreen
from lynkeus.text import clip
from lynkeus.widgets import Panel, colour
from pygments.token import Comment, Keyword, Name, Number, Operator, String, Token
from rich.style import Style
from rich.syntax import ANSISyntaxTheme, Syntax
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import DataTable, Input, ListItem, ListView, Static, Tree

from ..manifest import ManifestEntry
from ..primitives import Entity, Variable
from ..validation import ValidationResult
from .adapters import Materialization, Project

#: ``OP(`` at the head of a feature label — the naming contract ADR-0015
#: freezes (``AGG(entity.col|interval=W)``), read here for display only.
_LABEL_OP = re.compile(r"^([A-Za-z0-9_]+)\(")


def primitive_of(entry: ManifestEntry) -> str:
    """The outermost primitive of a manifest entry, or its kind when direct."""
    if entry.kind == "one_hot":
        return "one-hot"
    if entry.kind == "variable":
        return "direct"
    match = _LABEL_OP.match(entry.label)
    return match.group(1) if match else "derived"


# ------------------------------------------------------------------ Config


class ConfigScreen(ShellScreen):
    """The entity graph as a tree, and the validator's findings on ``v``.

    The tree is the config as the engine loaded it: the target at the root,
    every relationship under the entity it hangs from with its keys and
    temporal mode, each entity's variables, and the primitives and intervals
    the planner was given. It is read once; the shell shows the file the
    engine was started with, and a changed file needs a restart.
    """

    SLUG = "config"
    TITLE = "Config"
    KEYS = (
        ("v", "validate"),
        ("enter", "open source table on Data"),
        ("y", "copy as json"),
    )
    PRIMARY = "#config-tree"

    BINDINGS = [
        Binding("v", "validate", "validate", show=False),
        Binding("y", "copy", "copy as json", show=False),
    ]

    DEFAULT_CSS = """
    ConfigScreen Horizontal { height: 1fr; }
    ConfigScreen #config-left { width: 1fr; }
    ConfigScreen #config-left Tree { height: 1fr; background: transparent; }
    ConfigScreen #config-right { width: 48; padding: 0 0 0 1; }
    ConfigScreen #config-findings { height: auto; text-wrap: wrap; }
    """

    def __init__(self, project: Project, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.project = project
        self.result: Optional[ValidationResult] = None
        self.selected_alias: Optional[str] = None

    def compose(self) -> ComposeResult:
        """The graph on the left, the validator's findings on the right."""
        f = self.project.featurizer
        with Horizontal():
            with Panel(
                f"entity graph · {self.project.config_path.name}",
                id="config-left",
                classes="-fill",
            ):
                tree: Tree[str] = Tree(f.target.alias, id="config-tree")
                tree.show_root = True
                tree.guide_depth = 3
                yield tree
            with (
                Panel("validation", id="config-right", classes="-fill"),
                VerticalScroll(),
            ):
                yield Static(
                    "[$text-muted]press [b]v[/b] to run the validator[/]",
                    id="config-findings",
                )
        yield self.keys_bar()

    def on_mount(self) -> None:
        """The tree is the loaded config; build it once."""
        self.build_tree()

    def refresh_data(self) -> None:
        """Nothing to reload: the config is what the engine was started with."""

    def poll(self) -> None:
        """Static."""

    # -------------------------------------------------------------- tree
    def build_tree(self) -> None:
        """Target at the root; relationships hang from the entity they touch."""
        f = self.project.featurizer
        tree = self.query_one("#config-tree", Tree)
        tree.clear()
        muted = colour(self.app, "text-muted")
        root = tree.root
        root.set_label(self._entity_label(f.target, muted, "target"))
        root.data = f.target.alias
        self._add_entity(root, f.target, {f.target.alias}, muted)
        self._add_planner(root, muted)
        root.expand_all()

    def _add_planner(self, root: Any, muted: str) -> None:
        """What the planner was given: intervals, depth, boundary, primitives."""
        f = self.project.featurizer
        aggs = sorted(f.aggregations)
        txs = sorted(f.transformations)
        node = root.add(
            Text("planner  ").append(
                f"max_depth {f.max_depth} · boundary {f.as_of_boundary}", style=muted
            )
        )
        node.add_leaf(
            Text("intervals  ").append(
                ", ".join(f.intervals) if f.intervals else "none", style=muted
            )
        )
        node.add_leaf(
            Text(f"aggregations {len(aggs)}  ").append(
                clip(", ".join(aggs), 90), style=muted
            )
        )
        node.add_leaf(
            Text(f"transformations {len(txs)}  ").append(
                clip(", ".join(txs), 90), style=muted
            )
        )

    def _entity_label(self, entity: Entity, muted: str, role: str = "") -> Text:
        label = Text(entity.alias, style="bold")
        label.append(f"  {entity.table}", style=muted)
        parts = []
        if entity.id is not None:
            parts.append(f"id {entity.id.name}")
        if entity.temporal_ix is not None:
            parts.append(f"time {entity.temporal_ix.name}")
        if role:
            parts.append(role)
        if parts:
            label.append(f"  · {' · '.join(parts)}", style=muted)
        return label

    def _add_entity(
        self,
        node: Any,
        entity: Entity,
        seen: set[str],
        muted: str,
        via: Any = None,
    ) -> None:
        f = self.project.featurizer
        variables = [v for v in entity.features if isinstance(v, Variable)]
        if variables:
            vars_node = node.add(
                Text("variables  ", style="").append(f"{len(variables)}", style=muted),
                expand=len(variables) <= 12,
            )
            for var in variables:
                text = Text(var.name.replace('"', ""))
                text.append(f"  {var.type}", style=muted)
                if var.role:
                    text.append(f" · role {var.role}", style=muted)
                if var.vocabulary:
                    text.append(f" · {len(var.vocabulary)} values", style=muted)
                vars_node.add_leaf(text)
        # The relationship this entity was reached by is not listed again from
        # its other end; every other one is, and a genuine cycle stops at the
        # entity already on the path.
        for rel in sorted(f.graph.get_backward_relationships(entity), key=repr):
            if rel is not via:
                self._add_relationship(node, rel, rel.child, "←", seen, muted)
        for rel in sorted(f.graph.get_forward_relationships(entity), key=repr):
            if rel is not via:
                self._add_relationship(node, rel, rel.parent, "→", seen, muted)

    def _add_relationship(
        self, node: Any, rel: Any, other: Entity, arrow: str, seen: set[str], muted: str
    ) -> None:
        text = Text(f"{arrow} ", style=colour(self.app, "primary"))
        text.append(other.alias, style="bold")
        if rel.name:
            text.append(f" [{rel.name}]", style=muted)
        keys = (
            f"{rel.parent.alias}.{rel.parent_key} = {rel.child.alias}.{rel.child_key}"
        )
        text.append(f"  on {keys}", style=muted)
        if rel.temporal_mode:
            temporal = f"  · {rel.temporal_mode}"
            if rel.temporal_grace:
                temporal += f" grace {rel.temporal_grace}"
            if rel.temporal_child_field:
                temporal += f" on {rel.temporal_child_field}"
            text.append(temporal, style=colour(self.app, "warning"))
        child = node.add(text, data=other.alias)
        if other.alias in seen:
            child.add_leaf(Text("(cycle: already on this path)", style=muted))
            return
        self._add_entity(child, other, seen | {other.alias}, muted, via=rel)

    # ---------------------------------------------------------- validate
    def action_validate(self) -> None:
        """``v``: the ``featurizer validate`` verb's own check, shown inline."""
        self.query_one("#config-findings", Static).update("[$text-muted]validating…[/]")
        self.load(self.project.validate, self.show_result, group="validate")

    def show_result(self, result: ValidationResult) -> None:
        """Errors first, then warnings, or a single green line."""
        self.result = result
        lines: List[str] = []
        if result.is_valid:
            lines.append("[$success]✓[/] configuration is valid")
        else:
            lines.append(f"[$error]✗[/] {len(result.errors)} error(s)")
            for error in result.errors:
                where = f"[$text-muted]{error.location}[/] " if error.location else ""
                lines.append(f"  [$error]•[/] {where}{error.message}")
                if error.suggestion:
                    lines.append(f"    [$text-muted]→ {error.suggestion}[/]")
        if result.warnings:
            lines.append("")
            lines.append(f"[$warning]![/] {len(result.warnings)} warning(s)")
            for warning in result.warnings:
                where = (
                    f"[$text-muted]{warning.location}[/] " if warning.location else ""
                )
                lines.append(f"  [$warning]•[/] {where}{warning.message}")
        self.query_one("#config-findings", Static).update("\n".join(lines))

    # ----------------------------------------------------------- actions
    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted[str]) -> None:
        """Remember which entity the cursor is on."""
        data = event.node.data
        self.selected_alias = data if isinstance(data, str) else None

    def on_tree_node_selected(self, event: Tree.NodeSelected[str]) -> None:
        """``enter`` on an entity opens its source table on the Data screen."""
        alias = event.node.data
        if not isinstance(alias, str):
            return
        self.load(
            lambda: self.project.resolve_table(alias), self.open_on_data, group="open"
        )

    def open_on_data(self, facts: Any) -> None:
        """Select the resolved table on Data and switch to it."""
        if facts is None or facts.schema is None or facts.name is None:
            self.app.notify(
                f"{self.selected_alias}: table not found in the database",
                severity="warning",
                timeout=4,
            )
            return
        data = self.app.screen_for("data")  # type: ignore[attr-defined]
        if isinstance(data, DataScreen):
            data.selected = TableInfo(
                facts.schema, facts.name, "table", facts.rows_estimate
            )
            data.sample_rows = 3
        self.app.action_tab_slug("data")  # type: ignore[attr-defined]

    def action_copy(self) -> None:
        """``y``: the graph as JSON — entities, relationships, primitives."""
        f = self.project.featurizer
        self.copy_json(
            {
                "target": f.target.alias,
                "max_depth": f.max_depth,
                "intervals": list(f.intervals),
                "as_of_boundary": str(f.as_of_boundary),
                "aggregations": sorted(f.aggregations),
                "transformations": sorted(f.transformations),
                "entities": [
                    {
                        "alias": e.alias,
                        "table": e.table,
                        "id": e.id.name if e.id else None,
                        "temporal_ix": e.temporal_ix.name if e.temporal_ix else None,
                        "variables": {
                            v.name: {"type": v.type, "role": v.role}
                            for v in e.features
                            if isinstance(v, Variable)
                        },
                    }
                    for e in f.entities
                ],
                "relationships": [repr(r) for r in f.relationships],
                "validation": (
                    None
                    if self.result is None
                    else {
                        "errors": [e.message for e in self.result.errors],
                        "warnings": [w.message for w in self.result.warnings],
                    }
                ),
            }
        )

    def sql_for_selection(self) -> Optional[str]:
        """``4``: a ``select *`` over the selected entity's table, as configured."""
        alias = self.selected_alias
        if alias is None:
            return None
        entity = self.project.featurizer.graph.entities.get(alias)
        if entity is None:
            return None
        return f"select *\nfrom {entity.table}\nlimit 100;"


# ---------------------------------------------------------------- Manifest


class ManifestScreen(ShellScreen):
    """``feature_manifest()`` as a table; ``/`` filters through the library's glob.

    The filter is :meth:`Featurizer.manifest_matching` — the glob semantics
    stay the library's and are not implemented twice. The group column comes
    from the persisted manifest when this config has been materialized under
    its target alias, or is ``group_000`` when the config fits one query; a
    wide config that was never materialized shows ``?``.
    """

    SLUG = "manifest"
    TITLE = "Manifest"
    KEYS = (
        ("/", "filter (glob on label)"),
        ("y", "copy matching columns"),
        ("4", "open the group table on Query"),
    )
    PRIMARY = "#manifest-table"

    BINDINGS = [
        Binding("y", "copy", "copy matching columns", show=False),
        Binding("escape", "focus_table", "back", show=False),
    ]

    DEFAULT_CSS = """
    ManifestScreen #manifest-panel DataTable { height: 1fr; }
    ManifestScreen #manifest-panel Input {
        height: 1; border: none; padding: 0; background: $surface;
    }
    ManifestScreen #manifest-count { height: 1; color: $text-muted; }
    ManifestScreen #manifest-detail { height: auto; }
    """

    def __init__(self, project: Project, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.project = project
        self.entries: List[ManifestEntry] = []
        self.shown: List[ManifestEntry] = []
        self.pattern = ""
        self.groups: Dict[str, str] = {}
        self.materialization: Optional[Materialization] = None
        self.cursor = 0

    def compose(self) -> ComposeResult:
        """The table, its filter, the selected entry's full label under it."""
        with Panel("manifest", id="manifest-panel", classes="-fill"):
            table = DataTable(
                cursor_type="row", zebra_stripes=False, id="manifest-table"
            )
            table.add_columns("column", "entity", "primitive", "interval", "group")
            yield table
            yield Input(
                placeholder="/ glob on the full label, e.g. *orders.amount*",
                classes="filter",
                id="manifest-filter",
            )
            yield Static("", id="manifest-count")
        yield Panel("selected", Static("", id="manifest-detail"))
        yield self.keys_bar()

    # -------------------------------------------------------------- data
    def refresh_data(self) -> None:
        """Manifest from the engine; group assignment from the database."""
        self.load(self._gather, self.show, group="manifest")

    def _gather(self) -> Any:
        f = self.project.featurizer
        entries = f.feature_manifest
        groups: Dict[str, str] = {}
        materialization: Optional[Materialization] = None
        stem = f.target.alias
        for m in self.project.materializations():
            if m.stem == stem:
                materialization = m
                groups = {row.column: row.group for row in m.manifest}
                break
        if not groups and len(f.query_groups) == 1:
            groups = {e.column: "group_000" for e in entries}
        return entries, groups, materialization

    def show(self, gathered: Any) -> None:
        """Render, applying the current filter through ``manifest_matching``."""
        self.entries, self.groups, self.materialization = gathered
        self.render_rows()

    def render_rows(self) -> None:
        """Rows for the entries the pattern selects."""
        f = self.project.featurizer
        if self.pattern:
            self.shown = f.manifest_matching(self.pattern, allow_empty=True)
        else:
            self.shown = list(self.entries)
        table = self.query_one("#manifest-table", DataTable)
        table.clear()
        muted = colour(self.app, "text-muted")
        for entry in self.shown:
            table.add_row(
                Text(clip(entry.column, 48)),
                Text(entry.entity or "", style=muted),
                Text(primitive_of(entry)),
                Text(entry.interval or "", style=muted),
                Text(self.groups.get(entry.column, "?"), style=muted),
                key=entry.column,
            )
        self.query_one("#manifest-count", Static).update(
            f"{len(self.shown)} of {len(self.entries)}"
            + (f" · pattern {self.pattern}" if self.pattern else "")
        )
        if self.shown:
            table.move_cursor(row=min(self.cursor, len(self.shown) - 1))
        self.render_selected()

    @property
    def selected(self) -> Optional[ManifestEntry]:
        """The entry under the cursor."""
        if not self.shown or not 0 <= self.cursor < len(self.shown):
            return None
        return self.shown[self.cursor]

    def render_selected(self) -> None:
        """The full label (never truncated) and the description."""
        entry = self.selected
        detail = self.query_one("#manifest-detail", Static)
        if entry is None:
            detail.update("[$text-muted]no column selected[/]")
            return
        lines = [f"[b]{entry.label}[/b]"]
        if entry.truncated:
            lines.append(
                f"[$warning]![/] physical name capped at 63 bytes: "
                f"[$text-muted]{entry.column}[/]"
            )
        lines.append(f"[$text-muted]{entry.description}[/]")
        if entry.parents:
            lines.append(f"[$text-muted]from[/] {', '.join(entry.parents)}")
        detail.update("\n".join(lines))

    # ----------------------------------------------------------- actions
    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Follow the cursor."""
        self.cursor = int(event.cursor_row)
        self.render_selected()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Filter as the user types; the glob is the library's."""
        self.pattern = event.value.strip()
        self.cursor = 0
        self.render_rows()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Enter in the filter returns focus to the table."""
        del event
        self.action_focus_table()

    def action_focus_table(self) -> None:
        """Focus the table."""
        self.query_one("#manifest-table", DataTable).focus()

    def action_copy(self) -> None:
        """``y``: the physical column names the pattern selects, as JSON."""
        self.copy_json([entry.column for entry in self.shown])

    def sql_for_selection(self) -> Optional[str]:
        """``4``: the group table the selected column landed in, when it exists."""
        entry = self.selected
        m = self.materialization
        if entry is None or m is None:
            self.app.notify(
                "not materialized under this target alias yet — run "
                "`featurizer materialize` from Actions",
                timeout=5,
            )
            return None
        group = self.groups.get(entry.column)
        if group is None or group not in m.tables:
            self.app.notify(f"{group or 'its group'} has no table on disk", timeout=4)
            return None
        keys = ", ".join(f'"{k}"' for k in m.key_columns)
        return (
            f'select {keys}, "{entry.column}"\nfrom {m.table_name(group)}\nlimit 100;'
        )

    def poll(self) -> None:
        """The manifest does not change under the user."""


# --------------------------------------------------------------------- SQL


def _syntax_theme(app: Any) -> ANSISyntaxTheme:
    """A pygments theme drawn from the shell's theme variables, not hex codes."""
    return ANSISyntaxTheme(
        {
            Token: Style(color=colour(app, "foreground") or None),
            Comment: Style(color=colour(app, "text-muted") or None, italic=True),
            Keyword: Style(color=colour(app, "primary") or None, bold=True),
            Name.Builtin: Style(color=colour(app, "primary") or None),
            Name.Function: Style(color=colour(app, "secondary") or None),
            String: Style(color=colour(app, "success") or None),
            Number: Style(color=colour(app, "accent") or None),
            Operator: Style(color=colour(app, "warning") or None),
        }
    )


class SqlScreen(ShellScreen):
    """``query_groups()`` one group at a time, read-only — the live SQL preview.

    ``x`` explains the selected group through ``DataSource.explain``, which is
    ``explain (analyze, buffers)`` inside a transaction that is rolled back.
    Nothing here executes a group for its rows: that is the ``materialize``
    action, confirmed and streamed like any other.
    """

    SLUG = "sql"
    TITLE = "SQL"
    KEYS = (("x", "explain analyze (rolled back)"), ("y", "copy sql"))
    PRIMARY = "#sql-groups"

    BINDINGS = [
        Binding("x", "explain", "explain analyze", show=False),
        Binding("y", "copy", "copy sql", show=False),
        Binding("escape", "focus_groups", "back", show=False),
    ]

    DEFAULT_CSS = """
    SqlScreen Horizontal { height: 1fr; }
    SqlScreen #sql-left { width: 24; }
    SqlScreen #sql-left ListView { height: 1fr; background: transparent; }
    SqlScreen #sql-right { width: 1fr; padding: 0 0 0 1; }
    SqlScreen #sql-code-panel { height: 2fr; }
    SqlScreen #sql-code { height: auto; }
    SqlScreen #sql-plan-panel { height: 1fr; }
    SqlScreen #sql-plan { height: auto; }
    """

    def __init__(self, project: Project, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.project = project
        self.groups: Dict[str, str] = {}
        self.ddl: List[str] = []
        self.selected: Optional[str] = None

    def compose(self) -> ComposeResult:
        """Groups on the left; the SQL and, below it, the plan on the right."""
        with Horizontal():
            with Panel("groups", id="sql-left", classes="-fill"):
                yield ListView(id="sql-groups")
            with Vertical(id="sql-right"):
                with (
                    Panel("sql", id="sql-code-panel", classes="-fill"),
                    VerticalScroll(),
                ):
                    yield Static("", id="sql-code")
                with (
                    Panel("plan · press x", id="sql-plan-panel", classes="-fill"),
                    VerticalScroll(),
                ):
                    yield Static("", id="sql-plan")
        yield self.keys_bar()

    # -------------------------------------------------------------- data
    def refresh_data(self) -> None:
        """Render the groups; ``r`` re-renders (and re-highlights after ``t``)."""
        self.load(self._gather, self.show, group="sql")

    def _gather(self) -> Any:
        f = self.project.featurizer
        return dict(f.query_groups), list(f.materialization_ddl)

    def show(self, gathered: Any) -> None:
        """Fill the list once; keep the selection."""
        self.groups, self.ddl = gathered
        view = self.query_one("#sql-groups", ListView)
        view.clear()
        for gid in self.groups:
            view.append(ListItem(Static(gid), name=gid))
        if self.ddl:
            view.append(
                ListItem(
                    Static(f"[$text-muted]+ {len(self.ddl)} temp-table ddl[/]"),
                    name="ddl",
                )
            )
        if self.selected is None and self.groups:
            self.selected = next(iter(self.groups))
        self.render_sql()

    def render_sql(self) -> None:
        """Highlight the selected group with a theme-derived pygments style."""
        panel = self.query_one("#sql-code-panel", Panel)
        code = self.query_one("#sql-code", Static)
        if self.selected is None:
            panel.set_title("sql")
            code.update("[$text-muted]this config renders no groups[/]")
            return
        text = self.current_sql()
        n = len(self.groups)
        panel.set_title(
            f"sql · {self.selected} of {n} group{'s' if n != 1 else ''} · "
            f"{len(text):,} chars"
            + (" · presupposes the temp-table ddl" if self.ddl else "")
        )
        code.update(
            Syntax(
                text,
                "sql",
                theme=_syntax_theme(self.app),
                line_numbers=False,
                word_wrap=False,
            )
        )

    def current_sql(self) -> str:
        """The selected group's SQL, or the DDL preamble joined."""
        if self.selected == "ddl":
            return ";\n\n".join(self.ddl) + ";"
        if self.selected is None:
            return ""
        return self.groups.get(self.selected, "").strip()

    # ----------------------------------------------------------- actions
    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        """Selection follows the cursor."""
        if event.item is not None and event.item.name:
            self.selected = event.item.name
            self.render_sql()

    def action_focus_groups(self) -> None:
        """Focus the group list."""
        self.query_one("#sql-groups", ListView).focus()

    def action_explain(self) -> None:
        """``x``: ``explain (analyze, buffers)`` of the group, rolled back."""
        if self.selected is None or self.selected == "ddl":
            self.app.notify("select a group to explain", timeout=3)
            return
        sql_text = self.current_sql()
        self.query_one("#sql-plan-panel", Panel).set_title("plan · running…")
        self.load(
            lambda: self.project.source.explain(sql_text), self.show_plan, group="plan"
        )

    def show_plan(self, result: QueryResult) -> None:
        """The plan lines, or the error the server gave."""
        panel = self.query_one("#sql-plan-panel", Panel)
        plan = self.query_one("#sql-plan", Static)
        if result.error:
            panel.set_title("plan · error")
            plan.update(f"[$error]✗[/] {result.error}")
            return
        panel.set_title(f"plan · {self.selected} · {result.elapsed_ms:.0f} ms")
        plan.update(Text("\n".join(str(row[0]) for row in result.rows)))

    def action_copy(self) -> None:
        """``y``: the selected group's SQL to the clipboard."""
        if self.selected is None:
            return
        self.app.copy_to_clipboard(self.current_sql())
        self.app.notify(f"copied {self.selected}", timeout=2)

    def sql_for_selection(self) -> Optional[str]:
        """``4``: never the group itself — Query runs what it is given.

        A group query is the whole feature matrix; opening it on Query would
        execute it for its rows, which is the materialize action's job.
        """
        return None

    def poll(self) -> None:
        """The SQL does not change under the user."""


def project_screens(project: Project) -> List[ShellScreen]:
    """featurizer's tabs, in the order the number keys reach them: 6, 7, 8."""
    return [ConfigScreen(project), ManifestScreen(project), SqlScreen(project)]
