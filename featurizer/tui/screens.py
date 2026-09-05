"""featurizer's own screens: Config (tab 6), Manifest (tab 7), SQL (tab 8).

None of them holds business logic. They render what ``Featurizer``'s public
methods return and what the ``validate`` verb's own function finds; the only
thing that starts work is the Actions screen, through a subprocess.
"""

from __future__ import annotations

from typing import List

from lynkeus.screens import ShellScreen

from .adapters import Project


def project_screens(project: Project) -> List[ShellScreen]:
    """featurizer's tabs, in the order the number keys reach them."""
    del project
    return []
