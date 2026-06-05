import re
from pathlib import Path


def normalize_sql(sql: str) -> str:
    """Strip leading/trailing whitespace and collapse internal whitespace."""
    sql = sql.strip()
    sql = re.sub(r"\s+", " ", sql)
    return sql.lower()


def load_snapshot(name: str) -> str:
    path = Path(__file__).parent.parent / "snapshots" / name
    return path.read_text()
