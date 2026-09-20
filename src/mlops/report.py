"""Run summary: one file per run, one line appended to a shared history."""

from __future__ import annotations

import json
from pathlib import Path

from .extract import run_dir


def write_report(staging_root: Path, ds: str, sections: dict) -> dict:
    """Merge every task's return value into one summary and record it.

    Re-running a logical date replaces that date's line rather than appending a
    second one, so the history stays one row per date however often it is re-run.
    """
    summary = {"ds": ds, **sections}
    summary.pop("path", None)

    (run_dir(staging_root, ds) / "summary.json").write_text(json.dumps(summary, indent=2))

    history = staging_root / "history.jsonl"
    previous = history.read_text().splitlines() if history.exists() else []
    kept = [line for line in previous if line.strip() and json.loads(line).get("ds") != ds]
    history.write_text("\n".join([*kept, json.dumps(summary, sort_keys=True)]) + "\n")
    return summary
