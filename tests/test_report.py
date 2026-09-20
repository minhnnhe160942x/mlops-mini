import json

from mlops import report


def test_a_summary_is_written_for_the_run(staging):
    report.write_report(staging, "2026-08-25", {"clean_rows": 569, "train": 455})
    summary = json.loads((staging / "2026-08-25" / "summary.json").read_text())
    assert summary["ds"] == "2026-08-25"
    assert summary["clean_rows"] == 569


def test_each_run_appends_one_history_line(staging):
    report.write_report(staging, "2026-08-25", {"train": 455})
    report.write_report(staging, "2026-08-26", {"train": 460})
    lines = (staging / "history.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2


def test_rerunning_a_date_replaces_its_line_instead_of_adding_one(staging):
    report.write_report(staging, "2026-08-25", {"train": 455})
    report.write_report(staging, "2026-08-25", {"train": 999})
    lines = (staging / "history.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["train"] == 999


def test_the_internal_path_key_is_not_leaked_into_the_summary(staging):
    summary = report.write_report(staging, "2026-08-25", {"path": "/tmp/x", "train": 1})
    assert "path" not in summary
