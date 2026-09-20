"""Write the raw WDBC extract the pipeline ingests.

The CSV is committed to the repo so the pipeline has a real file to read, but it
is generated rather than hand-maintained: scikit-learn ships the same Wisconsin
Diagnostic Breast Cancer measurements, so this keeps the data reproducible.

Run:  python scripts/make_extract.py
"""

from __future__ import annotations

from pathlib import Path

from sklearn.datasets import load_breast_cancer

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "wdbc.csv"


def build() -> None:
    bundle = load_breast_cancer(as_frame=True)
    frame = bundle.data.copy()
    frame.columns = [c.replace(" ", "_") for c in frame.columns]

    # scikit-learn encodes 0 as malignant and 1 as benign; the extract uses the
    # M/B letters the original dataset ships with.
    frame.insert(0, "sample_id", [f"WDBC-{i + 1:04d}" for i in range(len(frame))])
    frame["diagnosis"] = ["M" if t == 0 else "B" for t in bundle.target]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    print(f"wrote {len(frame)} rows to {OUT}")


if __name__ == "__main__":
    build()
