"""Damage the raw extract on purpose, so the validation gate can be watched rejecting it.

`src/mlops/validate.py` quarantines bad rows and fails the run once more than
MAX_BAD_FRACTION (5%) of them are bad. This script damages 12% of the rows, spread over
the four defect kinds the gate looks for, so one run lights up every counter in the
validation report instead of only proving that nulls are caught.

Run:  python scripts/corrupt_extract.py           # damage 12% of the rows
      python scripts/corrupt_extract.py --repair  # put data/raw/wdbc.csv.orig back
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "wdbc.csv"
BACKUP = RAW.with_name(RAW.name + ".orig")

ID = "sample_id"
LABEL = "diagnosis"
BAD_LABEL = "X"
SEED = 20260920
DEFECTS = ("null", "negative", "bad_label", "duplicate")


def measurement_columns(frame: pd.DataFrame) -> list[str]:
    """Every column the gate range-checks, i.e. not the id and not the label."""
    return [c for c in frame.columns if c not in (ID, LABEL)]


def split_evenly(items: list[int], parts: int) -> list[list[int]]:
    """Cut `items` into `parts` chunks that differ in size by at most one."""
    size, extra = divmod(len(items), parts)
    chunks, start = [], 0
    for i in range(parts):
        stop = start + size + (1 if i < extra else 0)
        chunks.append(items[start:stop])
        start = stop
    return chunks


def corrupt(frame: pd.DataFrame, fraction: float, rng: random.Random) -> dict[str, int]:
    """Give each of `fraction` of the rows exactly one defect. Mutates `frame` in place.

    One defect per row keeps the arithmetic honest: the gate flags a row if *any* check
    fires, so overlapping defects would report fewer bad rows than were injected.
    """
    wanted = int(len(frame) * fraction)
    targets = rng.sample(range(len(frame)), wanted)
    nulls, negatives, labels, duplicates = split_evenly(targets, len(DEFECTS))

    columns = measurement_columns(frame)
    at = {name: frame.columns.get_loc(name) for name in frame.columns}

    for position in nulls:
        frame.iat[position, at[rng.choice(columns)]] = float("nan")

    for position in negatives:
        column = at[rng.choice(columns)]
        value = float(frame.iat[position, column])
        # -0.0 is not < 0, so a measurement that is already zero needs a real negative.
        frame.iat[position, column] = -abs(value) if value else -1.0

    for position in labels:
        frame.iat[position, at[LABEL]] = BAD_LABEL

    # Donors are drawn from rows with no other defect, so whichever of the pair the gate
    # keeps as "first", exactly one row per pair ends up flagged.
    healthy = sorted(set(range(len(frame))) - set(targets))
    for position, donor in zip(duplicates, rng.sample(healthy, len(duplicates)), strict=True):
        frame.iat[position, at[ID]] = frame.iat[donor, at[ID]]

    sizes = [len(nulls), len(negatives), len(labels), len(duplicates)]
    return dict(zip(DEFECTS, sizes, strict=True))


def repair() -> None:
    if not BACKUP.exists():
        sys.exit(f"error: no backup at {BACKUP} - nothing to restore from")
    shutil.copy(BACKUP, RAW)
    frame = pd.read_csv(RAW)
    print(f"restored {RAW.name} from {BACKUP.name}")
    print(f"  {len(frame)} rows, labels {sorted(frame[LABEL].unique())}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Damage the raw extract to exercise the gate.")
    parser.add_argument("--repair", action="store_true", help="restore the extract and exit")
    parser.add_argument("--fraction", type=float, default=0.12, help="share of rows to damage")
    args = parser.parse_args()

    if args.repair:
        repair()
        return

    if not 0 < args.fraction < 1:
        sys.exit(f"error: --fraction must be between 0 and 1, got {args.fraction}")
    if not RAW.exists():
        sys.exit(f"error: extract missing: {RAW} - run scripts/make_extract.py first")

    if BACKUP.exists():
        print(f"backup {BACKUP.name} already exists, keeping it")
    else:
        shutil.copy(RAW, BACKUP)
        print(f"backed up the original to {BACKUP.name}")

    # Always damage the pristine copy, never whatever is on disk: with the fixed seed that
    # makes a second run produce the same file rather than compounding the first run.
    frame = pd.read_csv(BACKUP)
    counts = corrupt(frame, args.fraction, random.Random(SEED))
    frame.to_csv(RAW, index=False)

    damaged = sum(counts.values())
    share = damaged / len(frame)
    print(f"damaged {damaged} of {len(frame)} rows ({share:.1%}) - the gate allows 5%")
    for kind, count in counts.items():
        print(f"  {kind:<9} {count}")
    print("now trigger the DAG and read the validate task log / validation_report.json")


if __name__ == "__main__":
    main()
