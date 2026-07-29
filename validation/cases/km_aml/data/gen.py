from __future__ import annotations

import csv
import hashlib
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
CSV_PATH = DATA_DIR / "aml.csv"
EXPECTED_SHA256 = "f42328a147f464ac64dbf5b193a6968b488d4e201e40cb210008bcad97774053"

ROWS = [
    (9, 1),
    (13, 1),
    (13, 0),
    (18, 1),
    (23, 1),
    (28, 0),
    (31, 1),
    (34, 1),
    (45, 0),
    (48, 1),
    (161, 0),
    (5, 1),
    (5, 1),
    (8, 1),
    (8, 1),
    (12, 1),
    (16, 0),
    (23, 1),
    (27, 1),
    (30, 1),
    (33, 1),
    (43, 1),
    (45, 1),
]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with CSV_PATH.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["time", "event"])
        writer.writerows(ROWS)

    payload = CSV_PATH.read_bytes()
    actual_sha = hashlib.sha256(payload).hexdigest()
    if actual_sha != EXPECTED_SHA256:
        raise RuntimeError(
            f"aml.csv SHA-256 mismatch: {actual_sha} != {EXPECTED_SHA256}"
        )

    event_count = sum(event for _time, event in ROWS)
    event_times = {time for time, event in ROWS if event == 1}
    if len(ROWS) != 23 or event_count != 18 or len(event_times) != 15:
        raise RuntimeError("aml.csv structural checks failed")


if __name__ == "__main__":
    main()
