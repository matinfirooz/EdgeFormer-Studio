"""Compatibility CSV append utility with schema checks."""

import csv
from pathlib import Path


def append_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    if not write_header:
        with path.open(newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))
        if header != list(row):
            raise ValueError("CSV column names/order changed")
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
