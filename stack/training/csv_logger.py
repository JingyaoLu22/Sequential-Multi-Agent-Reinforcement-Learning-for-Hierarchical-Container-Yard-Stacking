"""Lightweight local CSV next to W&B, which logs every statistic."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional, Sequence


class CsvLogger:
    """
    Append one row per log() call to ``path``, keeping only ``columns``
    (the first of which must be "timesteps").

    A fresh run (resume_step=None) starts a new file. A resumed run keeps
    the rows up to resume_step and drops rows an interrupted run logged
    after its last checkpoint, which the resumed run will log again.
    """

    def __init__(self, path: Path, columns: Sequence[str], resume_step: Optional[int] = None) -> None:
        self.path = Path(path)
        self.columns = list(columns)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        if resume_step is not None and self.path.exists():
            with self.path.open(newline="", encoding="utf-8") as file:
                rows = [row for row in csv.DictReader(file) if float(row["timesteps"]) <= resume_step]

        self.n_rows = 0
        self._write(rows, mode="w")

    def log(self, row: Dict) -> None:
        self._write([row], mode="a")

    def _write(self, rows, mode: str) -> None:
        with self.path.open(mode, newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.columns, extrasaction="ignore")
            if mode == "w":
                writer.writeheader()
            writer.writerows(rows)
        self.n_rows += len(rows)
