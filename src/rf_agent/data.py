"""Synthetic RF test data: ACLR measured per firmware build and temperature.

The dataset has a known, planted ground truth so the agent can be evaluated objectively:
  * build 25.3.0 regresses ACLR by ~5 dB at 60 C only
  * build 25.3.1 partially fixes it (~2 dB residual at 60 C)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BUILDS: list[str] = ["25.1.0", "25.2.0", "25.3.0", "25.3.1"]   # oldest -> newest
TEMPS_C: list[int] = [-20, 25, 60]
SPEC_LIMIT_DBC: float = -30.0        # pass if aclr_dbc <= limit
DEVICES_PER_CELL: int = 20

# Planted defects: (build, temperature) -> dB added to ACLR (higher = worse)
PLANTED_DEFECTS: dict[tuple[str, int], float] = {
    ("25.3.0", 60): 5.0,
    ("25.3.1", 60): 2.0,
}


@dataclass(frozen=True)
class RFDataset:
    df: pd.DataFrame
    builds: list[str]
    temps_c: list[int]
    spec_limit_dbc: float


def generate_dataset(seed: int = 7) -> RFDataset:
    """Generate the build x temperature ACLR table (one row per device measurement)."""
    rng = np.random.default_rng(seed)
    rows = []
    for build in BUILDS:
        for temp in TEMPS_C:
            nominal = -36.0 + 0.03 * (temp - 25)            # mild thermal slope
            defect = PLANTED_DEFECTS.get((build, temp), 0.0)
            for device in range(DEVICES_PER_CELL):
                rows.append({
                    "device_id": f"D{device:03d}",
                    "build": build,
                    "temperature_c": temp,
                    "aclr_dbc": round(nominal + defect + rng.normal(0, 0.8), 2),
                })
    return RFDataset(pd.DataFrame(rows), list(BUILDS), list(TEMPS_C), SPEC_LIMIT_DBC)
