"""Generate a reproducible synthetic CPCB-style air quality dataset."""
from pathlib import Path

import numpy as np
import pandas as pd


BREAKPOINTS = {
    # (concentration low, concentration high, AQI low, AQI high)
    "PM2.5": [(0, 30, 0, 50), (30, 60, 50, 100), (60, 90, 100, 200), (90, 120, 200, 300), (120, 250, 300, 400), (250, 500, 400, 500)],
    "PM10": [(0, 50, 0, 50), (50, 100, 50, 100), (100, 250, 100, 200), (250, 350, 200, 300), (350, 430, 300, 400), (430, 600, 400, 500)],
    "NO2": [(0, 40, 0, 50), (40, 80, 50, 100), (80, 180, 100, 200), (180, 280, 200, 300), (280, 400, 300, 400), (400, 1000, 400, 500)],
    "SO2": [(0, 40, 0, 50), (40, 80, 50, 100), (80, 380, 100, 200), (380, 800, 200, 300), (800, 1600, 300, 400), (1600, 2000, 400, 500)],
    # CO is measured in mg/m3; O3 and other gases in ug/m3.
    "CO": [(0, 1, 0, 50), (1, 2, 50, 100), (2, 10, 100, 200), (10, 17, 200, 300), (17, 34, 300, 400), (34, 50, 400, 500)],
    "O3": [(0, 50, 0, 50), (50, 100, 50, 100), (100, 168, 100, 200), (168, 208, 200, 300), (208, 748, 300, 400), (748, 1000, 400, 500)],
}


def sub_index(value: float, pollutant: str) -> float:
    """Linear interpolation within the CPCB concentration/AQI breakpoints."""
    for c_low, c_high, i_low, i_high in BREAKPOINTS[pollutant]:
        if value <= c_high:
            return float(np.clip((i_high - i_low) / (c_high - c_low) * (value - c_low) + i_low, 0, 500))
    return 500.0


def main() -> None:
    rng = np.random.default_rng(42)
    n = 2000
    # A shared pollution burden creates realistic cross-pollutant correlation.
    burden = rng.lognormal(mean=0.15, sigma=0.65, size=n)
    season = rng.uniform(0.75, 1.35, n)
    data = pd.DataFrame({
        "PM2.5": np.clip(rng.lognormal(3.35, 0.75, n) * burden ** 0.22 * season, 2, 450),
        "PM10": np.clip(rng.lognormal(4.05, 0.62, n) * burden ** 0.20 * season, 5, 550),
        "NO2": np.clip(rng.lognormal(3.35, 0.55, n) * burden ** 0.18, 2, 450),
        "SO2": np.clip(rng.lognormal(2.55, 0.70, n) * burden ** 0.16, 1, 350),
        "CO": np.clip(rng.lognormal(-0.15, 0.62, n) * burden ** 0.12, 0.1, 15),
        "O3": np.clip(rng.lognormal(3.85, 0.45, n) / burden ** 0.08, 5, 220),
        "Temperature": np.clip(rng.normal(25, 7, n), -5, 46),
        "Humidity": np.clip(rng.normal(58, 17, n), 12, 100),
    })
    # CPCB-style composite AQI is the maximum valid pollutant sub-index.
    scores = np.column_stack([[sub_index(v, p) for v in data[p]] for p in BREAKPOINTS])
    data["AQI"] = np.rint(scores.max(axis=1)).astype(int)
    # Small realistic sensor gaps; the target remains computed from full readings.
    for col in data.columns[:-1]:
        data.loc[rng.choice(n, size=round(n * 0.015), replace=False), col] = np.nan
    path = Path(__file__).resolve().parent / "air_quality_data.csv"
    data.to_csv(path, index=False)
    print(f"Generated {len(data):,} rows at {path}")


if __name__ == "__main__":
    main()
