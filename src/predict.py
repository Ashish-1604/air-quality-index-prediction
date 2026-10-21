"""Command-line single and batch inference for AQI readings."""
import argparse
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "aqi_xgboost_model.joblib"


def predict(model_path, readings):
    artifact = joblib.load(model_path)
    frame = pd.DataFrame(readings, columns=artifact["features"])
    return artifact["pipeline"].predict(frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL, help="Saved model artifact")
    parser.add_argument("--input", type=Path, help="CSV with pollutant and atmospheric feature columns")
    parser.add_argument("--pm25", type=float, default=35)
    parser.add_argument("--pm10", type=float, default=80)
    parser.add_argument("--no2", type=float, default=40)
    parser.add_argument("--so2", type=float, default=15)
    parser.add_argument("--co", type=float, default=0.8)
    parser.add_argument("--o3", type=float, default=60)
    args = parser.parse_args()
    if not args.model.exists():
        parser.error(f"Model not found at {args.model}; run python src/train.py first")
    if args.input:
        readings = pd.read_csv(args.input)
        artifact = joblib.load(args.model)
        missing = set(artifact["features"]) - set(readings.columns)
        if missing:
            parser.error(f"Input CSV missing columns: {', '.join(sorted(missing))}")
        readings["predicted_AQI"] = artifact["pipeline"].predict(readings[artifact["features"]])
        output = args.input.with_name(f"{args.input.stem}_predictions.csv")
        readings.to_csv(output, index=False)
        print(f"Saved {len(readings)} predictions to {output}")
    else:
        row = [[args.pm25, args.pm10, args.no2, args.so2, args.co, args.o3]]
        print(f"Predicted AQI: {predict(args.model, row)[0]:.1f}")


if __name__ == "__main__":
    main()
