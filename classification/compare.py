from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Compare completed model runs")
    p.add_argument("--output-dir", default="outputs")
    args = p.parse_args(); root = Path(args.output_dir)
    files = [root / name / "metrics.json" for name in ("resnet18", "convnext_tiny")]
    missing = [str(x) for x in files if not x.exists()]
    if missing:
        raise FileNotFoundError("Run both experiments first. Missing: " + ", ".join(missing))
    rows = [json.loads(x.read_text(encoding="utf-8")) for x in files]
    df = pd.DataFrame(rows)
    columns = ["model", "test_accuracy", "test_macro_f1", "parameters", "latency_ms_batch1"]
    table = df[columns].copy(); table["parameters_m"] = table.pop("parameters") / 1e6
    table.to_csv(root / "comparison.csv", index=False)
    ax = table.set_index("model")[["test_accuracy", "test_macro_f1"]].plot.bar(rot=0, ylim=(0, 1), figsize=(7, 4))
    ax.set_ylabel("Score"); ax.figure.tight_layout(); ax.figure.savefig(root / "comparison.png", dpi=180); plt.close(ax.figure)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

