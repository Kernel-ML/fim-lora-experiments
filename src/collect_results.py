"""Collect all results.json files into a single CSV + summary table.

Usage:
    uv run python src/collect_results.py --output-dir results --experiment glue
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


GLUE_METRIC = {
    "mnli": "accuracy", "sst2": "accuracy", "cola": "matthews_correlation",
    "mrpc": "f1", "qqp": "f1", "qnli": "accuracy", "stsb": "pearson", "rte": "accuracy",
}


def collect_glue(output_dir: Path) -> pd.DataFrame:
    records = []
    for result_file in output_dir.rglob("results.json"):
        with open(result_file) as f:
            data = json.load(f)
        if "task" not in data or data.get("task") not in GLUE_METRIC:
            continue
        records.append({
            "method": data["method"],
            "task": data["task"],
            "rank": data["rank"],
            "seed": data["seed"],
            "score": data.get("primary_score"),
            "metric": data.get("primary_metric"),
        })

    df = pd.DataFrame(records)
    if df.empty:
        print("No results found.")
        return df

    # Pivot: mean over seeds
    pivot = (
        df.groupby(["method", "task", "rank"])["score"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Main table: avg across tasks per method × rank
    avg = (
        pivot.groupby(["method", "rank"])["mean"]
        .mean()
        .reset_index()
        .rename(columns={"mean": "glue_avg"})
        .pivot(index="method", columns="rank", values="glue_avg")
    )

    print("\n=== GLUE Average Score by Method × Rank ===")
    print(avg.to_string(float_format="{:.4f}".format))

    return df


def collect_commonsense(output_dir: Path) -> pd.DataFrame:
    records = []
    for result_file in (output_dir / "lm_eval").rglob("results.json"):
        with open(result_file) as f:
            data = json.load(f)
        records.append(data)

    if not records:
        print("No commonsense eval results found. Run eval_commonsense.sh first.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    print("\n=== Commonsense Reasoning Results ===")
    print(df.to_string())
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--experiment", choices=["glue", "commonsense", "all"], default="all")
    args = parser.parse_args()

    out = Path(args.output_dir)

    if args.experiment in ("glue", "all"):
        df_glue = collect_glue(out)
        if not df_glue.empty:
            df_glue.to_csv(out / "glue_results.csv", index=False)
            print(f"\nSaved to {out / 'glue_results.csv'}")

    if args.experiment in ("commonsense", "all"):
        df_cs = collect_commonsense(out)
        if not df_cs.empty:
            df_cs.to_csv(out / "commonsense_results.csv", index=False)
