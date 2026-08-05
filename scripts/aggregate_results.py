#!/usr/bin/env python3
"""Aggregate repeated experiment seeds from experiment_summary.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="outputs/experiment_summary.csv")
    parser.add_argument("--profile", default="baseline")
    parser.add_argument("--viscosity-label", default="nu_0p1pi")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    frame = pd.read_csv(summary_path)
    selected = frame[
        (frame["profile"] == args.profile)
        & (frame["viscosity_label"] == args.viscosity_label)
    ].copy()
    if selected.empty:
        raise ValueError("No rows match the requested profile and viscosity label.")

    display_columns = [
        "seed",
        "relative_l2_error",
        "pde_residual_rms",
        "total_training_seconds",
        "trainable_parameters",
    ]
    statistics_columns = [
        "relative_l2_error",
        "pde_residual_rms",
        "total_training_seconds",
    ]

    print("\nPer-seed results\n")
    print(selected[display_columns].to_string(index=False))
    statistics = selected[statistics_columns].agg(["mean", "std", "min", "max"])
    print("\nAggregate statistics\n")
    print(statistics.to_string())

    output = Path(args.output) if args.output else summary_path.with_name(
        f"aggregate_{args.profile}_{args.viscosity_label}.csv"
    )
    statistics.to_csv(output)
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
