#!/usr/bin/env python
"""Create a CSV and Markdown summary from Lotka--Volterra result files."""

import argparse
import csv
import glob
import json
import os
from typing import Dict, List


def load_summary(path: str, threshold: float) -> Dict:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    records = data["iterations"]
    final = records[-1]
    best = min(records, key=lambda item: item["val_loss"])
    threshold_records = [
        item for item in records if item["val_loss"] <= threshold
    ]
    return {
        "mesh": data["mesh"],
        "precision": data["precision"],
        "beta": data["beta"],
        "step_size": data["step_size"],
        "n_train": data["n_train"],
        "n_val": data["n_val"],
        "seed": data["seed"],
        "final_iter": final["iter"],
        "final_train_loss": final["train_loss"],
        "final_val_loss": final["val_loss"],
        "best_val_loss": best["val_loss"],
        "best_val_iter": best["iter"],
        "final_param_err": final["param_err"],
        "peak_mem_mb": max(item["peak_mem_mb"] for item in records),
        "mean_iter_time_s": sum(item["iter_time_s"] for item in records) / len(records),
        "total_logged_time_s": sum(item["iter_time_s"] for item in records),
        "iter_to_val_threshold": (
            threshold_records[0]["iter"] if threshold_records else "NA"
        ),
        "final_params": ", ".join(f"{value:.6f}" for value in final["params"]),
    }


def write_csv(rows: List[Dict], path: str) -> None:
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: List[Dict], path: str, threshold: float) -> None:
    columns = [
        ("Mesh", "mesh"),
        ("Precision", "precision"),
        ("Final train loss", "final_train_loss"),
        ("Final val loss", "final_val_loss"),
        ("Best val loss", "best_val_loss"),
        ("Final parameter error", "final_param_err"),
        ("Peak memory (MB)", "peak_mem_mb"),
        ("Mean logged iteration time (s)", "mean_iter_time_s"),
        (f"Iteration to val loss ≤ {threshold:g}", "iter_to_val_threshold"),
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Lotka--Volterra Uniform versus Graded Mesh\n\n")
        handle.write(
            "Both mesh conditions use the predictor--corrector recurrence. "
            "The data, initialization, beta, optimizer, and step size are shared.\n\n"
        )
        handle.write("| " + " | ".join(label for label, _ in columns) + " |\n")
        handle.write("| " + " | ".join("---" for _ in columns) + " |\n")
        for row in rows:
            values = []
            for _, key in columns:
                value = row[key]
                if isinstance(value, float):
                    value = f"{value:.6g}"
                values.append(str(value))
            handle.write("| " + " | ".join(values) + " |\n")

        handle.write("\nFinal learned parameters:\n\n")
        for row in rows:
            handle.write(
                f"- `{row['mesh']}_{row['precision']}`: "
                f"[{row['final_params']}]\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="*", help="paths to results.json files")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--val-threshold", type=float, default=1e-3)
    args = parser.parse_args()

    paths = args.results or sorted(glob.glob(os.path.join(args.output_dir, "*/results.json")))
    if len(paths) < 2:
        parser.error("provide at least two results.json files")

    rows = [load_summary(path, args.val_threshold) for path in paths]
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "comparison.csv")
    markdown_path = os.path.join(args.output_dir, "comparison.md")
    write_csv(rows, csv_path)
    write_markdown(rows, markdown_path, args.val_threshold)

    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")
    print("\nMesh/precision comparison:")
    for row in rows:
        print(
            f"  {row['mesh']:>7} {row['precision']:>4} | "
            f"val={row['final_val_loss']:.6e} | "
            f"param_err={row['final_param_err']:.6e} | "
            f"peak_MB={row['peak_mem_mb']:.1f}"
        )


if __name__ == "__main__":
    main()
