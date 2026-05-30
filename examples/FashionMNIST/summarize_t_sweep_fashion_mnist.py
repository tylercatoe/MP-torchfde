#!/usr/bin/env python3
import argparse
import csv
import os
import re
from typing import Dict, List, Optional, Tuple, Union


BENCH_RE = re.compile(
    r"BENCHMARK\|epoch=(?P<epoch>\d+)\|time_s=(?P<time>[-+0-9.eE]+)\|peak_mem_mb=(?P<mem>[-+0-9.eE]+)\|mode=(?P<mode>[^|]+)\|T=(?P<T>[-+0-9.eE]+)"
)
BENCH_LAST_RE = re.compile(
    r"BENCHMARK_LAST\|epoch=(?P<epoch>\d+)\|time_s=(?P<time>[-+0-9.eE]+)\|peak_mem_mb=(?P<mem>[-+0-9.eE]+)\|mode=(?P<mode>[^|]+)\|T=(?P<T>[-+0-9.eE]+)"
)

MODE_ORDER: List[Tuple[str, str]] = [
    ("direct", "dir"),
    ("adjoint", "adj"),
    ("adjoint-mixed", "adj_fl16"),
    ("adjoint-mixed-bfloat", "adj_bfl16"),
]

RATIO_ROWS: List[Tuple[str, str, str]] = [
    ("dir/adj_fl16", "direct", "adjoint-mixed"),
    ("dir/adj_bfl16", "direct", "adjoint-mixed-bfloat"),
    ("adj/adj_fl16", "adjoint", "adjoint-mixed"),
    ("adj/adj_bfl16", "adjoint", "adjoint-mixed-bfloat"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize FashionMNIST T-sweep benchmark logs into memory/time tables.")
    parser.add_argument("--manifest", required=True, help="CSV manifest from submit_t_sweep_fashion_mnist.sh")
    parser.add_argument("--epoch", type=int, default=3, help="Target epoch to extract (typically last epoch)")
    parser.add_argument("--t_values", default="1,2,4,8,16,32,64,128", help="Comma-separated T values")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: manifest directory)")
    parser.add_argument("--output_prefix", default="fashion_mnist_t_sweep", help="Prefix for output files")
    return parser.parse_args()


def parse_t_values(raw: str) -> List[int]:
    out: List[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out


def parse_log_for_epoch(log_path: str, target_epoch: int) -> Optional[Tuple[float, float]]:
    if not os.path.exists(log_path):
        return None

    bench_match: Optional[Tuple[float, float]] = None
    bench_last_match: Optional[Tuple[int, float, float]] = None
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m_last = BENCH_LAST_RE.search(line)
                if m_last:
                    bench_last_match = (
                        int(m_last.group("epoch")),
                        float(m_last.group("time")),
                        float(m_last.group("mem")),
                    )
                    continue

                m = BENCH_RE.search(line)
                if m and int(m.group("epoch")) == target_epoch:
                    bench_match = (float(m.group("time")), float(m.group("mem")))
    except OSError:
        return None

    if bench_last_match is not None and bench_last_match[0] == target_epoch:
        return bench_last_match[1], bench_last_match[2]
    return bench_match


def make_empty_metric_map(t_values: List[int]) -> Dict[str, Dict[int, Union[float, str]]]:
    metrics: Dict[str, Dict[int, Union[float, str]]] = {}
    for mode, _ in MODE_ORDER:
        metrics[mode] = {t: "F" for t in t_values}
    return metrics


def parse_manifest(
    manifest_path: str,
    target_epoch: int,
    t_values: List[int],
) -> Tuple[Dict[str, Dict[int, Union[float, str]]], Dict[str, Dict[int, Union[float, str]]]]:
    mem_metrics = make_empty_metric_map(t_values)
    time_metrics = make_empty_metric_map(t_values)

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mode = row["mode"]
            if mode not in mem_metrics:
                continue

            t_str = row["T"].strip()
            t_val = int(float(t_str))
            if t_val not in mem_metrics[mode]:
                continue

            log_path = row["run_log"].strip()
            parsed = parse_log_for_epoch(log_path, target_epoch)
            if parsed is None:
                continue

            time_s, mem_mb = parsed
            time_metrics[mode][t_val] = time_s
            mem_metrics[mode][t_val] = mem_mb

    return mem_metrics, time_metrics


def ratio_value(num: Union[float, str], den: Union[float, str]) -> Union[float, str]:
    if num == "F" or den == "F":
        return "F"
    assert isinstance(num, float)
    assert isinstance(den, float)
    if den <= 0:
        return "F"
    return num / den


def build_rows(
    metrics: Dict[str, Dict[int, Union[float, str]]],
    t_values: List[int],
) -> List[Tuple[str, Dict[int, Union[float, str]]]]:
    rows: List[Tuple[str, Dict[int, Union[float, str]]]] = []
    for mode, label in MODE_ORDER:
        rows.append((label, {t: metrics[mode][t] for t in t_values}))

    for ratio_label, num_mode, den_mode in RATIO_ROWS:
        ratio_row: Dict[int, Union[float, str]] = {}
        for t in t_values:
            ratio_row[t] = ratio_value(metrics[num_mode][t], metrics[den_mode][t])
        rows.append((ratio_label, ratio_row))

    return rows


def fmt(v: Union[float, str], kind: str) -> str:
    if v == "F":
        return "F"
    assert isinstance(v, float)
    if kind == "ratio":
        return f"{v:.3f}"
    return f"{v:.2f}"


def write_csv_table(
    path: str,
    rows: List[Tuple[str, Dict[int, Union[float, str]]]],
    t_values: List[int],
    kind: str,
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", *[str(t) for t in t_values]])
        for label, values in rows:
            row = [label]
            for t in t_values:
                value_kind = "ratio" if "/" in label else kind
                row.append(fmt(values[t], value_kind))
            writer.writerow(row)


def write_markdown_table(
    path: str,
    title: str,
    rows: List[Tuple[str, Dict[int, Union[float, str]]]],
    t_values: List[int],
    kind: str,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        header = "| method | " + " | ".join(str(t) for t in t_values) + " |\n"
        sep = "|" + "---|" * (len(t_values) + 1) + "\n"
        f.write(header)
        f.write(sep)
        for label, values in rows:
            cells = [label]
            for t in t_values:
                value_kind = "ratio" if "/" in label else kind
                cells.append(fmt(values[t], value_kind))
            f.write("| " + " | ".join(cells) + " |\n")


def main() -> None:
    args = parse_args()
    t_values = parse_t_values(args.t_values)
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.manifest))
    os.makedirs(output_dir, exist_ok=True)

    mem_metrics, time_metrics = parse_manifest(args.manifest, args.epoch, t_values)
    mem_rows = build_rows(mem_metrics, t_values)
    time_rows = build_rows(time_metrics, t_values)

    mem_csv = os.path.join(output_dir, f"{args.output_prefix}_memory.csv")
    time_csv = os.path.join(output_dir, f"{args.output_prefix}_time.csv")
    mem_md = os.path.join(output_dir, f"{args.output_prefix}_memory.md")
    time_md = os.path.join(output_dir, f"{args.output_prefix}_time.md")

    write_csv_table(mem_csv, mem_rows, t_values, kind="mem")
    write_csv_table(time_csv, time_rows, t_values, kind="time")
    write_markdown_table(mem_md, "FashionMNIST T Sweep - Peak Memory (MB)", mem_rows, t_values, kind="mem")
    write_markdown_table(time_md, "FashionMNIST T Sweep - Epoch Time (s)", time_rows, t_values, kind="time")

    print(f"Wrote: {mem_csv}")
    print(f"Wrote: {time_csv}")
    print(f"Wrote: {mem_md}")
    print(f"Wrote: {time_md}")


if __name__ == "__main__":
    main()
