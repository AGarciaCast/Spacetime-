import argparse
import json
import os
from pathlib import Path
from statistics import mean, median
import csv


def _find_latest_run_dir(root: Path) -> Path:
    latest = root / "latest-run"
    if latest.exists():
        return latest
    run_dirs = [p for p in root.glob("run-*") if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No W&B runs found under {root}")
    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return run_dirs[0]


def _find_latest_metrics_csv(root: Path) -> Path:
    candidates = []
    for path in root.rglob("metrics.csv"):
        if path.is_file():
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No metrics.csv found under {root}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _load_csv_rows(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if value is None or value == "":
                    continue
                if key in ("step", "epoch"):
                    try:
                        parsed[key] = int(float(value))
                    except ValueError:
                        continue
                else:
                    try:
                        parsed[key] = float(value)
                    except ValueError:
                        continue
            if "step" in parsed:
                parsed["_step"] = parsed["step"]
            rows.append(parsed)
    return rows


def _load_json(path: Path):
    if not path.exists():
        return {}
    with path.open("r") as f:
        return json.load(f)


def _basic_stats(values):
    if not values:
        return {}
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def _extract_series(rows, key):
    series = []
    for row in rows:
        if key in row and row[key] is not None:
            series.append(row[key])
    return series


def _extract_series_with_steps(rows, key):
    series = []
    for row in rows:
        if key not in row or row[key] is None:
            continue
        step = row.get("_step")
        series.append({"step": step, "value": row[key]})
    return series


def _basic_stats_from_summary(summary, key):
    if key not in summary:
        return {}
    value = summary.get(key)
    if value is None:
        return {}
    return {
        "count": 1,
        "mean": value,
        "median": value,
        "min": value,
        "max": value,
        "source": "summary",
    }


def _suggestions(report):
    suggestions = []
    perf = report.get("perf", {})
    optim = report.get("optim", {})
    loss = report.get("loss", {})
    residual = report.get("residual", {})
    norm_grad = report.get("norm_grad", {})
    velocity = report.get("velocity", {})

    def _ratio(numerator, denominator, default=None):
        if numerator is None or denominator in (None, 0):
            return default
        return numerator / denominator

    grad = optim.get("grad_norm", {})
    grad_ratio = _ratio(grad.get("max"), grad.get("median"))
    if grad_ratio is not None and grad_ratio > 5:
        suggestions.append(
            "Grad norm spikes: max/median={:.2f}. Consider tighter gradient clipping or smaller LR.".format(
                grad_ratio
            )
        )

    step_time = perf.get("step_time_sec", {})
    step_time_mean = step_time.get("mean")
    if step_time_mean is not None:
        if step_time_mean > 0.2:
            suggestions.append(
                "Average step time is {:.3f}s. Consider profiling or reducing batch size.".format(
                    step_time_mean
                )
            )

    val = loss.get("val_eiko_epoch", {})
    train = loss.get("train_eiko_epoch", {})
    gap = _ratio(val.get("median"), train.get("median"))
    if gap is not None and gap > 1.1:
        suggestions.append(
            "Validation/train eikonal gap={:.2f}. Check regularization or sampling.".format(
                gap
            )
        )

    val_res_mean = residual.get("val_residual_mean", {})
    val_res_max = residual.get("val_residual_max", {})
    res_ratio = _ratio(val_res_max.get("max"), val_res_mean.get("mean"))
    if res_ratio is not None and res_ratio > 10:
        suggestions.append(
            "Residual is spiky: val max/mean={:.2f}. Consider residual weighting or curriculum.".format(
                res_ratio
            )
        )

    val_ng_mean = norm_grad.get("val_norm_grad_mean", {})
    val_ng_max = norm_grad.get("val_norm_grad_max", {})
    ng_ratio = _ratio(val_ng_max.get("max"), val_ng_mean.get("mean"))
    if ng_ratio is not None and ng_ratio > 5:
        suggestions.append(
            "Norm-grad spread is high: val max/mean={:.2f}. Check boundary conditions or scaling.".format(
                ng_ratio
            )
        )

    val_vel_mean = velocity.get("val_vel_mean", {})
    val_vel_std = velocity.get("val_vel_std", {})
    vel_cv = _ratio(val_vel_std.get("mean"), val_vel_mean.get("mean"))
    if vel_cv is not None and vel_cv > 0.5:
        suggestions.append(
            "Velocity variation is high: std/mean={:.2f}. Consider stabilizing speed parameterization.".format(
                vel_cv
            )
        )

    lr = optim.get("lr", {})
    lr_mean = lr.get("mean")
    if lr_mean is not None and lr_mean < 1e-6:
        suggestions.append(
            "Learning rate is very small (mean={:.2e}). Consider increasing decay_steps or min_lr.".format(
                lr_mean
            )
        )

    if not suggestions:
        suggestions.append(
            "No strong signals detected from current metrics; consider adding more steps or logging cadence."
        )
    return suggestions


def _format_human_summary(report):
    perf = report.get("perf", {})
    loss = report.get("loss", {})
    optim = report.get("optim", {})
    residual = report.get("residual", {})

    lines = []
    step = perf.get("step_time_sec", {})
    sps = perf.get("samples_per_sec", {})
    lines.append(
        "Performance: step_time_sec mean={:.4f} median={:.4f}, samples_per_sec mean={:.2f}.".format(
            step.get("mean", 0.0), step.get("median", 0.0), sps.get("mean", 0.0)
        )
    )

    train = loss.get("train_eiko_epoch", {})
    val = loss.get("val_eiko_epoch", {})
    lines.append(
        "Loss: train_eiko_epoch mean={:.4f}, val_eiko_epoch mean={:.4f}.".format(
            train.get("mean", 0.0), val.get("mean", 0.0)
        )
    )

    grad = optim.get("grad_norm", {})
    lines.append(
        "Optimization: grad_norm median={:.4f} max={:.4f}.".format(
            grad.get("median", 0.0), grad.get("max", 0.0)
        )
    )
    res = residual.get("val_residual_mean", {})
    if res:
        lines.append(
            "Residual: val_residual_mean mean={:.4f}.".format(res.get("mean", 0.0))
        )
    return " ".join(lines)


def _format_agent_summary(report):
    perf = report.get("perf", {})
    loss = report.get("loss", {})
    optim = report.get("optim", {})
    residual = report.get("residual", {})
    norm_grad = report.get("norm_grad", {})
    velocity = report.get("velocity", {})
    series = report.get("series", {})
    summary = {
        "perf_step_time_mean": perf.get("step_time_sec", {}).get("mean", None),
        "perf_step_time_median": perf.get("step_time_sec", {}).get("median", None),
        "perf_samples_per_sec_mean": perf.get("samples_per_sec", {}).get("mean", None),
        "loss_train_eiko_mean": loss.get("train_eiko_epoch", {}).get("mean", None),
        "loss_val_eiko_mean": loss.get("val_eiko_epoch", {}).get("mean", None),
        "loss_train_mse_mean": loss.get("train_mse_epoch", {}).get("mean", None),
        "loss_val_mse_mean": loss.get("val_mse_step_epoch", {}).get("mean", None),
        "optim_grad_norm_median": optim.get("grad_norm", {}).get("median", None),
        "optim_grad_norm_max": optim.get("grad_norm", {}).get("max", None),
        "optim_lr_mean": optim.get("lr", {}).get("mean", None),
        "optim_param_norm_mean": optim.get("param_norm", {}).get("mean", None),
        "residual_val_mean": residual.get("val_residual_mean", {}).get("mean", None),
        "residual_val_max": residual.get("val_residual_max", {}).get("max", None),
        "norm_grad_val_mean": norm_grad.get("val_norm_grad_mean", {}).get("mean", None),
        "norm_grad_val_max": norm_grad.get("val_norm_grad_max", {}).get("max", None),
        "vel_val_mean": velocity.get("val_vel_mean", {}).get("mean", None),
        "vel_val_std": velocity.get("val_vel_std", {}).get("mean", None),
        "series": series,
    }
    return summary


def analyze(run_dir: Path, out_path: Path, csv_root: Path):
    files_dir = run_dir / "files"
    history_path = files_dir / "wandb-history.jsonl"
    summary_path = files_dir / "wandb-summary.json"
    config_path = files_dir / "wandb-config.json"

    rows = _load_jsonl(history_path)
    if not rows and csv_root is not None:
        try:
            csv_path = _find_latest_metrics_csv(csv_root)
            rows = _load_csv_rows(csv_path)
        except FileNotFoundError:
            rows = []
    summary = _load_json(summary_path)
    config = _load_json(config_path)

    report = {
        "run_dir": str(run_dir),
        "summary": summary,
        "config": config,
        "loss": {
            "train_eiko_epoch": _basic_stats(
                _extract_series(rows, "train_eiko_epoch")
            ),
            "val_eiko_epoch": _basic_stats(_extract_series(rows, "val_eiko_epoch")),
            "train_mse_epoch": _basic_stats(_extract_series(rows, "train_mse_epoch")),
            "val_mse_step_epoch": _basic_stats(
                _extract_series(rows, "val_mse_step_epoch")
            ),
        },
        "perf": {
            "step_time_sec": _basic_stats(_extract_series(rows, "perf/step_time_sec")),
            "samples_per_sec": _basic_stats(_extract_series(rows, "perf/samples_per_sec")),
            "cuda_max_mem_alloc_mb": _basic_stats(
                _extract_series(rows, "perf/cuda_max_mem_alloc_mb")
            ),
            "cuda_max_mem_reserved_mb": _basic_stats(
                _extract_series(rows, "perf/cuda_max_mem_reserved_mb")
            ),
        },
        "optim": {
            "lr": _basic_stats(_extract_series(rows, "optim/lr")),
            "grad_norm": _basic_stats(_extract_series(rows, "optim/grad_norm")),
            "param_norm": _basic_stats(_extract_series(rows, "optim/param_norm")),
        },
        "residual": {
            "train_residual_mean": _basic_stats(
                _extract_series(rows, "train_residual_mean_epoch")
            ),
            "train_residual_max": _basic_stats(
                _extract_series(rows, "train_residual_max_epoch")
            ),
            "val_residual_mean": _basic_stats(
                _extract_series(rows, "val_residual_mean_epoch")
            ),
            "val_residual_max": _basic_stats(
                _extract_series(rows, "val_residual_max_epoch")
            ),
        },
        "norm_grad": {
            "train_norm_grad_mean": _basic_stats(
                _extract_series(rows, "train_norm_grad_mean_epoch")
            ),
            "train_norm_grad_max": _basic_stats(
                _extract_series(rows, "train_norm_grad_max_epoch")
            ),
            "val_norm_grad_mean": _basic_stats(
                _extract_series(rows, "val_norm_grad_mean_epoch")
            ),
            "val_norm_grad_max": _basic_stats(
                _extract_series(rows, "val_norm_grad_max_epoch")
            ),
        },
        "velocity": {
            "train_vel_mean": _basic_stats(
                _extract_series(rows, "train_vel_mean_epoch")
            ),
            "train_vel_std": _basic_stats(_extract_series(rows, "train_vel_std_epoch")),
            "val_vel_mean": _basic_stats(_extract_series(rows, "val_vel_mean_epoch")),
            "val_vel_std": _basic_stats(_extract_series(rows, "val_vel_std_epoch")),
        },
        "series": {
            "train_eiko_epoch": _extract_series_with_steps(rows, "train_eiko_epoch"),
            "val_eiko_epoch": _extract_series_with_steps(rows, "val_eiko_epoch"),
            "train_mse_epoch": _extract_series_with_steps(rows, "train_mse_epoch"),
            "val_mse_step_epoch": _extract_series_with_steps(
                rows, "val_mse_step_epoch"
            ),
            "perf_step_time_sec": _extract_series_with_steps(
                rows, "perf/step_time_sec"
            ),
            "optim_lr": _extract_series_with_steps(rows, "optim/lr"),
            "optim_grad_norm": _extract_series_with_steps(rows, "optim/grad_norm"),
            "train_residual_mean": _extract_series_with_steps(
                rows, "train_residual_mean_epoch"
            ),
            "val_residual_mean": _extract_series_with_steps(
                rows, "val_residual_mean_epoch"
            ),
            "val_norm_grad_mean": _extract_series_with_steps(
                rows, "val_norm_grad_mean_epoch"
            ),
            "val_vel_mean": _extract_series_with_steps(rows, "val_vel_mean_epoch"),
        },
    }

    if not rows:
        report["loss"]["train_eiko_epoch"] = _basic_stats_from_summary(
            summary, "train_eiko_epoch"
        )
        report["loss"]["val_eiko_epoch"] = _basic_stats_from_summary(
            summary, "val_eiko_epoch"
        )
        report["loss"]["train_mse_epoch"] = _basic_stats_from_summary(
            summary, "train_mse_epoch"
        )
        report["loss"]["val_mse_step_epoch"] = _basic_stats_from_summary(
            summary, "val_mse_step_epoch"
        )
        report["perf"]["step_time_sec"] = _basic_stats_from_summary(
            summary, "perf/step_time_sec"
        )
        report["perf"]["samples_per_sec"] = _basic_stats_from_summary(
            summary, "perf/samples_per_sec"
        )
        report["perf"]["cuda_max_mem_alloc_mb"] = _basic_stats_from_summary(
            summary, "perf/cuda_max_mem_alloc_mb"
        )
        report["perf"]["cuda_max_mem_reserved_mb"] = _basic_stats_from_summary(
            summary, "perf/cuda_max_mem_reserved_mb"
        )
        report["optim"]["lr"] = _basic_stats_from_summary(summary, "optim/lr")
        report["optim"]["grad_norm"] = _basic_stats_from_summary(
            summary, "optim/grad_norm"
        )
        report["optim"]["param_norm"] = _basic_stats_from_summary(
            summary, "optim/param_norm"
        )

    report["suggestions"] = _suggestions(report)
    report["human_summary"] = _format_human_summary(report)
    report["agent_summary"] = _format_agent_summary(report)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wandb-root",
        type=str,
        default="wandb",
        help="Path to the local W&B root directory.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Explicit W&B run directory (e.g., wandb/run-*/).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="toy_experiments/PathFinding/analysis/last_report.json",
        help="Output path for the analysis report.",
    )
    parser.add_argument(
        "--csv-root",
        type=str,
        default="toy_experiments/logs",
        help="Root directory to search for Lightning metrics.csv files.",
    )
    args = parser.parse_args()

    wandb_root = Path(args.wandb_root)
    run_dir = Path(args.run_dir) if args.run_dir else _find_latest_run_dir(wandb_root)
    report = analyze(run_dir, Path(args.out), Path(args.csv_root))
    print(f"Analysis report written to {args.out}")
    print(f"Suggestions: {report.get('suggestions')}")


if __name__ == "__main__":
    main()
