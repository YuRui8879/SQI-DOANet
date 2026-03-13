import argparse
import csv
import json
from pathlib import Path

import numpy as np

from SQI_DOANetTest import load_case_from_mat, parse_split_file


def safe_corr(x, y):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def safe_mean(vals):
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return float("nan")
    return float(np.mean(vals))


def parse_prediction_file(pred_file: Path):
    rows = []
    with pred_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if len(rows) == 0:
        return np.array([]), np.array([]), np.array([])

    def col(name, default=np.nan):
        out = []
        for r in rows:
            v = r.get(name, "")
            if v == "":
                out.append(default)
            else:
                out.append(float(v))
        return np.array(out, dtype=np.float32)

    bis_pred = col("bis_pred")
    sqi_pred = col("sqi_pred")
    bis_masked = col("bis_pred_masked")
    return bis_pred, sqi_pred, bis_masked


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SQI-DOANet prediction files.")
    parser.add_argument("--split-file", required=True, help="Path to split txt. Uses [test] section.")
    parser.add_argument("--data-dir", default="mat", help="Directory with case*.mat.")
    parser.add_argument("--pred-dir", default="predictions", help="Directory containing per-case prediction csv.")
    parser.add_argument("--output-dir", default="predictions", help="Directory to write evaluation outputs.")
    parser.add_argument("--sqi-th", type=float, default=20.0, help="SQI threshold for validity accuracy.")
    parser.add_argument("--use-masked-bis", action="store_true", help="Use bis_pred_masked for BIS metrics.")
    return parser.parse_args()


def main():
    args = parse_args()
    split_file = Path(args.split_file)
    pred_dir = Path(args.pred_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not split_file.exists():
        raise FileNotFoundError(f"split file not found: {split_file}")
    if not pred_dir.exists():
        raise FileNotFoundError(f"pred dir not found: {pred_dir}")

    sections = parse_split_file(split_file)
    test_set = sections.get("test", [])
    if len(test_set) == 0:
        raise ValueError("No [test] entries in split file.")

    per_case = []

    for case_item in test_set:
        case_name = Path(case_item).stem
        mat_path = data_dir / case_item
        if not mat_path.exists():
            mat_path = data_dir / f"{case_name}.mat"
        if not mat_path.exists():
            raise FileNotFoundError(f"mat file not found for case: {case_item}")

        pred_file = pred_dir / f"{case_name}.csv"
        if not pred_file.exists():
            raise FileNotFoundError(f"prediction file not found: {pred_file}")

        _, _, bis_gt, sqi_gt = load_case_from_mat(mat_path)
        bis_gt = np.where(np.isnan(np.array(bis_gt)), 0, bis_gt).astype(np.float32)
        sqi_gt = np.where(np.isnan(np.array(sqi_gt)), 0, sqi_gt).astype(np.float32)

        bis_pred, sqi_pred, bis_masked = parse_prediction_file(pred_file)
        bis_used = bis_masked if args.use_masked_bis else bis_pred

        n = int(min(len(bis_used), len(bis_gt)))
        if n == 0:
            continue
        bis_used = bis_used[:n]
        bis_gt_n = bis_gt[:n]

        bis_mse = float(np.mean((bis_used - bis_gt_n) ** 2))
        bis_mae = float(np.mean(np.abs(bis_used - bis_gt_n)))
        bis_corr = safe_corr(bis_used, bis_gt_n)

        sqi_n = int(min(len(sqi_pred), len(sqi_gt)))
        if sqi_n > 0:
            sqi_pred_n = sqi_pred[:sqi_n]
            sqi_gt_n = sqi_gt[:sqi_n]
            sqi_mae = float(np.mean(np.abs(sqi_pred_n - sqi_gt_n)))
            sqi_corr = safe_corr(sqi_pred_n, sqi_gt_n)
            sqi_acc_exact = float(np.mean(np.isclose(sqi_pred_n, sqi_gt_n)))
            sqi_valid_acc = float(
                np.mean((sqi_pred_n > args.sqi_th) == (sqi_gt_n > args.sqi_th))
            )
        else:
            sqi_mae = float("nan")
            sqi_corr = float("nan")
            sqi_acc_exact = float("nan")
            sqi_valid_acc = float("nan")

        per_case.append(
            {
                "case": case_name,
                "n_bis": n,
                "bis_mse": bis_mse,
                "bis_mae": bis_mae,
                "bis_corr": bis_corr,
                "n_sqi": sqi_n,
                "sqi_mae": sqi_mae,
                "sqi_corr": sqi_corr,
                "sqi_acc_exact": sqi_acc_exact,
                "sqi_valid_acc": sqi_valid_acc,
            }
        )

    if len(per_case) == 0:
        raise RuntimeError("No valid cases were evaluated.")

    summary = {
        "num_cases": len(per_case),
        "mean_bis_mse": safe_mean([c["bis_mse"] for c in per_case]),
        "mean_bis_mae": safe_mean([c["bis_mae"] for c in per_case]),
        "mean_bis_corr": safe_mean([c["bis_corr"] for c in per_case]),
        "mean_sqi_mae": safe_mean([c["sqi_mae"] for c in per_case]),
        "mean_sqi_corr": safe_mean([c["sqi_corr"] for c in per_case]),
        "mean_sqi_acc_exact": safe_mean([c["sqi_acc_exact"] for c in per_case]),
        "mean_sqi_valid_acc": safe_mean([c["sqi_valid_acc"] for c in per_case]),
    }

    per_case_file = output_dir / "metrics_per_case.csv"
    with per_case_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "n_bis",
                "bis_mse",
                "bis_mae",
                "bis_corr",
                "n_sqi",
                "sqi_mae",
                "sqi_corr",
                "sqi_acc_exact",
                "sqi_valid_acc",
            ],
        )
        writer.writeheader()
        for row in per_case:
            writer.writerow(row)

    summary_file = output_dir / "metrics_summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Evaluation summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"Saved: {per_case_file}")
    print(f"Saved: {summary_file}")


if __name__ == "__main__":
    main()
