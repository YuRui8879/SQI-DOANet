import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import scipy.io as scio

from vitaldb import load_trk


def parse_args():
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Download VitalDB tracks by case and save each case as a .mat file."
    )
    parser.add_argument(
        "--trks-csv",
        default=str(base_dir / "trks.csv"),
        help="Path to trks.csv (default: download/trks.csv).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=str(base_dir / "mat"),
        help="Directory to store output .mat files (default: download/mat).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of worker threads (default: 10).",
    )
    return parser.parse_args()


def case_sort_key(case_id):
    return (0, int(case_id)) if case_id.isdigit() else (1, case_id)


def build_download_queue(trks_csv, output_dir):
    cases = {}
    with trks_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = row.get("caseid", "").strip()
            track_name = row.get("tname", "").strip()
            tid = row.get("tid", "").strip()

            if not case_id or not tid:
                continue

            if case_id not in cases:
                cases[case_id] = {"caseid": case_id, "list_name": [], "tid": []}
            cases[case_id]["list_name"].append(track_name)
            cases[case_id]["tid"].append(tid)

    queue = []
    for case_id in sorted(cases.keys(), key=case_sort_key):
        output_file = output_dir / f"case{case_id}.mat"
        if not output_file.exists():
            queue.append(cases[case_id])
    return queue


def download_case_data(case_item, output_dir):
    case_id = case_item["caseid"]
    print(f"[START] case{case_id}")

    data_list = []
    time_list = []
    name_list = []

    for track_name, tid in zip(case_item["list_name"], case_item["tid"]):
        time_index, data = load_trk(tid)
        print(f"  case{case_id} | {track_name} | {tid}")
        data_list.append(data)
        time_list.append(time_index)
        name_list.append(track_name)

    output_file_path = output_dir / f"case{case_id}.mat"
    scio.savemat(
        str(output_file_path),
        {"name": name_list, "data": data_list, "time": time_list, "tid": case_item["tid"]},
    )
    print(f"[DONE]  {output_file_path}")


def main():
    args = parse_args()
    trks_csv = Path(args.trks_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not trks_csv.exists():
        raise FileNotFoundError(f"trks.csv not found: {trks_csv}")

    item_download = build_download_queue(trks_csv, output_dir)
    print(f"Cases to download: {len(item_download)}")

    if not item_download:
        return

    with ThreadPoolExecutor(max_workers=max(1, args.workers), thread_name_prefix="vitaldb") as pool:
        futures = {
            pool.submit(download_case_data, case_item, output_dir): case_item["caseid"]
            for case_item in item_download
        }
        for future in as_completed(futures):
            case_id = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"[ERROR] case{case_id}: {exc}")


if __name__ == "__main__":
    main()
