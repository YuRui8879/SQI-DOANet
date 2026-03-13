import argparse
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create 5-fold split txt files from .mat files under a directory."
    )
    parser.add_argument(
        "--mat-dir",
        default="mat",
        help="Directory containing case*.mat files (default: mat).",
    )
    parser.add_argument(
        "--output-dir",
        default="splits",
        help="Directory to save split txt files (default: splits).",
    )
    parser.add_argument(
        "--k-fold",
        type=int,
        default=5,
        help="Number of folds (default: 5).",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.2,
        help="Validation ratio from non-test data in each fold (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    return parser.parse_args()


def split_evenly(items, k):
    n = len(items)
    base = n // k
    rem = n % k
    folds = []
    start = 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        folds.append(items[start : start + size])
        start += size
    return folds


def split_train_valid(train_val, valid_ratio, seed):
    shuffled = train_val[:]
    random.Random(seed).shuffle(shuffled)

    if len(shuffled) <= 1 or valid_ratio <= 0:
        return shuffled, []

    n_valid = int(len(shuffled) * valid_ratio)
    n_valid = max(1, n_valid)
    n_valid = min(n_valid, len(shuffled) - 1)

    valid = shuffled[:n_valid]
    train = shuffled[n_valid:]
    return train, valid


def write_fold_file(path, fold_id, train_list, valid_list, test_list, total_size):
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# fold: {fold_id}\n")
        f.write(f"# total_cases: {total_size}\n")
        f.write(f"# train: {len(train_list)}\n")
        f.write(f"# valid: {len(valid_list)}\n")
        f.write(f"# test: {len(test_list)}\n\n")

        f.write("[train]\n")
        for item in train_list:
            f.write(f"{item}\n")

        f.write("\n[valid]\n")
        for item in valid_list:
            f.write(f"{item}\n")

        f.write("\n[test]\n")
        for item in test_list:
            f.write(f"{item}\n")


def main():
    args = parse_args()
    mat_dir = Path(args.mat_dir)
    output_dir = Path(args.output_dir)

    if args.k_fold < 2:
        raise ValueError("--k-fold must be >= 2.")
    if not (0 <= args.valid_ratio < 1):
        raise ValueError("--valid-ratio must be in [0, 1).")
    if not mat_dir.exists():
        raise FileNotFoundError(f"mat directory not found: {mat_dir}")

    files = sorted([p.name for p in mat_dir.glob("case*.mat") if p.is_file()])
    if len(files) < args.k_fold:
        raise ValueError(
            f"Need at least {args.k_fold} .mat files, but found {len(files)} in {mat_dir}."
        )

    random.Random(args.seed).shuffle(files)
    folds = split_evenly(files, args.k_fold)

    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.k_fold):
        test_set = folds[i]
        train_val = []
        for j, fold_items in enumerate(folds):
            if j != i:
                train_val.extend(fold_items)

        train_set, valid_set = split_train_valid(
            train_val, args.valid_ratio, seed=args.seed + i + 1
        )

        out_file = output_dir / f"fold_{i + 1}.txt"
        write_fold_file(
            out_file,
            fold_id=i + 1,
            train_list=train_set,
            valid_list=valid_set,
            test_list=test_set,
            total_size=len(files),
        )

        print(
            f"fold_{i + 1}.txt -> train: {len(train_set)}, valid: {len(valid_set)}, test: {len(test_set)}"
        )


if __name__ == "__main__":
    main()
