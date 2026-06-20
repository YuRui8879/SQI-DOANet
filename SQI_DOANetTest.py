import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from DOANetTrain import DoubleMRANet
from SQINetTrain import SQINet


def parse_split_file(split_file: Path):
    sections = {"train": [], "valid": [], "test": []}
    current = None
    with split_file.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                tag = line[1:-1].strip().lower()
                current = tag if tag in sections else None
                continue
            if current is None:
                continue
            sections[current].append(line)
    return sections


def choose_device(device_arg: str):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_1d(x):
    arr = np.asarray(x)
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return np.array([float(arr)], dtype=np.float32)
    return arr.reshape(-1).astype(np.float32)


def _unwrap(x):
    v = x
    while isinstance(v, np.ndarray) and v.size == 1:
        v = v.reshape(-1)[0]
    return v


def _get_from_obj(obj, key):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key, None)
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, np.ndarray) and obj.dtype.names is not None and key in obj.dtype.names:
        return obj[key]
    return None


def _try_keys(obj, keys):
    for key in keys:
        v = _get_from_obj(obj, key)
        if v is not None:
            return v
    return None


def _load_from_ref(ref):
    ref_obj = _unwrap(ref)

    eeg1 = _try_keys(ref_obj, ["EEG1", "eeg1", "eeg_1", "sig1", "signal1"])
    eeg2 = _try_keys(ref_obj, ["EEG2", "eeg2", "eeg_2", "sig2", "signal2"])
    bis = _try_keys(ref_obj, ["BIS", "bis", "label", "y"])
    sqi = _try_keys(ref_obj, ["SQI", "sqi"])

    if bis is None:
        try:
            bis = ref_obj[2][0]
        except Exception:
            bis = None
    if sqi is None:
        try:
            sqi = ref_obj[5][0]
        except Exception:
            sqi = None
    if eeg1 is None:
        try:
            eeg1 = ref_obj[0][0]
        except Exception:
            eeg1 = None
    if eeg2 is None:
        try:
            eeg2 = ref_obj[1][0]
        except Exception:
            eeg2 = None
    return eeg1, eeg2, bis, sqi


def _load_from_named_tracks(mat):
    if "name" not in mat or "data" not in mat:
        return None, None, None, None
    names_raw = np.asarray(mat["name"]).reshape(-1)
    data_raw = np.asarray(mat["data"]).reshape(-1)
    if len(names_raw) == 0 or len(data_raw) == 0:
        return None, None, None, None

    def find_track(keyword_list):
        for i, n in enumerate(names_raw):
            name = str(_unwrap(n)).lower()
            if any(k in name for k in keyword_list):
                if i < len(data_raw):
                    return data_raw[i]
        return None

    eeg1 = find_track(["eeg1", "eeg_1", "eeg ch1", "eeg ch 1"])
    eeg2 = find_track(["eeg2", "eeg_2", "eeg ch2", "eeg ch 2"])
    bis = find_track(["bis"])
    sqi = find_track(["sqi"])
    return eeg1, eeg2, bis, sqi


def load_case_from_mat(mat_path: Path):
    from scipy.io import loadmat

    mat = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)

    eeg1 = eeg2 = bis = sqi = None
    if "ref" in mat:
        eeg1, eeg2, bis, sqi = _load_from_ref(mat["ref"])

    if eeg1 is None or eeg2 is None or bis is None or sqi is None:
        ne1, ne2, nbis, nsqi = _load_from_named_tracks(mat)
        eeg1 = eeg1 if eeg1 is not None else ne1
        eeg2 = eeg2 if eeg2 is not None else ne2
        bis = bis if bis is not None else nbis
        sqi = sqi if sqi is not None else nsqi

    if eeg1 is None and "EEG1" in mat:
        eeg1 = mat["EEG1"]
    if eeg2 is None and "EEG2" in mat:
        eeg2 = mat["EEG2"]
    if bis is None and "BIS" in mat:
        bis = mat["BIS"]
    if sqi is None and "SQI" in mat:
        sqi = mat["SQI"]

    if eeg1 is None or eeg2 is None or bis is None or sqi is None:
        keys = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(f"Cannot parse EEG1/EEG2/BIS/SQI in {mat_path}. keys={keys}")

    eeg1 = _to_1d(_unwrap(eeg1))
    eeg2 = _to_1d(_unwrap(eeg2))
    bis = _to_1d(_unwrap(bis))
    sqi = _to_1d(_unwrap(sqi))
    eeg1 = np.where(np.isnan(eeg1), 0, eeg1)
    eeg2 = np.where(np.isnan(eeg2), 0, eeg2)
    return eeg1, eeg2, bis, sqi


def norm(x):
    return x - np.mean(x)


def schmidt_spike_removal(original_signal, fs):
    window_size = fs
    mod_num = np.mod(len(original_signal), window_size)
    original_signal = np.array(original_signal)
    original_signal = original_signal[: len(original_signal) - mod_num]

    segment_signal = np.reshape(original_signal, (window_size, int(len(original_signal) // window_size)))
    mean_segment_signal = np.mean(segment_signal, 1)
    std_segment_signal = np.std(segment_signal, 1)

    mean_segment_signal = mean_segment_signal[:, np.newaxis]
    std_segment_signal = std_segment_signal[:, np.newaxis]

    high_th = mean_segment_signal + 3 * std_segment_signal
    low_th = mean_segment_signal - 3 * std_segment_signal
    segment_signal = np.where(segment_signal > high_th, mean_segment_signal, segment_signal)
    segment_signal = np.where(segment_signal < low_th, mean_segment_signal, segment_signal)

    segment_signal = segment_signal.reshape(-1)
    despiked_signal = np.hstack((segment_signal, original_signal[len(original_signal) - mod_num :]))
    return despiked_signal


def build_case_windows(eeg1, eeg2, bis, sqi, fs=128, win_len=30):
    eeg1 = norm(schmidt_spike_removal(eeg1, fs))
    eeg2 = norm(schmidt_spike_removal(eeg2, fs))

    n = len(bis)
    if len(sqi) < n:
        pad = np.zeros((n - len(sqi),), dtype=np.float32)
        sqi = np.hstack((sqi, pad))
    elif len(sqi) > n:
        sqi = sqi[:n]

    windows = []
    for i in range(n):
        eeg1_seg = eeg1[max(0, (i - win_len + 1) * fs) : min(len(eeg1), (i + 1) * fs)]
        eeg2_seg = eeg2[max(0, (i - win_len + 1) * fs) : min(len(eeg2), (i + 1) * fs)]
        seg = np.stack([eeg1_seg, eeg2_seg])
        target_len = win_len * fs
        if seg.shape[1] != target_len:
            padding = np.zeros((2, target_len - seg.shape[1]))
            seg = np.hstack((padding, seg))
        windows.append(seg)

    windows = np.array(windows, dtype=np.float32)
    bis = np.where(np.isnan(np.array(bis)), 0, bis).astype(np.float32)
    sqi = np.where(np.isnan(np.array(sqi)), 0, sqi).astype(np.float32)
    return windows, bis, sqi


class SampleDataset(Dataset):
    def __init__(self, x):
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        return self.x[idx]

    def __len__(self):
        return len(self.x)


def quadratic_smooth5(sig):
    sig = np.asarray(sig, dtype=np.float32).reshape(-1)
    n = len(sig)
    if n < 5:
        return sig.copy()

    out = []
    out.append((31.0 * sig[0] + 9.0 * sig[1] - 3.0 * sig[2] - 5.0 * sig[3] + 3.0 * sig[4]) / 35.0)
    out.append((9.0 * sig[0] + 13.0 * sig[1] + 12 * sig[2] + 6.0 * sig[3] - 5.0 * sig[4]) / 35.0)
    for i in range(2, n - 2):
        out.append((-3.0 * (sig[i - 2] + sig[i + 2]) + 12.0 * (sig[i - 1] + sig[i + 1]) + 17 * sig[i]) / 35.0)
    out.append((9.0 * sig[n - 1] + 13.0 * sig[n - 2] + 12.0 * sig[n - 3] + 6.0 * sig[n - 4] - 5.0 * sig[n - 5]) / 35.0)
    out.append((31.0 * sig[n - 1] + 9.0 * sig[n - 2] - 3.0 * sig[n - 3] - 5.0 * sig[n - 4] + 3.0 * sig[n - 5]) / 35.0)
    return np.array(out, dtype=np.float32)


def _load_state_flexible(model, ckpt_path: Path, device):
    ckpt = torch.load(str(ckpt_path), map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    try:
        model.load_state_dict(state)
        return
    except Exception:
        pass

    keys = list(state.keys())
    if len(keys) == 0:
        raise RuntimeError(f"Empty checkpoint: {ckpt_path}")

    if keys[0].startswith("module."):
        new_state = {k[len("module."):]: v for k, v in state.items()}
    else:
        new_state = {f"module.{k}": v for k, v in state.items()}

    model.load_state_dict(new_state, strict=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Run combined DOANet+SQINet inference and output predictions only.")
    parser.add_argument("--split-file", required=True, help="Path to split txt. Uses [test] section.")
    parser.add_argument("--data-dir", default="mat", help="Directory with case*.mat.")
    parser.add_argument("--doanet-model", required=True, help="Path to DOANet model checkpoint.")
    parser.add_argument("--sqinet-model", required=True, help="Path to SQINet model checkpoint.")
    parser.add_argument("--output-dir", default="predictions", help="Output directory for per-case prediction files.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--fs", type=int, default=128)
    parser.add_argument("--win-len", type=int, default=30)
    parser.add_argument("--sqi-th", type=float, default=20.0, help="SQI threshold for masking BIS prediction.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data-parallel", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    split_file = Path(args.split_file)
    if not split_file.exists():
        raise FileNotFoundError(f"split file not found: {split_file}")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sections = parse_split_file(split_file)
    test_set = sections.get("test", [])
    if len(test_set) == 0:
        raise ValueError("No [test] entries in split file.")

    device = choose_device(args.device)

    doanet = DoubleMRANet(input_len=args.fs * args.win_len)
    sqinet = SQINet()
    if args.data_parallel and device.type == "cuda":
        doanet = torch.nn.DataParallel(doanet)
        sqinet = torch.nn.DataParallel(sqinet)
    doanet = doanet.to(device)
    sqinet = sqinet.to(device)
    _load_state_flexible(doanet, Path(args.doanet_model), device)
    _load_state_flexible(sqinet, Path(args.sqinet_model), device)
    doanet.eval()
    sqinet.eval()

    for case_item in test_set:
        case_name = Path(case_item).stem
        mat_path = data_dir / case_item
        if not mat_path.exists():
            mat_path = data_dir / f"{case_name}.mat"
        if not mat_path.exists():
            raise FileNotFoundError(f"mat file not found for case: {case_item}")

        eeg1, eeg2, bis_gt, sqi_gt = load_case_from_mat(mat_path)
        x, _, _ = build_case_windows(eeg1, eeg2, bis_gt, sqi_gt, fs=args.fs, win_len=args.win_len)
        loader = DataLoader(SampleDataset(x), batch_size=args.batch_size, shuffle=False, num_workers=0)

        bis_pred_raw = []
        sqi_pred_raw = np.zeros(0, dtype=np.float32)
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                do_pred = doanet(batch[:, 0, :], batch[:, 1, :]).squeeze()
                sqi_pred = sqinet(batch[:, 0, :], batch[:, 1, :]).squeeze()

                do_np = do_pred.detach().cpu().numpy()
                sqi_np = sqi_pred.detach().cpu().numpy()

                if np.ndim(do_np) == 0:
                    bis_pred_raw.append(float(do_np))
                else:
                    bis_pred_raw.extend(list(do_np))
                sqi_pred_raw = np.hstack((sqi_pred_raw, np.array(sqi_np).reshape(-1)))

        bis_pred_raw = np.round(np.array(bis_pred_raw, dtype=np.float32))
        bis_pred_raw = np.where(bis_pred_raw > 100, 100, bis_pred_raw)
        bis_pred_raw = np.where(np.isnan(bis_pred_raw), 0, bis_pred_raw)

        sqi_pred_smooth = quadratic_smooth5(sqi_pred_raw)
        bis_pred_final = np.where(np.array(sqi_pred_smooth) > args.sqi_th, bis_pred_raw, 0)

        out_file = output_dir / f"{case_name}.csv"
        with out_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "bis_pred", "sqi_pred", "bis_pred_masked"])
            for i in range(len(bis_pred_final)):
                writer.writerow([i, float(bis_pred_raw[i]), float(sqi_pred_smooth[i]), float(bis_pred_final[i])])

        print(f"saved: {out_file}")


if __name__ == "__main__":
    main()
