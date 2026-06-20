import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

np.set_printoptions(precision=3, suppress=False)


class SimpleLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = self.log_path.open("w", encoding="utf-8")

    def log(self, txt: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {txt}"
        print(line)
        self.fp.write(line + "\n")
        self.fp.flush()

    def close(self):
        self.fp.close()


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


def maybe_set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CrossValSampler:
    # This class mirrors the SQI sampling path in code/DataAdapter/EnsambleDataAdapter.py.
    # Difference: EEG1/EEG2/BIS/SQI are loaded from each .mat file instead of external .eeg files.
    def __init__(self, data_path: str, logger: SimpleLogger, fs=128, win_len=30):
        self.data_path = data_path
        self.fs = fs
        self.win_len = win_len
        self.log = logger

    def _to_1d(self, x):
        arr = np.asarray(x)
        arr = np.squeeze(arr)
        if arr.ndim == 0:
            return np.array([float(arr)], dtype=np.float32)
        return arr.reshape(-1).astype(np.float32)

    def _unwrap(self, x):
        v = x
        while isinstance(v, np.ndarray) and v.size == 1:
            v = v.reshape(-1)[0]
        return v

    def _get_from_obj(self, obj, key):
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(key, None)
        if hasattr(obj, key):
            return getattr(obj, key)
        if isinstance(obj, np.ndarray) and obj.dtype.names is not None and key in obj.dtype.names:
            return obj[key]
        return None

    def _try_keys(self, obj, keys):
        for key in keys:
            v = self._get_from_obj(obj, key)
            if v is not None:
                return v
        return None

    def _load_from_ref(self, ref):
        ref_obj = self._unwrap(ref)

        eeg1 = self._try_keys(ref_obj, ["EEG1", "eeg1", "eeg_1", "sig1", "signal1"])
        eeg2 = self._try_keys(ref_obj, ["EEG2", "eeg2", "eeg_2", "sig2", "signal2"])
        bis = self._try_keys(ref_obj, ["BIS", "bis", "label", "y"])
        sqi = self._try_keys(ref_obj, ["SQI", "sqi"])

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

    def _load_from_named_tracks(self, mat):
        if "name" not in mat or "data" not in mat:
            return None, None, None, None
        names_raw = np.asarray(mat["name"]).reshape(-1)
        data_raw = np.asarray(mat["data"]).reshape(-1)
        if len(names_raw) == 0 or len(data_raw) == 0:
            return None, None, None, None

        def find_track(keyword_list):
            for i, n in enumerate(names_raw):
                name = str(self._unwrap(n)).lower()
                if any(k in name for k in keyword_list):
                    if i < len(data_raw):
                        return data_raw[i]
            return None

        eeg1 = find_track(["eeg1", "eeg_1", "eeg ch1", "eeg ch 1"])
        eeg2 = find_track(["eeg2", "eeg_2", "eeg ch2", "eeg ch 2"])
        bis = find_track(["bis"])
        sqi = find_track(["sqi"])
        return eeg1, eeg2, bis, sqi

    def load_case_from_mat(self, labelpath):
        mat = loadmat(labelpath, squeeze_me=True, struct_as_record=False)

        eeg1 = eeg2 = bis = sqi = None

        if "ref" in mat:
            eeg1, eeg2, bis, sqi = self._load_from_ref(mat["ref"])

        if eeg1 is None or eeg2 is None or bis is None or sqi is None:
            ne1, ne2, nbis, nsqi = self._load_from_named_tracks(mat)
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

        if eeg1 is None or eeg2 is None or sqi is None:
            keys = [k for k in mat.keys() if not k.startswith("__")]
            raise KeyError(
                "Cannot find EEG1/EEG2/SQI in mat file: {}. Available keys: {}".format(
                    labelpath, keys
                )
            )

        eeg1 = self._to_1d(self._unwrap(eeg1))
        eeg2 = self._to_1d(self._unwrap(eeg2))
        bis = self._to_1d(self._unwrap(bis)) if bis is not None else np.zeros(1, dtype=np.float32)
        sqi = self._to_1d(self._unwrap(sqi))
        return eeg1, eeg2, bis, sqi

    def stat_sqi(self, data_set):
        sqi_count = np.zeros(100)
        file_count = 0
        for files in data_set:
            labelpath = os.path.join(self.data_path, files)
            if "case" not in files:
                continue
            _, _, _, sqi = self.load_case_from_mat(labelpath)
            sqi = np.round(sqi)
            file_count += 1
            for i in range(1, 101):
                sqi_count[i - 1] += np.sum(np.where(np.array(sqi) == i, 1, 0))

        self.log.log(str([int(i) for i in sqi_count]))
        return sqi_count, file_count

    def norm(self, x):
        return x - np.mean(x)

    def schmidt_spike_removal(self, original_signal, fs):
        window_size = fs
        mod_num = np.mod(len(original_signal), window_size)
        original_signal = np.array(original_signal)
        original_signal = original_signal[: len(original_signal) - mod_num]

        segment_signal = np.reshape(
            original_signal, (window_size, int(len(original_signal) // window_size))
        )
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

    def gen_sqi_data(self, data_set):
        random.seed(0)
        sample_num = 2000

        sqi_count, file_count = self.stat_sqi(data_set)
        sqi_rate = np.zeros(101)

        for i in range(len(sqi_count)):
            if sqi_count[i] < sample_num:
                sqi_rate[i] = sqi_count[i]
            else:
                sqi_rate[i] = 2

        real_count = sqi_rate.copy()
        res_sig = []
        res_label = []

        self.log.log("---------- Gen Data ----------")

        for files in data_set:
            re = files.split(".")
            name = re[0].split("_")[0]
            sample = []

            labelpath = os.path.join(self.data_path, files)
            self.log.log("----- {} -----".format(name))
            if "case" not in files:
                continue
            if not os.path.exists(labelpath):
                raise FileNotFoundError("Label mat not found: {}".format(labelpath))

            eeg1_ori_sig, eeg2_ori_sig, _, sqi = self.load_case_from_mat(labelpath)
            eeg1_ori_sig = np.where(np.isnan(np.array(eeg1_ori_sig)), 0, eeg1_ori_sig)
            eeg2_ori_sig = np.where(np.isnan(np.array(eeg2_ori_sig)), 0, eeg2_ori_sig)
            sqi = np.round(sqi)
            sqi = np.where(np.isnan(np.array(sqi)), 0, sqi)

            self.log.log("Invalid SQI: {}".format(np.sum(np.where(sqi <= 20, 1, 0))))
            self.log.log("Valid SQI: {}".format(np.sum(np.where(sqi > 20, 1, 0))))
            self.log.log("SQI Length: {}".format(len(sqi)))

            despiked_signal = self.schmidt_spike_removal(eeg1_ori_sig, self.fs)
            despiked_signal = self.norm(despiked_signal)
            eeg1_ori_sig = despiked_signal

            despiked_signal = self.schmidt_spike_removal(eeg2_ori_sig, self.fs)
            despiked_signal = self.norm(despiked_signal)
            eeg2_ori_sig = despiked_signal

            self.log.log("EEG1 Signal length: {}".format(len(eeg1_ori_sig)))
            self.log.log("EEG2 Signal length: {}".format(len(eeg2_ori_sig)))

            if len(eeg1_ori_sig) != len(eeg2_ori_sig):
                self.log.log("The length of EEG1 is not equal to EEG2")
                continue

            for i in range(101):
                idx = np.where(sqi == i)[0]
                if len(idx) == 0:
                    continue
                random.shuffle(idx)
                if real_count[i] > len(idx):
                    max_pos = idx
                    real_count[i] = real_count[i] - len(idx)
                else:
                    max_pos = idx[: int(real_count[i])]
                    real_count[i] = 0

                for j in max_pos:
                    eeg1 = eeg1_ori_sig[max(0, (j - self.win_len) * self.fs) : j * self.fs]
                    eeg2 = eeg2_ori_sig[max(0, (j - self.win_len) * self.fs) : j * self.fs]
                    sig = np.stack([eeg1, eeg2])
                    if j - self.win_len < 0:
                        padding = np.zeros((2, (self.win_len - j) * self.fs))
                        sig = np.hstack((padding, sig))
                    assert sig.shape[1] == self.win_len * self.fs
                    sample.append(j)
                    res_label.append(i)
                    res_sig.append(sig)

            real_count = sqi_rate + real_count

        assert len(res_sig) == len(res_label)

        new_label_count = np.zeros(101)
        for i in range(len(new_label_count)):
            new_label_count[i] += np.sum(np.where(np.array(res_label) == i, 1, 0))

        self.log.log("New label distribution")
        self.log.log(str(new_label_count))

        return np.array(res_sig, dtype=np.float32), np.array(res_label, dtype=np.float32)


class ArrayDataset(Dataset):
    def __init__(self, data, label):
        self.data = torch.FloatTensor(data)
        self.label = torch.FloatTensor(label)

    def __getitem__(self, index):
        return self.data[index, :], self.label[index]

    def __len__(self):
        return len(self.data)


class SQINet(nn.Module):
    def __init__(self):
        super(SQINet, self).__init__()
        self.branch1 = Branch(1, 64, 49)
        self.branch2 = Branch(1, 64, 49)
        self.conv1 = nn.Conv1d(128, 128, 7, 2, 3)
        self.conv2 = nn.Conv1d(128, 256, 7, 1, 3)
        self.convt = nn.Conv1d(128, 256, 1, 2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bnt = nn.BatchNorm1d(256)
        self.gp = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.ReLU(),
        )

    def forward(self, x1, x2):
        batch_size = x1.size(0)
        x1 = x1.view(x1.size(0), 1, x1.size(-1))
        x2 = x2.view(x2.size(0), 1, x2.size(-1))
        x1 = self.branch1(x1)
        x2 = self.branch2(x2)
        x = torch.cat((x1, x2), dim=1)
        xt = self.bnt(self.convt(x))
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.relu(x + xt)
        x = self.gp(x).view(batch_size, -1)
        x = self.fc(x)
        return x


class Branch(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7):
        super(Branch, self).__init__()
        self.conv1 = nn.Conv1d(in_ch, in_ch, kernel_size, 2, kernel_size // 2)
        self.conv2 = nn.Conv1d(in_ch, in_ch * 4, 1, 1)
        self.conv3 = nn.Conv1d(in_ch * 4, out_ch, 1, 1)
        self.convt = nn.Conv1d(in_ch, out_ch, 1, 2)
        self.bn1 = nn.BatchNorm1d(in_ch)
        self.bn2 = nn.BatchNorm1d(in_ch * 4)
        self.bn3 = nn.BatchNorm1d(out_ch)
        self.bnt = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        xt = self.bnt(self.convt(x))
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))
        x = self.relu(x + xt)
        return x


class EarlyStopping:
    def __init__(self, save_path, patience=7, verbose=False, delta=0, name=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path
        self.name = name

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        print("save best model..")
        if self.name is None:
            torch.save(model.state_dict(), os.path.join(self.save_path, "best_model.pt"))
        else:
            torch.save(model.state_dict(), os.path.join(self.save_path, self.name + "_best_model.pt"))
        self.val_loss_min = val_loss


class Regularization(nn.Module):
    def __init__(self, model, weight_decay, p=2):
        super(Regularization, self).__init__()
        self.model = model
        self.weight_decay = weight_decay
        self.p = p

    def forward(self, model):
        weight_list = self.get_weight(model)
        reg_loss = self.regularization_loss(weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):
        weight_list = []
        for name, param in model.named_parameters():
            if "weight" in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p=2):
        reg_loss = 0
        for _, w in weight_list:
            if p == 2 or p == 0:
                reg_loss += torch.sum(torch.pow(w, 2))
            else:
                reg_loss += torch.sum(torch.abs(w))
        reg_loss = weight_decay * reg_loss
        return reg_loss


def cal_metric(x_pred, x_real, is_cal_corr=True, is_cal_mae=True):
    assert len(x_pred) == len(x_real)
    corr = None
    mae = None
    if is_cal_corr:
        corr = np.corrcoef(x_pred, x_real)[0, 1]
        corr = np.round(corr, 3)
    if is_cal_mae:
        mae = np.mean(np.abs(x_pred - x_real))
    return corr, mae


def cal_regression_batch(loader, model, criterion, optimizer, reg_loss, device, types="train"):
    loss_list = []
    mae_list = []

    if types == "train":
        model.train()
    elif types == "valid":
        model.eval()

    for _, data in enumerate(loader, 0):
        inputs, labels = data[0].squeeze(), data[1].squeeze().to(device)
        if len(inputs.shape) < 2 or inputs.shape[0] < 5:
            continue
        inputs = inputs.to(device)

        outputs = model(inputs[:, 0, :], inputs[:, 1, :])
        outputs = outputs.squeeze()
        loss = criterion(outputs, labels)

        with torch.no_grad():
            _, mae = cal_metric(
                outputs.detach().cpu().numpy(),
                labels.detach().cpu().numpy(),
                is_cal_corr=False,
                is_cal_mae=True,
            )
            mae_list.append(mae)

        if reg_loss:
            loss += reg_loss(model)

        if types == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())

    loss_mean = float(np.mean(loss_list)) if len(loss_list) > 0 else float("nan")
    mae_mean = float(np.mean(mae_list)) if len(mae_list) > 0 else float("nan")
    return {"loss": loss_mean, "mae": mae_mean}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone SQINet trainer aligned with algorithm.train_sqi() and gen_sqi_data()."
    )
    parser.add_argument("--split-file", required=True, help="Path to one split txt file.")
    parser.add_argument(
        "--data-dir",
        default="mat",
        help="Directory containing case*.mat files with EEG1/EEG2/SQI/BIS.",
    )
    parser.add_argument(
        "--mat-dir",
        default="",
        help="Deprecated alias for --data-dir (kept for compatibility).",
    )
    parser.add_argument("--save-dir", default="runs/sqinet", help="Output root directory.")
    parser.add_argument("--run-name", default="", help="Run folder name.")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--l2-reg", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--t-max", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--fs", type=int, default=128)
    parser.add_argument("--win-len", type=int, default=30)
    parser.add_argument("--input-len", type=int, default=3840)

    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data-parallel", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.input_len != args.fs * args.win_len:
        raise ValueError(
            f"--input-len must equal fs*win-len ({args.fs * args.win_len}) to match original pipeline."
        )

    maybe_set_seed(args.seed)

    data_dir = args.data_dir
    if args.mat_dir.strip():
        data_dir = args.mat_dir.strip()

    split_file = Path(args.split_file)
    if not split_file.exists():
        raise FileNotFoundError(f"split file not found: {split_file}")

    run_name = args.run_name.strip() or time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.save_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = SimpleLogger(run_dir / "train.log")

    try:
        logger.log("-------- Parameters --------")
        logger.log(f"Split file: {split_file}")
        logger.log(f"Data dir: {data_dir}")
        logger.log(f"SQI Batch Size: {args.batch_size}")
        logger.log(f"SQI Learning Rate: {args.lr}")
        logger.log(f"Epochs: {args.epochs}")
        logger.log(f"Patience: {args.patience}")
        logger.log(f"L2 Reg: {args.l2_reg}")
        logger.log(f"Weight Decay: {args.weight_decay}")
        logger.log(f"Seed: {args.seed}")

        with (run_dir / "args.json").open("w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)

        split_sections = parse_split_file(split_file)
        train_set = split_sections.get("train", [])
        valid_set = split_sections.get("valid", [])

        if len(train_set) == 0:
            raise ValueError("No [train] entries in split file.")
        if len(valid_set) == 0:
            raise ValueError("No [valid] entries in split file.")

        sampler = CrossValSampler(data_path=data_dir, logger=logger, fs=args.fs, win_len=args.win_len)
        train_data, train_label = sampler.gen_sqi_data(train_set)
        valid_data, valid_label = sampler.gen_sqi_data(valid_set)

        logger.log(f"Train sample num: {len(train_label)}")
        logger.log(f"Valid sample num: {len(valid_label)}")

        train_adapter = ArrayDataset(train_data, train_label)
        valid_adapter = ArrayDataset(valid_data, valid_label)
        train_loader = DataLoader(
            train_adapter, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
        )
        valid_loader = DataLoader(
            valid_adapter, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )

        device = choose_device(args.device)
        logger.log(f"Device: {device}")

        model = SQINet()
        if args.data_parallel and device.type == "cuda":
            model = nn.DataParallel(model)
        model = model.to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        clr = CosineAnnealingLR(optimizer, T_max=args.t_max)
        reg_loss = Regularization(model, args.l2_reg) if args.l2_reg > 0 else None
        early_stopping = EarlyStopping(str(run_dir), args.patience, verbose=False, name="SQI")

        for epoch in range(1, args.epochs + 1):
            start_time = time.time()
            train_res = cal_regression_batch(
                train_loader, model, criterion, optimizer, reg_loss, device, "train"
            )
            valid_res = cal_regression_batch(
                valid_loader, model, criterion, optimizer, reg_loss, device, "valid"
            )
            end_time = time.time()
            logger.log(
                "- Epoch: {:d} - Train_loss: {:.5f} - Train_mae: {:.5f} - Val_loss: {:.5f} - Val_mae: {:.5f} - T_Time: {:.3f}".format(
                    epoch,
                    train_res["loss"],
                    train_res["mae"],
                    valid_res["loss"],
                    valid_res["mae"],
                    end_time - start_time,
                )
            )
            logger.log("Current Learning Rate: {:f}".format(optimizer.state_dict()["param_groups"][0]["lr"]))
            clr.step()

            if np.isnan(valid_res["loss"]) or np.isnan(train_res["loss"]):
                logger.log("Loss is NaN! Train stop!")
                break

            early_stopping(valid_res["loss"], model)
            if early_stopping.early_stop:
                logger.log("Early stopping")
                break

        torch.save(model.state_dict(), str(run_dir / "SQI_last_model.pt"))
        logger.log("Train SQI finished...")
        logger.log(f"Best model path: {run_dir / 'SQI_best_model.pt'}")
        logger.log(f"Last model path: {run_dir / 'SQI_last_model.pt'}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
