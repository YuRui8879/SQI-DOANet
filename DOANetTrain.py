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
from torch.nn import functional as F
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
    # This class mirrors the DOA sampling path in code/DataAdapter/EnsambleDataAdapter.py.
    # Difference: EEG1/EEG2/BIS/SQI are loaded from each .mat file instead of external .eeg files.
    def __init__(self, data_path: str, logger: SimpleLogger, fs=128, win_len=30):
        self.data_path = data_path
        self.fs = fs
        self.win_len = win_len
        self.log = logger

    def stat_bis(self, data_set):
        bis_count = np.zeros(100)
        file_count = 0
        for files in data_set:
            labelpath = os.path.join(self.data_path, files)
            if "case" not in files:
                continue
            data = loadmat(labelpath)["ref"][0][0]
            bis = data[2][0]
            file_count += 1
            for i in range(1, 101):
                bis_count[i - 1] += np.sum(np.where(np.array(bis) == i, 1, 0))

        self.log.log(str([int(i) for i in bis_count]))
        return bis_count, file_count

    def cal_guassian_dist(self, sel_sig, sel_idx):
        idx = 0
        guassian_dist = []
        while idx < len(sel_sig):
            if idx != sel_idx:
                dist = np.sqrt(np.sum(np.power(sel_sig[sel_idx] - sel_sig[idx], 2)))
                guassian_dist.append(dist)
            else:
                guassian_dist.append(0)
            idx += 1
        return guassian_dist

    def find_opt_time(self, bis, th=70):
        i = 2
        count = 0
        res = np.nan
        bis = np.where(bis < 1, np.nan, bis)
        while i < len(bis):
            a = bis[:i]

            if all(np.where(np.isnan(a), 0, a) < th):
                i += 1
                if i > len(bis) * 0.5:
                    return np.nan
                continue

            if len(a) > 5:
                ma = np.sum(a) / np.sum(np.where(np.isnan(a), 0, 1))
                if bis[i] - ma < -20 and np.mean(bis[i : i + 30]) < bis[i]:
                    res = i
                    break

            if bis[i] < th and bis[i] > 1:
                count += 1
                if count == 30:
                    res = i - 30
                    break
            else:
                count = 0
            i += 1
        return res

    def data_aug(self, sig, label):
        label_count = np.zeros(101)
        for i in range(101):
            label_count[i] += np.sum(np.where(np.array(label) == i, 1, 0))
        max_count = np.max(label_count)
        new_sig = []
        new_label = []
        last_len = 0
        for i in range(len(label_count)):
            if label_count[i] == 0:
                self.log.log("data enhancement of {} 0 --> 0".format(i))
                continue
            if label_count[i] < max_count // 2:
                rate = int(np.floor(max_count / label_count[i]))
                idx = np.where(np.array(label) == i)
                sel_sig = np.array(sig)[idx]
                for j in range(len(sel_sig)):
                    guassian_dist = self.cal_guassian_dist(sel_sig, j)
                    for k in range(rate):
                        max_pos = np.argmax(guassian_dist)
                        new_sig.append(
                            sel_sig[j]
                            + random.randint(1, 100) / 100 * (sel_sig[j] - sel_sig[max_pos])
                        )
                        new_label.append(i)
                        guassian_dist[max_pos] = 0
                        if k >= len(sel_sig):
                            break
            self.log.log(
                "data enhancement of {} {} --> {}".format(
                    i, label_count[i], label_count[i] + len(new_label) - last_len
                )
            )
            last_len = len(new_label)

        return new_sig, new_label

    def norm(self, x):
        return x - np.mean(x)

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

        # Fallback for old index-style ref layout used in the original project.
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
        # Support mat files exported as name/data lists.
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

        if eeg1 is None or eeg2 is None or bis is None:
            ne1, ne2, nbis, nsqi = self._load_from_named_tracks(mat)
            eeg1 = eeg1 if eeg1 is not None else ne1
            eeg2 = eeg2 if eeg2 is not None else ne2
            bis = bis if bis is not None else nbis
            sqi = sqi if sqi is not None else nsqi

        # Direct top-level key fallback
        if eeg1 is None and "EEG1" in mat:
            eeg1 = mat["EEG1"]
        if eeg2 is None and "EEG2" in mat:
            eeg2 = mat["EEG2"]
        if bis is None and "BIS" in mat:
            bis = mat["BIS"]
        if sqi is None and "SQI" in mat:
            sqi = mat["SQI"]

        if eeg1 is None or eeg2 is None or bis is None:
            keys = [k for k in mat.keys() if not k.startswith("__")]
            raise KeyError(
                "Cannot find EEG1/EEG2/BIS in mat file: {}. Available keys: {}".format(
                    labelpath, keys
                )
            )

        eeg1 = self._to_1d(self._unwrap(eeg1))
        eeg2 = self._to_1d(self._unwrap(eeg2))
        bis = self._to_1d(self._unwrap(bis))
        if sqi is None:
            sqi = np.zeros_like(bis, dtype=np.float32)
        else:
            sqi = self._to_1d(self._unwrap(sqi))

        return eeg1, eeg2, bis, sqi

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

    def gen_data(self, data_set, is_data_enhancement=False):
        th = 70
        sample_num = 1000

        bis_count, file_count = self.stat_bis(data_set)
        bis_rate = np.zeros(101)

        for i in range(len(bis_count)):
            if bis_count[i] < sample_num:
                bis_rate[i] = bis_count[i]
            else:
                bis_rate[i] = int(np.floor(file_count / sample_num))

        real_count = bis_rate.copy()

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

            eeg1, eeg2, bis, sqi = self.load_case_from_mat(labelpath)
            eeg1 = np.where(np.isnan(np.array(eeg1)), 0, eeg1)
            eeg2 = np.where(np.isnan(np.array(eeg2)), 0, eeg2)
            bis = np.where(np.isnan(np.array(bis)), 0, bis)

            self.log.log("Bis length: {}".format(len(bis)))
            self.log.log("SQI length: {}".format(len(sqi)))

            despiked_signal = self.schmidt_spike_removal(eeg1, self.fs)
            despiked_signal = self.norm(despiked_signal)
            eeg1 = despiked_signal

            despiked_signal = self.schmidt_spike_removal(eeg2, self.fs)
            despiked_signal = self.norm(despiked_signal)
            eeg2 = despiked_signal

            self.log.log("EEG1 Signal length: {}".format(len(eeg1)))
            self.log.log("EEG2 Signal length: {}".format(len(eeg2)))

            if len(eeg1) != len(eeg2):
                self.log.log("The length of EEG1 is not equal to EEG2")
                continue

            start_time = self.find_opt_time(bis, th)
            end_time = len(bis) - self.find_opt_time(bis[::-1], th)

            if np.isnan(start_time) and np.isnan(end_time):
                para = np.ones(len(bis))
                self.log.log("Start Position: Not found")
                self.log.log("End Position: Not found")
            elif np.isnan(start_time) and not np.isnan(end_time):
                para1 = np.linspace(0.1, 1, end_time)
                para2 = np.linspace(1, 0.1, len(bis) - end_time)
                para = np.hstack((para1, para2))
                self.log.log("Start Position: Not found")
                self.log.log("End Position: {}".format(end_time))
            elif not np.isnan(start_time) and np.isnan(end_time):
                para1 = np.linspace(0.1, 1, start_time)
                para2 = np.linspace(1, 0.1, len(bis) - start_time)
                para = np.hstack((para1, para2))
                self.log.log("Start Position: {}".format(start_time))
                self.log.log("End Position: Not found")
            else:
                para1 = np.linspace(0.1, 1, start_time)
                para2 = np.linspace(1, 0.1, int(np.ceil((end_time - start_time) / 2)))
                para3 = np.linspace(0.1, 1, int(np.floor((end_time - start_time) / 2)))
                para4 = np.linspace(1, 0.1, len(bis) - end_time)
                para = np.hstack((para1, para2, para3, para4))
                self.log.log("Start Position: {}".format(start_time))
                self.log.log("End Position: {}".format(end_time))

            for i in range(101):
                idx = np.where(bis == i)[0]
                if len(idx) == 0:
                    continue
                max_pos = []
                for _ in range(len(idx)):
                    pos = np.argmax(para[idx])
                    if idx[pos] < self.win_len:
                        continue
                    max_pos.append(idx[pos])
                    para[idx[pos]] = 0
                    real_count[i] -= 1
                    if real_count[i] <= 0:
                        break

                for j in max_pos:
                    sig1 = eeg1[max(0, (j - self.win_len) * self.fs) : j * self.fs]
                    sig2 = eeg2[max(0, (j - self.win_len) * self.fs) : j * self.fs]
                    sig = np.stack([sig1, sig2])
                    assert sig.shape[-1] == self.win_len * self.fs
                    sample.append(j)
                    res_label.append(i)
                    res_sig.append(sig)

            real_count = bis_rate + real_count

        if is_data_enhancement:
            self.log.log("----------- Data Enhancement ----------")
            new_sig, new_label = self.data_aug(res_sig, res_label)
            res_label.extend(new_label)
            res_sig.extend(new_sig)

        assert len(res_label) == len(res_sig)

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


class DoubleMRANet(nn.Module):
    def __init__(self, input_len=3840):
        super(DoubleMRANet, self).__init__()
        self.mtcnn1 = MTCNN()
        self.mtcnn2 = MTCNN()
        self.mra = MRA()
        self.gp = nn.AdaptiveAvgPool1d(1)
        self.convt1 = nn.Conv1d(1, 64, input_len, bias=False)
        self.convt2 = nn.Conv1d(1, 64, input_len, bias=False)
        self.bnt1 = nn.BatchNorm1d(64)
        self.bnt2 = nn.BatchNorm1d(64)
        self.gmlp1 = gMLPBlock(128, 256, 60)

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
        xt1 = self.bnt1(self.convt1(x1)).view(x1.size(0), -1)
        xt2 = self.bnt2(self.convt2(x2)).view(x2.size(0), -1)
        xt = torch.cat((xt1, xt2), dim=1)
        x1 = self.mtcnn1(x1)
        x2 = self.mtcnn2(x2)
        x = self.mra(x1, x2)
        x = x.view(x.size(0), x.size(2), x.size(1))
        x = self.gmlp1(x)
        x = x.reshape(x.size(0), x.size(2), x.size(1))
        x = self.gp(x).view(batch_size, -1)
        x = torch.cat((x, xt), 1)
        x = self.fc(x)
        return x


class MTCNN(nn.Module):
    def __init__(self):
        super(MTCNN, self).__init__()
        self.conv1 = CNNBlock(1, 64, 49, 2)
        self.conv2 = CNNBlock(64, 128, 7, 2)
        self.conv3 = CNNBlock(128, 128, 7, 2)
        self.maxpool1 = nn.MaxPool1d(2, 2)
        self.maxpool2 = nn.MaxPool1d(4, 4)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.dropout(x)
        return x


class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, filter_size, stride=2):
        super(CNNBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, filter_size, stride, filter_size // 2, bias=False)
        self.conv2 = nn.Conv1d(out_ch, out_ch * 2, 1, bias=False)
        self.conv3 = nn.Conv1d(out_ch * 2, out_ch, 1, bias=False)
        self.convt = nn.Conv1d(in_ch, out_ch, 1, stride, bias=False)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch * 2)
        self.bn3 = nn.BatchNorm1d(out_ch)
        self.bnt = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        shortcut = self.bnt(self.convt(x))
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x += shortcut
        x = self.relu(x)
        return x


class MRA(nn.Module):
    def __init__(self):
        super(MRA, self).__init__()
        self.conv1_1 = nn.Conv1d(128, 128, 7, 1, 3)
        self.conv1_2 = nn.Conv1d(128, 128, 25, 1, 12)
        self.conv2_1 = nn.Conv1d(128, 128, 7, 1, 3)
        self.conv2_2 = nn.Conv1d(128, 128, 25, 1, 12)
        self.bn1_1 = nn.BatchNorm1d(128)
        self.bn1_2 = nn.BatchNorm1d(128)
        self.bn2_1 = nn.BatchNorm1d(128)
        self.bn2_2 = nn.BatchNorm1d(128)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc1_1 = nn.Conv1d(128, 32, 1, bias=False)
        self.fc1_2 = nn.Conv1d(32, 128, 1, bias=False)
        self.fc1_3 = nn.Conv1d(32, 128, 1, bias=False)

        self.fc2_1 = nn.Conv1d(128, 32, 1, bias=False)
        self.fc2_2 = nn.Conv1d(32, 128, 1, bias=False)
        self.fc2_3 = nn.Conv1d(32, 128, 1, bias=False)

        self.fc = nn.Conv1d(128, 32, 1, bias=False)
        self.fc2 = nn.Conv1d(32, 128, 1, bias=False)
        self.fc3 = nn.Conv1d(32, 128, 1, bias=False)
        self.relu = nn.ReLU(True)
        self.softmax = nn.Softmax(-1)

    def forward(self, x1, x2):
        ch1_1 = self.bn1_1(self.conv1_1(x1))
        ch1_2 = self.bn1_2(self.conv1_2(x1))
        ch2_1 = self.bn2_1(self.conv2_1(x2))
        ch2_2 = self.bn2_2(self.conv2_2(x2))
        x1 = ch1_1 + ch1_2
        x2 = ch2_1 + ch2_2
        x1 = self.avgpool(x1)
        x2 = self.avgpool(x2)

        x1 = self.relu(self.fc1_1(x1))
        wch1_1 = self.fc1_2(x1)
        wch1_2 = self.fc1_3(x1)
        attn1 = self.softmax(torch.cat((wch1_1, wch1_2), -1))
        x1_1 = ch1_1 * attn1[:, :, 0].unsqueeze(-1)
        x1_2 = ch1_2 * attn1[:, :, 1].unsqueeze(-1)
        x1 = x1_1 + x1_2

        x2 = self.relu(self.fc2_1(x2))
        wch2_1 = self.fc2_2(x2)
        wch2_2 = self.fc2_3(x2)
        attn2 = self.softmax(torch.cat((wch2_1, wch2_2), -1))
        x2_1 = ch2_1 * attn2[:, :, 0].unsqueeze(-1)
        x2_2 = ch2_2 * attn2[:, :, 1].unsqueeze(-1)
        x2 = x2_1 + x2_2

        x1 = self.relu(self.fc1_1(x1))
        wch1_1 = self.fc1_2(x1)
        wch1_2 = self.fc1_3(x1)
        attn1 = self.softmax(torch.cat((wch1_1, wch1_2), -1))
        x1_1 = ch1_1 * attn1[:, :, 0].unsqueeze(-1)
        x1_2 = ch1_2 * attn1[:, :, 1].unsqueeze(-1)
        x1 = x1_1 + x1_2

        x = x1 + x2
        x = self.avgpool(x)
        x = self.relu(self.fc(x))
        wch1 = self.fc2(x)
        wch2 = self.fc3(x)
        attn = self.softmax(torch.cat((wch1, wch2), -1))
        x1 = x1 * attn[:, :, 0].unsqueeze(-1)
        x2 = x2 * attn[:, :, 1].unsqueeze(-1)
        x = x1 + x2
        return x


class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn)
        self.spatial_proj = nn.Conv1d(seq_len, seq_len, kernel_size=1)
        nn.init.constant_(self.spatial_proj.bias, 1.0)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.spatial_proj(v)
        out = u * v
        return out


class gMLPBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)
        self.channel_proj2 = nn.Linear(d_ffn, d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.gelu(self.channel_proj1(x))
        x = self.sgu(x)
        x = self.channel_proj2(x)
        out = x + residual
        return out


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
        description="Standalone DOANet trainer with sampling logic identical to EnsambleDataAdapter."
    )
    parser.add_argument("--split-file", required=True, help="Path to one split txt file.")
    parser.add_argument(
        "--data-dir",
        default="mat",
        help="Directory containing case*.mat files with EEG1/EEG2/BIS/SQI.",
    )
    parser.add_argument(
        "--mat-dir",
        default="",
        help="Deprecated alias for --data-dir (kept for compatibility).",
    )
    parser.add_argument("--save-dir", default="runs/doanet", help="Output root directory.")
    parser.add_argument("--run-name", default="", help="Run folder name.")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--l2-reg", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--t-max", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--fs", type=int, default=128)
    parser.add_argument("--win-len", type=int, default=30)
    parser.add_argument("--input-len", type=int, default=3840)
    parser.add_argument(
        "--disable-train-augmentation",
        action="store_true",
        help="Disable data augmentation for train set. Default matches EnsambleDataAdapter (enabled).",
    )

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
        logger.log(f"Batch Size: {args.batch_size}")
        logger.log(f"Learning Rate: {args.lr}")
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
        train_data, train_label = sampler.gen_data(
            train_set, is_data_enhancement=(not args.disable_train_augmentation)
        )
        valid_data, valid_label = sampler.gen_data(valid_set, is_data_enhancement=False)

        logger.log(f"Train sample num: {len(train_label)}")
        logger.log(f"Valid sample num: {len(valid_label)}")

        train_adapter = ArrayDataset(train_data, train_label)
        valid_adapter = ArrayDataset(valid_data, valid_label)
        train_loader = DataLoader(
            train_adapter,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        valid_loader = DataLoader(
            valid_adapter,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        device = choose_device(args.device)
        logger.log(f"Device: {device}")

        model = DoubleMRANet(input_len=args.input_len)
        if args.data_parallel and device.type == "cuda":
            model = nn.DataParallel(model)
        model = model.to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        clr = CosineAnnealingLR(optimizer, T_max=args.t_max)
        reg_loss = Regularization(model, args.l2_reg) if args.l2_reg > 0 else None
        early_stopping = EarlyStopping(str(run_dir), args.patience, verbose=False, name="DOANet")

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

        torch.save(model.state_dict(), str(run_dir / "DOANet_last_model.pt"))
        logger.log("Train finished...")
        logger.log(f"Best model path: {run_dir / 'DOANet_best_model.pt'}")
        logger.log(f"Last model path: {run_dir / 'DOANet_last_model.pt'}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
