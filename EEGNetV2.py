#!/usr/bin/env python3
"""
Train EEGNet v2-ish on MI epoch CSVs and save results to result/.

Dataset layout:
data/
  left/ right/   (extend if you want)
Each CSV: timestamp_sec, ch0..ch23 (T rows, typically 4s @ 250Hz => 1000)

Key features in this script:
- Robust delimiter detection (comma/tab/semicolon/space)
- Cropping window anchored to the END of file (pedal press is last row)
- Optional: drop tail (closest to pedal), drop head (earliest part of selected window)
- Optional bandpass (if SciPy installed)
- Optional per-channel z-score
- Optional channel selection: keep only MI-relevant channels (e.g., C3/C4/Cz/FC/Cp)
- Train/Val/Test split with early stopping
- Saves:
  result/confusion_matrix.png
  result/learning_curves.png
  result/best_model.pt
  result/metrics.json

Quick start:
  (conda env activated)
  python train_eegnet_mi.py --data_root data --labels left right --use_mi_channels --epochs 80 --lr 1e-3
"""

import argparse
from pathlib import Path
import json
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Channel map (your montage)
# -------------------------
CH_NAME_TO_IDX = {
    "FP1": 0, "FP2": 1, "F3": 2, "F4": 3, "C3": 4, "C4": 5, "FC5": 6, "FC6": 7,
    "O1": 8, "O2": 9, "F7": 10, "F8": 11, "T7": 12, "T8": 13, "P7": 14, "P8": 15,
    "AFZ": 16, "CZ": 17, "FZ": 18, "PZ": 19, "FPZ": 20, "OZ": 21, "AF3": 22, "AF4": 23
}

# MI-focused small set (common motor cortex-ish)
DEFAULT_MI_CH_NAMES = ["C3", "C4", "CZ", "FC5", "FC6", "F3", "F4"]  # tweak freely

def export_to_onnx(model, n_ch, n_samples, out_path, device):
    model.eval()

    # Dummy input: (B, 1, C, T)
    dummy = torch.randn(1, 1, n_ch, n_samples, device=device)

    torch.onnx.export(
        model,
        dummy,
        out_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["eeg_input"],
        output_names=["logits"],
        dynamic_axes={
            "eeg_input": {0: "batch"},
            "logits": {0: "batch"},
        }
    )

    print(f"ONNX model saved to: {out_path}")


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def butter_bandpass_filtfilt(x: np.ndarray, sr: int, low: float, high: float, order: int = 4) -> np.ndarray:
    """
    x: (C, T)
    If SciPy isn't available, returns input.
    """
    try:
        from scipy.signal import butter, filtfilt
    except Exception:
        return x

    nyq = 0.5 * sr
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    y = np.zeros_like(x, dtype=np.float32)
    for c in range(x.shape[0]):
        y[c] = filtfilt(b, a, x[c]).astype(np.float32)
    return y


def per_channel_zscore(x: np.ndarray) -> np.ndarray:
    # x: (C, T)
    m = x.mean(axis=1, keepdims=True)
    s = x.std(axis=1, keepdims=True) + 1e-8
    return ((x - m) / s).astype(np.float32)


def detect_sep(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            if line.lstrip().startswith("#"):
                continue
            sample = line
            break
        else:
            raise ValueError(f"{path}: empty file")

    if "\t" in sample and "," not in sample:
        return "\t"
    if "," in sample and "\t" not in sample:
        return ","

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", " "])
        return dialect.delimiter
    except Exception:
        return ","


def resolve_channel_indices(
    use_mi_channels: bool,
    mi_channel_names: list[str],
    drop_ch6: bool,
) -> tuple[list[str], int]:
    """
    Return:
      cols: list of column names to load from CSV (e.g., ["ch4","ch5",...])
      n_ch: number of channels
    """
    if use_mi_channels:
        idxs = []
        for name in mi_channel_names:
            name_u = name.strip().upper()
            if name_u not in CH_NAME_TO_IDX:
                raise ValueError(f"Unknown channel name: {name} (known keys: {list(CH_NAME_TO_IDX.keys())})")
            idxs.append(CH_NAME_TO_IDX[name_u])
    else:
        idxs = list(range(24))

    if drop_ch6 and 6 in idxs:
        idxs.remove(6)

    cols = [f"ch{i}" for i in idxs]
    return cols, len(cols)


def load_one_csv(
    path: Path,
    sr: int,
    window_sec: float,
    drop_head_sec: float,
    drop_tail_sec: float,
    cols: list[str],
) -> np.ndarray:
    """
    Returns X: (C, T_crop)
    Cropping is END-anchored:
      - take last window_sec seconds (window_samples)
      - then drop head and tail within that window
    Assumes pedal press is the LAST row in file.
    """
    sep = detect_sep(path)
    df = pd.read_csv(path, comment="#", sep=sep, engine="python")

    # validate columns exist
    if not all(c in df.columns for c in cols):
        raise ValueError(
            f"{path} delimiter guessed as {repr(sep)} but missing channel columns.\n"
            f"Need: {cols}\nFound: {list(df.columns)}"
        )

    data = df[cols].to_numpy(dtype=np.float32)  # (T, C)

    window_samples = int(round(window_sec * sr))
    drop_head = int(round(drop_head_sec * sr))
    drop_tail = int(round(drop_tail_sec * sr))

    if window_samples <= 0:
        raise ValueError("window_sec must be > 0")
    if drop_head + drop_tail >= window_samples:
        raise ValueError("drop_head_sec + drop_tail_sec must be < window_sec")

    # 1) take last window_samples ending at pedal (last row)
    if data.shape[0] < window_samples:
        pad = window_samples - data.shape[0]
        data_win = np.pad(data, ((pad, 0), (0, 0)), mode="edge")
    else:
        data_win = data[-window_samples:, :]  # (window_samples, C)

    # 2) drop head/tail inside window
    data_mid = data_win[drop_head: window_samples - drop_tail, :]  # (T_crop, C)

    X = data_mid.T.astype(np.float32)  # (C, T_crop)
    return X


def build_index(data_root: Path, labels: list[str]):
    files, y = [], []
    for li, lab in enumerate(labels):
        folder = data_root / lab
        if not folder.exists():
            continue
        for p in sorted(folder.glob("*.csv")):
            files.append(p)
            y.append(li)
    return files, np.array(y, dtype=np.int64)


class EEGCSVSet(Dataset):
    def __init__(
        self,
        files,
        y,
        sr: int,
        window_sec: float,
        drop_head_sec: float,
        drop_tail_sec: float,
        cols: list[str],
        bandpass: bool,
        bp_low: float,
        bp_high: float,
        zscore: bool,
    ):
        self.files = files
        self.y = y
        self.sr = sr
        self.window_sec = window_sec
        self.drop_head_sec = drop_head_sec
        self.drop_tail_sec = drop_tail_sec
        self.cols = cols
        self.bandpass = bandpass
        self.bp_low = bp_low
        self.bp_high = bp_high
        self.zscore = zscore

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        X = load_one_csv(
            self.files[idx],
            sr=self.sr,
            window_sec=self.window_sec,
            drop_head_sec=self.drop_head_sec,
            drop_tail_sec=self.drop_tail_sec,
            cols=self.cols,
        )
        if self.bandpass:
            X = butter_bandpass_filtfilt(X, self.sr, self.bp_low, self.bp_high, order=4)
        if self.zscore:
            X = per_channel_zscore(X)

        Xt = torch.from_numpy(X).unsqueeze(0)  # (1, C, T)
        yt = torch.tensor(self.y[idx], dtype=torch.long)
        return Xt, yt


class EEGNetV2(nn.Module):
    """
    EEGNet v2-ish for input (B, 1, C, T)
    """
    def __init__(
        self,
        n_ch: int,
        n_samples: int,
        n_classes: int,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kern_length: int = 64,
        sep_kern: int = 16,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, kern_length), padding=(0, kern_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        self.depthwise = nn.Conv2d(F1, F1 * D, kernel_size=(n_ch, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.act = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(dropout)

        self.sep_depth = nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, sep_kern), padding=(0, sep_kern // 2),
                                   groups=F1 * D, bias=False)
        self.sep_point = nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2 = nn.Dropout(dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_ch, n_samples)
            feat = self.forward_features(dummy)
            feat_dim = feat.shape[1]
        self.classifier = nn.Linear(feat_dim, n_classes)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.sep_depth(x)
        x = self.sep_point(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.pool2(x)
        x = self.drop2(x)

        return torch.flatten(x, start_dim=1)

    def forward(self, x):
        return self.classifier(self.forward_features(x))


def train_one_epoch(model, loader, opt, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ys, ps = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        ys.append(y.cpu().numpy())
        ps.append(pred.cpu().numpy())
    return total_loss / total, correct / total, np.concatenate(ys), np.concatenate(ps)


def plot_learning_curves(hist, out_path: Path):
    epochs = np.arange(1, len(hist["train_loss"]) + 1)
    plt.figure()
    plt.plot(epochs, hist["train_loss"], label="train_loss")
    plt.plot(epochs, hist["val_loss"], label="val_loss")
    plt.plot(epochs, hist["train_acc"], label="train_acc")
    plt.plot(epochs, hist["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion(cm, labels, out_path: Path, title="Confusion Matrix"):
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick = np.arange(len(labels))
    plt.xticks(tick, labels, rotation=45, ha="right")
    plt.yticks(tick, labels)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, f"{cm[i, j]:d}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()

    # data / labels
    ap.add_argument("--data_root", type=str, default="data")
    ap.add_argument("--labels", nargs="+", default=["left", "right"], help="Folder names under data_root")

    # sampling / cropping
    ap.add_argument("--sr", type=int, default=250)
    ap.add_argument("--window_sec", type=float, default=2.5, help="End-anchored window length (seconds)")
    ap.add_argument("--drop_head_sec", type=float, default=0.0, help="Drop from start of the selected window (seconds)")
    ap.add_argument("--drop_tail_sec", type=float, default=0.2, help="Drop closest to pedal (seconds)")
    ap.add_argument("--end_only_sec", type=float, default=None,
                    help="If set, override window_sec and use only last N seconds before drop_tail (e.g. 1.0)")

    # channels
    ap.add_argument("--use_mi_channels", action="store_true",
                    help="Use MI-relevant subset instead of all 24")
    ap.add_argument("--mi_channels", nargs="+", default=DEFAULT_MI_CH_NAMES,
                    help="Channel names to keep when --use_mi_channels (e.g. C3 C4 CZ FC5 FC6)")
    ap.add_argument("--drop_ch6", action="store_true", help="Drop ch6 (FC5 in your map) if present")

    # preprocessing
    ap.add_argument("--no_bandpass", action="store_true")
    ap.add_argument("--bp_low", type=float, default=8.0)
    ap.add_argument("--bp_high", type=float, default=30.0)
    ap.add_argument("--no_zscore", action="store_true")

    # training
    ap.add_argument("--result_dir", type=str, default="result")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--test_size", type=float, default=0.15)
    ap.add_argument("--val_size", type=float, default=0.15)
    ap.add_argument("--no_val", action="store_true",
                    help="Disable validation split + early stopping; train on train set only, evaluate on test set.")

    # model
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--F1", type=int, default=8)
    ap.add_argument("--D", type=int, default=2)
    ap.add_argument("--F2", type=int, default=16)

    args = ap.parse_args()
    set_seed(args.seed)

    labels = args.labels
    data_root = Path(args.data_root)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    # channels
    cols, n_ch = resolve_channel_indices(
        use_mi_channels=args.use_mi_channels,
        mi_channel_names=args.mi_channels,
        drop_ch6=args.drop_ch6
    )

    # window override for "끝 1초"
    window_sec = args.window_sec
    drop_head_sec = args.drop_head_sec
    drop_tail_sec = args.drop_tail_sec
    if args.end_only_sec is not None:
        # "end_only_sec" means: after dropping tail, keep last N seconds
        # simplest: set window_sec to end_only_sec + drop_tail_sec (+drop_head_sec if you still want head drop)
        window_sec = float(args.end_only_sec) + float(drop_tail_sec) + float(drop_head_sec)

    # build index
    files, y = build_index(data_root, labels)
    if len(files) == 0:
        raise SystemExit(f"No CSV files found under: {data_root.resolve()} for labels={labels}")

    idx_all = np.arange(len(files))

    # split
    tr_idx, te_idx = train_test_split(
        idx_all, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    if args.no_val:
        va_idx = np.array([], dtype=int)
    else:
        y_tr = y[tr_idx]
        val_rel = args.val_size / (1.0 - args.test_size)
        tr_idx, va_idx = train_test_split(
            tr_idx, test_size=val_rel, random_state=args.seed, stratify=y_tr
        )

    tr_files = [files[i] for i in tr_idx]
    te_files = [files[i] for i in te_idx]
    tr_y = y[tr_idx]
    te_y = y[te_idx]

    if args.no_val:
        va_files, va_y = [], np.array([], dtype=np.int64)
    else:
        va_files = [files[i] for i in va_idx]
        va_y = y[va_idx]

    bandpass = not args.no_bandpass
    zscore = not args.no_zscore

    # datasets
    tr_set = EEGCSVSet(
        tr_files, tr_y,
        sr=args.sr,
        window_sec=window_sec,
        drop_head_sec=drop_head_sec,
        drop_tail_sec=drop_tail_sec,
        cols=cols,
        bandpass=bandpass,
        bp_low=args.bp_low,
        bp_high=args.bp_high,
        zscore=zscore,
    )
    te_set = EEGCSVSet(
        te_files, te_y,
        sr=args.sr,
        window_sec=window_sec,
        drop_head_sec=drop_head_sec,
        drop_tail_sec=drop_tail_sec,
        cols=cols,
        bandpass=bandpass,
        bp_low=args.bp_low,
        bp_high=args.bp_high,
        zscore=zscore,
    )

    tr_loader = DataLoader(tr_set, batch_size=args.batch, shuffle=True, num_workers=0)
    te_loader = DataLoader(te_set, batch_size=args.batch, shuffle=False, num_workers=0)

    if not args.no_val:
        va_set = EEGCSVSet(
            va_files, va_y,
            sr=args.sr,
            window_sec=window_sec,
            drop_head_sec=drop_head_sec,
            drop_tail_sec=drop_tail_sec,
            cols=cols,
            bandpass=bandpass,
            bp_low=args.bp_low,
            bp_high=args.bp_high,
            zscore=zscore,
        )
        va_loader = DataLoader(va_set, batch_size=args.batch, shuffle=False, num_workers=0)
    else:
        va_set, va_loader = None, None

    # model
    # compute n_samples after cropping:
    window_samples = int(round(window_sec * args.sr))
    drop_head = int(round(drop_head_sec * args.sr))
    drop_tail = int(round(drop_tail_sec * args.sr))
    n_samples = window_samples - drop_head - drop_tail

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGNetV2(
        n_ch=n_ch,
        n_samples=n_samples,
        n_classes=len(labels),
        F1=args.F1,
        D=args.D,
        F2=args.F2,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val = float("inf")
    best_path = result_dir / "best_model.pt"
    wait = 0

    # training loop
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, tr_loader, opt, device)

        if args.no_val:
            # no early stopping; just log train
            hist["train_loss"].append(float(tr_loss))
            hist["train_acc"].append(float(tr_acc))
            print(f"Epoch {ep:03d}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.3f}")
            # save last
            torch.save(model.state_dict(), best_path)
            continue

        va_loss, va_acc, _, _ = eval_model(model, va_loader, device)
        hist["train_loss"].append(float(tr_loss))
        hist["val_loss"].append(float(va_loss))
        hist["train_acc"].append(float(tr_acc))
        hist["val_acc"].append(float(va_acc))

        print(f"Epoch {ep:03d}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")

        if va_loss < best_val - 1e-4:
            best_val = va_loss
            wait = 0
            torch.save(model.state_dict(), best_path)
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping at epoch {ep} (best val loss {best_val:.4f})")
                break

    # test
    model.load_state_dict(torch.load(best_path, map_location=device))

    # -----------------------------
    # Export to ONNX
    # -----------------------------
    onnx_path = result_dir / "eegnet.onnx"

    export_to_onnx(
        model,
        n_ch=n_ch,
        n_samples=n_samples,
        out_path=str(onnx_path),
        device=device,
    )


    te_loss, te_acc, y_true, y_pred = eval_model(model, te_loader, device)

    print("\nTest results")
    print(f"  loss: {te_loss:.4f}")
    print(f"  acc : {te_acc:.3f}\n")
    print(classification_report(y_true, y_pred, target_names=labels, digits=3))

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion(cm, labels, result_dir / "confusion_matrix.png")
    if not args.no_val:
        plot_learning_curves(hist, result_dir / "learning_curves.png")

    metrics = {
        "labels": labels,
        "test_loss": float(te_loss),
        "test_acc": float(te_acc),
        "confusion_matrix": cm.tolist(),
        "train_size": int(len(tr_set)),
        "val_size": int(0 if args.no_val else len(va_set)),
        "test_size": int(len(te_set)),
        "device": str(device),
        "sr": int(args.sr),
        "window_sec": float(window_sec),
        "drop_head_sec": float(drop_head_sec),
        "drop_tail_sec": float(drop_tail_sec),
        "n_samples": int(n_samples),
        "channels_cols": cols,
        "bandpass": bool(bandpass),
        "bp_low": float(args.bp_low),
        "bp_high": float(args.bp_high),
        "zscore": bool(zscore),
        "seed": int(args.seed),
        "lr": float(args.lr),
    }
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(f"  {(result_dir / 'confusion_matrix.png').resolve()}")
    if not args.no_val:
        print(f"  {(result_dir / 'learning_curves.png').resolve()}")
    print(f"  {best_path.resolve()}")
    print(f"  {(result_dir / 'metrics.json').resolve()}")


if __name__ == "__main__":
    main()
