import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pywt


################################################################################
# 数据预处理
################################################################################

def reduce_impedance_noise(imp: np.ndarray, window_length: int = 11, polyorder: int = 2) -> np.ndarray:
    if window_length % 2 == 0:
        window_length += 1
    return savgol_filter(imp, window_length=window_length, polyorder=polyorder)


################################################################################
# 加载Excel
################################################################################

def load_excel(
    file_path: Path,
    period_len: int,
    downsample_ratio: int = 1,
    use_wavelet: bool = False,
    wavelet_name: str = 'db4',
    wavelet_levels: int = 4,
    wavelet_max_coefs: int = 300,
    plot_curves: bool = False,
    plot_output_dir: Path = None
) -> Tuple[List[torch.Tensor], List[float]]:
    xl = pd.ExcelFile(file_path)
    sequences, targets = [], []

    if plot_curves and plot_output_dir:
        plot_output_dir.mkdir(parents=True, exist_ok=True)

    for sheet in xl.sheet_names:
        real_label = float(sheet)
        df = xl.parse(sheet, engine="openpyxl").dropna(how="all")
        if df.shape[1] < 3:
            raise ValueError(f"Sheet '{sheet}' has fewer than 3 columns.")

        imp = df.iloc[:, 2].to_numpy(dtype=float)
        if downsample_ratio > 1:
            imp = imp[::downsample_ratio]
        imp = np.round(imp, 2)

        if use_wavelet:
            coeffs = pywt.wavedec(imp, wavelet=wavelet_name, level=wavelet_levels)
            chans = []
            for c in coeffs:
                if len(c) >= wavelet_max_coefs:
                    ci = c[:wavelet_max_coefs]
                else:
                    ci = np.pad(c, (0, wavelet_max_coefs - len(c)), mode='constant')
                chans.append(ci)
            arr = np.stack(chans, axis=0).T
            mean = arr.mean(axis=0, keepdims=True)
            std = arr.std(axis=0, keepdims=True) + 1e-6
            arr = (arr - mean) / std
            seq_tensor = torch.tensor(arr, dtype=torch.float32)
        else:
            if imp.size >= period_len:
                arr = imp[:period_len]
            else:
                arr = np.pad(imp, (0, period_len - imp.size), mode='constant')
            seq_tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(-1)

        sequences.append(seq_tensor)
        targets.append(real_label)

    return sequences, targets


################################################################################
# Dataset
################################################################################

class PeriodDataset(Dataset):
    def __init__(self, sequences: List[torch.Tensor], targets: List[float], seg_len: int, random_segment: bool = True):
        self.sequences = sequences
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.seg_len = seg_len
        self.random_segment = random_segment

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        full_seq = self.sequences[idx]
        L = full_seq.size(0)
        if L < self.seg_len:
            raise ValueError(f"Sequence length ({L}) < segment length ({self.seg_len})")
        start = random.randint(0, L - self.seg_len) if self.random_segment else 0
        segment = full_seq[start: start + self.seg_len]
        return segment, self.targets[idx]


def collate_fn(batch):
    seqs, tgts = zip(*batch)
    seqs = torch.stack(seqs, dim=0)
    lengths = torch.full((len(seqs),), seqs.size(1), dtype=torch.long)
    targets = torch.stack(tgts, dim=0)
    return seqs, lengths, targets


################################################################################
# Transformer模型
################################################################################

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PeriodRegressorTransformer(nn.Module):
    def __init__(self, input_size=1, d_model=64, num_layers=2, nhead=4,
                 dim_feedforward=128, dropout=0.1, max_seq_len=2068):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.regressor = nn.Linear(d_model, 1)

    def forward(self, x, lengths):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        out = x.mean(dim=0)
        return self.regressor(out).squeeze(-1)


################################################################################
# 训练 & 预测
################################################################################

def train_epoch(model, loader, criterion, optim, sched, device):
    model.train()
    running = 0.0
    for x, lengths, y in loader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        optim.zero_grad()
        y_pred = model(x, lengths)
        loss = criterion(y_pred, y)
        loss.backward()
        optim.step()
        sched.step()
        running += loss.item() * y.size(0)
    return running / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    running = 0.0
    with torch.no_grad():
        for x, lengths, y in loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            y_pred = model(x, lengths)
            loss = criterion(y_pred, y)
            running += loss.item() * y.size(0)
    return running / len(loader.dataset)


################################################################################
# 主函数
################################################################################

def main():
    parser = argparse.ArgumentParser(description="Train/predict with DWT preprocessing (regression)")
    parser.add_argument("--wavelet", default=True)
    parser.add_argument("--wavelet-name", type=str, default="db4")
    parser.add_argument("--wavelet-levels", type=int, default=8)
    parser.add_argument("--wavelet-max-coefs", type=int, default=50)
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    parser.add_argument("--train-file", type=Path,
                        default=r"C:\Users\wsco\Desktop\velocity_pred\impedence\impedence2.xlsx")
    parser.add_argument("--eval-file", type=Path,
                        default=r"C:\Users\wsco\Desktop\velocity_pred\impedence\1x_test1.xlsx")
    parser.add_argument("--model", type=Path, default="best_model.pt")
    parser.add_argument("--period", type=int, default=50)
    parser.add_argument("--downsample", type=int, default=5)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--plot-dir", type=Path, default=Path("./plots"),
                        help="Directory to save output plots")  # ✅ 新增

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg_len = args.wavelet_max_coefs if args.wavelet else args.period
    in_size = (args.wavelet_levels + 1) if args.wavelet else 1

    label_norm_file = args.model.with_suffix('.label_norm.npy')

    load_kwargs = dict(
        period_len=args.period,
        downsample_ratio=args.downsample,
        use_wavelet=args.wavelet,
        wavelet_name=args.wavelet_name,
        wavelet_levels=args.wavelet_levels,
        wavelet_max_coefs=args.wavelet_max_coefs
    )

    if args.mode == "train":
        seq_tr, tgt_tr = load_excel(args.train_file, **load_kwargs)
        seq_ev, tgt_ev = load_excel(args.eval_file, **load_kwargs)

        label_min = min(tgt_tr)
        label_max = max(tgt_tr)
        np.save(label_norm_file, np.array([label_min, label_max]))
        print(f"[INFO] Label normalization: min={label_min}, max={label_max}")

        tgt_tr = [(x - label_min) / (label_max - label_min) for x in tgt_tr]
        tgt_ev = [(x - label_min) / (label_max - label_min) for x in tgt_ev]

        tr_loader = DataLoader(
            PeriodDataset(seq_tr, tgt_tr, seg_len=seg_len, random_segment=True),
            batch_size=args.batch, shuffle=True, collate_fn=collate_fn
        )
        ev_loader = DataLoader(
            PeriodDataset(seq_ev, tgt_ev, seg_len=seg_len, random_segment=False),
            batch_size=args.batch, shuffle=False, collate_fn=collate_fn
        )

        model = PeriodRegressorTransformer(
            input_size=in_size,
            d_model=64,
            num_layers=4,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            max_seq_len=seg_len
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs * len(tr_loader))

        best_val = float("inf")
        for epoch in range(1, args.epochs + 1):
            tr_loss = train_epoch(model, tr_loader, criterion, optimizer, sched, device)
            ev_loss = evaluate(model, ev_loader, criterion, device)

            if ev_loss < best_val:
                best_val = ev_loss
                torch.save(model.state_dict(), args.model)

            print(f"Epoch {epoch:04d} | train_loss={tr_loss:.4f} | val_loss={ev_loss:.4f}")

        print(f"Training complete. Best val MSE: {best_val:.4f}")

    else:
        label_min, label_max = np.load(label_norm_file)

        _, _ = load_excel(args.train_file, **load_kwargs)
        seq_ev, tgt_ev = load_excel(args.eval_file, **load_kwargs)

        tgt_ev = [(x - label_min) / (label_max - label_min) for x in tgt_ev]

        loader = DataLoader(
            PeriodDataset(seq_ev, tgt_ev, seg_len=seg_len, random_segment=False),
            batch_size=args.batch, shuffle=False, collate_fn=collate_fn
        )

        model = PeriodRegressorTransformer(
            input_size=in_size,
            d_model=64,
            num_layers=4,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            max_seq_len=seg_len
        ).to(device)
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.eval()

        preds, trues = [], []
        with torch.no_grad():
            for x, lengths, y in loader:
                x, lengths = x.to(device), lengths.to(device)
                out = model(x, lengths)
                preds.extend(out.cpu().tolist())
                trues.extend(y.tolist())

        preds = [p * (label_max - label_min) + label_min for p in preds]
        trues = [t * (label_max - label_min) + label_min for t in trues]

        df = pd.DataFrame({
            "index": range(len(preds)),
            "true_label": trues,
            "pred_label": preds
        })
        df.to_csv("predictions.csv", index=False)
        print(f"Saved predictions to predictions.csv")

        # ✅ 新增部分：预测 vs 实际对比图，并自动打开
        args.plot_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8, 5))
        plt.plot(trues, label="True", marker='o')
        plt.plot(preds, label="Predicted", marker='x')
        plt.xlabel("Sample Index")
        plt.ylabel("Flow Rate (or Regression Target)")
        plt.title("Prediction vs. Ground Truth")
        plt.legend()

        fig_out = args.plot_dir / "prediction_vs_truth.png"
        plt.tight_layout()
        plt.savefig(fig_out)
        print(f"Prediction vs truth plot saved to {fig_out.resolve()}")
        plt.close()

        import os
        os.startfile(fig_out.resolve())


if __name__ == "__main__":
    main()
