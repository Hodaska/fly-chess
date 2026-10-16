"""Быстрая проверка: даёт ли новая конфигурация реальный прирост точности хода.

probe.py показал, что сигнал о доске живёт в зрительной зоне (R2=0.86) и гибнет
до нисходящих нейронов (R2=-0.18). Отсюда решение — снимать широкую выборку по
всему мозгу, а не только с нисходящих.

Прежде чем гонять полный прогон на 150k (несколько часов), проверяем на выборке:
реально ли точность хода прыгает с ~10% до чего-то приличного.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore", message=".*Sparse CSR.*")

import chess  # noqa: E402

from flybrain.chess_encoding import (  # noqa: E402
    NUM_MOVE_CLASSES,
    board_to_bits,
    legal_move_mask,
)
from flybrain.connectome import load_connectome  # noqa: E402
from flybrain.features import BrainFeatureExtractor  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_POS = 20000
BATCH = 64
READOUT_SIZE = 2048


def train_head(X, y, tr, va, tag, epochs=60):
    mean, std = X[tr].mean(0, keepdims=True), X[tr].std(0, keepdims=True) + 1e-6
    Xt = torch.from_numpy((X - mean) / std).float()
    yt = torch.from_numpy(y).long()
    Xtr, ytr = Xt[tr].to(DEVICE), yt[tr].to(DEVICE)
    Xva, yva = Xt[va].to(DEVICE), yt[va].to(DEVICE)
    del Xt

    model = nn.Linear(X.shape[1], NUM_MOVE_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    best, best_lg = 0.0, None

    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(ytr), device=DEVICE)
        for lo in range(0, len(ytr), 4096):
            idx = order[lo : lo + 4096]
            opt.zero_grad()
            lossf(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            lg = torch.cat([model(Xva[i : i + 4096]).cpu() for i in range(0, len(yva), 4096)])
            acc = (lg.argmax(1) == yva.cpu()).float().mean().item()
            if acc > best:
                best, best_lg = acc, lg
    print(f"  [{tag}] без маски = {best * 100:.2f}%")
    del Xtr, Xva, model
    torch.cuda.empty_cache()
    return best, best_lg


def masked_acc(logits, y, fens):
    ok = 0
    for i, fen in enumerate(fens):
        m, _ = legal_move_mask(chess.Board(fen))
        lg = logits[i].numpy().copy()
        lg[~m] = -np.inf
        ok += int(np.argmax(lg)) == y[i]
    return ok / len(fens)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    df = pd.read_parquet("data/dataset.parquet")
    step = max(1, len(df) // N_POS)
    df = df.iloc[::step].head(N_POS).reset_index(drop=True)
    n = len(df)
    fens = df["fen"].tolist()
    labels = df["move_class"].to_numpy(np.int64)
    bits = np.stack([board_to_bits(chess.Board(f)) for f in fens]).astype(np.float32)

    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    n_va = int(n * 0.2)
    va, tr = perm[:n_va], perm[n_va:]
    fens_va = [fens[i] for i in va]
    y_va = labels[va]

    print(f"позиций: {n:,}")

    print("\nизвлекаю признаки мозга (широкий съём, замороженные пороги)...")
    c = load_connectome(min_weight=5)
    ex = BrainFeatureExtractor(
        c, batch=BATCH, readout="wide", readout_size=READOUT_SIZE,
        calibrate_steps=1500, signal="spikes",
    )
    print(f"частота после калибровки: {ex.calibrated_rate * 100:.2f}%")

    feats = np.zeros((n, ex.feature_dim), dtype=np.float32)
    t0 = time.time()
    for lo in range(0, n, BATCH):
        chunk = bits[lo : lo + BATCH]
        feats[lo : lo + len(chunk)] = (
            ex.extract(chunk) if len(chunk) == BATCH else ex.extract_padded(chunk)
        )
    el = time.time() - t0
    print(f"готово за {el / 60:.1f} мин ({n / el:.1f} поз/с)")
    del ex
    torch.cuda.empty_cache()

    print("\n=== точность предсказания хода ===")
    acc_b, lg_b = train_head(bits, labels, tr, va, "доска напрямую (ориентир)")
    acc_f, lg_f = train_head(feats, labels, tr, va, "МОЗГ, новый съём")

    print("\n=== с маской легальных ходов (реальная игра) ===")
    print(f"  [доска напрямую] {masked_acc(lg_b, y_va, fens_va) * 100:.2f}%")
    print(f"  [МОЗГ, новый съём] {masked_acc(lg_f, y_va, fens_va) * 100:.2f}%")
    print("\n  для сравнения, старый съём на полных 150k: 10.34% / 17.78% с маской")


if __name__ == "__main__":
    main()
