"""Сколько шагов симуляции реально нужно.

Сейчас на каждую позицию гоняется 200 шагов, и это главная статья расходов:
GPU сидит на 746 МГц (потребительская Pascal не разгоняется под разреженные
матрицы, залочить частоту нельзя), поэтому идёт ~4.7 поз/с, и полный прогон
на 150k занял бы ~9 часов.

Сигнал о доске живёт в зрительных нейронах, которым ток подаётся напрямую —
вероятно, длинное окно им не нужно. Проверяем.

Трюк: накопленные счётчики спайков на 25/50/100/200 шагах снимаются за ОДИН
прогон, а не четырьмя отдельными.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore", message=".*Sparse CSR.*")

import chess  # noqa: E402

from flybrain.chess_encoding import board_to_bits  # noqa: E402
from flybrain.connectome import load_connectome  # noqa: E402
from flybrain.features import build_pool_matrix, build_sensory_pools  # noqa: E402
from flybrain.lif import LIFBrain, LIFParams  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_POS = 3000
BATCH = 64
CHECKPOINTS = (25, 50, 100, 200)
READOUT = 2048
AMPLITUDE = 0.6


def ridge_r2(X, Y, tr, va, lam=10.0):
    Xt = torch.from_numpy(np.ascontiguousarray(X)).float().to(DEVICE)
    Yt = torch.from_numpy(Y).float().to(DEVICE)
    mu, sd = Xt[tr].mean(0, keepdim=True), Xt[tr].std(0, keepdim=True) + 1e-6
    Xt = torch.cat([(Xt - mu) / sd, torch.ones(len(Xt), 1, device=DEVICE)], 1)
    A, B = Xt[tr], Yt[tr]
    W = torch.linalg.solve(A.T @ A + lam * torch.eye(A.shape[1], device=DEVICE), A.T @ B)
    pred, true = Xt[va] @ W, Yt[va]
    ss_res = ((pred - true) ** 2).sum(0)
    ss_tot = ((true - true.mean(0, keepdim=True)) ** 2).sum(0)
    r2 = 1 - ss_res / (ss_tot + 1e-9)
    valid = ss_tot > 1e-6
    out = float(r2[valid].mean().item())
    del Xt, Yt, A, B, W, pred, true
    torch.cuda.empty_cache()
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    df = pd.read_parquet("data/dataset.parquet")
    df = df.iloc[:: max(1, len(df) // N_POS)].head(N_POS)
    bits = np.stack([board_to_bits(chess.Board(f)) for f in df["fen"]]).astype(np.float32)
    n = len(bits)
    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    va, tr = perm[: int(n * 0.2)], perm[int(n * 0.2) :]

    print(f"позиций: {n:,}")
    c = load_connectome(min_weight=5)

    probes = np.sort(rng.choice(c.num_neurons, READOUT, replace=False))
    probes_t = torch.from_numpy(probes.astype(np.int64)).to(DEVICE)

    bucket = build_sensory_pools(c, 768, seed=0)
    pool_m = build_pool_matrix(bucket, 768, DEVICE)

    brain = LIFBrain(c, params=LIFParams(), batch=BATCH, device=str(DEVICE))
    calib = np.zeros((BATCH, 768), dtype=np.float32)
    for b in range(BATCH):
        calib[b, rng.choice(768, 32, replace=False)] = 1.0
    ext_c = AMPLITUDE * torch.sparse.mm(pool_m, torch.from_numpy(calib.T).to(DEVICE))
    print(f"частота после калибровки: {brain.calibrate_and_freeze(ext_c, 1500) * 100:.2f}%")

    snaps = {k: np.zeros((n, READOUT), dtype=np.float32) for k in CHECKPOINTS}

    t0 = time.time()
    for lo in range(0, n, BATCH):
        chunk = bits[lo : lo + BATCH]
        pad = BATCH - len(chunk)
        if pad:
            chunk = np.vstack([chunk, np.zeros((pad, 768), dtype=np.float32)])
        ext = AMPLITUDE * torch.sparse.mm(pool_m, torch.from_numpy(chunk.T).to(DEVICE))

        brain.reset_state()
        counts = torch.zeros((brain.n, BATCH), device=DEVICE)
        for s in range(1, max(CHECKPOINTS) + 1):
            counts += brain.step(ext)
            if s in CHECKPOINTS:
                take = BATCH - pad
                snaps[s][lo : lo + take] = (
                    counts.index_select(0, probes_t).T[:take].cpu().numpy()
                )
    el = time.time() - t0
    print(f"симуляция: {el / 60:.1f} мин ({n / el:.1f} поз/с при 200 шагах)\n")

    print(f"{'шагов':>7} {'R2 доски':>10} {'спайков/нейрон':>16} {'оценка на 150k':>16}")
    print("-" * 54)
    for k in CHECKPOINTS:
        r2 = ridge_r2(snaps[k], bits, tr, va)
        rate = n / el * (200 / k)  # линейно по числу шагов
        print(f"{k:>7} {r2:>+10.3f} {snaps[k].mean():>16.2f} {150000 / rate / 3600:>14.1f} ч")


if __name__ == "__main__":
    main()
