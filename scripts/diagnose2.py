"""Где теряется информация: можно ли восстановить доску из признаков мозга.

diagnose.py показал: линейная модель ПРЯМО ПО ДОСКЕ даёт 43%, случайная
проекция доски в 1312 чисел — тоже 43%, а мозг — 10%. Значит дело не в
размерности и не в сложности задачи: мозг теряет информацию по дороге.

Здесь меряем это напрямую и дёшево, без переобучения классификатора на 4096
классов: пробуем линейно восстановить 768 бит доски из признаков.
Это гораздо более sample-efficient метрика, чем точность хода, поэтому её
можно использовать для быстрого перебора конфигураций экстрактора.

Заодно проверяем гипотезу «вся дисперсия признаков — это общий уровень
активности, а не то, какая на доске позиция».
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess  # noqa: E402

from flybrain.chess_encoding import board_to_bits  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ridge_r2(X: np.ndarray, Y: np.ndarray, tr, va, lam: float = 1.0) -> tuple[float, float]:
    """Гребневая регрессия X -> Y в замкнутой форме. Возвращает (R2, доля бит
    угаданных лучше, чем константой)."""
    Xt = torch.from_numpy(X).float().to(DEVICE)
    Yt = torch.from_numpy(Y).float().to(DEVICE)

    mu, sd = Xt[tr].mean(0, keepdim=True), Xt[tr].std(0, keepdim=True) + 1e-6
    Xt = (Xt - mu) / sd
    ones = torch.ones(len(Xt), 1, device=DEVICE)
    Xt = torch.cat([Xt, ones], 1)

    A = Xt[tr]
    B = Yt[tr]
    G = A.T @ A + lam * torch.eye(A.shape[1], device=DEVICE)
    W = torch.linalg.solve(G, A.T @ B)

    pred = Xt[va] @ W
    true = Yt[va]
    ss_res = ((pred - true) ** 2).sum(0)
    ss_tot = ((true - true.mean(0, keepdim=True)) ** 2).sum(0)
    r2_per_bit = 1 - ss_res / (ss_tot + 1e-9)
    valid = ss_tot > 1e-6

    r2 = float(r2_per_bit[valid].mean().item())
    good = float((r2_per_bit[valid] > 0.5).float().mean().item())

    del Xt, Yt, A, B, G, W, pred, true
    torch.cuda.empty_cache()
    return r2, good


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    cache = np.load("data/features_cache.npz")
    feats = cache["features"]
    df = pd.read_parquet("data/dataset.parquet")
    n = len(df)

    print("кодирую доски...")
    bits = np.stack([board_to_bits(chess.Board(f)) for f in df["fen"]]).astype(np.float32)

    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    n_va = int(n * 0.15)
    va, tr = perm[:n_va], perm[n_va:]

    # ограничим обучающую выборку, чтобы матрица Грама считалась быстро
    tr = tr[:60000]

    print("\n=== насколько доска восстановима из признаков (линейно) ===")

    r2_b, good_b = ridge_r2(feats, bits, tr, va)
    print(f"  из признаков МОЗГА:            R2 = {r2_b:+.3f}   "
          f"бит с R2>0.5: {good_b * 100:.1f}%")

    proj = rng.randn(768, 1312).astype(np.float32) / np.sqrt(768)
    rand_feats = np.maximum(bits @ proj, 0)
    r2_r, good_r = ridge_r2(rand_feats, bits, tr, va)
    print(f"  из СЛУЧАЙНОЙ проекции доски:   R2 = {r2_r:+.3f}   "
          f"бит с R2>0.5: {good_r * 100:.1f}%")
    del rand_feats

    print("\n=== сколько дисперсии признаков — это просто общий уровень активности ===")
    total = feats.sum(1, keepdims=True)
    z = feats - feats.mean(0, keepdims=True)
    t = total - total.mean()
    # доля дисперсии каждого нейрона, объяснимая общей активностью
    beta = (z * t).sum(0) / ((t**2).sum() + 1e-9)
    explained = (beta**2) * (t**2).sum(0) / ((z**2).sum(0) + 1e-9)
    print(f"  медианная доля дисперсии нейрона, объяснимая общей активностью: "
          f"{np.median(explained) * 100:.1f}%")

    # остаток после вычитания общей активности — есть ли там сигнал о доске
    resid = z - beta[None, :] * t
    r2_res, good_res = ridge_r2(np.ascontiguousarray(resid), bits, tr, va)
    print(f"  доска из ОСТАТКА (без общей активности): R2 = {r2_res:+.3f}   "
          f"бит с R2>0.5: {good_res * 100:.1f}%")

    print("\n=== насколько признаки зависят от позиции в прогоне (дрейф порогов) ===")
    idx = (np.arange(n, dtype=np.float32) / n)[:, None]
    r2_drift, _ = ridge_r2(feats, idx, tr, va)
    print(f"  индекс позиции предсказывается из признаков: R2 = {r2_drift:+.3f}")
    print("  (высокий R2 = признаки сильно помнят, когда их посчитали, а не что на доске)")


if __name__ == "__main__":
    main()
