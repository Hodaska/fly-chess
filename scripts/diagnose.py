"""Диагностика Фазы 3: откуда берутся 10.5% и много это или мало.

Считает на ОДНОМ И ТОМ ЖЕ train/val сплите несколько моделей:

  A. prior      — всегда самый частый класс хода (пол, floor)
  B. board      — линейная голова прямо по 768 битам доски, БЕЗ мозга
                  (потолок для линейной политики с идеальным доступом к доске)
  C. brain      — линейная голова по 1312 признакам мозга (то, что у нас есть)
  D. random     — линейная голова по 1312 случайным проекциям битов доски
                  (контроль на размерность: столько же чисел, но без мозга)

Плюс «честная» метрика игры: accuracy с маской легальных ходов (в обучении
маска не применяется, поэтому val_acc из train_readout.py занижен).

Плюс проверка дрейфа признаков: в lif.py reset_state() НЕ сбрасывает пороги
гомеостаза, поэтому они уезжают по ходу прогона 150k позиций. Смотрим, есть
ли систематический тренд признаков по индексу позиции.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess  # noqa: E402

from flybrain.chess_encoding import (  # noqa: E402
    NUM_MOVE_CLASSES,
    board_to_bits,
    legal_move_mask,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0
VAL_FRACTION = 0.15
BATCH = 4096
EPOCHS = 40


def split_indices(n: int):
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n)
    n_val = int(n * VAL_FRACTION)
    return perm[n_val:], perm[:n_val]


def train_linear(X: np.ndarray, y: np.ndarray, tr, va, tag: str, epochs: int = EPOCHS):
    """Линейная голова X -> 4096 классов. Возвращает (val_acc, логиты валидации)."""
    mean = X[tr].mean(0, keepdims=True)
    std = X[tr].std(0, keepdims=True) + 1e-6

    Xt = torch.from_numpy((X - mean) / std).float()
    yt = torch.from_numpy(y).long()

    Xtr, ytr = Xt[tr].to(DEVICE), yt[tr].to(DEVICE)
    Xva, yva = Xt[va].to(DEVICE), yt[va].to(DEVICE)
    del Xt

    model = nn.Linear(X.shape[1], NUM_MOVE_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()

    best_acc, best_logits = 0.0, None
    n_tr = len(ytr)

    for ep in range(epochs):
        model.train()
        order = torch.randperm(n_tr, device=DEVICE)
        for lo in range(0, n_tr, BATCH):
            idx = order[lo : lo + BATCH]
            opt.zero_grad()
            lossf(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            correct = 0
            logits_chunks = []
            for lo in range(0, len(yva), BATCH):
                lg = model(Xva[lo : lo + BATCH])
                correct += (lg.argmax(1) == yva[lo : lo + BATCH]).sum().item()
                logits_chunks.append(lg.cpu())
            acc = correct / len(yva)
            if acc > best_acc:
                best_acc = acc
                best_logits = torch.cat(logits_chunks)

    print(f"  [{tag}] val_acc (без маски) = {best_acc * 100:.2f}%")
    del Xtr, Xva, model
    torch.cuda.empty_cache()
    return best_acc, best_logits


def masked_accuracy(logits: torch.Tensor, y: np.ndarray, fens: list[str]) -> float:
    """Точность после наложения маски легальных ходов — честная метрика игры."""
    correct = 0
    for i, fen in enumerate(fens):
        mask, _ = legal_move_mask(chess.Board(fen))
        lg = logits[i].numpy().copy()
        lg[~mask] = -np.inf
        if int(np.argmax(lg)) == y[i]:
            correct += 1
    return correct / len(fens)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    print("загружаю кэш и датасет...")
    cache = np.load("data/features_cache.npz")
    feats = cache["features"]
    labels = cache["labels"]
    df = pd.read_parquet("data/dataset.parquet")
    fens_all = df["fen"].tolist()
    n = len(labels)
    print(f"{n:,} позиций, признаков мозга: {feats.shape[1]}")

    tr, va = split_indices(n)
    y_va = labels[va]
    fens_va = [fens_all[i] for i in va]

    # ---------- проверка дрейфа гомеостаза ----------
    print("\n=== дрейф признаков по ходу прогона (пороги не сбрасывались) ===")
    idx = np.arange(n, dtype=np.float64)
    total = feats.sum(1)
    drift_r = np.corrcoef(idx, total)[0, 1]
    early, late = total[: n // 10].mean(), total[-n // 10 :].mean()
    print(f"корреляция(индекс позиции, суммарная активность) = {drift_r:+.3f}")
    print(f"средняя активность: первые 10% = {early:.0f}, последние 10% = {late:.0f}"
          f"  (изменение {100 * (late - early) / early:+.1f}%)")

    # ---------- структура признаков ----------
    print("\n=== структура признаков мозга ===")
    std_per_neuron = feats.std(0)
    dead = int((std_per_neuron < 1e-6).sum())
    print(f"мёртвых нейронов (нулевая дисперсия по всем позициям): {dead} из {feats.shape[1]}")
    print(f"медианный std по нейронам: {np.median(std_per_neuron):.2f}, "
          f"медианное среднее: {np.median(feats.mean(0)):.1f}")
    alive = feats[:, std_per_neuron > 1e-6]
    if alive.shape[1] > 1:
        z = (alive - alive.mean(0)) / (alive.std(0) + 1e-9)
        sv = np.linalg.svd(z[:5000], compute_uv=False)
        var = sv**2 / (sv**2).sum()
        rank95 = int(np.searchsorted(np.cumsum(var), 0.95) + 1)
        print(f"эффективная размерность (95% дисперсии): {rank95} из {alive.shape[1]}")

    # ---------- базовые линии ----------
    print("\n=== сравнение моделей (один и тот же сплит) ===")

    counts = np.bincount(labels[tr], minlength=NUM_MOVE_CLASSES)
    prior_cls = int(counts.argmax())
    prior_acc = float((y_va == prior_cls).mean())
    print(f"  [A prior] val_acc = {prior_acc * 100:.2f}%  (всегда один класс)")

    print("  кодирую доски в биты...")
    bits = np.stack([board_to_bits(chess.Board(f)) for f in fens_all]).astype(np.float32)

    acc_board, lg_board = train_linear(bits, labels, tr, va, "B board 768")

    rng = np.random.RandomState(1)
    proj = rng.randn(bits.shape[1], feats.shape[1]).astype(np.float32) / np.sqrt(bits.shape[1])
    rand_feats = np.maximum(bits @ proj, 0)  # ReLU случайная проекция той же размерности
    acc_rand, lg_rand = train_linear(rand_feats, labels, tr, va, "D random 1312")
    del rand_feats

    acc_brain, lg_brain = train_linear(feats, labels, tr, va, "C brain 1312")

    # ---------- честная метрика с маской ----------
    print("\n=== с маской легальных ходов (честная метрика игры) ===")
    t0 = time.time()
    for tag, lg in (("B board", lg_board), ("D random", lg_rand), ("C brain", lg_brain)):
        m = masked_accuracy(lg, y_va, fens_va)
        print(f"  [{tag}] masked val_acc = {m * 100:.2f}%")
    print(f"  (посчитано за {time.time() - t0:.0f} с)")

    # случайный выбор среди легальных — настоящий пол для игры
    rng2 = np.random.RandomState(7)
    hits = 0
    for i, fen in enumerate(fens_va):
        mask, _ = legal_move_mask(chess.Board(fen))
        legal_cls = np.flatnonzero(mask)
        if len(legal_cls) and rng2.choice(legal_cls) == y_va[i]:
            hits += 1
    print(f"  [пол] случайный легальный ход = {hits / len(fens_va) * 100:.2f}%")


if __name__ == "__main__":
    main()
