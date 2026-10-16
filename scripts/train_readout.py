"""Обучить линейный readout на кэшированных признаках (behavior cloning от Stockfish).

Граф мозга здесь вообще не участвует — только кэш из cache_features.py.
Поэтому весь цикл обучения занимает минуты, а не часы (см. заметки проекта, Фаза 2, п.5).

Метрика — top-1 accuracy по классу хода, который выбрал Stockfish. Ожидаемый
результат по плану: 30-50% (см. заметки проекта, ориентир BPU: 60% на 10k партий).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flybrain.readout import MoveReadout  # noqa: E402


# Конфигурация ночного прогона 23.09 — для кэшей, посчитанных до того, как
# настройки экстрактора начали сохраняться в сам файл.
LEGACY_EXTRACTOR_CONFIG = {
    "min_weight": 5,
    "steps": 100,
    "readout": "wide",
    "readout_size": 2048,
    "calibrate_steps": 1500,
    "batch": 64,
    "seed": 0,
}


def load_cache(path: str):
    data = np.load(path)
    if "extractor_config" in data.files:
        config = json.loads(str(data["extractor_config"]))
    else:
        config = LEGACY_EXTRACTOR_CONFIG
    return data["features"], data["labels"], int(data["feature_dim"]), config


@torch.no_grad()
def _batched_accuracy(model: MoveReadout, X: torch.Tensor, y: torch.Tensor, batch_size: int) -> float:
    """Точность по чанкам — полный форвард на 150k x 4096 классов сам по себе
    требует ~2.5 ГБ на одни логиты и не влезает в 3 ГБ видеопамяти вместе
    с остальным (см. баг ниже про OOM)."""
    model.eval()
    correct = 0
    for lo in range(0, len(y), batch_size):
        hi = min(lo + batch_size, len(y))
        correct += (model(X[lo:hi]).argmax(1) == y[lo:hi]).sum().item()
    return correct / len(y)


def train(
    features: np.ndarray,
    labels: np.ndarray,
    feature_dim: int,
    epochs: int = 30,
    lr: float = 1e-3,
    val_fraction: float = 0.15,
    seed: int = 0,
    device: str | None = None,
    batch_size: int = 4096,
) -> tuple[MoveReadout, dict]:
    """Мини-батчевое обучение.

    🐛 БАГ, НАЙДЕННЫЙ НА РЕАЛЬНОМ ПРОГОНЕ (150k позиций, GTX 1060 3GB):
    исходная версия делала full-batch градиентный спуск — весь датасет одним
    форвардом. При 4096 классах хода логиты на 150k позиций занимают
    150000*4096*4 байта ≈ 2.5 ГБ САМИ ПО СЕБЕ, плюс градиенты того же размера
    на backward — итог `CUDA out of memory` на карте с 3 ГБ. На дымовом тесте
    (300 позиций) это не всплывало, потому что 300*4096*4 ≈ 5 МБ — ловушка
    "работает на маленьком, падает на настоящем размере". Мораль: full-batch
    training с большим числом классов масштабируется по памяти как
    N_позиций x N_классов, а не как N_позиций — это стоило проверить сразу.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    n = len(labels)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = int(n * val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    # Нормализация признаков: сырые спайковые счётчики имеют произвольный
    # масштаб (см. features.py — суммы порядка тысяч), стандартизация
    # заметно стабилизирует обучение линейного слоя.
    mean = features[train_idx].mean(axis=0, keepdims=True)
    std = features[train_idx].std(axis=0, keepdims=True) + 1e-6

    # ⚠️ Нарезаем train/val НА CPU и только потом переносим на GPU.
    # Если сделать наоборот (X целиком на GPU, а потом X[train_idx]), на карте
    # окажется и полная матрица, и обе её копии: при 150k x 2048 это
    # 1.23 + 1.04 + 0.18 = 2.46 ГБ при доступных ~2.37 ГБ -> OOM в самом конце
    # многочасового прогона. Тот же класс ошибки уже один раз положил пайплайн.
    X_cpu = torch.from_numpy((features - mean) / std).float()
    y_cpu = torch.from_numpy(labels.copy()).long()

    Xtr = X_cpu[train_idx].to(device)
    ytr = y_cpu[train_idx].to(device)
    Xval = X_cpu[val_idx].to(device)
    yval = y_cpu[val_idx].to(device)
    del X_cpu, y_cpu

    model = MoveReadout(feature_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    n_train = len(ytr)

    history = {"train_acc": [], "val_acc": [], "train_loss": []}
    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        epoch_order = torch.randperm(n_train, device=device)
        running_loss = 0.0
        num_batches = 0

        for lo in range(0, n_train, batch_size):
            idx = epoch_order[lo : lo + batch_size]
            optim.zero_grad()
            logits = model(Xtr[idx])
            loss = loss_fn(logits, ytr[idx])
            loss.backward()
            optim.step()
            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / num_batches
        train_acc = _batched_accuracy(model, Xtr, ytr, batch_size)
        val_acc = _batched_accuracy(model, Xval, yval, batch_size)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(avg_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(
            f"эпоха {epoch + 1:3d}/{epochs}  loss={avg_loss:.4f}  "
            f"train_acc={train_acc * 100:5.2f}%  val_acc={val_acc * 100:5.2f}%"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\nлучшая val_acc: {best_val_acc * 100:.2f}%")

    return model, {"mean": mean, "std": std, "history": history, "best_val_acc": best_val_acc}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="data/features_cache.npz")
    parser.add_argument("--out", default="data/readout_model.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    print(f"загружаю кэш {args.cache}...")
    features, labels, feature_dim, extractor_config = load_cache(args.cache)
    print(f"{len(labels):,} позиций, feature_dim={feature_dim}")

    model, meta = train(
        features, labels, feature_dim,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_dim": feature_dim,
            "norm_mean": meta["mean"],
            "norm_std": meta["std"],
            "best_val_acc": meta["best_val_acc"],
            "extractor_config": extractor_config,
        },
        out_path,
    )
    print(f"\nмодель сохранена -> {out_path}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
