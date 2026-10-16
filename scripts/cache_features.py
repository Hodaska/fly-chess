"""Прогнать мозг по всем позициям датасета ОДИН РАЗ, сохранить признаки на диск.

Это тот самый ночной прогон из заметок проекта (Фаза 2, п.5): граф заморожен, поэтому
отклик мозга на позицию не меняется между эпохами обучения. Прогоняем один раз,
дальше голову обучаем на кэше за минуты вместо часов.

Батч 64 (батч 8 аномально медленный из-за смены алгоритма cuSPARSE).
Реальная скорость на GTX 1060 ~9 поз/с при 100 шагах: карта не разгоняется выше 746 МГц.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore", message=".*Sparse CSR.*")

from flybrain.chess_encoding import board_to_bits  # noqa: E402
from flybrain.connectome import load_connectome  # noqa: E402
from flybrain.features import BrainFeatureExtractor  # noqa: E402

import chess  # noqa: E402

BATCH = 64  # см. заметки проекта — оптимум для GTX 1060 3GB


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/dataset.parquet")
    parser.add_argument("--out", default="data/features_cache.npz")
    parser.add_argument("--min-weight", type=int, default=5)
    parser.add_argument("--steps", type=int, default=100)  # 100 — замеренный оптимум, на нём обучена модель
    parser.add_argument("--readout", default="wide", choices=["descending", "wide", "optic"])
    parser.add_argument("--readout-size", type=int, default=2048)
    args = parser.parse_args()

    print("загружаю коннектом...")
    c = load_connectome(min_weight=args.min_weight)
    print(c.summary())

    print(f"\nсобираю extractor (batch={BATCH}, steps={args.steps}, "
          f"readout={args.readout}/{args.readout_size})...")
    extractor = BrainFeatureExtractor(
        c,
        batch=BATCH,
        steps=args.steps,
        readout=args.readout,
        readout_size=args.readout_size,
        calibrate_steps=1500,
    )
    print(f"устройство: {extractor.device}, размер признаков: {extractor.feature_dim}")
    print(f"частота после калибровки: {extractor.calibrated_rate * 100:.2f}%")

    print(f"\nчитаю датасет {args.dataset}...")
    df = pd.read_parquet(args.dataset)
    n = len(df)
    print(f"{n:,} позиций")

    print("кодирую доски в биты (это быстро, CPU)...")
    t0 = time.time()
    bits_all = np.stack(
        [board_to_bits(chess.Board(fen)) for fen in df["fen"]]
    ).astype(np.float32)
    print(f"  готово за {time.time() - t0:.1f} с")

    labels = df["move_class"].to_numpy(dtype=np.int64)

    print(f"\nпрогоняю мозг батчами по {BATCH}...")
    features = np.zeros((n, extractor.feature_dim), dtype=np.float32)

    t0 = time.time()
    num_batches = (n + BATCH - 1) // BATCH
    for bi in range(num_batches):
        lo, hi = bi * BATCH, min((bi + 1) * BATCH, n)
        chunk = bits_all[lo:hi]
        if len(chunk) == BATCH:
            features[lo:hi] = extractor.extract(chunk)
        else:
            features[lo:hi] = extractor.extract_padded(chunk)

        if (bi + 1) % 50 == 0 or bi + 1 == num_batches:
            done = hi
            elapsed = time.time() - t0
            rate = done / elapsed
            remaining = (n - done) / rate if rate > 0 else 0
            print(
                f"  {done:,}/{n:,}  {rate:.1f} поз/с  "
                f"осталось ~{remaining / 60:.1f} мин"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        features=features,
        labels=labels,
        feature_dim=extractor.feature_dim,
        num_positions=n,
        # Настройки экстрактора едут вместе с признаками, дальше — в чекпоинт
        # модели, чтобы play_move собрал мозг ровно так же. Иначе модель,
        # обученная на одном съёме, получает признаки с другого.
        extractor_config=json.dumps({
            "min_weight": args.min_weight,
            "steps": args.steps,
            "readout": args.readout,
            "readout_size": args.readout_size,
            "calibrate_steps": 1500,
            "batch": BATCH,
            "seed": 0,
        }),
    )
    print(f"\nсохранено -> {out_path} ({out_path.stat().st_size / 2**20:.1f} МБ)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
