"""Генерация датасета для behavior cloning: случайные партии + разметка Stockfish.

Почему не reinforcement learning — см. заметки проекта, Фаза 2, п.4: RL на замороженном
коннектоме разваливается из-за credit assignment (пример nfly на Pong: -20.5 vs
+8.0 у supervised). Stockfish уже стоит в проекте как отдельный "серьёзный"
оппонент — используем его же как бесплатного учителя.

Позиции берутся из self-play случайными легальными ходами (не через сам мозг —
на этом этапе мозг ещё не readout, ему нечем выбирать). Такой подход даёт
разнообразные позиции по всем стадиям партии, без необходимости первым делом
иметь обученную модель. Ограничение длины партии не даёт вырождаться в вечные
эндшпили с случайным блужданием короля.

Результат: parquet с колонками fen, best_move_uci, move_class (0..4095).
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flybrain.chess_encoding import move_to_class  # noqa: E402

MAX_GAME_PLIES = 80  # обрубаем случайные партии, чтобы не вырождались в шум
MIN_PLY_FOR_SAMPLE = 2  # не берём совсем пустые стартовые позиции


def random_self_play_positions(num_positions: int, seed: int = 0) -> list[str]:
    """Случайные легальные партии, из каждой берём несколько позиций (FEN)."""
    rng = random.Random(seed)
    fens: list[str] = []

    while len(fens) < num_positions:
        board = chess.Board()
        ply = 0
        game_fens = []
        while not board.is_game_over() and ply < MAX_GAME_PLIES:
            if ply >= MIN_PLY_FOR_SAMPLE:
                game_fens.append(board.fen())
            legal = list(board.legal_moves)
            board.push(rng.choice(legal))
            ply += 1
        fens.extend(game_fens)

    rng.shuffle(fens)
    return fens[:num_positions]


def label_with_stockfish(
    fens: list[str],
    stockfish_path: str,
    movetime_ms: int = 50,
    threads: int = 1,
) -> pd.DataFrame:
    """Для каждой позиции спросить Stockfish лучший ход, вернуть DataFrame."""
    # На Windows asyncio-обвязка python-chess (в отличие от обычного
    # subprocess.run) не резолвит относительный путь и падает с
    # FileNotFoundError — приводим путь к абсолютному заранее.
    stockfish_path = str(Path(stockfish_path).resolve())
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    try:
        if threads > 1:
            engine.configure({"Threads": threads})

        rows = []
        t0 = time.time()
        for i, fen in enumerate(fens):
            board = chess.Board(fen)
            result = engine.play(board, chess.engine.Limit(time=movetime_ms / 1000))
            best = result.move
            if best is None:
                continue
            rows.append(
                {
                    "fen": fen,
                    "best_move_uci": best.uci(),
                    "move_class": move_to_class(board, best),
                }
            )
            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                remaining = (len(fens) - i - 1) / rate
                print(
                    f"  {i + 1}/{len(fens)}  {rate:.1f} поз/с  "
                    f"осталось ~{remaining / 60:.1f} мин"
                )
    finally:
        engine.quit()

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stockfish", required=True, help="путь к бинарнику stockfish")
    parser.add_argument("--num-positions", type=int, default=200_000)
    parser.add_argument("--movetime-ms", type=int, default=50)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", default="data/dataset.parquet", help="куда сохранить результат"
    )
    args = parser.parse_args()

    print(f"генерирую {args.num_positions:,} позиций случайным self-play...")
    fens = random_self_play_positions(args.num_positions, seed=args.seed)
    print(f"получено {len(fens):,} позиций")

    print(f"\nразметка Stockfish (movetime={args.movetime_ms}мс, threads={args.threads})...")
    df = label_with_stockfish(
        fens, args.stockfish, movetime_ms=args.movetime_ms, threads=args.threads
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\nсохранено {len(df):,} размеченных позиций -> {out_path}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
