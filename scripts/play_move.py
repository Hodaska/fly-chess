"""Инференс: одна позиция -> ход мухи. Связывает мозг + обученный readout.

Использование:
    python -m scripts.play_move --model data/readout_model.pt --fen "<FEN>"
    python -m scripts.play_move --model data/readout_model.pt  # стартовая позиция

Это то, что в итоге дёргает манипулятор (см. заметки проекта, Фаза 4): на входе FEN,
на выходе UCI-строка хода. Мозг реально прогоняется (не кэш) — инференс
единичной позиции, поэтому скорость кэша здесь не нужна.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore", message=".*Sparse CSR.*")

import chess  # noqa: E402

from flybrain.chess_encoding import board_to_bits, decode_move, legal_move_mask  # noqa: E402
from flybrain.connectome import load_connectome  # noqa: E402
from flybrain.features import BrainFeatureExtractor  # noqa: E402
from flybrain.readout import MoveReadout  # noqa: E402
from scripts.train_readout import LEGACY_EXTRACTOR_CONFIG  # noqa: E402


def load_model(path: str, device: torch.device):
    """Модель + нормализация + настройки мозга, на которых её обучали."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MoveReadout(ckpt["feature_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    config = ckpt.get("extractor_config", LEGACY_EXTRACTOR_CONFIG)
    print(f"модель загружена, val_acc при обучении: {ckpt['best_val_acc'] * 100:.2f}%")
    return model, ckpt["norm_mean"], ckpt["norm_std"], config


def build_extractor(config: dict, device: torch.device) -> BrainFeatureExtractor:
    """Мозг ровно в той конфигурации, на признаках которой учили голову.

    Батч берётся из конфига (64), а не 1: калибровка порогов зависит от батча,
    и мозг с батчем 1 откалибровался бы иначе и выдавал бы слегка другие
    признаки. Одиночную позицию прогоняем через extract_padded — после
    заморозки порогов столбцы батча независимы, лишние просто нули.
    """
    c = load_connectome(min_weight=config["min_weight"])
    return BrainFeatureExtractor(
        c,
        batch=config["batch"],
        device=str(device),
        steps=config["steps"],
        readout=config["readout"],
        readout_size=config["readout_size"],
        calibrate_steps=config["calibrate_steps"],
        seed=config["seed"],
    )


def choose_move(
    board: chess.Board,
    extractor: BrainFeatureExtractor,
    model: MoveReadout,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> chess.Move:
    bits = board_to_bits(board)[None, :]  # [1, 768]
    features = extractor.extract_padded(bits)  # [1, feature_dim]
    features = (features - mean) / std

    mask, legal = legal_move_mask(board)
    if not legal:
        raise ValueError("нет легальных ходов (мат/пат)")

    with torch.no_grad():
        x = torch.from_numpy(features).float().to(device)
        logits = model(x).squeeze(0).cpu().numpy()

    logits[~mask] = -np.inf
    best_class = int(np.argmax(logits))
    return decode_move(board, best_class, legal)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fen", default=chess.STARTING_FEN)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, mean, std, config = load_model(args.model, device)
    print(f"собираю мозг: {config}")
    extractor = build_extractor(config, device)

    board = chess.Board(args.fen)
    print(f"\nпозиция: {board.fen()}")
    print(board)

    move = choose_move(board, extractor, model, mean, std, device)
    print(f"\nход мухи: {move.uci()}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
