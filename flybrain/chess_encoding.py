"""Кодирование шахматной позиции в сенсорный ток и декодирование хода из readout.

Никакой логики правил здесь нет и не должно быть — `python-chess` уже умеет
генерировать легальные ходы, наша задача только перевести FEN в ток и score
обратно в ход. Конь ходит буквой Г с нулевой итерации, потому что маска
легальности накладывается ПОСЛЕ readout, а не потому что мозг это выучил.
"""

from __future__ import annotations

import chess
import numpy as np

# 64 клетки x 12 типов фигур (6 своих + 6 чужих, если play as white).
# Порядок фигур: P N B R Q K, сначала свои, потом чужие.
PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
NUM_SQUARES = 64
NUM_PIECE_PLANES = 12  # 6 own + 6 opponent
BOARD_BITS = NUM_SQUARES * NUM_PIECE_PLANES  # 768

# Ход кодируется как (from_square, to_square) без учёта превращения —
# 64*64 = 4096 возможных пар. Промоция всегда в ферзя (см. decode_move).
NUM_MOVE_CLASSES = 64 * 64


def board_to_bits(board: chess.Board) -> np.ndarray:
    """FEN/Board -> вектор 768 бит (float32 0/1).

    Плоскости 0-5   — фигуры хода, чей сейчас ход (P,N,B,R,Q,K)
    Плоскости 6-11  — фигуры противника
    Ориентация доски нормализуется: если ходят чёрные, доска переворачивается
    так, что "свои" фигуры всегда идут первыми и снизу — иначе сети пришлось
    бы отдельно учить два зеркальных представления одной и той же идеи.
    """
    bits = np.zeros(BOARD_BITS, dtype=np.float32)
    stm = board.turn  # side to move

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        plane = PIECE_ORDER.index(piece.piece_type)
        if piece.color != stm:
            plane += 6

        sq = square if stm == chess.WHITE else chess.square_mirror(square)
        bits[sq * NUM_PIECE_PLANES + plane] = 1.0

    return bits


def bits_to_current_indices(bits: np.ndarray) -> np.ndarray:
    """Индексы установленных битов — то, что реально толкается в сенсорные нейроны."""
    return np.flatnonzero(bits)


def move_to_class(board: chess.Board, move: chess.Move) -> int:
    """Ход -> индекс класса (0..4095), в той же нормализованной ориентации,
    что и board_to_bits (если ходят чёрные, координаты зеркалятся)."""
    stm = board.turn
    frm = move.from_square if stm == chess.WHITE else chess.square_mirror(move.from_square)
    to = move.to_square if stm == chess.WHITE else chess.square_mirror(move.to_square)
    return frm * 64 + to


def legal_move_mask(board: chess.Board) -> tuple[np.ndarray, list[chess.Move]]:
    """Булева маска [4096] легальных классов + список исходных ходов в том же порядке
    появления масок (для случаев, когда несколько ходов бьют в один класс —
    т.е. под-промоции все схлопываются в один класс with move_to_class)."""
    mask = np.zeros(NUM_MOVE_CLASSES, dtype=bool)
    legal = list(board.legal_moves)
    for mv in legal:
        mask[move_to_class(board, mv)] = True
    return mask, legal


def decode_move(board: chess.Board, class_idx: int, legal_moves: list[chess.Move]) -> chess.Move:
    """Класс -> конкретный ход из списка легальных. Если под класс попадает
    несколько ходов (например e7e8q/e7e8r/e7e8b/e7e8n), приоритет ферзю —
    так поступает большинство engine-подобных энкодеров (AlphaZero и др.)."""
    candidates = [mv for mv in legal_moves if move_to_class(board, mv) == class_idx]
    if not candidates:
        raise ValueError(f"класс {class_idx} не соответствует ни одному легальному ходу")
    if len(candidates) == 1:
        return candidates[0]
    for mv in candidates:
        if mv.promotion == chess.QUEEN:
            return mv
    return candidates[0]


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    board = chess.Board()
    bits = board_to_bits(board)
    print(f"стартовая позиция: {bits.sum():.0f} фигур на доске (ожидается 32)")
    print(f"сенсорных индексов: {len(bits_to_current_indices(bits))}")

    mask, legal = legal_move_mask(board)
    print(f"легальных ходов из старта: {len(legal)}, уникальных классов: {mask.sum():.0f}")

    e4 = chess.Move.from_uci("e2e4")
    cls = move_to_class(board, e4)
    print(f"e2e4 -> класс {cls}")
    decoded = decode_move(board, cls, legal)
    print(f"класс {cls} -> {decoded.uci()} (совпадает: {decoded == e4})")

    # Проверка симметрии: тот же ход после хода чёрных должен кодироваться
    # в те же координаты относительно стороны хода
    board.push(chess.Move.from_uci("e2e4"))
    board.push(chess.Move.from_uci("e7e5"))
    bits2 = board_to_bits(board)
    print(f"\nпосле 1.e4 e5: {bits2.sum():.0f} фигур, ход белых снова")
