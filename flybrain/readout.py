"""Readout: спайки замороженного мозга -> скор по 4096 классам хода.

Единственная обучаемая часть всего пайплайна (см. заметки проекта, Фаза 2, п.1).
Коннектом не трогается вообще — сюда попадает уже готовый вектор признаков
(накопленные спайки нисходящих + моторных нейронов), а откуда он взят,
readout не знает и знать не должен.

Обучение — behavior cloning от Stockfish (кросс-энтропия по классу лучшего
хода), НЕ reinforcement learning. Обоснование в заметках проекта со ссылкой на nfly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .chess_encoding import NUM_MOVE_CLASSES


class MoveReadout(nn.Module):
    """Линейный слой поверх вектора признаков мозга.

    features -> logits[4096]. Маска легальности накладывается СНАРУЖИ,
    на этапе инференса (см. select_move), а не внутри модели — так обучение
    видит настоящее распределение мозга, а не то, что осталось после маски.
    """

    def __init__(self, feature_dim: int, hidden: int | None = None):
        super().__init__()
        if hidden:
            self.net = nn.Sequential(
                nn.Linear(feature_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, NUM_MOVE_CLASSES),
            )
        else:
            # По умолчанию — именно ОДИН линейный слой. Так честнее для защиты:
            # "мозг + линейная проекция", а не "мозг + маленькая MLP-голова".
            self.net = nn.Linear(feature_dim, NUM_MOVE_CLASSES)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


@torch.no_grad()
def select_move_class(
    model: MoveReadout, features: torch.Tensor, legal_mask: torch.Tensor
) -> int:
    """logits -> индекс класса, только среди легальных (маска bool [4096])."""
    logits = model(features.unsqueeze(0)).squeeze(0)
    logits = logits.masked_fill(~legal_mask, float("-inf"))
    return int(torch.argmax(logits).item())


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    # Дымовой тест без мозга и без сети: случайный вектор фич правильной размерности.
    feature_dim = 1024  # 512 descending + 512 моторных, как в плане Фазы 2
    model = MoveReadout(feature_dim)

    dummy_features = torch.randn(feature_dim)
    legal = torch.zeros(NUM_MOVE_CLASSES, dtype=torch.bool)
    legal[[796, 812, 828]] = True  # e2e4, d2d4, что-то ещё, для примера

    cls = select_move_class(model, dummy_features, legal)
    print(f"выбранный класс: {cls} (легален: {bool(legal[cls])})")
    print(f"параметров в readout: {sum(p.numel() for p in model.parameters()):,}")
