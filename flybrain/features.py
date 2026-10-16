"""Доска -> ток в мозг -> прогон LIF -> вектор признаков для readout.

Сенсорное отображение: 768 бит доски (chess_encoding.board_to_bits) толкаются
в 768 непересекающихся пулов нейронов зрительной группы (optic). Отображение
детерминированное и фиксируется сидом — так же в духе того, как это делают
существующие проекты (StockFly, FLYT3: "fixed sensory populations").

Выходная группа: DESCENDING-нейроны (те, что реально несут сигнал из мозга
в тело). В нашем графе VNC исключён (см. заметки проекта, решение №3), поэтому
классических "моторных" нейронов нет — берём все descending (1312 штук)
целиком, без урезания до произвольных 512. Это отклонение от ориентира
FLYT3 ("512 descending + 512 VNC-моторных"), но оно честнее: не обрезаем
данные, чтобы подогнать число под чужой проект.
"""

from __future__ import annotations

import numpy as np
import torch

from .chess_encoding import BOARD_BITS
from .connectome import Connectome, GROUP_NAMES
from .lif import LIFBrain, LIFParams

DEFAULT_STEPS = 200  # ~100 мс биологического времени при dt=0.5, как в Фазе 1
DEFAULT_AMPLITUDE = 0.6  # та же величина, что использовалась в просмотрщике


def build_sensory_pools(
    connectome: Connectome, num_bits: int = BOARD_BITS, seed: int = 0
) -> np.ndarray:
    """Детерминированно разбить нейроны optic-группы на num_bits непересекающихся
    пулов. Возвращает bucket_of[optic_local_idx] -> id пула (0..num_bits-1)."""
    optic_idx = np.flatnonzero(connectome.group == GROUP_NAMES.index("optic"))
    if len(optic_idx) < num_bits:
        raise ValueError(
            f"нейронов optic ({len(optic_idx)}) меньше числа бит доски ({num_bits})"
        )

    rng = np.random.RandomState(seed)
    shuffled = optic_idx.copy()
    rng.shuffle(shuffled)

    chunks = np.array_split(shuffled, num_bits)
    bucket_of = np.full(connectome.num_neurons, -1, dtype=np.int32)
    for bucket_id, chunk in enumerate(chunks):
        bucket_of[chunk] = bucket_id
    return bucket_of


def build_pool_matrix(
    bucket_of: np.ndarray, num_bits: int, device: torch.device
) -> torch.Tensor:
    """Разреженная [N, num_bits] матрица принадлежности нейрон->пул (0/1).
    Pool @ bits_batch[num_bits, B] -> внешний ток [N, B]."""
    neuron_idx = np.flatnonzero(bucket_of >= 0)
    bit_idx = bucket_of[neuron_idx]

    indices = torch.tensor(
        np.stack([neuron_idx, bit_idx]), dtype=torch.long, device=device
    )
    values = torch.ones(len(neuron_idx), dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(
        indices, values, size=(len(bucket_of), num_bits), device=device
    ).coalesce()


class BrainFeatureExtractor:
    """Прогоняет мозг батчами позиций, возвращает признаки для readout.

    Граф заморожен (см. заметки проекта) — этот класс НИКОГДА не трогает веса
    коннектома. Единственное состояние, которое меняется между вызовами —
    мембранный потенциал и пороги гомеостаза внутри LIFBrain, которые
    сбрасываются перед каждым батчем через reset_state().
    """

    def __init__(
        self,
        connectome: Connectome,
        batch: int,
        device: str | None = None,
        steps: int = DEFAULT_STEPS,
        amplitude: float = DEFAULT_AMPLITUDE,
        seed: int = 0,
        readout: str = "descending",
        readout_size: int = 4096,
        calibrate_steps: int = 1500,
        signal: str = "spikes",
        target_rate: float | None = None,
    ):
        """readout: откуда снимать признаки.
            "descending" — 1312 нисходящих нейронов (биологический выход в тело)
            "wide"       — случайная выборка readout_size нейронов по всему мозгу
            "optic"      — случайная выборка из зрительной группы (куда бьём
                           входом); нужна как проверка «сигнал вообще есть?»

        signal: что именно снимать.
            "spikes"  — счётчики спайков за окно (квантуются грубо при низкой
                        активности, см. run_graded в lif.py)
            "voltage" — средний мембранный потенциал, непрерывный
        """
        self.c = connectome
        self.steps = steps
        self.amplitude = amplitude
        self.signal = signal

        params = LIFParams()
        if target_rate is not None:
            params.target_rate = target_rate
        self.brain = LIFBrain(connectome, params=params, batch=batch, device=device)
        self.device = self.brain.device

        bucket_of = build_sensory_pools(connectome, BOARD_BITS, seed=seed)
        self.pool_matrix = build_pool_matrix(bucket_of, BOARD_BITS, self.device)

        if readout == "descending":
            self.output_idx = np.flatnonzero(
                connectome.group == GROUP_NAMES.index("descending")
            )
        elif readout in ("wide", "optic"):
            rng = np.random.RandomState(seed + 1)
            pool = (
                np.arange(connectome.num_neurons)
                if readout == "wide"
                else np.flatnonzero(connectome.group == GROUP_NAMES.index("optic"))
            )
            self.output_idx = np.sort(
                rng.choice(pool, size=min(readout_size, len(pool)), replace=False)
            )
        else:
            raise ValueError(f"неизвестный readout: {readout}")

        self.feature_dim = len(self.output_idx)
        self._output_idx_t = torch.from_numpy(self.output_idx.astype(np.int64)).to(
            self.device
        )

        # Калибруем пороги на типичной по статистике доске (~32 фигуры) и
        # замораживаем их. Без этого гомеостаз дрейфует через весь прогон и
        # признаки начинают кодировать порядковый номер позиции — подробности
        # в docstring LIFBrain.calibrate_and_freeze.
        if calibrate_steps > 0:
            rng = np.random.RandomState(seed + 2)
            typical = np.zeros((batch, BOARD_BITS), dtype=np.float32)
            for b in range(batch):
                typical[b, rng.choice(BOARD_BITS, size=32, replace=False)] = 1.0
            ext = self._external_from_bits(typical)
            self.calibrated_rate = self.brain.calibrate_and_freeze(ext, calibrate_steps)
        else:
            self.calibrated_rate = float("nan")

    def _external_from_bits(self, bits_batch: np.ndarray) -> torch.Tensor:
        bits_t = torch.from_numpy(bits_batch.T.astype(np.float32)).to(self.device)
        return self.amplitude * torch.sparse.mm(self.pool_matrix, bits_t)

    def extract(self, bits_batch: np.ndarray) -> np.ndarray:
        """bits_batch: [B, 768] float32 0/1 -> признаки [B, feature_dim] float32.

        B должен совпадать с batch, заданным в конструкторе (для последнего
        неполного батча см. extract_padded)."""
        b = bits_batch.shape[0]
        if b != self.brain.batch:
            raise ValueError(
                f"размер батча {b} не совпадает с brain.batch={self.brain.batch}; "
                f"используйте extract_padded для неполных батчей"
            )

        ext = self._external_from_bits(bits_batch)

        self.brain.reset_state()
        if self.signal == "voltage":
            _, raw = self.brain.run_graded(self.steps, external=ext)
        else:
            raw = self.brain.run(self.steps, external=ext)  # [N, B]

        features = raw.index_select(0, self._output_idx_t)  # [feature_dim, B]
        return features.T.cpu().numpy()

    def extract_with_trace(
        self, bits_batch: np.ndarray, column: int = 0
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Как extract(), но ещё и запись: какие нейроны спайкнули на каждом
        шаге (для столбца `column`). Это то, что показывает сайт, пока муха
        думает — настоящая активность именно на этот ход, а не анимация."""
        ext = self._external_from_bits(bits_batch)
        self.brain.reset_state()
        counts = torch.zeros(
            (self.brain.n, self.brain.batch), device=self.device, dtype=self.brain.dtype
        )
        trace_gpu = []
        for _ in range(self.steps):
            spikes = self.brain.step(ext)
            counts += spikes
            trace_gpu.append(torch.nonzero(spikes[:, column], as_tuple=True)[0])
        features = counts.index_select(0, self._output_idx_t).T.cpu().numpy()
        trace = [t.cpu().numpy().astype(np.uint32) for t in trace_gpu]
        return features, trace

    def rebatch(self, batch: int) -> None:
        """Пересобрать мозг под другой размер батча, сохранив пороги.

        Калибровать надо на батче 64 (так же, как при обучении), а играть
        удобнее на батче 1: шаг в десятки раз дешевле. После заморозки порогов
        столбцы батча независимы, поэтому признаки те же.
        """
        threshold = self.brain.threshold.clone()
        params = self.brain.p
        del self.brain
        torch.cuda.empty_cache()
        self.brain = LIFBrain(self.c, params=params, batch=batch, device=str(self.device))
        self.brain.threshold = threshold

    def extract_padded(self, bits_batch: np.ndarray) -> np.ndarray:
        """Как extract(), но допускает батч меньше self.brain.batch —
        дополняет нулями и обрезает результат. Удобно для последнего
        неполного батча в датасете."""
        b = bits_batch.shape[0]
        full = self.brain.batch
        if b == full:
            return self.extract(bits_batch)
        if b > full:
            raise ValueError(f"батч {b} больше, чем brain.batch={full}")

        padded = np.zeros((full, bits_batch.shape[1]), dtype=np.float32)
        padded[:b] = bits_batch
        return self.extract(padded)[:b]


if __name__ == "__main__":
    import sys
    import time
    import warnings

    warnings.filterwarnings("ignore", message=".*Sparse CSR.*")
    sys.stdout.reconfigure(encoding="utf-8")

    from .chess_encoding import board_to_bits
    from .connectome import load_connectome

    import chess

    print("загружаю коннектом...")
    c = load_connectome(min_weight=5)
    print(c.summary())

    BATCH = 8
    print(f"\nсобираю extractor (batch={BATCH})...")
    extractor = BrainFeatureExtractor(c, batch=BATCH)
    print(f"устройство: {extractor.device}")
    print(f"размер выходного вектора признаков (descending): {extractor.feature_dim}")

    boards = [chess.Board() for _ in range(BATCH)]
    boards[1].push(chess.Move.from_uci("e2e4"))
    boards[2].push(chess.Move.from_uci("d2d4"))

    bits_batch = np.stack([board_to_bits(b) for b in boards])
    print(f"\nbits_batch: {bits_batch.shape}")

    t0 = time.time()
    feats = extractor.extract(bits_batch)
    if extractor.device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    print(f"признаки: {feats.shape}, извлечены за {dt:.2f} с")
    print(f"позиция 0 (старт) vs позиция 1 (после e4): "
          f"совпадают ли признаки? {np.allclose(feats[0], feats[1])}")
    print(f"сумма спайков по позициям: {feats.sum(axis=1)}")
