"""Leaky integrate-and-fire симуляция коннектома на GPU.

Модель по духу Shiu et al. 2024 (Nature), но в безразмерных единицах:
  v[t+1] = leak * v[t] + I_syn + I_ext
  спайк, если v > threshold, затем сброс и рефрактерный период

Граф заморожен — обучаются только внешние слои (см. Фазу 2).

Устойчивость держится на гомеостатической подстройке порогов: без неё сеть
либо замолкает, либо уходит в насыщение (самая частая причина «не работает
без причины»). Порог каждого нейрона медленно ползёт к целевой частоте.

Состояние хранится как [N, B], где B — размер батча, чтобы на Фазе 2 можно
было прогонять много позиций за один проход.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .connectome import Connectome


@dataclass
class LIFParams:
    dt: float = 0.5  # мс на шаг
    tau_m: float = 20.0  # мс, постоянная времени мембраны
    v_threshold: float = 1.0  # стартовый порог
    v_reset: float = 0.0
    refractory_ms: float = 2.0
    weight_gain: float = 0.02  # общий масштаб синаптического тока

    # Гомеостаз порогов
    homeostasis: bool = True
    target_rate: float = 0.02  # доля нейронов, спайкующих за шаг
    homeostasis_gain: float = 0.02
    threshold_min: float = 0.05
    threshold_max: float = 50.0

    @property
    def leak(self) -> float:
        return float(np.exp(-self.dt / self.tau_m))

    @property
    def refractory_steps(self) -> int:
        return max(1, int(round(self.refractory_ms / self.dt)))


class LIFBrain:
    """Симулятор. Матрица связей строится один раз и не меняется."""

    def __init__(
        self,
        connectome: Connectome,
        params: LIFParams | None = None,
        batch: int = 1,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.c = connectome
        self.p = params or LIFParams()
        self.batch = batch
        self.dtype = dtype
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        n = connectome.num_neurons
        self.n = n

        # Разреженная матрица M[post, pre] = знаковый вес.
        # CSR по строкам-post: ток I = M @ spikes.
        post = torch.from_numpy(connectome.edge_post.astype(np.int64))
        pre = torch.from_numpy(connectome.edge_pre.astype(np.int64))
        w = torch.from_numpy(connectome.edge_weight) * self.p.weight_gain

        order = torch.argsort(post)
        post, pre, w = post[order], pre[order], w[order]
        crow = torch.zeros(n + 1, dtype=torch.int64)
        crow[1:] = torch.bincount(post, minlength=n).cumsum(0)

        self.W = torch.sparse_csr_tensor(
            crow.to(self.device),
            pre.to(self.device),
            w.to(self.device, dtype),
            size=(n, n),
        )

        self.threshold = torch.full(
            (n, 1), self.p.v_threshold, device=self.device, dtype=dtype
        )
        self.reset_state()

    def reset_state(self) -> None:
        z = lambda d=self.dtype: torch.zeros(  # noqa: E731
            (self.n, self.batch), device=self.device, dtype=d
        )
        self.v = z()
        self.spikes = z()
        self.refractory = z(torch.int16)
        self.step_count = 0

    # ------------------------------------------------------------------
    def step(self, external: torch.Tensor | None = None) -> torch.Tensor:
        """Один шаг. external — ток [N, B] или None. Возвращает спайки [N, B] (0/1)."""
        current = torch.sparse.mm(self.W, self.spikes)
        if external is not None:
            current = current + external

        self.v = self.p.leak * self.v + current
        self.v.clamp_(min=-5.0)  # не даём уйти в глубокий минус

        free = self.refractory <= 0
        self.spikes = ((self.v > self.threshold) & free).to(self.dtype)

        fired = self.spikes > 0
        self.v = torch.where(fired, torch.full_like(self.v, self.p.v_reset), self.v)
        self.refractory = torch.where(
            fired,
            torch.full_like(self.refractory, self.p.refractory_steps),
            (self.refractory - 1).clamp_(min=0),
        )

        if self.p.homeostasis:
            rate = self.spikes.mean(dim=1, keepdim=True)
            self.threshold += self.p.homeostasis_gain * (rate - self.p.target_rate)
            self.threshold.clamp_(self.p.threshold_min, self.p.threshold_max)

        self.step_count += 1
        return self.spikes

    # ------------------------------------------------------------------
    def make_external(self, indices: np.ndarray, amplitude: float) -> torch.Tensor:
        """Ток [N, B], подаваемый на указанные нейроны (одинаково по батчу)."""
        ext = torch.zeros((self.n, self.batch), device=self.device, dtype=self.dtype)
        if len(indices):
            idx = torch.from_numpy(np.asarray(indices, dtype=np.int64)).to(self.device)
            ext[idx] = amplitude
        return ext

    def run(self, steps: int, external: torch.Tensor | None = None) -> torch.Tensor:
        """Прогнать N шагов, вернуть суммарное число спайков на нейрон [N, B]."""
        counts = torch.zeros(
            (self.n, self.batch), device=self.device, dtype=self.dtype
        )
        for _ in range(steps):
            counts += self.step(external)
        return counts

    def run_graded(
        self, steps: int, external: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Как run(), но дополнительно копит средний мембранный потенциал.

        Зачем: при низкой частоте активности (после калибровки вышло 0.73%)
        нейрон выдаёт за окно 1-2 спайка, и счётчик спайков квантуется в
        «0, 1, 2» — информационная ёмкость почти нулевая. Средний потенциал
        непрерывен и несёт заметно больше. Так же поступают rate-based
        коннектомные модели (nfly: «one scalar rate per neuron»).

        Возвращает (счётчики спайков [N,B], средний потенциал [N,B]).
        """
        counts = torch.zeros((self.n, self.batch), device=self.device, dtype=self.dtype)
        v_sum = torch.zeros((self.n, self.batch), device=self.device, dtype=self.dtype)
        for _ in range(steps):
            counts += self.step(external)
            v_sum += self.v
        return counts, v_sum / steps

    @property
    def firing_rate(self) -> float:
        """Доля нейронов, спайкнувших на последнем шаге."""
        return float(self.spikes.mean().item())

    def spiking_indices(self, column: int = 0) -> np.ndarray:
        """Индексы спайкнувших нейронов — то, что рисует просмотрщик."""
        return torch.nonzero(self.spikes[:, column], as_tuple=True)[0].cpu().numpy()

    # ------------------------------------------------------------------
    def calibrate_and_freeze(self, external: torch.Tensor, steps: int = 1500) -> float:
        """Дать гомеостазу сойтись на типичном входе, потом ЗАМОРОЗИТЬ пороги.

        Зачем: `reset_state()` намеренно не трогает пороги, поэтому при
        включённом гомеостазе они дрейфуют через весь прогон и признаки
        начинают кодировать «когда их посчитали», а не «что на входе».
        Замер на прогоне 150k позиций: активность упала на 21%, а номер
        позиции восстанавливался из признаков с R2=0.62 против R2=0.13 для
        самой доски. То есть дрейф был сильнее полезного сигнала.

        После вызова пороги фиксированы, и extract() становится чистой
        функцией от входа. Возвращает достигнутую частоту активности.
        """
        self.p.homeostasis = True
        self.reset_state()
        for _ in range(steps):
            self.step(external)
        self.p.homeostasis = False  # пороги зафиксированы на найденных значениях
        rate = self.firing_rate
        self.reset_state()
        return rate


if __name__ == "__main__":
    import sys
    import time

    from .connectome import load_connectome

    sys.stdout.reconfigure(encoding="utf-8")

    print("загружаю коннектом...")
    c = load_connectome(min_weight=5)
    print(c.summary())

    brain = LIFBrain(c)
    print(f"\nустройство: {brain.device}, рёбер в матрице: {brain.W._nnz():,}")
    if brain.device.type == "cuda":
        print(f"VRAM занято: {torch.cuda.memory_allocated() / 2**20:.0f} МБ")

    # Толкаем зрительные нейроны, как будто что-то увидели
    optic = np.flatnonzero(c.group == 0)[:2000]
    ext = brain.make_external(optic, amplitude=0.6)

    print("\nпрогрев 200 шагов...")
    t0 = time.time()
    for i in range(200):
        brain.step(ext)
        if i % 40 == 0:
            print(f"  шаг {i:3d}  активность {brain.firing_rate * 100:5.2f}%  "
                  f"порог ср. {brain.threshold.mean().item():.3f}")
    if brain.device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    bio_ms = 200 * brain.p.dt
    print(f"\n200 шагов ({bio_ms:.0f} мс биологического времени) за {dt:.2f} с")
    print(f"  {dt / 200 * 1000:.2f} мс на шаг")
    print(f"  {dt / (bio_ms / 1000):.2f} с на биологическую секунду "
          f"({(bio_ms / 1000) / dt:.2f}x реального времени)")
