"""Один прогон симуляции — много замеров.

Предыдущий перебор (sweep_extractor.py) гонял отдельную симуляцию на каждую
конфигурацию, ~40 минут за штуку. Но дорогая часть — только сама симуляция;
снять с неё разные группы нейронов и разные типы сигнала можно почти даром.

Здесь: прогоняем мозг ОДИН раз на набор позиций, записывая и счётчики спайков,
и средний потенциал для большого набора «зондов» (descending + optic + central),
а потом оффлайн считаем восстановимость доски для всех комбинаций.

Ответ, который ищем: где именно теряется сигнал о доске.
  - если R2 высок у optic и низок у descending -> сигнал гибнет по дороге
  - если низок везде -> проблема во входе или в динамике
  - если counts плохо, а voltage хорошо -> дело в грубом квантовании счётчиков
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore", message=".*Sparse CSR.*")

import chess  # noqa: E402

from flybrain.chess_encoding import board_to_bits  # noqa: E402
from flybrain.connectome import GROUP_NAMES, load_connectome  # noqa: E402
from flybrain.features import build_pool_matrix, build_sensory_pools  # noqa: E402
from flybrain.lif import LIFBrain, LIFParams  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_POS = 6000
BATCH = 64
STEPS = 200
AMPLITUDE = 0.6


def ridge_r2(X: np.ndarray, Y: np.ndarray, tr, va, lam: float = 10.0) -> float:
    Xt = torch.from_numpy(np.ascontiguousarray(X)).float().to(DEVICE)
    Yt = torch.from_numpy(Y).float().to(DEVICE)
    mu, sd = Xt[tr].mean(0, keepdim=True), Xt[tr].std(0, keepdim=True) + 1e-6
    Xt = torch.cat([(Xt - mu) / sd, torch.ones(len(Xt), 1, device=DEVICE)], 1)
    A, B = Xt[tr], Yt[tr]
    W = torch.linalg.solve(A.T @ A + lam * torch.eye(A.shape[1], device=DEVICE), A.T @ B)
    pred, true = Xt[va] @ W, Yt[va]
    ss_res = ((pred - true) ** 2).sum(0)
    ss_tot = ((true - true.mean(0, keepdim=True)) ** 2).sum(0)
    r2 = 1 - ss_res / (ss_tot + 1e-9)
    valid = ss_tot > 1e-6
    out = float(r2[valid].mean().item())
    del Xt, Yt, A, B, W, pred, true
    torch.cuda.empty_cache()
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    df = pd.read_parquet("data/dataset.parquet")
    # берём позиции РАВНОМЕРНО по всему датасету, а не первые N подряд:
    # подряд идущие позиции приходят из одних и тех же партий и похожи друг
    # на друга, что завышало бы оценку.
    step = len(df) // N_POS
    df = df.iloc[::step].head(N_POS)
    bits = np.stack([board_to_bits(chess.Board(f)) for f in df["fen"]]).astype(np.float32)
    n = len(bits)

    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    n_va = int(n * 0.2)
    va, tr = perm[:n_va], perm[n_va:]

    print(f"позиций: {n:,}")
    c = load_connectome(min_weight=5)

    # ---- набор зондов ----
    g_desc = np.flatnonzero(c.group == GROUP_NAMES.index("descending"))
    g_optic = np.flatnonzero(c.group == GROUP_NAMES.index("optic"))
    g_centr = np.flatnonzero(c.group == GROUP_NAMES.index("central"))
    probe_optic = rng.choice(g_optic, 2000, replace=False)
    probe_centr = rng.choice(g_centr, 2000, replace=False)

    probes = np.concatenate([g_desc, probe_optic, probe_centr])
    spans = {
        "descending (1312)": (0, len(g_desc)),
        "optic (2000)": (len(g_desc), len(g_desc) + 2000),
        "central (2000)": (len(g_desc) + 2000, len(probes)),
        "всё вместе (5312)": (0, len(probes)),
    }
    probes_t = torch.from_numpy(probes.astype(np.int64)).to(DEVICE)

    bucket = build_sensory_pools(c, 768, seed=0)
    pool_m = build_pool_matrix(bucket, 768, DEVICE)

    for target_rate in (0.02, 0.10):
        print(f"\n{'=' * 78}")
        print(f"ЦЕЛЕВАЯ ЧАСТОТА АКТИВНОСТИ ПРИ КАЛИБРОВКЕ: {target_rate * 100:.0f}%")
        print("=" * 78)

        p = LIFParams()
        p.target_rate = target_rate
        brain = LIFBrain(c, params=p, batch=BATCH, device=str(DEVICE))

        calib = np.zeros((BATCH, 768), dtype=np.float32)
        for b in range(BATCH):
            calib[b, rng.choice(768, 32, replace=False)] = 1.0
        ext_c = AMPLITUDE * torch.sparse.mm(
            pool_m, torch.from_numpy(calib.T).to(DEVICE)
        )
        rate = brain.calibrate_and_freeze(ext_c, 1500)
        print(f"частота после калибровки: {rate * 100:.2f}%")

        cnt = np.zeros((n, len(probes)), dtype=np.float32)
        vol = np.zeros((n, len(probes)), dtype=np.float32)

        t0 = time.time()
        for lo in range(0, n, BATCH):
            chunk = bits[lo : lo + BATCH]
            pad = BATCH - len(chunk)
            if pad:
                chunk = np.vstack([chunk, np.zeros((pad, 768), dtype=np.float32)])
            ext = AMPLITUDE * torch.sparse.mm(
                pool_m, torch.from_numpy(chunk.T).to(DEVICE)
            )
            brain.reset_state()
            cc, vv = brain.run_graded(STEPS, external=ext)
            take = len(chunk) - pad
            cnt[lo : lo + take] = cc.index_select(0, probes_t).T[:take].cpu().numpy()
            vol[lo : lo + take] = vv.index_select(0, probes_t).T[:take].cpu().numpy()
        print(f"симуляция: {time.time() - t0:.0f} с ({n / (time.time() - t0):.1f} поз/с)")
        print(f"спайков на нейрон за окно: среднее {cnt.mean():.2f}, макс {cnt.max():.0f}")

        print(f"\n{'группа зондов':<22} {'R2 (спайки)':>12} {'R2 (потенциал)':>15}")
        print("-" * 52)
        for name, (a, b) in spans.items():
            r_c = ridge_r2(cnt[:, a:b], bits, tr, va)
            r_v = ridge_r2(vol[:, a:b], bits, tr, va)
            print(f"{name:<22} {r_c:>+12.3f} {r_v:>+15.3f}")

        del brain, cnt, vol
        torch.cuda.empty_cache()

    proj = rng.randn(768, 2000).astype(np.float32) / np.sqrt(768)
    print(f"\nориентир — случайная проекция доски: "
          f"R2 = {ridge_r2(np.maximum(bits @ proj, 0), bits, tr, va):+.3f}")


if __name__ == "__main__":
    main()
