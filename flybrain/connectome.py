"""Загрузка коннектома MaleCNS v1.0 в компактное представление для симуляции.

Данные: https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/
Лицензия данных: CC BY 4.0 — FlyEM / HHMI Janelia, University of Cambridge,
MRC Laboratory of Molecular Biology, Google Research.

Из трёх файлов собирается граф:
  body-annotations      -> bodyId, somaLocation (3D), type, superclass
  body-neurotransmitters -> consensus_nt  -> знак синапса (правило Дейла)
  connectome-weights    -> body_pre, body_post, weight

Файл связей отсортирован по weight по убыванию, поэтому отсечка по min_weight
обрезает длинный хвост одиночных контактов почти без потери структуры.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

# Знак синапса по нейромедиатору пресинаптического нейрона (правило Дейла).
# Как в Shiu et al. 2024: ACh возбуждающий, GABA и глутамат тормозные.
# Моноамины — модуляторы, отдельной динамики для них здесь нет, рёбра отбрасываются.
NT_SIGN = {
    "acetylcholine": +1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "histamine": -1.0,
}

# Группировка superclass -> крупная область (совпадает с разбиением, которое
# использует официальный атлас MaleCNS).
SUPERCLASS_GROUP = {
    "ol_intrinsic": "optic",
    "ol_sensory": "optic",
    "visual_projection": "optic",
    "visual_centrifugal": "optic",
    "visual_projection_tbc": "optic",
    "cb_intrinsic": "central",
    "cb_sensory": "central",
    "cb_motor": "central",
    "cb_endocrine": "central",
    "cb_sensory_tbc": "central",
    "cb_efferent": "central",
    "descending_neuron": "descending",
    "descending_neuron_tbc": "descending",
    "sensory_descending": "descending",
    "efferent_descending": "descending",
    "vnc_intrinsic": "vnc",
    "vnc_sensory": "vnc",
    "vnc_motor": "vnc",
    "vnc_efferent": "vnc",
    "vnc_tbc": "vnc",
    "vnc_sensory_tbc": "vnc",
    "vnc_endocrine": "vnc",
    "ascending_neuron": "vnc",
    "sensory_ascending": "vnc",
    "sensory_ascending_tbc": "vnc",
    "efferent_ascending": "vnc",
}

GROUP_NAMES = ["optic", "central", "descending", "vnc", "other"]
GROUP_COLORS = ["#81b5c8", "#cfcac0", "#dfb672", "#af9bc3", "#6c7777"]


@dataclass
class Connectome:
    """Граф после фильтрации. Индексы нейронов — позиции в этих массивах."""

    body_ids: np.ndarray  # int64  [N]
    positions: np.ndarray  # float32 [N, 3] — нативные 8-нм воксели
    group: np.ndarray  # int8   [N] — индекс в GROUP_NAMES
    types: np.ndarray  # object [N] — строковый тип клетки
    edge_pre: np.ndarray  # int32  [E]
    edge_post: np.ndarray  # int32  [E]
    edge_weight: np.ndarray  # float32 [E] — знаковый (синапсы × знак медиатора)

    @property
    def num_neurons(self) -> int:
        return len(self.body_ids)

    @property
    def num_edges(self) -> int:
        return len(self.edge_pre)

    def index_of(self, body_id: int) -> int:
        """Индекс нейрона по bodyId. -1, если нейрон отфильтрован."""
        hit = np.searchsorted(self.body_ids, body_id)
        if hit < len(self.body_ids) and self.body_ids[hit] == body_id:
            return int(hit)
        return -1

    def find_types(self, pattern: str) -> np.ndarray:
        """Индексы нейронов, чей тип начинается с pattern (например 'MBON')."""
        return np.flatnonzero(
            np.char.startswith(self.types.astype(str), pattern)
        )

    def summary(self) -> str:
        lines = [
            f"нейронов: {self.num_neurons:,}",
            f"рёбер:    {self.num_edges:,}",
        ]
        for gi, name in enumerate(GROUP_NAMES):
            n = int((self.group == gi).sum())
            if n:
                lines.append(f"  {name:<11} {n:>7,}")
        exc = int((self.edge_weight > 0).sum())
        inh = int((self.edge_weight < 0).sum())
        lines.append(f"возбуждающих рёбер: {exc:,}  тормозных: {inh:,}")
        return "\n".join(lines)


def _load_neurons(data_dir: Path, include_vnc: bool):
    """Аннотации -> bodyId, позиция сомы, тип, группа. Только с измеренной сомой."""
    tbl = feather.read_table(
        data_dir / "body-annotations.feather",
        columns=["bodyId", "somaLocation", "type", "superclass"],
    )
    body = tbl.column("bodyId").to_numpy()
    soma = tbl.column("somaLocation").to_pylist()
    types = np.array(tbl.column("type").to_pylist(), dtype=object)
    supercls = tbl.column("superclass").to_pylist()

    keep = np.array(
        [s is not None and len(s) == 3 for s in soma],
        dtype=bool,
    )

    group = np.array(
        [GROUP_NAMES.index(SUPERCLASS_GROUP.get(s or "", "other")) for s in supercls],
        dtype=np.int8,
    )
    if not include_vnc:
        keep &= group != GROUP_NAMES.index("vnc")

    idx = np.flatnonzero(keep)
    body = body[idx].astype(np.int64)
    pos = np.array([soma[i] for i in idx], dtype=np.float32)
    types = types[idx]
    group = group[idx]

    # bodyId должен быть отсортирован — index_of() полагается на searchsorted
    order = np.argsort(body, kind="stable")
    return body[order], pos[order], types[order], group[order]


def _load_signs(data_dir: Path, body_ids: np.ndarray) -> np.ndarray:
    """Знак каждого нейрона по consensus_nt. 0 — медиатор неизвестен/модулятор."""
    tbl = feather.read_table(
        data_dir / "body-neurotransmitters.feather",
        columns=["body", "consensus_nt"],
    )
    nt_body = tbl.column("body").to_numpy()
    nt_name = tbl.column("consensus_nt").to_pylist()

    sign_by_body = np.zeros(len(body_ids), dtype=np.float32)
    hit = np.searchsorted(body_ids, nt_body)
    valid = (hit < len(body_ids)) & (body_ids[np.clip(hit, 0, len(body_ids) - 1)] == nt_body)

    for pos_in_nt in np.flatnonzero(valid):
        sign_by_body[hit[pos_in_nt]] = NT_SIGN.get(nt_name[pos_in_nt] or "", 0.0)
    return sign_by_body


def load_connectome(
    data_dir: str | Path = "data",
    min_weight: int = 5,
    include_vnc: bool = False,
    max_edges: int | None = None,
) -> Connectome:
    """Собрать граф из трёх feather-файлов.

    min_weight  — отсечка по числу синапсов. 5 убирает миллионы одиночных
                  контактов и делает граф пригодным для 3 ГБ видеопамяти.
    include_vnc — включать ли брюшную нервную цепочку (для шахмат не нужна).
    max_edges   — жёсткий потолок: оставить только N самых сильных рёбер.
    """
    data_dir = Path(data_dir)

    body_ids, positions, types, group = _load_neurons(data_dir, include_vnc)
    sign = _load_signs(data_dir, body_ids)

    tbl = feather.read_table(
        data_dir / "connectome-weights.feather",
        columns=["body_pre", "body_post", "weight"],
    )
    w = tbl.column("weight").to_numpy()

    # Файл отсортирован по убыванию веса: достаточно найти границу отсечки.
    cut = int(np.searchsorted(-w, -min_weight, side="right"))
    pre = tbl.column("body_pre").to_numpy()[:cut]
    post = tbl.column("body_post").to_numpy()[:cut]
    w = w[:cut].astype(np.float32)

    # bodyId -> индекс; рёбра к отфильтрованным нейронам выбрасываем
    ipre = np.searchsorted(body_ids, pre)
    ipost = np.searchsorted(body_ids, post)
    n = len(body_ids)
    ok = (
        (ipre < n)
        & (ipost < n)
        & (body_ids[np.clip(ipre, 0, n - 1)] == pre)
        & (body_ids[np.clip(ipost, 0, n - 1)] == post)
    )

    ipre = ipre[ok].astype(np.int32)
    ipost = ipost[ok].astype(np.int32)
    w = w[ok] * sign[ipre]  # знак задаёт пресинаптический нейрон

    # Рёбра без известного медиатора (знак 0) не несут тока — убираем
    nonzero = w != 0
    ipre, ipost, w = ipre[nonzero], ipost[nonzero], w[nonzero]

    if max_edges is not None and len(w) > max_edges:
        strongest = np.argpartition(-np.abs(w), max_edges)[:max_edges]
        strongest.sort()
        ipre, ipost, w = ipre[strongest], ipost[strongest], w[strongest]

    return Connectome(
        body_ids=body_ids,
        positions=positions,
        group=group,
        types=types,
        edge_pre=ipre,
        edge_post=ipost,
        edge_weight=w,
    )


if __name__ == "__main__":
    import sys

    min_w = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    c = load_connectome(min_weight=min_w)
    print(f"--- коннектом, min_weight={min_w} ---")
    print(c.summary())
    for probe in ("MBON", "DNp", "KC"):
        print(f"  нейронов типа {probe}*: {len(c.find_types(probe)):,}")
