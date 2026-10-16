"""Собрать для сайта тело мухи + всю ЦНС (мозг и нервный тяж) в одной системе координат.

Выход (кладётся в viewer/assets/, отдаётся статикой — сервер перезапускать не надо):
  flybody.bin  тело: uint32 nverts, uint32 ntris,
               float32[nverts*3] позиции, uint8[nverts] материал, uint32[ntris*3] индексы
  cns.bin      нейроны: uint32 N, float32[N*3] позиции, uint8[N] группа,
               uint32 Nsim, uint32[Nsim] индекс в этом массиве для каждого нейрона симуляции
  fly_parts.bin тело БЕЗ крыльев + два крыла отдельно (для 3D-сцены Фазы 5, где
               крылья должны махать): uint32 nBodyVerts, uint32 nBodyTris,
               float32[nBodyVerts*3] позиции, uint8[nBodyVerts] материал (0 тело/1 глаз),
               uint32[nBodyTris*3] индексы (+паддинг до 4 байт после uint8-массива);
               затем на каждое крыло: float32[3] точка шарнира (сцена),
               uint32 nVerts, uint32 nTris, float32[nVerts*3] позиции ОТНОСИТЕЛЬНО
               шарнира, uint32[nTris*3] индексы. Порядок: левое, затем правое.

Тело — flybody (TuragaLab / Google DeepMind + HHMI Janelia, Apache 2.0),
собирается через MuJoCo в позе по умолчанию, сетка упрощается для браузера.

ЦНС — MaleCNS v1.0 (CC BY 4.0), вместе с нервным тяжом (VNC). Симулируется
только мозг (модель обучена без тяжа), поэтому нужна таблица соответствия
«нейрон симуляции -> точка на экране».

Совмещение мозга с телом — РУЧНОЕ, но в настоящем масштабе: воксель 8 нм,
модель тела в сантиметрах. Оси EM: x — поперёк, +y — вентрально, +z — назад
(тяж лежит на 0.55 мм позади и 0.21 мм ниже мозга). Оси тела MuJoCo: +x —
вперёд, +z — вверх. Центр мозга ставится в центр головной капсулы; тяж при
этом сам попадает в нижнюю часть груди. Анатомической регистрации нет —
это разные особи и разные данные.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import fast_simplification
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flybrain.connectome import GROUP_NAMES, _load_neurons, load_connectome  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# MuJoCo (библиотека на C) не открывает пути с кириллицей — только относительный
# путь от корня проекта, поэтому в main() делаем chdir в ROOT.
FLYBODY_XML = "cache/flybody/flybody/fruitfly/assets/fruitfly.xml"
OUT = ROOT / "viewer/assets"

VOXEL_CM = 8e-7          # 8 нм в сантиметрах
SCENE_SCALE = 15.0       # см -> единицы сцены (муха ~4.5 единицы в длину)
KEEP = 0.25              # доля треугольников после упрощения
MAT_BODY, MAT_EYE, MAT_WING = 0, 1, 2

# слои-дубликаты для текстур (щетинки, тёмные пятна) для стеклянного тела не нужны
SKIP_SUFFIX = ("_black", "_brown")
SKIP_NAMES = {"head_ocelli"}


def build_body(m, d, exclude_bodies: frozenset[str] = frozenset()):
    verts, mats, tris = [], [], []
    base = 0
    head_pts = []
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
        if body_name in exclude_bodies:
            continue
        mi = m.geom_dataid[g]
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mi)
        if name.endswith(SKIP_SUFFIX) or name in SKIP_NAMES:
            continue

        va, vn = m.mesh_vertadr[mi], m.mesh_vertnum[mi]
        fa, fn = m.mesh_faceadr[mi], m.mesh_facenum[mi]
        v = m.mesh_vert[va:va + vn].astype(np.float64)
        f = m.mesh_face[fa:fa + fn].astype(np.int64)
        v = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]

        # сшиваем совпадающие вершины — иначе упрощение не видит общих рёбер
        key = np.round(v / 1e-6).astype(np.int64)
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        v, f = v[first], inv.reshape(-1)[f]
        f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]

        if len(f) > 400:
            v, f = fast_simplification.simplify(
                v.astype(np.float32), f.astype(np.int32), target_reduction=1 - KEEP
            )

        if name == "head":
            head_pts.append(v)
        mat = MAT_EYE if name == "head_red" else MAT_WING if "membrane" in name else MAT_BODY
        verts.append(np.asarray(v, np.float64))
        mats.append(np.full(len(v), mat, np.uint8))
        tris.append(np.asarray(f, np.int64) + base)
        base += len(v)

    V = np.concatenate(verts)
    head = np.concatenate(head_pts)
    head_center = (head.min(0) + head.max(0)) / 2
    return V, np.concatenate(mats), np.concatenate(tris), head_center


def build_wing(m, d, body_name: str):
    """Меш одного крыла (мембрана+каркас) в мировых координатах + точка шарнира.

    Точка шарнира — d.xpos тела крыла (там, где оно крепится к груди);
    вершины возвращаются ОТНОСИТЕЛЬНО неё, чтобы в three.js можно было
    просто вращать группу вокруг локального нуля для взмаха.
    """
    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
    pivot = d.xpos[body_id].copy()

    verts, tris = [], []
    base = 0
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if m.geom_bodyid[g] != body_id:
            continue
        mi = m.geom_dataid[g]
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mi)
        if name.endswith("_black"):
            continue
        va, vn = m.mesh_vertadr[mi], m.mesh_vertnum[mi]
        fa, fn = m.mesh_faceadr[mi], m.mesh_facenum[mi]
        v = m.mesh_vert[va:va + vn].astype(np.float64)
        f = m.mesh_face[fa:fa + fn].astype(np.int64)
        v = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]

        key = np.round(v / 1e-6).astype(np.int64)
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        v, f = v[first], inv.reshape(-1)[f]
        f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]

        verts.append(v - pivot)
        tris.append(f + base)
        base += len(v)

    V = np.concatenate(verts)
    T = np.concatenate(tris)
    return pivot, V, T


def build_cns(head_center: np.ndarray):
    data = ROOT / "data"
    body_ids, pos, _, group = _load_neurons(data, include_vnc=True)

    # EM (x поперёк, +y вентрально, +z назад) -> тело (+x вперёд, +y поперёк, +z вверх)
    p = pos.astype(np.float64) * VOXEL_CM
    fly = np.stack([-p[:, 2], p[:, 0], -p[:, 1]], axis=1)

    brain = group != GROUP_NAMES.index("vnc")
    fly += head_center - fly[brain].mean(0)

    # соответствие нейронам симуляции (граф без тяжа, тот же порядок по bodyId)
    sim = load_connectome(data, min_weight=5, include_vnc=False)
    idx = np.searchsorted(body_ids, sim.body_ids)
    assert np.all(body_ids[idx] == sim.body_ids), "нейроны симуляции не нашлись в полной ЦНС"
    return fly, group, idx.astype(np.uint32)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    import os
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)

    print("собираю тело (MuJoCo, поза по умолчанию)...")
    m = mujoco.MjModel.from_xml_path(FLYBODY_XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    V, M, T, head_center = build_body(m, d)
    print(f"  вершин {len(V):,}, треугольников {len(T):,}")

    print("собираю тело без крыльев + крылья отдельно (для 3D-сцены)...")
    Vb, Mb, Tb, _ = build_body(m, d, exclude_bodies=frozenset({"wing_left", "wing_right"}))
    wl_pivot, wl_V, wl_T = build_wing(m, d, "wing_left")
    wr_pivot, wr_V, wr_T = build_wing(m, d, "wing_right")
    print(f"  тело без крыльев: вершин {len(Vb):,}; крыло L {len(wl_V):,} верш., R {len(wr_V):,} верш.")

    print("размещаю ЦНС в голове...")
    C, G, sim_map = build_cns(head_center)
    vnc = G == GROUP_NAMES.index("vnc")
    print(f"  нейронов на экране {len(C):,} (из них тяж {vnc.sum():,}), в симуляции {len(sim_map):,}")

    # общий центр и масштаб сцены — по телу
    center = (V.min(0) + V.max(0)) / 2
    Vs = ((V - center) * SCENE_SCALE).astype(np.float32)
    Cs = ((C - center) * SCENE_SCALE).astype(np.float32)

    # проверка «на глаз», что тяж попал в грудь, а не торчит наружу
    bb_lo, bb_hi = V.min(0), V.max(0)
    inside = np.all((C >= bb_lo) & (C <= bb_hi), axis=1).mean()
    print(f"  нейронов внутри габаритов тела: {inside * 100:.1f}%")

    with open(OUT / "flybody.bin", "wb") as f:
        f.write(struct.pack("<II", len(Vs), len(T)))
        f.write(Vs.tobytes())
        f.write(M.tobytes())
        pad = (-(len(Vs) * 13) % 4)  # выравнивание индексов на 4 байта
        f.write(b"\0" * pad)
        f.write(T.astype(np.uint32).tobytes())
    with open(OUT / "cns.bin", "wb") as f:
        f.write(struct.pack("<I", len(Cs)))
        f.write(Cs.tobytes())
        f.write(G.astype(np.uint8).tobytes())
        f.write(b"\0" * (-(4 + len(Cs) * 13) % 4))
        f.write(struct.pack("<I", len(sim_map)))
        f.write(sim_map.tobytes())

    # то же самое центрирование/масштаб, что у основного тела и ЦНС
    Vbs = ((Vb - center) * SCENE_SCALE).astype(np.float32)
    wl_pivot_s = ((wl_pivot - center) * SCENE_SCALE).astype(np.float32)
    wr_pivot_s = ((wr_pivot - center) * SCENE_SCALE).astype(np.float32)
    wl_Vs = (wl_V * SCENE_SCALE).astype(np.float32)  # уже относительно шарнира, центр не вычитаем
    wr_Vs = (wr_V * SCENE_SCALE).astype(np.float32)

    with open(OUT / "fly_parts.bin", "wb") as f:
        f.write(struct.pack("<II", len(Vbs), len(Tb)))
        f.write(Vbs.tobytes())
        f.write(Mb.tobytes())
        f.write(b"\0" * (-(len(Vbs) * 13) % 4))
        f.write(Tb.astype(np.uint32).tobytes())
        for pivot, Vw, Tw in ((wl_pivot_s, wl_Vs, wl_T), (wr_pivot_s, wr_Vs, wr_T)):
            f.write(pivot.tobytes())
            f.write(struct.pack("<II", len(Vw), len(Tw)))
            f.write(Vw.tobytes())
            f.write(Tw.astype(np.uint32).tobytes())

    for name in ("flybody.bin", "cns.bin", "fly_parts.bin"):
        print(f"  {name}: {(OUT / name).stat().st_size / 2**20:.1f} МБ")


if __name__ == "__main__":
    main()
