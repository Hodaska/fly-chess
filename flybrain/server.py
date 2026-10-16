"""Сервер просмотрщика: крутит LIF-симуляцию и стримит спайки в браузер.

Запуск:
    python -m flybrain.server
    -> http://127.0.0.1:8000

Протокол:
    GET  /atlas   бинарь: positions float32[N,3] + groups uint8[N]
    WS   /ws      кадры: uint32[] индексы спайкнувших нейронов
"""

from __future__ import annotations

import asyncio
import json
import struct
import warnings
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response

from .connectome import GROUP_COLORS, GROUP_NAMES, load_connectome
from .lif import LIFBrain, LIFParams

warnings.filterwarnings("ignore", message=".*Sparse CSR.*")

VIEWER_DIR = Path(__file__).resolve().parent.parent / "viewer"
MIN_WEIGHT = 5
FPS = 30
STEPS_PER_FRAME = 2  # шагов симуляции на кадр (dt=0.5мс -> 1мс биологического)

app = FastAPI(title="fly brain viewer")

state: dict = {}


def viewer_positions(c) -> np.ndarray:
    """Координаты сомы -> компактные координаты для WebGL, как в атласе MaleCNS."""
    pos = c.positions.copy()
    pos[:, 1] *= -1
    pos[:, 2] *= -1
    # Центрируем по центру масс, а не по bounding box: 90k нейронов оптических
    # долей смещают видимую массу к одному краю, и мозг уезжает из кадра.
    pos -= pos.mean(0)
    pos *= 5.0 / float((pos.max(0) - pos.min(0)).max())
    return np.ascontiguousarray(pos, dtype=np.float32)


@app.on_event("startup")
def _startup() -> None:
    print("загружаю коннектом...")
    c = load_connectome(min_weight=MIN_WEIGHT)
    print(c.summary())

    brain = LIFBrain(c, params=LIFParams(), batch=1)
    print(f"устройство: {brain.device}")

    state.update(
        connectome=c,
        brain=brain,
        positions=viewer_positions(c),
        stimulus=brain.make_external(np.flatnonzero(c.group == 0)[:2000], 0.6),
    )
    print("готово -> http://127.0.0.1:8000")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(VIEWER_DIR / "index.html")


@app.get("/meta")
def meta() -> dict:
    c = state["connectome"]
    return {
        "neurons": int(c.num_neurons),
        "edges": int(c.num_edges),
        "device": str(state["brain"].device),
        "minWeight": MIN_WEIGHT,
        "groups": [
            {"key": k, "color": col, "count": int((c.group == i).sum())}
            for i, (k, col) in enumerate(zip(GROUP_NAMES, GROUP_COLORS))
            if (c.group == i).sum()
        ],
        "dataset": "male-cns v1.0",
        "license": "CC BY 4.0 — FlyEM/HHMI Janelia, Cambridge, MRC LMB, Google Research",
    }


@app.get("/atlas")
def atlas() -> Response:
    """positions float32[N,3], затем groups uint8[N]."""
    c = state["connectome"]
    payload = (
        struct.pack("<I", c.num_neurons)
        + state["positions"].tobytes()
        + c.group.astype(np.uint8).tobytes()
    )
    return Response(content=payload, media_type="application/octet-stream")


@app.websocket("/ws")
async def ws(sock: WebSocket) -> None:
    await sock.accept()
    brain: LIFBrain = state["brain"]
    interval = 1.0 / FPS
    try:
        while True:
            for _ in range(STEPS_PER_FRAME):
                brain.step(state["stimulus"])
            idx = brain.spiking_indices().astype(np.uint32)
            await sock.send_bytes(idx.tobytes())
            await asyncio.sleep(interval)
    except (WebSocketDisconnect, RuntimeError):
        pass


def main() -> None:
    import sys

    import uvicorn

    sys.stdout.reconfigure(encoding="utf-8")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
