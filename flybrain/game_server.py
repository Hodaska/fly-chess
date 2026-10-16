"""Игра против мухи в браузере: доска слева, живой мозг справа.

Запуск:
    PYTHONUTF8=1 ./.venv/Scripts/python.exe -m flybrain.game_server
    -> http://127.0.0.1:8000   (3D-сцена; старая 2D-страница, если есть, — /2d)

Мозг загружается ОДИН раз при старте, дальше ход считается за доли секунды.
Первый старт калибрует пороги (~2 мин) и сохраняет их в data/brain_thresholds.pt,
следующие старты берут их с диска и поднимаются за секунды.

Протокол:
    POST /new    {"color": "w"|"b"|"none"}  новая партия; "none" — муха против мухи
    POST /move   {"uci": "e2e4"}     ход человека
    POST /think                      ход мухи (+ запись активности мозга)
    POST /engine                     ход Stockfish (режимы "sf" и "fly_sf" в /new)
    GET  /trace                      бинарь с активностью последнего хода мухи
    GET  /atlas, /meta               геометрия мозга для просмотрщика
"""

from __future__ import annotations

import asyncio
import shutil
import struct
import sys
import threading
import warnings
from pathlib import Path

import chess
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore", message=".*Sparse CSR.*")
warnings.filterwarnings("ignore", message="(?s).*on_event is deprecated.*")

from flybrain.chess_encoding import board_to_bits, decode_move, legal_move_mask  # noqa: E402
from flybrain.connectome import GROUP_COLORS, GROUP_NAMES  # noqa: E402
from flybrain.server import viewer_positions  # noqa: E402
from scripts.play_move import build_extractor, load_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VIEWER_DIR = ROOT / "viewer"
MODEL_PATH = ROOT / "data" / "readout_model.pt"
THRESHOLDS_PATH = ROOT / "data" / "brain_thresholds.pt"
TOP_K = 3
SF_MOVE_TIME = 0.4  # секунд на ход Stockfish

app = FastAPI(title="fly chess")
app.mount("/static", StaticFiles(directory=VIEWER_DIR), name="static")

S: dict = {}  # мозг, модель, партия
lock = threading.Lock()


# --------------------------------------------------------------------------
# загрузка мозга
# --------------------------------------------------------------------------
def _load_brain():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std, config = load_model(str(MODEL_PATH), device)

    cached = None
    if THRESHOLDS_PATH.exists():
        blob = torch.load(THRESHOLDS_PATH, map_location=device, weights_only=False)
        if blob.get("config") == config:
            cached = blob["threshold"]

    if cached is not None:
        print("пороги найдены на диске, калибровка не нужна")
        ex = build_extractor(dict(config, batch=1, calibrate_steps=0), device)
        ex.brain.threshold = cached.to(device)
        ex.brain.p.homeostasis = False
    else:
        print("первый запуск: калибрую пороги мозга (~2 мин, дальше не понадобится)...")
        ex = build_extractor(config, device)
        torch.save({"config": config, "threshold": ex.brain.threshold.cpu()}, THRESHOLDS_PATH)
        ex.rebatch(1)

    S.update(
        device=device,
        model=model,
        mean=mean,
        std=std,
        ex=ex,
        positions=viewer_positions(ex.c),
        board=chess.Board(),
        human=chess.WHITE,
        mode="fly",
        fly_color=None,
        sf_elo=None,
        engine=None,
        engine_elo=-1,
        san=[],
        last=None,
        fly_top=[],
        trace=b"",
        traces={},
        rng=np.random.default_rng(),
    )


@app.on_event("startup")
def _startup() -> None:
    _load_brain()
    print("готово -> http://127.0.0.1:8000")


# --------------------------------------------------------------------------
# партия
# --------------------------------------------------------------------------
def _state() -> dict:
    b: chess.Board = S["board"]
    # claim_draw: троекратное повторение и правило 50 ходов заканчивают партию
    # сразу — иначе две мухи могут часами гонять одни и те же ходы
    outcome = b.outcome(claim_draw=True)
    human = S["human"]
    fc = S["fly_color"]
    return {
        "fen": b.fen(),
        "turn": "w" if b.turn == chess.WHITE else "b",
        "human": "none" if human is None else ("w" if human == chess.WHITE else "b"),
        "mode": S["mode"],
        "next": _next(),
        "fly_color": None if fc is None else ("w" if fc == chess.WHITE else "b"),
        "sf_elo": S["sf_elo"],
        "legal": [m.uci() for m in b.legal_moves],
        "check": b.is_check(),
        "last": S["last"],
        "history": S["san"],
        "over": outcome is not None,
        "result": outcome.result() if outcome else None,
        "reason": outcome.termination.name.lower() if outcome else None,
        "fly_top": S["fly_top"],
    }


def _next() -> str:
    """Кто ходит сейчас: "human" | "fly" | "sf"."""
    turn = S["board"].turn
    if S["mode"] == "fly_sf":
        return "fly" if turn == S["fly_color"] else "sf"
    if S["human"] == turn:
        return "human"
    return "sf" if S["mode"] == "sf" else "fly"


def _stockfish_path() -> Path | None:
    for f in sorted((ROOT / "tools" / "stockfish").glob("stockfish*")):
        if f.is_file() and f.suffix.lower() in ("", ".exe"):
            return f.resolve()  # относительный путь ломает asyncio-подпроцесс на Windows
    found = shutil.which("stockfish")
    return Path(found) if found else None


def _engine():
    if S["engine"] is None:
        path = _stockfish_path()
        if path is None:
            raise HTTPException(503, "Stockfish не найден (tools/stockfish/)")
        import chess.engine

        S["engine"] = chess.engine.SimpleEngine.popen_uci(str(path))
        S["engine"].configure({"Threads": 1, "Hash": 32})
        S["engine_elo"] = -1
    return S["engine"]


def _sf_move() -> None:
    import chess.engine

    eng = _engine()
    elo = S["sf_elo"]
    if elo != S["engine_elo"]:
        opt = eng.options.get("UCI_Elo")
        if elo and opt is not None:
            eng.configure({"UCI_LimitStrength": True, "UCI_Elo": max(opt.min, min(opt.max, elo))})
        else:
            eng.configure({"UCI_LimitStrength": False})
        S["engine_elo"] = elo
    result = eng.play(S["board"], chess.engine.Limit(time=SF_MOVE_TIME))
    _push(result.move)


@app.on_event("shutdown")
def _shutdown() -> None:
    if S.get("engine") is not None:
        S["engine"].quit()


def _push(move: chess.Move) -> None:
    b: chess.Board = S["board"]
    S["san"].append(b.san(move))
    b.push(move)
    S["last"] = move.uci()


def _pack_trace(trace: list[np.ndarray]) -> bytes:
    """uint32 число шагов, затем uint32 длины по шагам, затем все индексы подряд."""
    lengths = np.array([len(t) for t in trace], dtype=np.uint32)
    flat = np.concatenate(trace).astype(np.uint32) if trace else np.zeros(0, np.uint32)
    return struct.pack("<I", len(trace)) + lengths.tobytes() + flat.tobytes()


def _over() -> bool:
    return S["board"].outcome(claim_draw=True) is not None


def _fly_move(sample: bool = False) -> None:
    """Муха думает: прогон мозга с записью, readout, маска легальных ходов.

    sample=False — берём самый вероятный ход (игра против человека).
    sample=True  — случайно из топ-3 по их вероятностям (муха против мухи):
    мозг детерминирован, и без случайности каждая партия была бы копией
    предыдущей.
    """
    b: chess.Board = S["board"]
    ex = S["ex"]

    feats, trace = ex.extract_with_trace(board_to_bits(b)[None, :])
    x = torch.from_numpy((feats - S["mean"]) / S["std"]).float().to(S["device"])
    with torch.no_grad():
        logits = S["model"](x).squeeze(0).cpu().numpy().astype(np.float64)

    mask, legal = legal_move_mask(b)
    logits[~mask] = -np.inf
    probs = np.exp(logits - logits[mask].max())
    probs /= probs.sum()

    top = np.argsort(-probs)[:TOP_K]
    S["fly_top"] = [
        {
            "uci": decode_move(b, int(c), legal).uci(),
            "san": b.san(decode_move(b, int(c), legal)),
            "p": float(probs[c]),
        }
        for c in top
        if probs[c] > 0
    ]
    choice = top[0]
    if sample:
        p = probs[top] / probs[top].sum()
        choice = S["rng"].choice(top, p=p)
    _push(decode_move(b, int(choice), legal))

    # Запись привязана к номеру полухода: страница просит /trace?ply=N и
    # получает именно запись этого хода, даже если между /think и /trace
    # другая вкладка успела начать новую партию или сделать ход.
    ply = len(S["san"])
    S["traces"][ply] = S["trace"] = _pack_trace(trace)
    for old in [k for k in S["traces"] if k < ply - 8]:
        del S["traces"][old]


class NewGame(BaseModel):
    # mode "fly": человек против мухи, color — цвет человека ("none" — муха против мухи, только для 2D)
    # mode "sf": человек против Stockfish, color — цвет человека
    # mode "fly_sf": муха против Stockfish, color — цвет МУХИ
    color: str = "w"
    mode: str = "fly"
    elo: int | None = None  # сила Stockfish (UCI_Elo); None/0 — на полную
    fen: str | None = None  # только для тестов особых ходов (рокировка, взятие на проходе)


class HumanMove(BaseModel):
    uci: str


@app.post("/new")
def new_game(req: NewGame) -> dict:
    with lock:
        S["board"] = chess.Board(req.fen) if req.fen else chess.Board()
        if req.mode not in ("fly", "sf", "fly_sf"):
            raise HTTPException(400, "неизвестный режим")
        side = {"w": chess.WHITE, "b": chess.BLACK}.get(req.color)  # None: муха vs муха
        if req.mode != "fly" and side is None:
            side = chess.WHITE
        S["mode"] = req.mode
        S["human"] = None if req.mode == "fly_sf" else side
        S["fly_color"] = side if req.mode == "fly_sf" else None
        S["sf_elo"] = req.elo or None
        S["san"], S["last"], S["fly_top"], S["trace"] = [], None, [], b""
        S["traces"] = {}
        return _state()


@app.post("/move")
def human_move(req: HumanMove) -> dict:
    with lock:
        b: chess.Board = S["board"]
        if S["human"] is None or b.turn != S["human"] or _over():
            raise HTTPException(409, "сейчас не ваш ход")
        try:
            move = chess.Move.from_uci(req.uci)
        except ValueError:
            raise HTTPException(400, "непонятный ход") from None
        if move not in b.legal_moves:
            raise HTTPException(400, "нелегальный ход")
        _push(move)
        return _state()


@app.post("/think")
async def think() -> dict:
    def work():
        with lock:
            if _over() or _next() != "fly":
                raise HTTPException(409, "сейчас не ход мухи")
            _fly_move(sample=S["mode"] == "fly" and S["human"] is None)
            return _state()

    return await asyncio.to_thread(work)


@app.post("/engine")
async def engine_move() -> dict:
    def work():
        with lock:
            if _over() or _next() != "sf":
                raise HTTPException(409, "сейчас не ход Stockfish")
            _sf_move()
            return _state()

    return await asyncio.to_thread(work)


@app.get("/trace")
def trace(ply: int | None = None) -> Response:
    data = S["trace"] if ply is None else S["traces"].get(ply)
    if data is None:
        raise HTTPException(404, "записи этого хода нет")
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/state")
def state() -> dict:
    with lock:
        return _state()


# --------------------------------------------------------------------------
# геометрия мозга и страница
# --------------------------------------------------------------------------
@app.get("/")
@app.get("/3d")
def index() -> FileResponse:
    return FileResponse(VIEWER_DIR / "play3d.html")


# старая 2D-версия: лежит только в рабочей копии проекта, в пакет для друга не входит
if (VIEWER_DIR / "play.html").exists():
    @app.get("/2d")
    def index_2d() -> FileResponse:
        return FileResponse(VIEWER_DIR / "play.html")


@app.get("/meta")
def meta() -> dict:
    c = S["ex"].c
    return {
        "neurons": int(c.num_neurons),
        "edges": int(c.num_edges),
        "steps": int(S["ex"].steps),
        "dtMs": float(S["ex"].brain.p.dt),
        "stockfish": _stockfish_path() is not None,
        "groups": [
            {"key": k, "color": col, "count": int((c.group == i).sum())}
            for i, (k, col) in enumerate(zip(GROUP_NAMES, GROUP_COLORS))
            if (c.group == i).sum()
        ],
    }


@app.get("/atlas")
def atlas() -> Response:
    c = S["ex"].c
    payload = (
        struct.pack("<I", c.num_neurons)
        + S["positions"].tobytes()
        + c.group.astype(np.uint8).tobytes()
    )
    return Response(content=payload, media_type="application/octet-stream")


def main() -> None:
    import uvicorn

    sys.stdout.reconfigure(encoding="utf-8")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
