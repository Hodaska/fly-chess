"""Установка и запуск одной командой:  python play.py

Первый запуск сам: создаёт .venv, ставит зависимости, скачивает коннектом мозга мухи
(~570 МБ, публичные данные FlyEM/Janelia), Stockfish (только Windows) и запускает игру.
Следующие запуски пропускают всё готовое и сразу стартуют сервер.
"""
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
import webbrowser
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
DATA = ROOT / "data"
URL = "http://127.0.0.1:8000"
BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
# локальное имя -> (имя на сервере, размер в байтах)
FILES = {
    "body-annotations.feather": ("body-annotations-male-cns-v1.0-minconf-0.5.feather", 14483314),
    "body-neurotransmitters.feather": ("body-neurotransmitters-male-cns-v1.0.feather", 43282834),
    "connectome-weights.feather": ("connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather", 508025642),
}
STOCKFISH_WIN = "https://github.com/official-stockfish/Stockfish/releases/download/sf_19/stockfish-windows-x86-64-universal.zip"
IS_WIN = platform.system() == "Windows"
PY = VENV / ("Scripts/python.exe" if IS_WIN else "bin/python")


def say(msg):
    print(f"\n=== {msg}", flush=True)


def run(cmd, **kw):
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def download(url, dest, size=None):
    if dest.exists() and (size is None or dest.stat().st_size == size):
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  скачиваю {dest.name} ...", flush=True)
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, last = 0, 0.0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total and time.time() - last > 2:
                print(f"    {done / 1e6:.0f} / {total / 1e6:.0f} МБ", flush=True)
                last = time.time()
    tmp.replace(dest)


def ensure_venv():
    if PY.exists() and subprocess.run([str(PY), "-c", "import torch, fastapi, chess"], capture_output=True).returncode == 0:
        return
    say("Ставлю зависимости (первый раз 5-10 минут, ~1,5 ГБ)")
    if not PY.exists():
        run([sys.executable, "-m", "venv", VENV])
    run([PY, "-m", "pip", "install", "--upgrade", "pip"])
    has_gpu = shutil.which("nvidia-smi") is not None
    torch_cmd = [PY, "-m", "pip", "install", "torch==2.6.0"]
    if not IS_WIN and not has_gpu:
        torch_cmd += ["--index-url", "https://download.pytorch.org/whl/cpu"]  # на Linux без GPU — лёгкая сборка
    run(torch_cmd)
    rest = [l.strip() for l in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#") and not l.startswith("torch")]
    run([PY, "-m", "pip", "install", *rest])


def ensure_data():
    missing = [n for n, (_, size) in FILES.items() if not (DATA / n).exists() or (DATA / n).stat().st_size != size]
    if missing:
        say("Скачиваю коннектом мозга мухи (~570 МБ, один раз)")
        for name in missing:
            remote, size = FILES[name]
            download(BASE + remote, DATA / name, size)
    for need in ("readout_model.pt",):
        if not (DATA / need).exists():
            sys.exit(f"Нет data/{need} — репозиторий скачан не полностью.")


def ensure_stockfish():
    sf_dir = ROOT / "tools" / "stockfish"
    if any(sf_dir.glob("stockfish*")) or shutil.which("stockfish"):
        return
    if IS_WIN and platform.machine().lower() in ("amd64", "x86_64"):
        say("Скачиваю Stockfish (шахматный движок, для режимов против него)")
        zpath = ROOT / "cache" / "stockfish.zip"
        download(STOCKFISH_WIN, zpath)
        sf_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath) as z:
            for m in z.namelist():
                base = os.path.basename(m)
                if base and (base.lower().startswith("stockfish") and base.lower().endswith(".exe") or base.lower() == "copying.txt"):
                    (sf_dir / base).write_bytes(z.read(m))
    else:
        print("\n(Stockfish не найден. Режим «я против мухи» работает и без него. Для остальных: "
              "Linux — `sudo apt install stockfish`, macOS — `brew install stockfish`.)")


def wait_and_open(proc):
    for _ in range(180):
        if proc.poll() is not None:
            return
        try:
            urllib.request.urlopen(URL + "/state", timeout=1).read()
            say(f"Готово! Открываю {URL}  (закрыть игру: Ctrl+C в этом окне)")
            webbrowser.open(URL)
            return
        except Exception:
            time.sleep(1)


def main():
    if sys.version_info < (3, 10):
        sys.exit("Нужен Python 3.10 или новее (лучше 3.12/3.13): https://www.python.org/downloads/")
    ensure_venv()
    ensure_data()
    ensure_stockfish()
    say("Запускаю сервер (загрузка мозга 10-30 секунд)")
    env = dict(os.environ, PYTHONUTF8="1")
    proc = subprocess.Popen([str(PY), "-m", "flybrain.game_server"], cwd=ROOT, env=env)
    try:
        wait_and_open(proc)
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()


if __name__ == "__main__":
    main()
