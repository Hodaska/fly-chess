#!/bin/bash
# Полный прогон Фазы 2: датасет -> кэш признаков -> обучение головы.
#
# Генерация распараллелена на 5 процессов Stockfish (i5-9400F, 6 ядер, одно
# оставляем системе) — экономит ~2.3 часа против однопоточного прогона,
# без единой строчки нового кода в generate_dataset.py: просто N параллельных
# вызовов уже протестированного скрипта с разными seed на разные файлы.
#
# Кэш признаков (GPU, batch=64) остаётся последовательным — см. заметки проекта,
# батч 64 это уже насыщение GTX 1060 3GB, параллелить процессы GPU нет смысла.
set -e
cd "$(dirname "$0")/.."

PY="./.venv/Scripts/python.exe"
SF="tools/stockfish/stockfish.exe"
TOTAL=150000
WORKERS=5
PER_WORKER=$((TOTAL / WORKERS))
MOVETIME=50

echo "=== этап 1/3: генерация датасета (${TOTAL} позиций, ${WORKERS} параллельных Stockfish) ==="
pids=()
for i in $(seq 0 $((WORKERS - 1))); do
    PYTHONUTF8=1 "$PY" scripts/generate_dataset.py \
        --stockfish "$SF" \
        --num-positions "$PER_WORKER" \
        --movetime-ms "$MOVETIME" \
        --seed "$i" \
        --out "data/dataset_part${i}.parquet" &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done

echo ""
echo "=== склеиваю ${WORKERS} частей датасета ==="
PYTHONUTF8=1 "$PY" -c "
import pandas as pd
import glob
parts = sorted(glob.glob('data/dataset_part*.parquet'))
df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
df.to_parquet('data/dataset.parquet', index=False)
print(f'итого {len(df):,} позиций -> data/dataset.parquet')
"
rm -f data/dataset_part*.parquet

echo ""
echo "=== этап 2/3: прогон мозга по датасету (GPU, ночной прогон) ==="
PYTHONUTF8=1 "$PY" scripts/cache_features.py \
    --dataset data/dataset.parquet \
    --out data/features_cache.npz

echo ""
echo "=== этап 3/3: обучение readout-головы ==="
PYTHONUTF8=1 "$PY" scripts/train_readout.py \
    --cache data/features_cache.npz \
    --out data/readout_model.pt \
    --epochs 30

echo ""
echo "=== ГОТОВО. Проверка на стартовой позиции: ==="
PYTHONUTF8=1 "$PY" -m scripts.play_move --model data/readout_model.pt
