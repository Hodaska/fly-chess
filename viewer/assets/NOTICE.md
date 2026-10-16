# Источники данных для 3D-мухи

Файлы генерирует `scripts/build_body_assets.py` (нужны `cache/flybody` — sparse-клон
TuragaLab/flybody — и пакеты `mujoco`, `fast-simplification`).

## flybody.bin — тело мухи

Упрощённая сетка модели **flybody**, собранная в позе по умолчанию.

- Проект: https://github.com/TuragaLab/flybody
- Авторы: Roman Vaxenburg, Igor Siwanowicz, Josh Merel, Alice A. Robie, Carmen Morrow,
  Guido Novati, Zinovia Stefanidi, Gert-Jan Both, Gwyneth M. Card, Michael B. Reiser,
  Matthew M. Botvinick, Kristin M. Branson, Yuval Tassa, Srinivas C. Turaga —
  Google DeepMind и HHMI Janelia Research Campus
- Публикация: «Whole-body physics simulation of fruit fly locomotion»,
  Nature 643, 1312–1320 (2025), doi:10.1038/s41586-025-09029-4
- Лицензия: Apache License 2.0

Изменения: сборка частей через MuJoCo в одну сетку, удаление слоёв-дубликатов
(`*_black`, `*_brown`, `head_ocelli`), упрощение сетки до ~25% треугольников.

## cns.bin — нейроны мозга и нервного тяжа

Позиции сомы из **MaleCNS v1.0** — FlyEM / HHMI Janelia, University of Cambridge,
MRC Laboratory of Molecular Biology, Google Research. Лицензия CC BY 4.0.
https://male-cns.janelia.org/

## Совмещение

Мозг вставлен в тело ВРУЧНУЮ, но в настоящем масштабе (воксель 8 нм, модель тела
в сантиметрах): центр мозга — в центр головной капсулы. Анатомической регистрации
между коннектомом и моделью тела нет — это разные особи и разные данные.
