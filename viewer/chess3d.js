// 3D-доска: трава, доска, фигуры, свет/тени, камера, клики.
// Правила шахмат тут не считаются — только рендер того, что говорит сервер
// (st.fen, st.legal, st.last). Истина всегда FEN с сервера.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const FILES = 'abcdefgh';
export const SQUARE = 1;
const TRAY_INNER_X = 4.9, TRAY_OUTER_X = 6.9;  // подставки для съеденных фигур по бокам доски

export function squareToXZ(sq) {
  const file = FILES.indexOf(sq[0]);
  const rank = +sq[1] - 1;
  return [file - 3.5, 3.5 - rank];
}

function xzToSquare(x, z) {
  const file = Math.round(x + 3.5);
  const rank = Math.round(3.5 - z);
  if (file < 0 || file > 7 || rank < 0 || rank > 7) return null;
  return FILES[file] + (rank + 1);
}

const COLORS = {
  light: 0xe8e0cc, dark: 0x8a6a4a,
  white: 0xece6d8, black: 0x2e2b36,
  frame: 0x5b3d24,
  lastMove: 0xffc45c, selected: 0xffb85c, target: 0x6fd08c, check: 0xeb5046,
};

// -------------------------------------------------------------------------
// геометрия фигур
// -------------------------------------------------------------------------
function lathe(points, segments = 28) {
  const pts = points.map(([x, y]) => new THREE.Vector2(x, y));
  const g = new THREE.LatheGeometry(pts, segments);
  g.computeVertexNormals();
  return g;
}

function pawnGeom() {
  return lathe([
    [0, 0], [0.34, 0], [0.30, 0.06], [0.16, 0.10], [0.14, 0.30], [0.22, 0.36],
    [0.22, 0.42], [0.10, 0.48], [0.20, 0.55], [0.20, 0.62], [0, 0.65],
  ]);
}

function rookBodyGeom() {
  return lathe([
    [0, 0], [0.38, 0], [0.34, 0.08], [0.20, 0.12], [0.20, 0.60], [0.32, 0.66],
    [0.32, 0.78], [0.26, 0.80], [0, 0.80],
  ]);
}

function bishopGeom() {
  return lathe([
    [0, 0], [0.36, 0], [0.32, 0.07], [0.17, 0.11], [0.15, 0.55], [0.28, 0.66],
    [0.22, 0.78], [0.10, 0.88], [0.05, 0.96], [0, 0.98],
  ]);
}

function queenGeom() {
  return lathe([
    [0, 0], [0.40, 0], [0.35, 0.08], [0.19, 0.13], [0.17, 0.62], [0.30, 0.72],
    [0.30, 0.85], [0.20, 0.92], [0.20, 1.0], [0, 1.02],
  ]);
}

function kingGeom() {
  return lathe([
    [0, 0], [0.40, 0], [0.35, 0.08], [0.19, 0.13], [0.17, 0.64], [0.30, 0.74],
    [0.30, 0.88], [0.22, 0.94], [0.22, 1.02], [0, 1.05],
  ]);
}

function knightGeom() {
  // силуэт головы коня в плоскости XY, экструдированный по Z и утоньшённый
  const shape = new THREE.Shape();
  shape.moveTo(-0.20, 0.00);
  shape.lineTo(0.20, 0.00);
  shape.lineTo(0.22, 0.20);
  shape.lineTo(0.16, 0.34);
  shape.lineTo(0.26, 0.46);
  shape.lineTo(0.30, 0.58);
  shape.lineTo(0.20, 0.60);
  shape.lineTo(0.10, 0.52);
  shape.lineTo(-0.02, 0.56);
  shape.lineTo(-0.16, 0.50);
  shape.lineTo(-0.22, 0.34);
  shape.lineTo(-0.20, 0.16);
  shape.closePath();
  const g = new THREE.ExtrudeGeometry(shape, { depth: 0.16, bevelEnabled: true, bevelSize: 0.02, bevelThickness: 0.02, bevelSegments: 2 });
  g.translate(0, 0, -0.08);
  g.computeVertexNormals();
  return g;
}

function knightBaseGeom() {
  return lathe([[0, 0], [0.34, 0], [0.30, 0.07], [0.20, 0.10], [0.18, 0.18], [0.12, 0.20]]);
}

const PIECE_HEIGHT = { p: 0.55, n: 0.62, b: 0.75, r: 0.62, q: 0.92, k: 1.1 };
const GEOM_CACHE = {};
function baseGeom(type) {
  if (GEOM_CACHE[type]) return GEOM_CACHE[type];
  const fn = { p: pawnGeom, r: rookBodyGeom, b: bishopGeom, q: queenGeom, k: kingGeom, n: knightBaseGeom }[type];
  return (GEOM_CACHE[type] = fn());
}

// Король (человек в кресле) и конь (всадник) — готовые glb из Blender.
// Какой файл не загрузился — та фигура остаётся точёной.
const CUSTOM = {};  // 'k_w' -> Group, масштабированная и поставленная на y=0
const CUSTOM_FIT = { k: { h: 1.1, w: 0.8 }, n: { h: 0.95, w: 0.85 }, q: { h: 1.0, w: 0.8 }, b: { h: 0.95, w: 0.8 }, r: { h: 0.9, w: 0.9 } };
const CUSTOM_PLINTH = new Set(['k', 'n']);  // у остальных своя подставка в модели
{
  const loader = new GLTFLoader();
  const files = {
    k_w: 'person_white', k_b: 'person_black', n_w: 'horse_white', n_b: 'horse_black',
    q_w: 'queen_white', q_b: 'queen_black', b_w: 'bishop_white', b_b: 'bishop_black',
    r_w: 'rook_white', r_b: 'rook_black',
  };
  await Promise.all(Object.entries(files).map(async ([key, name]) => {
    try {
      const gltf = await loader.loadAsync(`/static/assets/${name}.glb`);
      const model = gltf.scene;
      const box = new THREE.Box3().setFromObject(model);
      const size = box.getSize(new THREE.Vector3());
      const fit = CUSTOM_FIT[key[0]];
      const s = Math.min(fit.h / size.y, fit.w / Math.max(size.x, size.z));
      model.scale.setScalar(s);
      model.position.set(-(box.min.x + box.max.x) / 2 * s, -box.min.y * s + (CUSTOM_PLINTH.has(key[0]) ? 0.06 : 0), -(box.min.z + box.max.z) / 2 * s);
      const wrap = new THREE.Group();
      wrap.add(model);
      // белые смотрят на соперника (к -z), чёрные — к +z; модель из Blender смотрит к +z
      wrap.rotation.y = key.endsWith('_w') ? Math.PI : 0;
      model.traverse(o => { if (o.isMesh) o.castShadow = true; });
      CUSTOM[key] = wrap;
    } catch (e) {
      console.warn(`${name}.glb не загрузился, эта фигура остаётся точёной`, e);
    }
  }));
}

function buildCustomPiece(type, color) {
  const group = new THREE.Group();
  const mat = new THREE.MeshStandardMaterial({
    color: color === 'w' ? COLORS.white : COLORS.black,
    roughness: color === 'w' ? 0.55 : 0.4,
    metalness: 0.05,
  });
  const plinth = new THREE.Mesh(lathe([[0, 0], [0.38, 0], [0.34, 0.06], [0, 0.06]]), mat);
  plinth.castShadow = true;
  if (CUSTOM_PLINTH.has(type)) group.add(plinth);
  group.add(CUSTOM[`${type}_${color}`].clone());
  group.traverse(o => { if (o.isMesh) o.receiveShadow = true; });
  group.userData.type = type;
  group.userData.color = color;
  return group;
}

function buildPieceMesh(type, color) {
  if (CUSTOM[`${type}_${color}`]) return buildCustomPiece(type, color);
  const mat = new THREE.MeshStandardMaterial({
    color: color === 'w' ? COLORS.white : COLORS.black,
    roughness: color === 'w' ? 0.55 : 0.4,
    metalness: 0.05,
  });
  const group = new THREE.Group();
  const scale = PIECE_HEIGHT[type] / (type === 'k' ? 1.05 : type === 'q' ? 1.02 : type === 'n' ? 0.6 : type === 'b' ? 0.98 : type === 'r' ? 0.8 : 0.65);
  const body = new THREE.Mesh(baseGeom(type), mat);
  body.castShadow = true;
  group.add(body);

  if (type === 'n') {
    const head = new THREE.Mesh(knightGeom(), mat);
    head.position.set(0, 0.20, 0);
    head.rotation.y = Math.PI / 2;
    head.scale.setScalar(1.05);
    head.castShadow = true;
    group.add(head);
  } else if (type === 'r') {
    for (let i = 0; i < 4; i++) {
      const b = new THREE.Mesh(new THREE.BoxGeometry(0.13, 0.10, 0.13), mat);
      const a = (i / 4) * Math.PI * 2 + Math.PI / 4;
      b.position.set(Math.cos(a) * 0.24, 0.83, Math.sin(a) * 0.24);
      b.castShadow = true;
      group.add(b);
    }
  } else if (type === 'q') {
    for (let i = 0; i < 6; i++) {
      const spike = new THREE.Mesh(new THREE.ConeGeometry(0.045, 0.16, 6), mat);
      const a = (i / 6) * Math.PI * 2;
      spike.position.set(Math.cos(a) * 0.16, 1.08, Math.sin(a) * 0.16);
      spike.castShadow = true;
      group.add(spike);
    }
  } else if (type === 'k') {
    const v = new THREE.Mesh(new THREE.BoxGeometry(0.045, 0.22, 0.045), mat);
    v.position.set(0, 1.16, 0);
    const h = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.045, 0.045), mat);
    h.position.set(0, 1.13, 0);
    v.castShadow = h.castShadow = true;
    group.add(v, h);
  } else if (type === 'b') {
    const slit = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.10, 0.16), mat);
    slit.position.set(0, 0.90, 0);
    slit.rotation.z = 0.5;
    slit.castShadow = true;
    group.add(slit);
  }

  group.scale.setScalar(scale);
  group.traverse(o => { if (o.isMesh) o.receiveShadow = true; });
  group.userData.type = type;
  group.userData.color = color;
  return group;
}

// -------------------------------------------------------------------------
// доска + трава
// -------------------------------------------------------------------------
function grassTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 256;
  const ctx = c.getContext('2d');
  // мягкий базовый градиент вместо плоского цвета — меньше похоже на текстуру-плитку
  const base = ctx.createLinearGradient(0, 0, 256, 256);
  base.addColorStop(0, '#4c7a3c');
  base.addColorStop(1, '#446f38');
  ctx.fillStyle = base;
  ctx.fillRect(0, 0, 256, 256);
  // редкие мягкие пятна вместо резкого шума — читается как трава, не как шум
  for (let i = 0; i < 900; i++) {
    const x = Math.random() * 256, y = Math.random() * 256;
    const r = 3 + Math.random() * 5;
    const g = 100 + Math.random() * 40;
    ctx.fillStyle = `rgba(${50 + Math.random() * 25},${g},${40 + Math.random() * 20},0.35)`;
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
  }
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(10, 10);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function buildSky() {
  const sky = new THREE.Mesh(
    new THREE.SphereGeometry(45, 24, 16),
    new THREE.ShaderMaterial({
      side: THREE.BackSide,
      fog: false,
      uniforms: {
        top: { value: new THREE.Color(0x8fc7e8) },
        bottom: { value: new THREE.Color(0xdcefd6) },
      },
      vertexShader: /* glsl */`
        varying vec3 vPos;
        void main() {
          vPos = position;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }`,
      fragmentShader: /* glsl */`
        uniform vec3 top; uniform vec3 bottom;
        varying vec3 vPos;
        void main() {
          float h = clamp(normalize(vPos).y * 0.8 + 0.35, 0.0, 1.0);
          gl_FragColor = vec4(mix(bottom, top, h), 1.0);
        }`,
    }),
  );
  return sky;
}

// одна травинка: тонкий изогнутый клин из 3 треугольников (без текстуры)
function bladeGeometry() {
  const w = 0.05, h = 0.32, bend = 0.10;
  const pos = new Float32Array([
    -w / 2, 0, 0, w / 2, 0, 0, w / 4, h * 0.55, bend * 0.5,
    -w / 2, 0, 0, w / 4, h * 0.55, bend * 0.5, -w / 4, h * 0.55, bend * 0.5,
    -w / 4, h * 0.55, bend * 0.5, w / 4, h * 0.55, bend * 0.5, 0, h, bend,
  ]);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.computeVertexNormals();
  return g;
}

// поле травинок вокруг доски (InstancedMesh — дёшево даже в больших количествах).
// exclude — полуширина полосы вдоль доски (по z), где травинки не сажаем:
// доска и обе подставки для съеденных фигур (до TRAY_OUTER_X по x).
function buildGrassField(count, half, exclude) {
  const mesh = new THREE.InstancedMesh(
    bladeGeometry(),
    new THREE.MeshStandardMaterial({ vertexColors: true, side: THREE.DoubleSide, roughness: 1 }),
    count,
  );
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3);
  const m = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const s = new THREE.Vector3();
  const col = new THREE.Color();
  let i = 0;
  let guard = 0;
  while (i < count && guard < count * 20) {
    guard++;
    const x = (Math.random() * 2 - 1) * half;
    const z = (Math.random() * 2 - 1) * half;
    if (Math.abs(z) < exclude && Math.abs(x) < TRAY_OUTER_X) continue;  // доска + подставки
    q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.random() * Math.PI * 2);
    const scale = 0.7 + Math.random() * 0.8;
    s.set(scale, scale * (0.8 + Math.random() * 0.5), scale);
    m.compose(new THREE.Vector3(x, -0.375, z), q, s);
    mesh.setMatrixAt(i, m);
    const tone = 0.55 + Math.random() * 0.4;
    col.setRGB(0.20 * tone, (0.42 + Math.random() * 0.12) * tone, 0.16 * tone);
    mesh.instanceColor.setXYZ(i, col.r, col.g, col.b);
    i++;
  }
  mesh.count = i;
  mesh.castShadow = false;
  mesh.receiveShadow = false;
  return mesh;
}

function checkerTexture() {
  const c = document.createElement('canvas');
  const cell = 64;
  c.width = c.height = cell * 8;
  const ctx = c.getContext('2d');
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      ctx.fillStyle = (f + r) % 2 === 0 ? '#8a6a4a' : '#e8e0cc';
      ctx.fillRect(f * cell, (7 - r) * cell, cell, cell);
    }
  }
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

const TRAY_TOP = 0.01;

// низкая деревянная подставка сбоку от доски: сюда ставятся съеденные фигуры
function buildTray(sign) {
  const g = new THREE.Group();
  const width = TRAY_OUTER_X - TRAY_INNER_X;
  const cx = sign * (TRAY_INNER_X + TRAY_OUTER_X) / 2;
  const base = new THREE.Mesh(
    new THREE.BoxGeometry(width, 0.40, 8.4),
    new THREE.MeshStandardMaterial({ color: COLORS.frame, roughness: 0.8 }),
  );
  base.position.set(cx, -0.195, 0);  // верх на 0.005, низ уходит под траву
  const top = new THREE.Mesh(
    new THREE.BoxGeometry(width - 0.2, 0.02, 8.2),
    new THREE.MeshStandardMaterial({ color: 0x7c5a3a, roughness: 0.7 }),
  );
  top.position.set(cx, TRAY_TOP - 0.01, 0);
  for (const m of [base, top]) { m.castShadow = true; m.receiveShadow = true; g.add(m); }
  return g;
}

function buildBoard() {
  const group = new THREE.Group();

  const frameMat = new THREE.MeshStandardMaterial({ color: COLORS.frame, roughness: 0.8 });
  const frame = new THREE.Mesh(new THREE.BoxGeometry(9.3, 0.35, 9.3), frameMat);
  frame.position.y = -0.20;
  frame.receiveShadow = true;
  frame.castShadow = true;
  group.add(frame);

  const topMat = new THREE.MeshStandardMaterial({ map: checkerTexture(), roughness: 0.6 });
  const sideMat = new THREE.MeshStandardMaterial({ color: COLORS.frame, roughness: 0.8 });
  const tiles = new THREE.Mesh(
    new THREE.BoxGeometry(8, 0.16, 8),
    [sideMat, sideMat, topMat, sideMat, sideMat, sideMat],
  );
  tiles.position.y = -0.08;
  tiles.receiveShadow = true;
  tiles.castShadow = true;
  group.add(tiles);

  // невидимые плоскости-мишени для клика, по одной на клетку
  const cellGeom = new THREE.PlaneGeometry(SQUARE, SQUARE);
  const cells = {};
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const sq = FILES[f] + (r + 1);
      const m = new THREE.Mesh(cellGeom, new THREE.MeshBasicMaterial({ visible: false }));
      m.rotation.x = -Math.PI / 2;
      const [x, z] = squareToXZ(sq);
      m.position.set(x, 0.001, z);
      m.userData.square = sq;
      group.add(m);
      cells[sq] = m;
    }
  }

  // подсветка клеток (последний ход / выбор / цели / шах) — тонкие плашки чуть выше доски
  const highlightMat = (color, opacity) => new THREE.MeshBasicMaterial({ color, transparent: true, opacity, depthWrite: false });
  const highlights = { last: [], sel: null, targets: [], check: null };
  function clearHighlights() {
    for (const k of ['last', 'targets']) { for (const m of highlights[k]) group.remove(m); highlights[k] = []; }
    if (highlights.sel) { group.remove(highlights.sel); highlights.sel = null; }
    if (highlights.check) { group.remove(highlights.check); highlights.check = null; }
  }
  function addHighlight(sq, mat, list) {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(SQUARE * 0.94, SQUARE * 0.94), mat);
    m.rotation.x = -Math.PI / 2;
    const [x, z] = squareToXZ(sq);
    m.position.set(x, 0.011, z);
    group.add(m);
    if (list) list.push(m); else return m;
  }

  return { group, cells, clearHighlights, addHighlight, highlights, highlightMat };
}

// -------------------------------------------------------------------------
// сцена
// -------------------------------------------------------------------------
export function createChessScene(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0xcfe6d8, 16, 34);
  scene.add(buildSky());

  const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
  camera.position.set(0, 7.5, 8.5);

  const hemi = new THREE.HemisphereLight(0xeaf4ff, 0x4a7a3a, 1.0);
  scene.add(hemi);
  const sun = new THREE.DirectionalLight(0xfff2da, 1.5);
  sun.position.set(6, 10, 4);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.left = -8; sun.shadow.camera.right = 8;
  sun.shadow.camera.top = 8; sun.shadow.camera.bottom = -8;
  sun.shadow.camera.near = 1; sun.shadow.camera.far = 30;
  sun.shadow.bias = -0.0015;
  scene.add(sun);
  const fill = new THREE.DirectionalLight(0xcfe0ff, 0.35);
  fill.position.set(-8, 6, -6);
  scene.add(fill);

  const grass = new THREE.Mesh(
    new THREE.PlaneGeometry(60, 60),
    new THREE.MeshStandardMaterial({ map: grassTexture(), roughness: 1 }),
  );
  grass.rotation.x = -Math.PI / 2;
  grass.position.y = -0.375;
  grass.receiveShadow = true;
  scene.add(grass);

  // травинки поверх плоскости — от квадратного «газона» до объёмного поля
  scene.add(buildGrassField(9000, 18, 4.9));

  const board = buildBoard();
  scene.add(board.group);

  const pieces = new THREE.Group();
  scene.add(pieces);
  const graveyard = new THREE.Group();
  scene.add(graveyard);
  for (const sign of [1, -1]) scene.add(buildTray(sign));

  const controls = new OrbitControls(camera, canvas);
  controls.target.set(0, 0, 0.5);
  controls.enablePan = false;
  controls.minPolarAngle = 0.3;
  controls.maxPolarAngle = 1.3;
  controls.minDistance = 6;
  controls.maxDistance = 16;
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(canvas);
  resize();

  // ---- фигуры на доске: карта square -> mesh group ----
  const onBoard = {};  // square -> group
  let graveW = 0, graveB = 0;

  function spawn(sq, ch) {
    const type = ch.toLowerCase();
    const color = ch === ch.toUpperCase() ? 'w' : 'b';
    const mesh = buildPieceMesh(type, color);
    const [x, z] = squareToXZ(sq);
    mesh.position.set(x, 0, z);
    pieces.add(mesh);
    onBoard[sq] = mesh;
    return mesh;
  }

  function removeFromBoard(sq) {
    const mesh = onBoard[sq];
    if (mesh) { pieces.remove(mesh); delete onBoard[sq]; }
    return mesh;
  }

  function sendToGraveyard(mesh) {
    const color = mesh.userData.color;
    const row = color === 'w' ? graveW++ : graveB++;
    const sign = color === 'w' ? 1 : -1;
    const x = sign * (TRAY_INNER_X + 0.55 + Math.floor(row / 8) * 0.9);
    const z = -3.5 + (row % 8) * 1.0;
    mesh.position.set(x, TRAY_TOP, z);
    mesh.scale.multiplyScalar(0.7);
    graveyard.add(mesh);
  }

  // новая партия: убрать сбитые фигуры прошлой партии, иначе они остаются
  // висеть по бокам доски
  function clearGraveyard() {
    for (const mesh of [...graveyard.children]) graveyard.remove(mesh);
    graveW = 0;
    graveB = 0;
  }

  function parseFen(fen) {
    const rows = fen.split(' ')[0].split('/');
    const grid = {};
    rows.forEach((row, ri) => {
      let f = 0;
      for (const ch of row) {
        if (/\d/.test(ch)) { f += +ch; continue; }
        grid[FILES[f] + (8 - ri)] = ch;
        f++;
      }
    });
    return grid;
  }

  // строит доску с нуля по FEN, без анимации
  function setFen(fen) {
    for (const sq of Object.keys(onBoard)) removeFromBoard(sq);
    const grid = parseFen(fen);
    for (const sq of Object.keys(grid)) spawn(sq, grid[sq]);
  }

  // сверка: досоздать/убрать меши так, чтобы доска точно совпала с FEN
  function reconcile(fen) {
    const grid = parseFen(fen);
    for (const sq of Object.keys(onBoard)) {
      if (!grid[sq]) sendToGraveyard(removeFromBoard(sq));
    }
    for (const sq of Object.keys(grid)) {
      const want = grid[sq];
      const have = onBoard[sq];
      const haveCh = have ? (have.userData.color === 'w' ? have.userData.type.toUpperCase() : have.userData.type) : null;
      if (haveCh === want) continue;
      if (have) sendToGraveyard(removeFromBoard(sq));
      spawn(sq, want);
    }
  }

  // мухa перенесла фигуру: обновить карту square->mesh (позицию уже поставила муха)
  function commitMove(from, to, mesh) {
    delete onBoard[from];
    onBoard[to] = mesh;
  }

  function highlightLast(uci) {
    board.clearHighlights();
    if (uci) {
      board.addHighlight(uci.slice(0, 2), board.highlightMat(COLORS.lastMove, 0.45), board.highlights.last);
      board.addHighlight(uci.slice(2, 4), board.highlightMat(COLORS.lastMove, 0.45), board.highlights.last);
    }
  }

  function setSelection(sq, targets) {
    if (board.highlights.sel) { board.group.remove(board.highlights.sel); board.highlights.sel = null; }
    for (const m of board.highlights.targets) board.group.remove(m);
    board.highlights.targets = [];
    if (sq) board.highlights.sel = board.addHighlight(sq, board.highlightMat(COLORS.selected, 0.55));
    for (const t of targets || []) board.addHighlight(t, board.highlightMat(COLORS.target, 0.5), board.highlights.targets);
  }

  function setCheck(sq) {
    if (board.highlights.check) { board.group.remove(board.highlights.check); board.highlights.check = null; }
    if (sq) board.highlights.check = board.addHighlight(sq, board.highlightMat(COLORS.check, 0.55));
  }

  // ---- клики ----
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  let downX = 0, downY = 0;

  function pick(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    pointer.x = ((clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const pieceHits = raycaster.intersectObjects(pieces.children, true);
    if (pieceHits.length) {
      let o = pieceHits[0].object;
      while (o.parent && o.parent !== pieces) o = o.parent;
      return { type: 'piece', square: xzToSquare(o.position.x, o.position.z) };
    }
    const cellHits = raycaster.intersectObjects(Object.values(board.cells));
    if (cellHits.length) return { type: 'square', square: cellHits[0].object.userData.square };
    return null;
  }

  let onSquareClick = null;
  canvas.addEventListener('pointerdown', e => { downX = e.clientX; downY = e.clientY; });
  canvas.addEventListener('pointerup', e => {
    const moved = Math.hypot(e.clientX - downX, e.clientY - downY);
    if (moved > 5) return;  // это было вращение, не клик
    const hit = pick(e.clientX, e.clientY);
    if (hit && onSquareClick) onSquareClick(hit.square, hit.type);
  });

  // камера со стороны игрока: своя сторона ближе, соперник (муха) — далеко
  function setViewSide(side) {
    const k = side === 'b' ? -1 : 1;
    camera.position.set(0, 7.5, 8.5 * k);
    controls.target.set(0, 0, 0.5 * k);
    controls.update();
  }

  const frameCallbacks = [];
  const clock = new THREE.Clock();
  function loop() {
    requestAnimationFrame(loop);
    const dt = Math.min(0.1, clock.getDelta());
    for (const cb of frameCallbacks) cb(dt);
    controls.update();
    renderer.render(scene, camera);
  }
  loop();

  return {
    scene, camera, pieces, onBoard,
    setFen, reconcile, highlightLast, setSelection, setCheck, commitMove, clearGraveyard, setViewSide,
    getMesh: sq => onBoard[sq],
    removeFromBoard, sendToGraveyard, spawn,
    squareToXZ,
    addFrameCallback: fn => frameCallbacks.push(fn),
    set onSquareClick(fn) { onSquareClick = fn; },
  };
}
