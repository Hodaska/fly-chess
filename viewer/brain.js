// Мозг мухи + вспышки на спайках. Используется страницей игры (play.html).
// Координаты — измеренные позиции сомы MaleCNS v1.0, активность — настоящий
// выход LIF-симуляции.
//
// Два вида сцены:
//   view: 'brain' — только мозг, как раньше (геометрия с сервера, /atlas);
//   view: 'fly'   — вся муха: стеклянное тело flybody + мозг + нервный тяж
//                   (статические файлы из /static/assets, собирает
//                   scripts/build_body_assets.py). Тяж только анатомия:
//                   симулируется мозг, тяж на экране не вспыхивает.
import * as THREE from 'three';

const GROUP_HEX = [0x81b5c8, 0xcfcac0, 0xdfb672, 0xaf9bc3, 0x6c7777];

const pointVert = /* glsl */`
  uniform float uBoost;
  uniform float uSize;
  attribute float intensity;
  attribute vec3 baseColor;
  attribute float baseAlpha;
  varying float vI;
  varying vec3 vC;
  varying float vA;
  void main() {
    vI = intensity; vC = baseColor; vA = baseAlpha;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = uSize * (0.85 + uBoost * intensity) * (9.0 / -mv.z);
    gl_Position = projectionMatrix * mv;
  }`;

const pointFrag = /* glsl */`
  uniform float uPeak;
  uniform float uBase;
  varying float vI;
  varying vec3 vC;
  varying float vA;
  void main() {
    vec2 d = gl_PointCoord - vec2(0.5);
    float r = dot(d, d);
    if (r > 0.25) discard;
    float soft = 1.0 - smoothstep(0.0, 0.25, r);
    vec3 hot = mix(vec3(1.0, 0.62, 0.22), vec3(1.0, 0.97, 0.90), vI * vI);
    vec3 col = mix(vC, hot, smoothstep(0.0, 0.35, vI));
    // база почти прозрачна: >100k точек в аддитивном режиме иначе выбеливают кадр
    // нервный тяж (яркий постоянный фон, vA > 0.3) не тускнеет — он редкий и не сливается
    float a = (vA < 0.3 ? vA * uBase : vA) + uPeak * vI * vI;
    gl_FragColor = vec4(col, soft * a);
  }`;

// Стеклянное тело: почти прозрачное «в лоб», светится по краям (эффект Френеля).
const bodyVert = /* glsl */`
  attribute float mat;
  varying vec3 vN;
  varying vec3 vV;
  varying float vMat;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vN = normalize(normalMatrix * normal);
    vV = normalize(-mv.xyz);
    vMat = mat;
    gl_Position = projectionMatrix * mv;
  }`;

const bodyFrag = /* glsl */`
  varying vec3 vN;
  varying vec3 vV;
  varying float vMat;
  void main() {
    float rim = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), 2.2);
    vec3 col; float base; float edge;
    if (vMat < 0.5)      { col = vec3(0.62, 0.70, 0.78); base = 0.012; edge = 0.30; } // тело
    else if (vMat < 1.5) { col = vec3(0.90, 0.30, 0.24); base = 0.020; edge = 0.38; } // глаза
    else                 { col = vec3(0.55, 0.72, 0.90); base = 0.006; edge = 0.16; } // крылья
    gl_FragColor = vec4(col, base + edge * rim);
  }`;

async function loadBrainOnly() {
  const buf = await fetch('/atlas').then(r => r.arrayBuffer());
  const N = new Uint32Array(buf, 0, 1)[0];
  return {
    N,
    pos: new Float32Array(buf, 4, N * 3),
    grp: new Uint8Array(buf, 4 + N * 12, N),
    simMap: null,
  };
}

async function loadFly() {
  const [cns, body] = await Promise.all([
    fetch('/static/assets/cns.bin').then(r => r.arrayBuffer()),
    fetch('/static/assets/flybody.bin').then(r => r.arrayBuffer()),
  ]);
  const N = new Uint32Array(cns, 0, 1)[0];
  const afterGroups = 4 + N * 13;
  const mapOff = afterGroups + ((4 - (afterGroups % 4)) % 4);
  const nSim = new Uint32Array(cns, mapOff, 1)[0];

  const nv = new Uint32Array(body, 0, 1)[0];
  const nt = new Uint32Array(body, 4, 1)[0];
  const afterMats = 8 + nv * 13;
  const idxOff = afterMats + ((4 - (afterMats % 4)) % 4);

  return {
    N,
    pos: new Float32Array(cns, 4, N * 3),
    grp: new Uint8Array(cns, 4 + N * 12, N),
    simMap: new Uint32Array(cns, mapOff + 4, nSim),
    body: {
      pos: new Float32Array(body, 8, nv * 3),
      mat: new Uint8Array(body, 8 + nv * 12, nv),
      idx: new Uint32Array(body, idxOff, nt * 3),
    },
  };
}

export async function createBrain(canvas, { view = 'brain' } = {}) {
  const fly = view === 'fly';
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x07090c, fly ? 0.02 : 0.045);
  const camera = new THREE.PerspectiveCamera(fly ? 40 : 50, 1, 0.01, 100);
  const rig = new THREE.Group();
  scene.add(rig);
  // в модели тела ось «вверх» — z (MuJoCo), в Three.js — y
  const world = new THREE.Group();
  if (fly) world.rotation.x = -Math.PI / 2;
  rig.add(world);
  // центр вращения и приближения — нервная система, а не середина всей мухи
  // (иначе при зуме голова с мозгом уезжает за край кадра)
  const inner = new THREE.Group();
  world.add(inner);

  let zoom = 1;
  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    const dist = fly ? 6.0 : 5.2;
    // в узком окне отъезжаем, чтобы всё влезало по ширине
    camera.position.set(0, 0, zoom * dist * Math.max(1, 1.15 / camera.aspect));
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(canvas);
  resize();
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    zoom = Math.max(0.3, Math.min(2.5, zoom * Math.exp(e.deltaY * 0.001)));
    resize();
  }, { passive: false });

  // Покачивание у удобного ракурса вместо полного оборота. Мозг отдельно —
  // фронтально (сбоку он бесформенное пятно); муха — в три четверти спереди.
  let drag = false, px = 0, py = 0, t0 = performance.now();
  let userYaw = fly ? -0.75 : 0, userPitch = fly ? 0.35 : -0.1;
  canvas.addEventListener('pointerdown', e => { drag = true; px = e.clientX; py = e.clientY; });
  addEventListener('pointerup', () => { drag = false; });
  addEventListener('pointermove', e => {
    if (!drag) return;
    userYaw += (e.clientX - px) * 0.006;
    userPitch = Math.max(-1.3, Math.min(1.3, userPitch + (e.clientY - py) * 0.006));
    px = e.clientX; py = e.clientY;
  });

  const [meta, data] = await Promise.all([
    fetch('/meta').then(r => r.json()),
    fly ? loadFly() : loadBrainOnly(),
  ]);
  const { N, pos, grp, simMap } = data;
  if (fly) {
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < N; i++) { cx += pos[i * 3]; cy += pos[i * 3 + 1]; cz += pos[i * 3 + 2]; }
    inner.position.set(-cx / N, -cy / N, -cz / N);
  }

  if (fly) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(data.body.pos, 3));
    g.setAttribute('mat', new THREE.BufferAttribute(Float32Array.from(data.body.mat), 1));
    g.setIndex(new THREE.BufferAttribute(data.body.idx, 1));
    g.computeVertexNormals();
    const bodyMesh = new THREE.Mesh(g, new THREE.ShaderMaterial({
      vertexShader: bodyVert, fragmentShader: bodyFrag,
      transparent: true, depthWrite: false, side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
    }));
    bodyMesh.renderOrder = 0;  // тело раньше точек — иначе подерутся за прозрачность
    inner.add(bodyMesh);
  }

  const colors = new Float32Array(N * 3);
  const c = new THREE.Color();
  for (let i = 0; i < N; i++) {
    c.setHex(GROUP_HEX[grp[i]]);
    colors[i * 3] = c.r; colors[i * 3 + 1] = c.g; colors[i * 3 + 2] = c.b;
  }
  const intensity = new Float32Array(N);
  // Базовая (фоновая) яркость точки. Нервный тяж не симулируется и никогда не
  // вспыхивает — чтобы его было видно, он светится постоянно своим фиолетовым
  // цветом, а не оранжевым «активным», чтобы не притворяться активностью.
  const VNC = 3;
  const baseAlpha = new Float32Array(N);
  for (let i = 0; i < N; i++) baseAlpha[i] = fly ? (grp[i] === VNC ? 0.55 : 0.05) : 0.06;

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geom.setAttribute('baseColor', new THREE.BufferAttribute(colors, 3));
  geom.setAttribute('baseAlpha', new THREE.BufferAttribute(baseAlpha, 1));
  geom.setAttribute('intensity', new THREE.BufferAttribute(intensity, 1));

  // Два режима показа одной и той же активности:
  //  show — эффектный, как в рилсах: вспышка тянется несколько шагов и
  //         раздувается, поэтому кажется, что горит весь мозг;
  //  real — честный: виден только текущий шаг (~1.5% нейронов), точки почти
  //         не раздуваются. Так активность выглядит на самом деле.
  const MODES = {
    show: { boost: 5.8, peak: 0.92, fade: 0.004, clear: false },
    real: { boost: 1.4, peak: 0.94, fade: 1e-4, clear: true },
  };
  let mode = MODES.show;
  const material = new THREE.ShaderMaterial({
    uniforms: {
      uBoost: { value: mode.boost },
      uPeak: { value: mode.peak },
      uBase: { value: 1 },
      // внутри тела мозг в ~4 раза мельче на экране — точки плотнее, делаем
      // их меньше и тусклее, иначе мозг сливается в сплошное пятно
      uSize: { value: fly ? 0.42 : 1.0 },
    },
    vertexShader: pointVert, fragmentShader: pointFrag,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  });
  const points = new THREE.Points(geom, material);
  points.renderOrder = 1;
  inner.add(points);

  function setMode(name) {
    mode = MODES[name];
    material.uniforms.uBoost.value = mode.boost;
    material.uniforms.uPeak.value = mode.peak;
  }

  let last = performance.now();
  let paused = false;  // скрытый вид не рисуем (страница 3D держит два вида на двух холстах)
  (function loop() {
    requestAnimationFrame(loop);
    const now = performance.now(), dt = (now - last) / 1000; last = now;
    if (paused) return;
    // Вся муха на маленьком экранчике: мозг занимает десятки пикселей, и фон из
    // 126 тысяч точек сливается в белое пятно, на котором вспышек не видно.
    // Фон тускнеет вместе с площадью мозга на экране (размер холста, приближение).
    if (fly) material.uniforms.uBase.value = Math.max(0.04, Math.min(1, Math.pow(canvas.clientHeight / 700 / zoom, 2)));
    const t = (now - t0) / 1000;
    rig.rotation.set(userPitch, userYaw + (fly ? 0.35 : 0.5) * Math.sin(t * 0.22), 0);
    const decay = Math.pow(mode.fade, dt);
    for (let i = 0; i < N; i++) intensity[i] *= decay;
    geom.attributes.intensity.needsUpdate = true;
    renderer.render(scene, camera);
  })();

  // индексы из записи — это нейроны симуляции; в режиме мухи на экране их
  // больше (плюс тяж), поэтому переводим через таблицу соответствия
  function flash(idx) {
    if (simMap) for (let i = 0; i < idx.length; i++) intensity[simMap[idx[i]]] = 1.0;
    else for (let i = 0; i < idx.length; i++) intensity[idx[i]] = 1.0;
  }

  // Бинарь с сервера: uint32 число шагов, uint32 длины по шагам, индексы подряд.
  // every: ждать кадр только на каждом every-м шаге (ускорение без потери
  // шагов — промежуточные тоже вспыхивают, просто не успевают отрисоваться)
  async function playTrace(buffer, onStep, frameMs = 32, every = 1) {
    if (!buffer || buffer.byteLength < 4) return;  // записи нет — нечего играть
    const steps = new Uint32Array(buffer, 0, 1)[0];
    const lens = new Uint32Array(buffer, 4, steps);
    let off = 4 + steps * 4;
    for (let s = 0; s < steps; s++) {
      const idx = new Uint32Array(buffer, off, lens[s]);
      off += lens[s] * 4;
      if (mode.clear) intensity.fill(0);  // только текущий шаг, без накопления
      flash(idx);
      if ((s + 1) % every === 0 || s + 1 === steps) {
        onStep?.(s + 1, steps, lens[s]);
        await new Promise(r => setTimeout(r, frameMs));
      }
    }
  }

  return { N, meta, flash, playTrace, setMode, fly, setPaused: v => { paused = v; } };
}
