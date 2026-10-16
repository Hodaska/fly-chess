// Муха для 3D-сцены: непрозрачное тело + отдельно анимируемые крылья,
// парение и полёт с переносом фигуры. Модель — flybody (см. NOTICE.md),
// геометрия — viewer/assets/fly_parts.bin (собирает scripts/build_body_assets.py).
import * as THREE from 'three';

const BODY_COLOR = new THREE.Color(0x6e4a2a);
const EYE_COLOR = new THREE.Color(0xc21f1f);
const TARGET_LENGTH = 1.5;  // клеток
const HOME_Z = 4.7;    // дом чуть за краем доски (край рамки ~4.65)
const HOME_Y = 1.7;

export function tween(ms, fn, ease = easeInOutCubic) {
  return new Promise(resolve => {
    const t0 = performance.now();
    (function step() {
      const t = Math.min(1, (performance.now() - t0) / ms);
      fn(ease(t), t);
      if (t < 1) requestAnimationFrame(step); else resolve();
    })();
  });
}
export function easeInOutCubic(t) { return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; }

async function loadParts() {
  const buf = await fetch('/static/assets/fly_parts.bin').then(r => r.arrayBuffer());
  const nBodyVerts = new Uint32Array(buf, 0, 1)[0];
  const nBodyTris = new Uint32Array(buf, 4, 1)[0];
  let off = 8;
  const bodyPos = new Float32Array(buf, off, nBodyVerts * 3); off += nBodyVerts * 12;
  const bodyMat = new Uint8Array(buf, off, nBodyVerts); off += nBodyVerts;
  off += (-off) % 4 === 0 ? 0 : 4 - (off % 4);
  const bodyIdx = new Uint32Array(buf, off, nBodyTris * 3); off += nBodyTris * 12;

  function readWing() {
    const pivot = new Float32Array(buf, off, 3); off += 12;
    const nV = new Uint32Array(buf, off, 1)[0]; off += 4;
    const nT = new Uint32Array(buf, off, 1)[0]; off += 4;
    const pos = new Float32Array(buf, off, nV * 3); off += nV * 12;
    const idx = new Uint32Array(buf, off, nT * 3); off += nT * 12;
    return { pivot, pos, idx };
  }
  const wingLeft = readWing();
  const wingRight = readWing();
  return { bodyPos, bodyMat, bodyIdx, wingLeft, wingRight };
}

function buildBodyMesh(bodyPos, bodyMat, bodyIdx) {
  const n = bodyPos.length / 3;
  const colors = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const c = bodyMat[i] > 0 ? EYE_COLOR : BODY_COLOR;
    colors[i * 3] = c.r; colors[i * 3 + 1] = c.g; colors[i * 3 + 2] = c.b;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(bodyPos, 3));
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  g.setIndex(new THREE.BufferAttribute(bodyIdx, 1));
  g.computeVertexNormals();
  const mesh = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.65 }));
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

function buildWingMesh(pos, idx) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  g.computeVertexNormals();
  const mesh = new THREE.Mesh(g, new THREE.MeshStandardMaterial({
    color: 0xcfe0ee, transparent: true, opacity: 0.3, depthWrite: false, side: THREE.DoubleSide, roughness: 0.3,
  }));
  mesh.castShadow = false;
  return mesh;
}

export async function createFly(scene) {
  const parts = await loadParts();

  const root = new THREE.Group();      // мировая позиция мухи (полёт, парение)
  const orient = new THREE.Group();    // перевод осей MuJoCo (z-вверх) -> Three (y-вверх)
  orient.rotation.x = -Math.PI / 2;
  root.add(orient);

  const bodyMesh = buildBodyMesh(parts.bodyPos, parts.bodyMat, parts.bodyIdx);
  orient.add(bodyMesh);

  const wingPivotL = new THREE.Group();
  wingPivotL.position.set(...parts.wingLeft.pivot);
  wingPivotL.add(buildWingMesh(parts.wingLeft.pos, parts.wingLeft.idx));
  orient.add(wingPivotL);

  const wingPivotR = new THREE.Group();
  wingPivotR.position.set(...parts.wingRight.pivot);
  wingPivotR.add(buildWingMesh(parts.wingRight.pos, parts.wingRight.idx));
  orient.add(wingPivotR);

  // масштаб: длина тела (без крыльев) -> TARGET_LENGTH клеток
  const box = new THREE.Box3().setFromObject(bodyMesh);
  const size = box.getSize(new THREE.Vector3());
  const length = Math.max(size.x, size.z);
  const scale = TARGET_LENGTH / length;
  orient.scale.setScalar(scale);

  scene.add(root);

  let carried = null;   // фигура сейчас в лапках
  let flapPhase = 0;
  let flapSpeed = 14;    // рад/с базово, ускоряется в полёте
  let bobPhase = Math.random() * 10;
  let resting = true;    // покачивание на месте включено только вне полёта
  let baseY = 2.2;

  function update(dt) {
    flapPhase += dt * flapSpeed;
    const a = Math.sin(flapPhase) * THREE.MathUtils.degToRad(40);
    wingPivotL.rotation.x = a;
    wingPivotR.rotation.x = -a;
    bobPhase += dt;
    if (resting) root.position.y = baseY + Math.sin(bobPhase * 1.6) * 0.06;
  }

  function setHome(side) {
    const z = side === 'w' ? HOME_Z : -HOME_Z;
    baseY = HOME_Y;
    root.position.set(0, baseY, z);
    root.rotation.y = side === 'w' ? 0 : Math.PI;
    resting = true;
  }

  async function flyTo(x, z, y, ms, faceMotion = true) {
    resting = false;
    const from = root.position.clone();
    const to = new THREE.Vector3(x, y, z);
    const mid = from.clone().add(to).multiplyScalar(0.5);
    mid.y = Math.max(from.y, to.y) + Math.max(1.2, from.distanceTo(to) * 0.15);
    if (faceMotion) {
      const dir = Math.atan2(to.x - from.x, to.z - from.z);
      root.rotation.y = dir;
    }
    flapSpeed = 26;
    await tween(ms, t => {
      const a = from.clone().lerp(mid, t);
      const b = mid.clone().lerp(to, t);
      root.position.copy(a.lerp(b, t));
      const bank = Math.sin(t * Math.PI) * 0.25;
      root.rotation.z = bank;
    });
    root.rotation.z = 0;
    flapSpeed = 14;
    baseY = y;
    resting = true;
  }

  async function moveY(targetY, ms) {
    resting = false;
    const y0 = root.position.y;
    await tween(ms, t => { root.position.y = THREE.MathUtils.lerp(y0, targetY, t); });
    baseY = targetY;
    resting = true;
  }

  function attachPiece(mesh) {
    if (!mesh) return;
    root.attach(mesh);      // сохраняет мировую трансформацию, дальше едет с мухой
    carried = mesh;
  }

  function detachPiece(parentGroup, worldX, worldZ) {
    if (!carried) return null;
    const mesh = carried;
    parentGroup.attach(mesh);
    mesh.position.set(worldX, 0, worldZ);
    mesh.rotation.set(0, mesh.rotation.y, 0);
    mesh.scale.setScalar(1);
    carried = null;
    return mesh;
  }

  const HOVER_Y = HOME_Y, PICKUP_Y = 0.85, CARRY_Y = 1.8;

  // полная хореография хода: подлёт -> вниз -> взять -> вверх -> нести ->
  // вниз -> отпустить -> вверх домой. pieceGroup — THREE.Group с фигурами
  // сцены (chess3d.pieces), чтобы фигуру можно было туда вернуть.
  async function carryMove(mesh, fromXZ, toXZ, pieceGroup) {
    const [fx, fz] = fromXZ, [tx, tz] = toXZ;
    await flyTo(fx, fz, HOVER_Y, 900);
    await moveY(PICKUP_Y, 400);
    attachPiece(mesh);
    await moveY(CARRY_Y, 350);
    root.rotation.y = Math.atan2(tx - fx, tz - fz);
    await flyTo(tx, tz, CARRY_Y, 1000, false);
    await moveY(PICKUP_Y, 400);
    detachPiece(pieceGroup, tx, tz);
    await moveY(HOVER_Y, 350);
  }

  async function goHome(side, ms = 900) {
    const z = side === 'w' ? HOME_Z : -HOME_Z;
    await flyTo(0, z, HOVER_Y, ms);
  }

  return {
    root, update, setHome, flyTo, goHome, carryMove,
    get carrying() { return carried; },
  };
}
