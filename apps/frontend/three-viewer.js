/* Interactive 3D room, rendered from the pipeline's own OBJ/MTL artifacts.
 *
 * Two things make this a review tool rather than a model viewer. A room is a
 * closed box, so drawn honestly from outside it is an opaque solid that shows
 * nothing; the cutaway below hides whichever surfaces stand between the eye and
 * the interior. And every mesh keeps its canonical surface ID, so a click here
 * selects the same wall the plan and the inspector are talking about.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { MTLLoader } from 'three/addons/loaders/MTLLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

const artifactUrl = (scanId, name) =>
  `/api/scans/${encodeURIComponent(scanId)}/artifacts/${encodeURIComponent(name)}`;

const SELECTED_COLOR = 0x0a6cff;
const reducedMotion = () =>
  window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;

export class SpatialThreeViewer {
  constructor(host, { scanId, entityMap, onSelect, onContextLost }) {
    this.host = host;
    this.scanId = scanId;
    this.surfaceIds = new Set(Object.keys(entityMap.surfaces || {}));
    this.onSelect = onSelect;
    this.onContextLost = onContextLost;
    this.surfaces = new Map();       // surfaceId -> { meshes, center, outward }
    this.meshRecords = new Map();    // mesh -> { isArray, base, highlight }
    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.pointerDown = null;
    this.selected = null;
    this.cutaway = true;
    this.disposed = false;
    this.needsRender = true;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.01, 1000);
    this.camera.up.set(0, 1, 0);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;

    const canvas = this.renderer.domElement;
    canvas.className = 'three-canvas';
    // Focusable and described, so the view is reachable and explained without a mouse.
    canvas.tabIndex = 0;
    canvas.setAttribute('role', 'application');
    canvas.setAttribute('aria-label',
      'Interactive 3D room. Drag or use the arrow keys to orbit, hold Shift to pan, '
      + 'plus and minus to zoom, F to fit, Enter to select the surface in the centre, '
      + 'and Escape to clear the selection.');
    host.append(canvas);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = !reducedMotion();
    this.controls.dampingFactor = 0.08;
    this.controls.screenSpacePanning = true;
    this.controls.minDistance = 0.1;
    this.controls.maxDistance = 200;
    this.onControlsChange = () => { this.updateCutaway(); this.invalidate(); };
    this.controls.addEventListener('change', this.onControlsChange);

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x7d8791, 2.25));
    this.key = new THREE.DirectionalLight(0xffffff, 2.1);
    this.key.position.set(4, 8, 6);
    this.scene.add(this.key);

    // The clear colour is a theme token; read it on theme change, not every frame.
    this.themeQuery = window.matchMedia?.('(prefers-color-scheme: dark)');
    this.onThemeChange = () => { this.readBackground(); this.invalidate(); };
    this.themeQuery?.addEventListener?.('change', this.onThemeChange);
    this.readBackground();

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(host);

    this.onPointerDown = (event) => {
      this.pointerDown = { x: event.clientX, y: event.clientY };
    };
    this.onPointerUp = (event) => this.pick(event);
    this.onKeyDown = (event) => this.handleKey(event);
    this.onContextLoss = (event) => {
      event.preventDefault();
      cancelAnimationFrame(this.animationFrame);
      this.onContextLost?.();
    };
    canvas.addEventListener('pointerdown', this.onPointerDown);
    canvas.addEventListener('pointerup', this.onPointerUp);
    canvas.addEventListener('keydown', this.onKeyDown);
    canvas.addEventListener('webglcontextlost', this.onContextLoss);

    this.animate = this.animate.bind(this);
    this.animationFrame = requestAnimationFrame(this.animate);
    this.resize();
  }

  invalidate() { this.needsRender = true; }

  readBackground() {
    this.background = getComputedStyle(document.documentElement)
      .getPropertyValue('--bg').trim() || '#f6f6f7';
    this.renderer.setClearColor(this.background, 1);
  }

  async load() {
    const materials = await new MTLLoader().loadAsync(artifactUrl(this.scanId, 'room_model.mtl'));
    materials.preload();
    const model = await new OBJLoader().setMaterials(materials)
      .loadAsync(artifactUrl(this.scanId, 'room_model.obj'));
    if (this.disposed) return;

    model.updateMatrixWorld(true);
    const roomCenter = new THREE.Box3().setFromObject(model).getCenter(new THREE.Vector3());

    model.traverse((node) => {
      if (!node.isMesh) return;
      const surfaceId = this.findSurfaceId(node);
      if (!surfaceId) return;
      node.userData.surfaceId = surfaceId;

      const wasArray = Array.isArray(node.material);
      const source = wasArray ? node.material : [node.material];
      const base = source.map((material) => {
        const copy = material.clone();
        copy.side = THREE.DoubleSide;
        return copy;
      });
      // A fixed highlight, so selection reads the same over any base colour
      // instead of tinting toward whatever the material happened to be.
      const highlight = base.map((material) => {
        const copy = material.clone();
        copy.color?.set(SELECTED_COLOR);
        copy.emissive?.set(SELECTED_COLOR);
        copy.emissiveIntensity = 0.35;
        return copy;
      });
      node.material = wasArray ? base : base[0];
      this.meshRecords.set(node, { isArray: wasArray, base, highlight });

      const record = this.surfaces.get(surfaceId)
        || { meshes: [], center: new THREE.Vector3(), outward: new THREE.Vector3() };
      record.meshes.push(node);
      this.surfaces.set(surfaceId, record);
    });

    if (!this.surfaces.size) {
      throw new Error('The 3D artifact loaded, but no canonical surface IDs were recoverable.');
    }

    for (const record of this.surfaces.values()) {
      this.measureSurface(record, roomCenter);
    }

    this.model = model;
    this.roomCenter = roomCenter;
    this.scene.add(model);
    this.fitCamera(true);
  }

  /* Where a surface sits, and which way it faces away from the room. Both are
   * needed to decide whether it is standing in front of what you want to see. */
  measureSurface(record, roomCenter) {
    const box = new THREE.Box3();
    const normal = new THREE.Vector3();
    const scratch = new THREE.Vector3();
    for (const mesh of record.meshes) {
      box.expandByObject(mesh);
      const attribute = mesh.geometry.getAttribute('normal');
      if (!attribute) continue;
      const matrix = new THREE.Matrix3().getNormalMatrix(mesh.matrixWorld);
      for (let i = 0; i < attribute.count; i += 1) {
        scratch.fromBufferAttribute(attribute, i).applyMatrix3(matrix).normalize();
        normal.add(scratch);
      }
    }
    box.getCenter(record.center);
    if (normal.lengthSq() < 1e-8) {
      normal.copy(record.center).sub(roomCenter);
    }
    normal.normalize();
    // Orient it away from the room, whichever way the exporter wound the faces.
    if (normal.dot(scratch.copy(record.center).sub(roomCenter)) < 0) normal.negate();
    record.outward.copy(normal.lengthSq() < 1e-8 ? new THREE.Vector3(0, 1, 0) : normal);
  }

  /* Hide surfaces the eye is behind, so the room opens up as it is orbited.
   * The selected surface always stays drawn — it is the thing being examined. */
  updateCutaway() {
    if (!this.model) return;
    const toward = new THREE.Vector3();
    for (const [id, record] of this.surfaces) {
      const occluding = this.cutaway
        && toward.copy(this.camera.position).sub(record.center).dot(record.outward) > 0;
      const visible = id === this.selected || !occluding;
      for (const mesh of record.meshes) {
        if (mesh.visible !== visible) { mesh.visible = visible; this.invalidate(); }
      }
    }
  }

  setCutaway(enabled) {
    this.cutaway = enabled;
    this.updateCutaway();
    this.invalidate();
  }

  findSurfaceId(node) {
    let current = node;
    while (current) {
      if (this.surfaceIds.has(current.name)) return current.name;
      current = current.parent;
    }
    return null;
  }

  pick(event) {
    if (!this.model || !this.pointerDown) return;
    const moved = Math.hypot(event.clientX - this.pointerDown.x,
      event.clientY - this.pointerDown.y);
    this.pointerDown = null;
    if (moved > 5) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this.onSelect(this.surfaceAt(this.pointer));
  }

  surfaceAt(pointer) {
    this.raycaster.setFromCamera(pointer, this.camera);
    // Only visible meshes, so a cut-away wall cannot be picked through.
    const hit = this.raycaster.intersectObject(this.model, true)
      .find((item) => item.object.visible && item.object.userData.surfaceId);
    return hit?.object.userData.surfaceId || null;
  }

  handleKey(event) {
    const step = event.shiftKey ? 0 : 0.09;
    const keys = {
      ArrowLeft: () => this.orbit(-step, 0, event),
      ArrowRight: () => this.orbit(step, 0, event),
      ArrowUp: () => this.orbit(0, -step, event),
      ArrowDown: () => this.orbit(0, step, event),
      '+': () => this.dolly(1 / 1.15),
      '=': () => this.dolly(1 / 1.15),
      '-': () => this.dolly(1.15),
      f: () => this.fitCamera(false),
      F: () => this.fitCamera(true),
      Enter: () => this.onSelect(this.surfaceAt(new THREE.Vector2(0, 0))),
      Escape: () => this.onSelect(null),
    };
    const action = keys[event.key];
    if (!action) return;
    event.preventDefault();
    action();
  }

  orbit(deltaAzimuth, deltaPolar, event) {
    if (event?.shiftKey) {  // Shift turns the arrow keys into a pan.
      const offset = new THREE.Vector3().copy(this.camera.position).sub(this.controls.target);
      const distance = offset.length() * 0.06;
      const right = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 0);
      const up = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 1);
      const pan = right.multiplyScalar(-Math.sign(deltaAzimuth) * distance)
        .addScaledVector(up, Math.sign(deltaPolar) * distance);
      this.camera.position.add(pan);
      this.controls.target.add(pan);
    } else {
      const offset = new THREE.Vector3().copy(this.camera.position).sub(this.controls.target);
      const spherical = new THREE.Spherical().setFromVector3(offset);
      spherical.theta -= deltaAzimuth;
      spherical.phi = THREE.MathUtils.clamp(spherical.phi + deltaPolar, 0.05, Math.PI - 0.05);
      this.camera.position.copy(this.controls.target)
        .add(new THREE.Vector3().setFromSpherical(spherical));
    }
    this.controls.update();
    this.updateCutaway();
    this.invalidate();
  }

  dolly(factor) {
    const offset = new THREE.Vector3().copy(this.camera.position).sub(this.controls.target);
    const distance = THREE.MathUtils.clamp(offset.length() * factor,
      this.controls.minDistance, this.controls.maxDistance);
    this.camera.position.copy(this.controls.target)
      .add(offset.setLength(distance));
    this.controls.update();
    this.updateCutaway();
    this.invalidate();
  }

  setSelection(surfaceId) {
    this.selected = surfaceId;
    for (const [id, record] of this.surfaces) {
      for (const mesh of record.meshes) {
        const stored = this.meshRecords.get(mesh);
        if (!stored) continue;
        const set = id === surfaceId ? stored.highlight : stored.base;
        mesh.material = stored.isArray ? set : set[0];
      }
    }
    this.updateCutaway();
    this.invalidate();
  }

  fitCamera(resetOrientation = false) {
    if (!this.model) return;
    const box = new THREE.Box3().setFromObject(this.model);
    const sphere = box.getBoundingSphere(new THREE.Sphere());
    if (!Number.isFinite(sphere.radius) || sphere.radius <= 0) return;
    const halfFov = THREE.MathUtils.degToRad(this.camera.fov / 2);
    const aspect = Math.max(this.camera.aspect, 0.01);
    const verticalDistance = sphere.radius / Math.sin(halfFov);
    const horizontalFov = Math.atan(Math.tan(halfFov) * aspect);
    const distance = Math.max(verticalDistance, sphere.radius / Math.sin(horizontalFov)) * 1.12;
    let direction = new THREE.Vector3(1, 0.72, 1);
    if (!resetOrientation) {
      direction = this.camera.position.clone().sub(this.controls.target);
      if (direction.lengthSq() < 0.001) direction.set(1, 0.72, 1);
    }
    direction.normalize();
    this.controls.target.copy(sphere.center);
    this.camera.position.copy(sphere.center).addScaledVector(direction, distance);
    this.camera.near = Math.max(distance / 1000, 0.01);
    this.camera.far = Math.max(distance * 20, 100);
    this.camera.updateProjectionMatrix();
    this.controls.minDistance = Math.max(sphere.radius * 0.08, 0.05);
    this.controls.maxDistance = Math.max(sphere.radius * 20, 20);
    this.controls.update();
    this.updateCutaway();
    this.invalidate();
  }

  resize() {
    const width = this.host.clientWidth;
    const height = this.host.clientHeight;
    if (!width || !height) return;
    const aspect = width / height;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = aspect;
    this.camera.updateProjectionMatrix();
    // Rotating a phone or dragging a pane changes the aspect enough to push the
    // room outside the frustum, so reframe it while keeping the angle the user
    // chose. Small changes leave their zoom alone.
    const changed = !this.lastAspect
      || Math.abs(aspect - this.lastAspect) / this.lastAspect > 0.02;
    this.lastAspect = aspect;
    if (changed && this.model) this.fitCamera(false);
    this.invalidate();
  }

  animate() {
    if (this.disposed) return;
    this.animationFrame = requestAnimationFrame(this.animate);
    // update() reports whether damping moved the camera; combined with the dirty
    // flag it keeps an idle view from redrawing sixty times a second.
    if (this.controls.update()) this.needsRender = true;
    if (!this.needsRender) return;
    this.needsRender = false;
    this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    this.disposed = true;
    cancelAnimationFrame(this.animationFrame);
    this.resizeObserver.disconnect();
    this.themeQuery?.removeEventListener?.('change', this.onThemeChange);
    this.controls.removeEventListener('change', this.onControlsChange);
    const canvas = this.renderer.domElement;
    canvas.removeEventListener('pointerdown', this.onPointerDown);
    canvas.removeEventListener('pointerup', this.onPointerUp);
    canvas.removeEventListener('keydown', this.onKeyDown);
    canvas.removeEventListener('webglcontextlost', this.onContextLoss);
    this.controls.dispose();
    this.model?.traverse((node) => node.geometry?.dispose());
    for (const stored of this.meshRecords.values()) {
      stored.base.forEach((material) => material.dispose());
      stored.highlight.forEach((material) => material.dispose());
    }
    this.renderer.dispose();
    canvas.remove();
  }
}
