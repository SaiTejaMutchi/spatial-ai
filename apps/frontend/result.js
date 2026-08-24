/* Spatial Result — the primary product screen.
 *
 * The room is the object. Everything else is apparatus. The one distinction the
 * interface must make on its own, without anyone explaining it:
 *
 *   AI interprets visual evidence.  Geometry measures physical quantities.
 *   Humans review consequential findings.
 *
 * So every metric value carries its producer, and every AI finding carries the
 * surface and evidence it refers to.
 */

import {
  announce, api, badge, classificationWord, conditionWord, disclosure, evidenceQualityWord, h, kv, metres, metric, mount, observationWord, remount, squareMetres, stateTag, surfaceName,
} from './lib.js';
import { SpatialThreeViewer } from './three-viewer.js';

export async function renderResult(main, topbar, navigate, scanId) {
  const state = {
    scanId, record: null, model: null, artifacts: [], loss: null, condition: null,
    selected: null, view: '2d',
    tab: location.hash === '#ai' ? 'ai'
      : location.hash === '#damage' ? 'damage' : 'overview',
    zoom: 1, panX: 0, panY: 0, baseScale: 1,
    reload: () => renderResult(main, topbar, navigate, scanId),
  };

  let record;
  try {
    record = await api.get(`/api/scans/${scanId}`);
  } catch (error) {
    main.replaceChildren(h('div', { class: 'page' },
      h('div', { class: 'notice' },
        h('h3', {}, 'That space could not be opened'),
        h('p', { class: 'muted' }, error.message),
        h('button', { class: 'btn btn-secondary', onClick: () => navigate('/') },
          'Back to spaces'))));
    return;
  }
  state.record = record;

  if (record.status !== 'complete') {
    navigate(`/scans/${scanId}/processing`, true);
    return;
  }

  state.model = await api.get(`/api/scans/${scanId}/model`);
  state.artifacts = (await api.get(`/api/scans/${scanId}/artifacts`)).artifacts;

  buildChrome(state, topbar, navigate);
  buildBody(state, main);
  announce(`${record.name} opened`);
  return () => {
    state.viewer3d?.dispose();
    state.resizeObserver?.disconnect();
  };
}

/* ------------------------------------------------------------------ chrome */

function buildChrome(state, topbar, navigate) {
  const isDev = state.record.classification === 'public_development_fixture';
  const seg = h('div', { class: 'segmented', role: 'tablist', 'aria-label': 'View mode' },
    h('button', {
      role: 'tab', 'aria-selected': 'true', id: 'tab-2d',
      onClick: () => setView(state, '2d'),
    }, '2D plan'),
    h('button', {
      role: 'tab', 'aria-selected': 'false', id: 'tab-3d',
      onClick: () => setView(state, '3d'),
    }, '3D model'));

  topbar.replaceChildren(
    h('button', { class: 'btn btn-quiet', onClick: () => navigate('/') }, '‹ Spaces'),
    h('div', { class: 'title-block' },
      // The space is the page's subject, so its name is the page heading.
      h('h1', { class: 't', id: 'space-name' }, state.record.name),
      h('span', { class: 's' },
        `${state.model.surfaces.length} surfaces · ${classificationWord(state.record.classification)}`)),
    isDev ? badge('Development sample', 'dev') : badge('Final capture', 'final'),
    h('div', { class: 'spacer' }),
    seg,
    h('button', {
      class: 'btn btn-secondary',
      onClick: () => openExport(state),
    }, 'Export'));
  state.seg = seg;
}

function setView(state, view) {
  state.view = view;
  state.seg.children[0].setAttribute('aria-selected', String(view === '2d'));
  state.seg.children[1].setAttribute('aria-selected', String(view === '3d'));
  document.getElementById('pane-2d').classList.toggle('hidden', view !== '2d');
  document.getElementById('pane-3d').classList.toggle('hidden', view !== '3d');
  document.getElementById('view-tools').classList.toggle('hidden', view !== '2d');
  document.getElementById('view-tools-3d').classList.toggle('hidden', view !== '3d');
  if (view === '3d') requestAnimationFrame(() => state.viewer3d?.resize());
}

/* -------------------------------------------------------------------- body */

function buildBody(state, main) {
  const stage = h('div', { class: 'stage', id: 'pane-2d' });
  const pane3d = h('div', { class: 'stage hidden', id: 'pane-3d' });
  const inspector = h('aside', {
    class: 'inspector', 'aria-label': 'Details',
  },
    h('div', { class: 'sheet-grip', 'aria-hidden': 'true' }, h('span')),
    h('div', { class: 'insp-head', id: 'insp-head' }),
    h('div', { class: 'insp-tabs', role: 'tablist', 'aria-label': 'Detail sections', id: 'insp-tabs' }),
    h('div', { class: 'insp-body', id: 'insp-body', role: 'tabpanel' }));

  remount(main, h('div', { class: 'result' },
    h('div', { class: 'canvas-col' },
      h('div', { class: 'canvas-toolbar' },
        h('div', { id: 'view-tools', style: 'display:flex;gap:8px' },
          h('button', { class: 'btn btn-quiet', onClick: () => fit(state) }, 'Fit'),
          h('button', {
            class: 'btn btn-quiet', 'aria-label': 'Zoom out',
            onClick: () => zoomBy(state, 1 / 1.25),
          }, '−'),
          h('button', {
            class: 'btn btn-quiet', 'aria-label': 'Zoom in',
            onClick: () => zoomBy(state, 1.25),
          }, '+')),
        h('div', { id: 'view-tools-3d', class: 'hidden', style: 'display:flex;gap:8px' },
          h('button', {
            class: 'btn btn-quiet', onClick: () => state.viewer3d?.fitCamera(false),
          }, 'Fit'),
          h('button', {
            class: 'btn btn-quiet', onClick: () => state.viewer3d?.fitCamera(true),
          }, 'Reset camera'),
          h('button', {
            class: 'btn btn-quiet', id: 'cutaway-toggle', 'aria-pressed': 'true',
            title: 'Hide the surfaces between you and the room',
            onClick: (event) => {
              const on = event.currentTarget.getAttribute('aria-pressed') !== 'true';
              event.currentTarget.setAttribute('aria-pressed', String(on));
              state.viewer3d?.setCutaway(on);
              announce(on ? 'Cutaway on' : 'Showing the closed room');
            },
          }, 'Cutaway')),
        h('div', { class: 'spacer' }),
        h('span', { class: 'tiny muted' }, 'Files stay on this machine')),
      stage, pane3d,
      h('div', { class: 'canvas-foot' },
        h('span', {}, `${state.artifacts.filter((a) => a.available).length} artifacts generated`),
        h('span', {}, `Geometry config ${(state.model.provenance.geometryConfigHash || '').slice(0, 8)}…`),
        state.model.aiAssessments?.[0]?.model
          ? h('span', {}, `AI ${state.model.aiAssessments[0].model}`) : null)),
    inspector));

  loadPlan(state, stage);
  loadModel3d(state, pane3d);
  renderInspector(state);
}

/* ------------------------------------------------------------------ 2D plan */

async function loadPlan(state, stage) {
  let svgText;
  try {
    svgText = await api.text(`/api/scans/${state.scanId}/artifacts/floorplan.svg`);
  } catch {
    stage.append(h('p', { class: 'muted' }, 'The floor plan artifact is unavailable.'));
    return;
  }
  const layer = h('div', { class: 'pan-layer' });
  layer.innerHTML = svgText;
  stage.replaceChildren(layer);
  state.layer = layer;
  state.stage = stage;

  const svg = layer.querySelector('svg');
  if (!svg) return;
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label',
    `Floor plan of ${state.record.name}. Use the surface list to select a wall.`);

  // Make the generated plan's wall labels selectable, so the drawing and the
  // inspector always agree about what is selected.
  const walls = state.model.surfaces.filter((s) => s.type === 'wall');
  for (const group of svg.querySelectorAll('g')) {
    const text = group.textContent || '';
    const wall = walls.find((w) => text.startsWith(`${w.id} `));
    if (!wall) continue;
    group.dataset.surface = wall.id;
    group.setAttribute('role', 'button');
    group.setAttribute('tabindex', '0');
    group.setAttribute('aria-label',
      `${surfaceName(wall.id)}, ${observationWord(wall.observationState)}, `
      + `${wall.dimensions.width_m.toFixed(2)} metres`);
    group.setAttribute('aria-selected', 'false');
    for (const node of group.querySelectorAll('text')) node.classList.add('lbl');
    group.addEventListener('click', () => select(state, wall.id));
    group.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault(); select(state, wall.id);
      }
    });
  }

  wirePanZoom(state);
  requestAnimationFrame(() => fit(state));

  // Re-fit when the pane changes size, so rotating a phone or resizing a window
  // does not leave the plan cropped or adrift.
  if (typeof ResizeObserver !== 'undefined') {
    const observer = new ResizeObserver(() => fit(state));
    observer.observe(stage);
    state.resizeObserver = observer;
  }
}

function wirePanZoom(state) {
  const { stage } = state;
  let dragging = false;
  let lastX = 0;
  let lastY = 0;

  stage.addEventListener('pointerdown', (event) => {
    if (event.target.closest('[data-surface]')) return;
    dragging = true; lastX = event.clientX; lastY = event.clientY;
    stage.setPointerCapture(event.pointerId);
    stage.style.cursor = 'grabbing';
  });
  stage.addEventListener('pointermove', (event) => {
    if (!dragging) return;
    state.panX += event.clientX - lastX;
    state.panY += event.clientY - lastY;
    lastX = event.clientX; lastY = event.clientY;
    applyTransform(state);
  });
  const end = (event) => {
    dragging = false;
    stage.style.cursor = '';
    if (event.pointerId !== undefined && stage.hasPointerCapture?.(event.pointerId)) {
      stage.releasePointerCapture(event.pointerId);
    }
  };
  stage.addEventListener('pointerup', end);
  stage.addEventListener('pointercancel', end);
  stage.addEventListener('wheel', (event) => {
    event.preventDefault();
    zoomBy(state, event.deltaY < 0 ? 1.1 : 1 / 1.1);
  }, { passive: false });
}

function applyTransform(state) {
  if (!state.layer) return;
  const scale = state.baseScale * state.zoom;
  state.layer.style.transform =
    `translate(${state.panX}px, ${state.panY}px) scale(${scale})`;
}

function fit(state) {
  const svg = state.layer?.querySelector('svg');
  if (!svg || !state.stage) return;
  const width = parseFloat(svg.getAttribute('width')) || svg.viewBox.baseVal.width;
  const height = parseFloat(svg.getAttribute('height')) || svg.viewBox.baseVal.height;
  const box = state.stage.getBoundingClientRect();
  const pad = 32;
  state.baseScale = Math.min((box.width - pad) / width, (box.height - pad) / height);
  state.zoom = 1;
  state.panX = (box.width - width * state.baseScale) / 2;
  state.panY = (box.height - height * state.baseScale) / 2;
  applyTransform(state);
}

function zoomBy(state, factor) {
  const next = Math.min(Math.max(state.zoom * factor, 0.4), 8);
  const box = state.stage.getBoundingClientRect();
  const cx = box.width / 2;
  const cy = box.height / 2;
  const ratio = next / state.zoom;
  state.panX = cx - (cx - state.panX) * ratio;
  state.panY = cy - (cy - state.panY) * ratio;
  state.zoom = next;
  applyTransform(state);
}

/* ---------------------------------------------------------------------- 3D */

async function loadModel3d(state, pane) {
  const required = ['room_model.obj', 'room_model.mtl', 'room_model_entity_map.json'];
  const available = new Set(state.artifacts.filter((artifact) => artifact.available)
    .map((artifact) => artifact.name));
  if (!required.every((name) => available.has(name))) {
    pane.replaceChildren(model3dError('The complete OBJ, MTL, and entity-map artifact set is unavailable.'));
    return;
  }

  const status = h('div', { class: 'model3d-status', role: 'status' },
    h('span', { class: 'spinner', 'aria-hidden': 'true' }),
    h('div', {}, h('h2', {}, 'Loading 3D room'),
      h('p', { class: 'muted small' }, 'Reading the generated OBJ, materials, and canonical surface map…')));
  pane.replaceChildren(status);
  try {
    const entityMap = await api.get(
      `/api/scans/${state.scanId}/artifacts/room_model_entity_map.json`);
    const host = h('div', { class: 'three-host' });
    pane.replaceChildren(host, h('div', { class: 'three-help' },
      'Drag to orbit · Secondary drag to pan · Scroll or pinch to zoom · Click a surface for details'));
    const viewer = new SpatialThreeViewer(host, {
      scanId: state.scanId,
      entityMap,
      onSelect: (surfaceId) => select(state, surfaceId, true),
      onContextLost: () => {
        // The GPU can drop the context on sleep or a driver reset. Say so and
        // offer the rebuild rather than leaving a blank canvas.
        state.viewer3d = null;
        remount(pane, model3dError('The graphics context was lost, which can happen '
          + 'after the display sleeps.', () => loadModel3d(state, pane)));
      },
    });
    state.viewer3d = viewer;
    await viewer.load();
    if (viewer.disposed) return;
    const toggle = document.getElementById('cutaway-toggle');
    if (toggle) viewer.setCutaway(toggle.getAttribute('aria-pressed') === 'true');
    viewer.setSelection(state.selected);
    announce('Interactive 3D model loaded');
  } catch (error) {
    state.viewer3d?.dispose();
    state.viewer3d = null;
    pane.replaceChildren(model3dError(error.message, () => loadModel3d(state, pane)));
    announce('The 3D model could not be loaded');
  }
}

function model3dError(message, retry) {
  return h('div', { class: 'model3d-status notice', role: 'alert' },
    h('div', {}, h('h2', {}, '3D model unavailable'),
      h('p', { class: 'muted small' }, message),
      retry ? h('button', { class: 'btn btn-secondary', onClick: retry }, 'Try again') : null));
}

/* --------------------------------------------------------------- selection */

function select(state, surfaceId, direct = false) {
  state.selected = direct ? surfaceId : (state.selected === surfaceId ? null : surfaceId);
  for (const node of document.querySelectorAll('[data-surface]')) {
    node.setAttribute('aria-selected', String(node.dataset.surface === state.selected));
  }
  state.viewer3d?.setSelection(state.selected);
  renderInspector(state);
  const name = state.selected ? surfaceName(state.selected) : state.record.name;
  announce(`${name} selected`);
}

function measurement(state, type, entityId) {
  return state.model.measurements.find(
    (m) => m.type === type && (entityId === undefined || m.entityId === entityId));
}

/* --------------------------------------------------------------- inspector */

function availableTabs(state) {
  return [
    ['overview', 'Overview'],
    ['ai', 'AI review'],
    ['evidence', 'Evidence'],
    ['damage', 'Damage intelligence'],
  ];
}

function renderInspector(state) {
  const head = document.getElementById('insp-head');
  const tabsHost = document.getElementById('insp-tabs');
  const body = document.getElementById('insp-body');
  const surface = state.selected
    ? state.model.surfaces.find((s) => s.id === state.selected) : null;

  remount(head, 
    h('div', { class: 'row' },
      h('h2', {}, surface ? surfaceName(surface.id) : state.record.name),
      h('div', { style: 'flex:1' }),
      surface ? h('button', {
        class: 'btn btn-quiet tiny', onClick: () => select(state, surface.id),
      }, 'Clear') : null),
    surface
      ? stateTag(surface.observationState)
      : h('p', { class: 'muted small' },
        `${classificationWord(state.record.classification)} · `
        + 'Select a wall on the plan to focus'));

  const tabs = availableTabs(state);
  if (!tabs.some(([id]) => id === state.tab)) state.tab = 'overview';
  tabsHost.replaceChildren(...tabs.map(([id, label]) => h('button', {
    role: 'tab', 'aria-selected': String(state.tab === id),
    onClick: () => { state.tab = id; renderInspector(state); },
  }, label)));

  const painters = {
    overview: paintOverview, ai: paintAI, evidence: paintEvidence,
    damage: paintDamage,
  };
  body.replaceChildren();
  painters[state.tab](state, body, surface);
}

/* ---- overview ------------------------------------------------------------ */

function paintOverview(state, body, surface) {
  if (!surface) {
    const observed = state.model.surfaces.filter(
      (s) => s.observationState === 'directly_observed').length;
    const inferred = state.model.surfaces.filter(
      (s) => s.observationState === 'inferred').length;
    const openings = state.model.openings || [];
    const confirmedOpenings = openings.filter(
      (o) => o.observationState !== 'unresolved' && o.dimensions);
    const unresolvedOpenings = openings.filter(
      (o) => o.observationState === 'unresolved').length;

    mount(body, 
      h('div', { class: 'metric-row' },
        metric('Length', metres(measurement(state, 'room_length')?.value_m)),
        metric('Width', metres(measurement(state, 'room_width')?.value_m)),
        metric('Height', metres(measurement(state, 'room_height')?.value_m)),
        metric('Floor area', squareMetres(measurement(state, 'floor_area')?.value_m))),
      h('p', { class: 'producer' }, 'All measurements produced by geometry'),
      h('div', {},
        h('p', { class: 'section-label', style: 'margin-bottom:8px' }, 'Surfaces'),
        h('div', { class: 'kv' },
          kv('Total', String(state.model.surfaces.length)),
          kv('Directly observed', String(observed)),
          inferred ? kv('Inferred closure', String(inferred)) : null,
          kv('Doorways and windows', confirmedOpenings.length
            ? confirmedOpenings.map((item) => conditionWord(item.type)).join(', ')
            : unresolvedOpenings
              ? 'None confirmed'
              : 'None found'))),
      inferred ? h('div', { class: 'notice warn' },
        h('h3', {}, 'Part of this outline is inferred'),
        h('p', { class: 'small' },
          `${inferred} of ${state.model.rooms[0].footprintEdgeStates.length} outline `
          + 'edges have no wall behind them. Those edges follow the observed floor '
          + 'extent and are not measured walls.')) : null,
      h('div', {},
        h('p', { class: 'section-label', style: 'margin-bottom:8px' }, 'All surfaces'),
        h('div', { class: 'surface-list' },
          ...state.model.surfaces.map((s) => surfaceButton(state, s)))));
    return;
  }

  const findings = aiFindingsFor(state, surface.id);
  const views = state.model.evidence.filter(
    (v) => v.visibleSurfaceIds.includes(surface.id));
  const fit = surface.provenance.fitDiagnostics;

  mount(body, 
    h('div', { class: 'metric-row' },
      metric(surface.type === 'wall' ? 'Length' : 'Extent',
        metres(surface.dimensions.width_m)),
      metric('Height', metres(surface.dimensions.height_m))),
    h('p', { class: 'producer' }, 'Produced by geometry'),
    h('div', { class: 'kv' },
      kv('Observation', observationWord(surface.observationState)),
      kv('Evidence quality', evidenceQualityWord(surface.confidence.label)),
      fit ? kv('Coverage', `${Math.round(fit.coverageFraction * 100)}%`) : null,
      fit ? kv('Registered views', String(views.length)) : kv('Registered views', String(views.length)),
      kv('AI findings', String(findings.length))),
    disclosure('About evidence quality',
      h('p', { class: 'small' },
        'Rule-based from fit residual, coverage and how many camera positions saw '
        + 'the surface. Not a calibrated probability, and it carries no interval.'),
      h('p', { class: 'small mono', style: 'margin-top:6px' },
        surface.confidence.ruleTriggered)),
    fit ? disclosure('Technical provenance',
      h('div', { class: 'kv' },
        kv('Algorithm', h('span', { class: 'mono' }, surface.provenance.algorithm)),
        kv('Inlier points', String(fit.inlierCount)),
        kv('Contributing frames', String(fit.contributingFrames)),
        kv('RMS residual', `${(fit.rmsResidual_m * 100).toFixed(2)} cm`),
        kv('Config', h('span', { class: 'mono' },
          `${(surface.provenance.geometryConfigHash || '').slice(0, 12)}…`)))) : null,
    findings.length ? h('div', {},
      h('p', { class: 'section-label', style: 'margin:6px 0 8px' }, 'AI findings here'),
      ...findings.map((f) => findingCard(state, f))) : null);
}

function surfaceButton(state, surface) {
  return h('button', {
    class: 'surface-btn',
    'aria-pressed': String(state.selected === surface.id),
    onClick: () => select(state, surface.id),
  },
    h('span', { class: 'nm' }, surfaceName(surface.id)),
    stateTag(surface.observationState));
}

/* ---- AI review ----------------------------------------------------------- */

function parsePastedReview(text) {
  let cleaned = String(text || '').trim();
  if (cleaned.startsWith('```')) {
    cleaned = cleaned.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '');
  }
  const parsed = JSON.parse(cleaned);
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('Paste a JSON object, not an array or a sentence.');
  }
  return parsed;
}

function assessment(state) { return state.model.aiAssessments?.[0] || null; }

function aiFindingsFor(state, surfaceId) {
  const a = assessment(state);
  if (!a || a.status !== 'completed') return [];
  return a.findings.filter((f) => f.surfaceId === surfaceId);
}

const groqAsk = new Map();

function isPublishedExample(state) {
  const kind = state.record?.authenticity?.kind;
  const cls = state.record?.classification;
  return kind === 'public_sample' || kind === 'public_review_example'
    || cls === 'public_sample' || cls === 'public_review_example';
}

function paintAskingGroq(body) {
  mount(body,
    h('p', { class: 'ai-lede' },
      'AI looks at the photographs. It does not change any measurement.'),
    h('div', { class: 'empty', style: 'padding:28px 16px' },
      h('h2', {}, 'Asking Groq'),
      h('p', { class: 'small' },
        'The approved model is reviewing the photographs. Measurements stay '
        + 'produced by geometry.')));
}

function paintQwenPaste(state, body) {
  mount(body,
    h('p', { class: 'ai-lede' },
      'AI looks at the photographs. It does not change any measurement.'),
    h('div', { class: 'empty', style: 'padding:28px 16px' },
      h('h2', {}, 'Paste the Qwen review'),
      h('p', { class: 'small' },
        'Groq did not return a review. Use the zip and prompt. Attach the '
        + 'three photographs, then paste only the JSON Qwen returns. '
        + 'Measurements stay produced by geometry.'),
      h('textarea', {
        id: 'qwen-review-paste',
        class: 'qwen-paste',
        rows: 12,
        spellcheck: 'false',
        placeholder: '{ "schemaVersion": "0.1", "status": "completed", ... }',
      }),
      h('button', {
        class: 'btn btn-primary',
        style: 'margin-top:14px',
        onClick: async () => {
          const node = document.getElementById('qwen-review-paste');
          const text = (node && node.value || '').trim();
          if (!text) {
            window.alert('Paste the Qwen JSON first.');
            return;
          }
          try {
            announce('Storing the Qwen review');
            const payload = parsePastedReview(text);
            try {
              await api.post(
                `/api/scans/${state.scanId}/ai-review/import`, payload);
            } catch (error) {
              if (error.status !== 404 && error.status !== 405) throw error;
              await api.post(
                `/api/scans/${state.scanId}/ai-review`, payload);
            }
            state.reload();
          } catch (error) {
            window.alert(error.message);
          }
        },
      }, 'Store review')));
}

async function requestGroqReview(state) {
  announce('Asking Groq for a review');
  try {
    await api.post(`/api/scans/${state.scanId}/ai-review`);
  } catch {
    /* Groq failed; the Qwen paste is the fallback. */
  }
  groqAsk.set(state.scanId, 'done');
  try {
    const model = await api.get(`/api/scans/${state.scanId}/model`);
    if (model.aiAssessments?.[0]?.status === 'completed') {
      state.reload();
      return;
    }
  } catch {
    /* stay on the Qwen paste */
  }
  if (state.tab === 'ai') renderInspector(state);
}

function paintAI(state, body, surface) {
  const a = assessment(state);

  if (!a || a.status !== 'completed') {
    if (isPublishedExample(state)) {
      paintQwenPaste(state, body);
      return;
    }
    const phase = groqAsk.get(state.scanId);
    if (phase !== 'done') {
      if (phase !== 'asking') {
        groqAsk.set(state.scanId, 'asking');
        void requestGroqReview(state);
      }
      paintAskingGroq(body);
      return;
    }
    paintQwenPaste(state, body);
    return;
  }

  const all = a.findings || [];
  const shown = surface ? all.filter((f) => f.surfaceId === surface.id) : all;
  const review = shown.filter((f) => f.status === 'review_recommended');
  const quiet = shown.filter((f) => f.status !== 'review_recommended');
  const roomReview = all.filter((f) => f.status === 'review_recommended');

  mount(body,
    h('p', { class: 'ai-lede' },
      'AI looks at the photographs. It does not change any measurement. '
      + 'A doorway or window is confirmed only from a crop of a geometry gap, '
      + 'not from watching the video.'),
    surface
      ? h('div', { class: 'scope-bar' },
        h('p', {}, `Notes for ${surfaceName(surface.id)}`),
        h('button', {
          class: 'text-btn', type: 'button',
          onClick: () => select(state, surface.id),
        }, 'Show the whole space'))
      : h('p', { class: 'scope-note' },
        'The list below is for the whole space. Select a wall to read only that wall.'),
    !surface && a.roomTypeHypothesis
      ? h('div', { class: 'type-chip' },
        h('div', {},
          h('p', { class: 'section-label human' }, 'Suggested room type'),
          h('p', { class: 'type-value' }, conditionWord(a.roomTypeHypothesis))),
        badge('AI interpretation', 'ai'))
      : null,
    review.length
      ? h('div', { class: 'review-block' },
        h('p', { class: 'section-label human' },
          review.length === 1 ? 'Needs a look' : 'Need a look'),
        ...review.map((f) => findingCard(state, f)))
      : surface
        ? h('p', { class: 'muted small' },
          'Nothing on this wall needs a person to review it.')
        : roomReview.length === 0
          ? h('p', { class: 'muted small' },
            'No photograph note needs a person to review it.')
          : null,
    openingsBlock(state, surface),
    quiet.length
      ? disclosure(
        surface
          ? `${quiet.length} other ${quiet.length === 1 ? 'note' : 'notes'} on this wall`
          : `${quiet.length} other notes · no action needed`,
        h('p', { class: 'small muted', style: 'margin-bottom:8px' },
          surface
            ? 'These are AI notes, not measurements.'
            : 'Select a wall on the plan to open a note in place.'),
        h('div', { class: 'note-list' },
          ...quiet.map((f) => noteRow(state, f))))
      : surface && !shown.length
        ? h('p', { class: 'muted small' },
          'AI did not leave a note on this wall.')
        : null,
    disclosure('Model and prompt provenance',
      h('div', { class: 'kv' },
        kv('Model', h('span', { class: 'mono' }, a.model || '—')),
        kv('Provider', a.provider || '—'),
        kv('Prompt', h('span', { class: 'mono' }, a.promptVersion || '—')),
        kv('Generated', a.generatedAt || '—'))));
}

function noteRow(state, finding) {
  const statusWordMap = {
    verified: 'Looks consistent',
    review_recommended: 'Needs a look',
    occluded: 'Too obscured',
    not_visible: 'Not in the photo',
  };
  return h('button', {
    class: 'note-row', type: 'button',
    onClick: () => select(state, finding.surfaceId, true),
  },
    h('span', { class: 'nm' }, surfaceName(finding.surfaceId)),
    h('span', { class: 'st' }, statusWordMap[finding.status] || conditionWord(finding.status)));
}

function openingsBlock(state, surface) {
  const openings = state.model.openings || [];
  const onWall = surface
    ? openings.filter((item) => item.surfaceId === surface.id)
    : openings;
  const rejected = rejectedOnSurface(openings, surface);

  return h('div', { class: 'openings-block' },
    h('p', { class: 'section-label human' }, 'Doorways and windows'),
    h('p', { class: 'small muted', style: 'margin-bottom:10px' },
      'Geometry finds a gap on a named wall. AI classifies a crop of that '
      + 'nominated region. A door seen elsewhere in the room does not confirm '
      + 'this gap. Dimensions stay with geometry.'),
    ...onWall.map((item) => openingCard(state, item)),
    !onWall.length && surface && rejected
      ? h('article', { class: 'finding quiet' },
        h('div', { class: 'top' },
          h('div', { class: 'hrow' },
            h('h3', {}, 'No doorway on this wall'),
            badge('Geometry', 'geometry', true)),
          h('p', { class: 'reason' }, humanOpeningReason(rejected.reason))))
      : null,
    !onWall.length && surface && !rejected
      ? h('p', { class: 'muted small' },
        'No doorway or window is attached to this wall.')
      : null,
    !surface && !openings.length
      ? h('p', { class: 'muted small' }, 'No doorway or window candidates were recorded.')
      : null);
}

function rejectedOnSurface(openings, surface) {
  if (!surface) return null;
  for (const item of openings) {
    const rejected = item.provenance?.rejectedCandidates || [];
    const hit = rejected.find((row) => row.surfaceId === surface.id);
    if (hit) return hit;
  }
  return null;
}

function humanOpeningReason(reason) {
  if (!reason) return 'This wall was checked and no opening was confirmed.';
  if (/no empty region/i.test(reason)) {
    return 'This wall has no gap large enough to be a door or window.';
  }
  if (/edge of the observed region/i.test(reason)) {
    return 'A gap here sits at the edge of the scan, so it is treated as missing '
      + 'coverage rather than a doorway.';
  }
  return reason;
}

function findingCard(state, finding) {
  const statusWordMap = {
    verified: 'Looks consistent',
    review_recommended: 'Needs a look',
    occluded: 'Too obscured to judge',
    not_visible: 'Not in the photograph',
  };
  const view = state.model.evidence.find(
    (v) => finding.evidenceFrameIds.includes(v.id));

  return h('article', { class: 'finding' },
    h('div', { class: 'top' },
      h('div', { class: 'hrow' },
        h('h3', {}, statusWordMap[finding.status] || conditionWord(finding.status)),
        badge('AI interpretation', 'ai')),
      h('div', { class: 'where' },
        h('button', {
          class: 'btn btn-quiet tiny', style: 'padding:2px 6px;min-height:0',
          onClick: () => select(state, finding.surfaceId),
        }, surfaceName(finding.surfaceId)),
        finding.evidenceFrameIds.length
          ? h('span', {}, `Evidence ${finding.evidenceFrameIds.join(', ')}`) : null),
      h('p', { class: 'reason' }, finding.reason),
      finding.occlusionDescription
        ? h('p', { class: 'small muted' }, `Occlusion: ${finding.occlusionDescription}`) : null),
    view ? h('figure', {},
      h('img', {
        loading: 'lazy',
        alt: `Evidence frame ${view.id} showing ${surfaceName(finding.surfaceId)}`,
        src: evidenceSrc(state, view),
      })) : null);
}

function openingCard(state, opening) {
  const resolved = opening.observationState !== 'unresolved' && opening.dimensions;
  const provenance = opening.provenance || {};
  const ai = provenance.aiResolution || {};
  const resolution = ai.resolution || provenance.resolution || null;
  const cropName = (ai.cropPath || '').split('/').pop();
  const reason = resolved
    ? (resolution?.reason || provenance.reason
      || 'A crop of this nominated gap supports a doorway or window.')
    : (resolution?.reason
      || 'No door or window was confirmed. Geometry searched the walls and left '
        + 'this as unresolved rather than inventing a size.');

  return h('article', { class: 'finding' },
    h('div', { class: 'top' },
      h('div', { class: 'hrow' },
        h('h3', {}, resolved ? conditionWord(opening.type) : 'Not confirmed'),
        badge(resolved ? 'Crop corroborated' : 'Unresolved', resolved ? 'complete' : 'review'),
        resolution?.semanticClass
          ? badge('AI interpretation', 'ai') : null),
      opening.surfaceId ? h('div', { class: 'where' },
        h('button', {
          class: 'btn btn-quiet tiny', style: 'padding:2px 6px;min-height:0',
          onClick: () => select(state, opening.surfaceId, true),
        }, surfaceName(opening.surfaceId)),
        ai.evidenceFrameId
          ? h('span', {}, `Crop from ${ai.evidenceFrameId}`) : null) : null,
      h('p', { class: 'reason' }, reason)),
    cropName
      ? h('figure', {},
        h('img', {
          loading: 'lazy',
          alt: `Nominated crop for ${opening.id}`,
          src: `/api/scans/${state.scanId}/evidence/${encodeURIComponent(cropName)}`,
        }))
      : null,
    resolved ? h('div', { class: 'quantity' },
      h('div', {},
        h('div', { class: 'n' },
          `${opening.dimensions.width_m.toFixed(2)} × ${opening.dimensions.height_m.toFixed(2)} m`),
        h('span', { class: 'producer' }, 'Dimensions produced by geometry')),
      badge('Geometry', 'geometry', true))
      : h('div', { class: 'quantity' },
        h('span', { class: 'small muted' }, 'No opening dimensions reported'),
        badge('No quantity', 'geometry', true)),
    !resolved && (provenance.rejectedCandidates || []).length
      ? disclosure('Walls that were checked',
        h('div', { class: 'note-list' },
          ...(provenance.rejectedCandidates || []).map((row) => h('button', {
            class: 'note-row', type: 'button',
            onClick: () => row.surfaceId && select(state, row.surfaceId, true),
          },
            h('span', { class: 'nm' }, surfaceName(row.surfaceId)),
            h('span', { class: 'st' },
              /no empty region/i.test(row.reason || '') ? 'No gap' : 'Scan edge')))))
      : null);
}

/* ---- evidence ------------------------------------------------------------ */

function evidenceSrc(state, view) {
  const name = view.path.split('/').pop();
  return `/api/scans/${state.scanId}/evidence/${encodeURIComponent(name)}`;
}

function paintEvidence(state, body, surface) {
  const views = surface
    ? state.model.evidence.filter((v) => v.visibleSurfaceIds.includes(surface.id))
    : state.model.evidence;

  if (!views.length) {
    body.append(h('div', { class: 'empty', style: 'padding:30px 18px' },
      h('h2', {}, 'No registered views'),
      h('p', { class: 'small' }, surface
        ? `No photograph in this capture is registered to ${surfaceName(surface.id)}.`
        : 'This capture has no registered evidence.')));
    return;
  }

  body.append(
    h('p', { class: 'small muted' },
      'Each view is tied to the camera position that recorded it, so a photograph '
      + 'can be traced to the surfaces it shows.'),
    h('div', { class: 'evidence-grid' }, ...views.map((view) => h('button', {
      class: 'ev-card',
      onClick: () => openViewer(state, view),
    },
      h('img', {
        loading: 'lazy', src: evidenceSrc(state, view),
        alt: `Evidence frame ${view.id}`,
      }),
      h('div', { class: 'cap' },
        h('strong', {}, view.id),
        h('span', { class: 'tiny muted' },
          `Registered · ${view.visibleSurfaceIds.map(surfaceName).join(', ')}`))))));
}

function openViewer(state, view) {
  const dialog = document.getElementById('viewer');
  dialog.replaceChildren(
    h('div', { class: 'vhead' },
      h('h2', {}, view.id),
      h('div', { class: 'spacer' }),
      h('button', { class: 'btn btn-quiet', onClick: () => dialog.close() }, 'Close')),
    h('div', { class: 'vbody' },
      h('img', { src: evidenceSrc(state, view), alt: `Evidence frame ${view.id}` })),
    h('div', { class: 'vfoot' },
      h('div', { class: 'kv' },
        kv('Surfaces', view.visibleSurfaceIds.map(surfaceName).join(', ')),
        kv('Timestamp', `${view.timestamp_s.toFixed(2)} s`),
        kv('Registration', conditionWord(view.registration))),
      disclosure('Technical provenance',
        h('div', { class: 'kv' },
          kv('Source frame', h('span', { class: 'mono' }, view.sourceFrame)),
          kv('Pose offset', `${(view.poseTimeOffset_s * 1000).toFixed(1)} ms`),
          kv('Producer', view.producer)),
        h('p', { class: 'tiny muted', style: 'margin-top:8px' },
          'The 4×4 camera transform is recorded in evidence_manifest.json.'))));
  dialog.showModal();
}

/* ---- damage intelligence (future Track B/C; not implemented) ------------ */

function paintDamage(state, body) {
  const damage = state.model.damage || [];
  const scope = state.model.scope || [];

  mount(body,
    h('div', { class: 'notice warn' },
      h('h3', {}, 'Track B and Track C are not implemented'),
      h('p', { class: 'small' },
        'This tab is the planned path from today’s room model into damage '
        + 'intelligence. It is not a damage detector, a repair estimate, a '
        + 'coverage decision, or automated claims handling.')),
    h('h3', {}, 'The contract that stays'),
    h('p', {},
      'AI identifies what may be damaged. Geometry determines how much. '
      + 'Humans approve consequential claim decisions.'),
    h('p', { class: 'small muted' },
      'An AI interpretation may mark a visible condition on a registered '
      + 'photograph. A Geometry measurement is produced by geometry after that '
      + 'region is registered to a named surface. Nothing is uploaded.'),
    h('p', { class: 'section-label human' }, 'How the layers stack'),
    h('div', { class: 'kv' },
      kv('Track A now',
        'Room → named surfaces → geometry measurements → evidence → AI semantic review'),
      kv('Track B later',
        'AI detects and classifies visible water, fire, or mold on a named surfaceId; '
        + 'geometry then measures affected area after registration'),
      kv('Track C later',
        'Sourced restoration rules turn those damage records into line items and scope')),
    h('p', { class: 'section-label human' }, 'What this space already supplies'),
    h('p', { class: 'small' },
      'Stable surface IDs, geometry-owned lengths and area, registered photographs, '
      + 'and a bounded photograph review. That is the grounding layer Track B '
      + 'would attach to. Production damage[] and scope[] stay empty until a '
      + 'later implementation writes them.'),
    h('div', { class: 'kv' },
      kv('Production damage[]', damage.length ? String(damage.length) : 'Empty'),
      kv('Production scope[]', scope.length ? String(scope.length) : 'Empty'),
      kv('Track B status', 'Deferred — not evaluated'),
      kv('Track C status', 'Not implemented')),
    h('p', { class: 'section-label human' }, 'What would be added, not what exists'),
    h('p', { class: 'small' },
      'Week 3 in the implementation plan would evolve photograph review into '
      + 'damage proposals: classify staged water, fire, or mold; associate each '
      + 'proposal with one surface; fuse repeated views without double counting; '
      + 'and let geometry compute metric affected area and vertical extent. AI '
      + 'would never invent square footage or linear quantities.'),
    h('p', { class: 'small' },
      'Week 4 would add sourced restoration actions — flood-cut, baseboard run, '
      + 'drying equipment, containment, and PPE — as line items keyed to surface '
      + 'and damage IDs, with citations. Hidden damage, coverage, reserve, and '
      + 'settlement stay out of scope.'),
    h('p', { class: 'tiny muted' },
      'Source: implementation plan §§8, 14, 16.2/16.3 and the validation ledger. '
      + 'A Development sample or Final capture uses the same contract.'));
}

/* ------------------------------------------------------------------ export */

function openExport(state) {
  const dialog = document.getElementById('export');
  const statusWordMap = {
    'development-only': 'Development only',
    'final-verified': 'Final verified',
    experimental: 'Experimental',
    pending: 'Pending',
    unavailable: 'Unavailable',
  };
  remount(dialog, 
    h('div', { class: 'vhead' },
      h('h2', {}, 'Export'),
      h('div', { class: 'spacer' }),
      h('button', { class: 'btn btn-quiet', onClick: () => dialog.close() }, 'Done')),
    h('div', { class: 'vfoot' },
      h('p', { class: 'small muted' },
        'Generated artifacts for this space. Files stay on this machine.'),
      ...state.artifacts.map((artifact) => h('div', { class: 'artifact-row' },
        h('div', {},
          h('div', { class: 'n' }, artifact.name),
          h('div', { class: 'tiny muted' }, artifact.description)),
        h('div', { style: 'display:flex;gap:10px;align-items:center' },
          badge(statusWordMap[artifact.status] || artifact.status,
            artifact.status === 'experimental' ? 'experimental'
              : artifact.status === 'development-only' ? 'dev'
                : artifact.status === 'final-verified' ? 'final' : 'geometry', true),
          artifact.available ? h('a', {
            class: 'btn btn-secondary',
            href: `/api/scans/${state.scanId}/artifacts/${encodeURIComponent(artifact.name)}`,
            download: artifact.name,
          }, 'Download') : null)))));
  dialog.showModal();
}
