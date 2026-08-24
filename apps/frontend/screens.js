/* Library, Add capture, and Processing screens. */

import {
  ApiError, announce, api, badge, classificationWord, dimensions, elapsed, h, hostPlatform, metric, mount, relativeDate, remount, squareMetres, statusTone, statusWord,
} from './lib.js';

/* Internal stage names are grouped into phases a person can follow. The raw
 * identifiers stay available under "View processing details". */
const PHASES = [
  { title: 'Reading capture', blurb: 'Checking the recorded colour, depth and camera-pose records.',
    stages: ['upload_received', 'source_detected', 'connector_validation'] },
  { title: 'Aligning evidence', blurb: 'Matching each depth frame to the camera position that recorded it.',
    stages: ['frame_alignment', 'normalized_capture'] },
  { title: 'Building room geometry', blurb: 'Turning depth into a metric point cloud in a gravity-aligned frame.',
    stages: ['geometry'] },
  { title: 'Identifying surfaces', blurb: 'Creating walls, floor, ceiling and openings from the captured evidence.',
    stages: ['canonical_model'] },
  { title: 'Drawing the result', blurb: 'Generating the floor plan and the semantic 3D model.',
    stages: ['floorplan', 'model_3d'] },
  { title: 'Reviewing with AI', blurb: 'Checking the reconstruction against the registered photographs.',
    stages: ['benchmark', 'ai_review', 'loss_preview'] },
  { title: 'Preparing result', blurb: 'Saving artifacts so you can return to them later.',
    stages: ['complete'] },
];

/* ------------------------------------------------------------------ library */

export async function renderLibrary(main, topbar, navigate) {
  topbar.replaceChildren(
    h('div', { class: 'brand' }, h('span', { class: 'mark', 'aria-hidden': 'true' }), 'Spatial AI'),
    h('div', { class: 'spacer' }),
    h('button', { class: 'btn btn-primary', onClick: () => navigate('/scans/new') },
      'Add capture'));

  const page = h('div', { class: 'page' });
  main.replaceChildren(page);

  let scans = [];
  try {
    scans = (await api.get('/api/scans')).scans;
  } catch (error) {
    page.append(h('div', { class: 'notice' },
      h('h3', {}, 'Cannot reach the local service'),
      h('p', { class: 'muted' }, error.message),
      h('p', { class: 'small muted' }, 'Start it with ./run_local.sh and reload.')));
    return;
  }

  page.append(h('div', { class: 'page-head' },
    h('div', { class: 'stack' },
      h('h1', {}, 'Spaces'),
      h('p', { class: 'muted' },
        scans.length
          ? 'Measured spatial models with registered visual evidence.'
          : 'Nothing processed yet.')),
    h('div', { class: 'spacer' })));

  if (!scans.length) {
    page.append(h('div', { class: 'empty' },
      h('h2', {}, 'Your spaces will appear here'),
      h('p', {}, 'Process a room capture to create a measured spatial model with '
        + 'registered visual evidence.'),
      h('button', { class: 'btn btn-primary', onClick: () => navigate('/scans/new') },
        'Add capture')));
    return;
  }

  const grid = h('div', { class: 'space-grid' });
  page.append(grid);
  for (const scan of scans) {
    grid.append(spaceCard(scan, navigate, () => renderLibrary(main, topbar, navigate)));
  }
  announce(`${scans.length} saved ${scans.length === 1 ? 'space' : 'spaces'}`);
}

function spaceCard(scan, navigate, onDeleted) {
  const isDev = scan.classification === 'public_development_fixture';
  const authenticity = scan.authenticity || {};
  const summary = scan.summary || {};
  const failed = scan.status === 'failed';
  const target = scan.status === 'processing'
    ? `/scans/${scan.id}/processing` : `/scans/${scan.id}`;

  const thumb = h('div', { class: 'thumb' },
    h('span', { class: 'placeholder' },
      failed ? 'No result' : 'Preparing…'));
  if (scan.thumbnailArtifact) {
    api.text(`/api/scans/${scan.id}/artifacts/${scan.thumbnailArtifact}`)
      .then((svg) => {
        thumb.innerHTML = svg;
        const node = thumb.querySelector('svg');
        if (node) {
          node.removeAttribute('width');
          node.removeAttribute('height');
          node.setAttribute('preserveAspectRatio', 'xMidYMid meet');
          node.setAttribute('role', 'img');
          node.setAttribute('aria-label', `Floor plan preview of ${scan.name}`);
        }
      })
      .catch(() => { thumb.textContent = ''; });
  }

  const meta = [badge(statusWord(scan), statusTone(scan))];
  if (authenticity.label) {
    meta.push(badge(authenticity.label, authenticityTone(authenticity.kind)));
  } else if (isDev) {
    meta.push(badge('Development sample', 'dev'));
  }
  if (summary.needsReviewCount > 0) {
    meta.push(h('span', { class: 'tiny muted' },
      `${summary.needsReviewCount} ${summary.needsReviewCount === 1 ? 'finding needs' : 'findings need'} review`));
  } else if (summary.aiFindingCount > 0) {
    meta.push(h('span', { class: 'tiny muted' },
      `${summary.aiFindingCount} AI ${summary.aiFindingCount === 1 ? 'finding' : 'findings'}`));
  } else if (scan.status === 'complete') {
    meta.push(h('span', { class: 'tiny muted' }, 'No findings need review'));
  }

  const dims = dimensions(summary);
  return h('article', {
    class: `space-card${failed ? ' is-failed' : ''}`,
    dataset: { kind: authenticity.kind || 'unclassified' },
  },
    h('button', {
      class: 'space-card-open', type: 'button',
      onClick: () => navigate(target),
    },
      thumb,
      h('div', { class: 'body' },
        h('div', { class: 'row1' },
          h('span', { class: 'name' }, scan.name),
          h('span', { class: 'spacer', style: 'flex:1' }),
          h('span', { class: 'tiny muted nowrap' }, relativeDate(scan.createdAt))),
        dims && h('div', { class: 'dims' },
          `${dims} · ${squareMetres(summary.floorArea_m2)}`),
        authenticity.cite
          ? h('p', { class: 'cite' }, authenticity.cite)
          : null,
        isDatasetTile(authenticity)
          ? h('p', { class: 'cite' }, datasetAccuracyCite(authenticity, dims))
          : null,
        h('div', { class: 'meta' }, ...meta))),
    h('div', { class: 'space-card-actions' },
      h('button', {
        class: 'btn btn-secondary', type: 'button',
        'aria-label': `View ${scan.name}`,
        onClick: () => navigate(target),
      }, authenticity.kind === 'local_capture' ? 'View' : 'View sample'),
      isDatasetTile(authenticity)
        ? null
        : h('button', {
          class: 'btn btn-quiet space-card-delete', type: 'button',
          'aria-label': `Delete ${scan.name}`,
          onClick: async (event) => {
            event.preventDefault();
            if (!window.confirm(
              `Delete “${scan.name}”? This removes the processed result from Spaces.`)) {
              return;
            }
            try {
              await api.delete(`/api/scans/${scan.id}`);
              announce(`${scan.name} deleted`);
              onDeleted();
            } catch (error) {
              window.alert(error instanceof ApiError ? error.message : String(error));
            }
          },
        }, 'Delete')));
}

function isDatasetTile(authenticity) {
  return authenticity.kind === 'published_dataset'
    || authenticity.kind === 'public_sample'
    || authenticity.kind === 'public_review_example';
}

function datasetAccuracyCite(authenticity, dims) {
  if (authenticity.kind === 'published_dataset') {
    return dims
      ? 'Geometry metrics are from this model. FARO correspondence to this room was not established, so no accuracy is written. Tape accuracy is not attached to a dataset.'
      : 'FARO correspondence to this room was not established, so no accuracy is written. Tape accuracy is not attached to a dataset.';
  }
  return dims
    ? 'Geometry metrics are from this model. Tape accuracy is not attached to a dataset.'
    : 'Tape accuracy is not attached to a dataset. Process the sample to see geometry metrics.';
}

function authenticityTone(kind) {
  return {
    published_dataset: 'dev',
    public_sample: 'sample',
    public_review_example: 'sample',
    local_capture: 'local',
    unclassified: 'plain',
  }[kind] || 'plain';
}

/* -------------------------------------------------------------- add capture */

function cleanCapturePath(value) {
  // The path field is plain text. A pasted quoted shell path is a common miss.
  let text = String(value || '').trim();
  while (text.length >= 2 && (
    (text.startsWith("'") && text.endsWith("'")) ||
    (text.startsWith('"') && text.endsWith('"'))
  )) {
    text = text.slice(1, -1).trim();
  }
  return text;
}

export async function renderAddCapture(main, topbar, navigate) {
  topbar.replaceChildren(
    h('button', { class: 'btn btn-quiet', onClick: () => navigate('/') }, '‹ Spaces'),
    h('div', { class: 'title-block' }, h('span', { class: 't' }, 'Add capture')),
    h('div', { class: 'spacer' }));

  const page = h('div', { class: 'page', style: 'max-width:720px' });
  main.replaceChildren(page);

  const fixtureWrap = h('div', { class: 'choice-list' });
  const platform = hostPlatform();

  let busy = false;
  let session = null;

  function enterProcessing() {
    session = presentProcessing(main, topbar, navigate);
    return session;
  }

  async function begin(scan, view) {
    if (view.isStopped()) return;
    if (scan.status === 'failed') {
      view.showFailure(scan);
      return;
    }
    api.post(`/api/scans/${scan.scanId}/process`).catch(() => {});
    history.replaceState({}, '', `/scans/${scan.scanId}/processing`);
    view.watch(scan.scanId);
  }

  async function start(sourcePath, label, classification) {
    if (busy) return;
    busy = true;
    const view = enterProcessing();
    await yieldToPaint();
    try {
      const scan = await api.post('/api/scans', {
        source_path: cleanCapturePath(sourcePath), label: label || null,
        classification: classification || 'final_private_capture',
      });
      await begin(scan, view);
    } catch (error) {
      if (view.isStopped()) return;
      view.failStart(error);
      busy = false;
    }
  }

  async function startFromFolder(files) {
    if (busy || !files?.length) return;
    busy = true;
    const view = enterProcessing();
    await yieldToPaint();
    try {
      const form = new FormData();
      form.append('classification', 'final_private_capture');
      for (const file of files) {
        const relative = (file.webkitRelativePath || file.name).replace(/\\/g, '/');
        form.append('files', file, file.name);
        form.append('paths', relative);
      }
      const scan = await api.postForm('/api/scans/from-folder', form);
      await begin(scan, view);
    } catch (error) {
      if (view.isStopped()) return;
      view.failStart(error);
      busy = false;
    }
  }

  page.append(
    h('div', { class: 'stack', style: 'margin-bottom:6px' },
      h('h1', {}, 'Add capture'),
      h('p', { class: 'muted' }, 'Choose a capture source. Everything is processed '
        + 'on this machine — nothing is uploaded to any service.')),
    captureGuide(),
    h('h2', { style: 'margin-top:28px' }, 'Upload the export'),
    h('p', { class: 'muted small' }, 'Drop the Stray folder or a zip of that folder. '
      + 'Do not rename files inside it.'),
    folderDropzone(platform, start, startFromFolder),
    laterDamageNote(),
    h('h2', { style: 'margin-top:30px' }, 'Or try a development sample'),
    h('p', { class: 'muted small' }, 'A prepared public capture, clearly marked so it '
      + 'is never mistaken for real property evidence.'),
    fixtureWrap);

  try {
    const { fixtures } = await api.get('/api/fixtures');
    if (!fixtures.length) {
      fixtureWrap.append(h('p', { class: 'muted small' },
        'No development samples are present on this machine.'));
    }
    for (const fixture of fixtures) {
      fixtureWrap.append(h('button', {
        class: 'choice',
        onClick: () => start(fixture.path, fixture.label, fixture.classification),
      },
        h('span', { class: 'h' }, fixture.label, badge('Development sample', 'dev')),
        h('span', { class: 'muted small mono' }, fixture.path)));
    }
  } catch (error) {
    fixtureWrap.append(h('p', { class: 'muted small' },
      `Could not list samples: ${error.message}`));
  }

  return () => { session?.stop(); };
}

function captureGuide() {
  return h('section', { class: 'capture-guide', 'aria-labelledby': 'iphone-guide' },
    h('h2', { id: 'iphone-guide' }, 'Capture with an iPhone'),
    h('p', { class: 'muted small' },
      'LiDAR has to be part of the recording. The accepted final device in this '
      + 'project is iPhone 13 Pro Max. Other Pro models with LiDAR also produce '
      + 'a usable Stray export.'),
    h('h3', {}, 'Phones and tablets that have LiDAR'),
    h('ul', { class: 'guide-list' },
      h('li', {}, 'iPhone 12 Pro and 12 Pro Max'),
      h('li', {}, 'iPhone 13 Pro and 13 Pro Max'),
      h('li', {}, 'iPhone 14 Pro and 14 Pro Max'),
      h('li', {}, 'iPhone 15 Pro and 15 Pro Max'),
      h('li', {}, 'iPhone 16 Pro and 16 Pro Max'),
      h('li', {}, 'iPad Pro 11-inch (2020 or later) and 12.9-inch (2020 or later)')),
    h('p', { class: 'muted small' },
      'Non-Pro iPhones do not have LiDAR. The regular Camera app does not write '
      + 'the files this pipeline needs.'),
    h('h3', {}, 'Record the room'),
    h('ol', { class: 'guide-list' },
      h('li', {}, 'Install ',
        h('a', {
          href: 'https://apps.apple.com/us/app/stray-scanner/id1557051662',
          target: '_blank', rel: 'noreferrer',
        }, 'Stray Scanner'),
        ' and allow camera access.'),
      h('li', {}, 'Stand near the centre of the room and start recording.'),
      h('li', {}, 'Sweep the floor, then each wall slowly (wall 1, 2, 3, 4), '
        + 'then the ceiling. Pause at doors and windows.'),
      h('li', {}, 'Keep the phone about 0.5–2 m from the surface. Walk; do not '
        + 'only spin in place.'),
      h('li', {}, 'Stop, then export or share the capture. Copy the folder to '
        + 'this computer (AirDrop, Finder, or File Explorer).')),
    h('h3', {}, 'What the export must contain'),
    h('p', { class: 'muted small' },
      'Leave names as Stray wrote them. Required records are colour, LiDAR depth, '
      + 'and camera pose.'),
    h('div', { class: 'file-table', role: 'table', 'aria-label': 'Required export files' },
      fileRow('odometry.csv', 'Camera pose for every frame', 'Required'),
      fileRow('depth/000000.png …', 'LiDAR depth, millimetres', 'Required'),
      fileRow('camera_matrix.csv', 'Colour-camera intrinsics, or fx/fy/cx/cy in odometry.csv', 'Required'),
      fileRow('rgb.mp4', 'Colour video, one frame per depth frame', 'Recommended'),
      fileRow('confidence/000000.png …', 'Depth confidence 0 / 1 / 2', 'Recommended'),
      fileRow('imu.csv', 'Device motion', 'Optional')),
    h('p', { class: 'muted small' },
      'Upload that folder, or a zip of that folder, below. Nothing is uploaded '
      + 'to any service.'));
}

function fileRow(name, meaning, need) {
  return h('div', { class: 'file-row', role: 'row' },
    h('span', { class: 'mono' }, name),
    h('span', {}, meaning),
    h('span', { class: 'tiny muted' }, need));
}

function laterDamageNote() {
  return h('section', { class: 'later-card', 'aria-labelledby': 'later-damage' },
    h('p', { class: 'section-label human' }, 'Later — not in this pass'),
    h('h3', { id: 'later-damage' }, 'Damage analysis'),
    h('p', {},
      'Today the capture builds named surfaces, measurements, and photograph '
      + 'review. Production damage[] and scope[] stay empty.'),
    h('p', {},
      'When damage analysis is added, it uses this same room: AI may mark a '
      + 'visible condition on a registered photograph; geometry then measures '
      + 'the area on that wall, floor, or ceiling; a person reviews it. The '
      + 'Damage intelligence tab on a finished space states that path.'),
    h('p', { class: 'muted small' },
      'Still out of scope: hidden damage, restoration scope, coverage decisions, '
      + 'and automated claims. That is Track B/C in the implementation plan, '
      + 'not this upload.'));
}

function folderDropzone(platform, onPath, onFiles) {
  const windows = platform === 'windows';
  const chooseLabel = windows ? 'Select folder' : 'Choose Folder…';
  const dropHint = windows ? 'Drag a folder here' : 'Drop a folder here';
  const browseHint = windows ? 'or browse this PC' : 'or choose one from the Finder';
  const pathPlaceholder = windows
    ? 'C:\\Users\\…\\stray-capture'
    : '/Users/…/stray-capture';

  const picker = h('input', {
    type: 'file',
    class: 'sr-only',
    id: 'capture-folder',
    multiple: true,
    webkitdirectory: true,
    directory: true,
    'aria-label': chooseLabel,
  });
  picker.webkitdirectory = true;
  picker.directory = true;

  const pathInput = h('input', {
    type: 'text', id: 'capture-path', spellcheck: 'false',
    placeholder: pathPlaceholder,
    'aria-describedby': 'capture-path-help',
  });
  const idle = h('div', { class: 'dropzone-idle' });
  const selected = h('div', { class: 'folder-picked hidden' });
  const processBtn = h('button', {
    class: 'btn btn-primary', type: 'submit', disabled: true,
  }, 'Process capture');
  const chooseBtn = h('button', {
    class: 'btn btn-secondary btn-choose-folder', type: 'button',
    onClick: () => picker.click(),
  }, chooseLabel);
  const zipPicker = h('input', {
    type: 'file', class: 'sr-only', id: 'capture-zip',
    accept: '.zip,application/zip',
    'aria-label': 'Choose a zip of a capture folder',
  });
  const zipBtn = h('button', {
    class: 'btn btn-secondary btn-choose-folder', type: 'button',
    onClick: () => zipPicker.click(),
  }, windows ? 'Select zip' : 'Choose Zip…');

  let pendingFiles = null;
  let pendingPath = '';

  function clearSelection() {
    pendingFiles = null;
    pendingPath = '';
    picker.value = '';
    pathInput.value = '';
    selected.classList.add('hidden');
    selected.replaceChildren();
    idle.classList.remove('hidden');
    processBtn.disabled = true;
  }

  function showSelection({ name, kind, count, bytes, path }) {
    pendingPath = path || '';
    remount(selected,
      folderIcon(platform),
      h('div', { class: 'folder-picked-copy' },
        h('strong', {}, name),
        h('p', { class: 'muted small' },
          [kind, count ? `${count} ${count === 1 ? 'file' : 'files'}` : null,
            bytes ? formatBytes(bytes) : null].filter(Boolean).join(' · ')),
        path && h('p', { class: 'muted tiny mono' }, path)),
      h('button', {
        class: 'btn btn-quiet', type: 'button', onClick: clearSelection,
      }, 'Clear'));
    selected.classList.remove('hidden');
    idle.classList.add('hidden');
    processBtn.disabled = false;
  }

  pathInput.addEventListener('input', () => {
    processBtn.disabled = !(cleanCapturePath(pathInput.value)
      || pendingPath || pendingFiles?.length);
  });

  picker.addEventListener('change', () => {
    if (!picker.files?.length) return;
    pendingFiles = [...picker.files];
    pendingPath = '';
    showSelection(describeCaptureFiles(pendingFiles));
  });
  zipPicker.addEventListener('change', () => {
    const file = zipPicker.files?.[0];
    if (!file) return;
    pendingFiles = [file];
    pendingPath = '';
    showSelection({
      name: file.name, kind: 'Zip archive', count: 1, bytes: file.size, path: '',
    });
  });

  const zone = h('form', {
    class: `dropzone dropzone-${windows ? 'windows' : 'mac'}`,
    onSubmit: (event) => {
      event.preventDefault();
      if (processBtn.disabled) return;
      processBtn.disabled = true;
      processBtn.textContent = 'Starting…';
      processBtn.setAttribute('aria-busy', 'true');
      const typed = cleanCapturePath(pathInput.value);
      if (typed) onPath(typed);
      else if (pendingPath) onPath(pendingPath);
      else if (pendingFiles?.length) onFiles(pendingFiles);
    },
  },
    picker,
    zipPicker,
    remount(idle,
      folderIcon(platform),
      h('p', { class: 'dropzone-title' }, dropHint),
      h('p', { class: 'muted small' }, `${browseHint}, or drop a zip`),
      h('div', { class: 'dropzone-actions' }, chooseBtn, zipBtn)),
    selected,
    processBtn,
    h('details', { class: 'disclosure dropzone-path' },
      h('summary', {}, 'Or enter a folder path'),
      h('div', { class: 'body' },
        pathInput,
        h('p', { class: 'muted tiny', id: 'capture-path-help' },
          'The source type is detected automatically.'))),
    h('p', { class: 'muted tiny' },
      'The folder stays on this machine — nothing is uploaded to any service.'));

  zone.addEventListener('dragover', (event) => {
    event.preventDefault(); zone.classList.add('over');
  });
  zone.addEventListener('dragleave', () => zone.classList.remove('over'));
  zone.addEventListener('drop', (event) => {
    event.preventDefault();
    zone.classList.remove('over');
    const local = localPathFromDrop(event);
    const files = [...(event.dataTransfer?.files || [])];
    if (files.length === 1 && /\.zip$/i.test(files[0].name)) {
      pendingFiles = files;
      pendingPath = '';
      showSelection({
        name: files[0].name, kind: 'Zip archive',
        count: 1, bytes: files[0].size, path: '',
      });
      return;
    }
    if (local) {
      pendingFiles = null;
      pathInput.value = local;
      showSelection({
        name: local.replace(/[/\\]+$/, '').split(/[/\\]/).pop() || local,
        kind: 'Folder on this machine',
        count: files.length, bytes: 0, path: local,
      });
      return;
    }
    if (files.length) {
      pendingFiles = files;
      pendingPath = '';
      showSelection(describeCaptureFiles(files));
    }
  });
  return zone;
}

function describeCaptureFiles(files) {
  const paths = [...files].map((file) =>
    (file.webkitRelativePath || file.name).replace(/\\/g, '/'));
  const name = paths[0]?.split('/').filter(Boolean)[0] || 'Capture';
  const has = (fileName) => paths.some((path) =>
    path === fileName || path.endsWith(`/${fileName}`));
  let kind = 'Capture folder';
  if (has('odometry.csv') && paths.some((path) => /\/depth\/[^/]+\.png$/i.test(path))) {
    kind = 'Stray Scanner export';
  } else if (paths.some((path) => path.endsWith('.traj'))) {
    kind = 'ARKitScenes sequence';
  }
  const bytes = [...files].reduce((sum, file) => sum + (file.size || 0), 0);
  return { name, kind, count: files.length, bytes, path: '' };
}

function localPathFromDrop(event) {
  const raw = event.dataTransfer?.getData('text/uri-list')
    || event.dataTransfer?.getData('text/plain')
    || '';
  const line = raw.split(/\r?\n/).find((row) => row.startsWith('file:'));
  if (line) {
    try {
      const url = new URL(line.trim());
      if (url.protocol === 'file:') {
        let path = decodeURIComponent(url.pathname);
        if (/^\/[A-Za-z]:\//.test(path)) path = path.slice(1);
        return path;
      }
    } catch { /* not a file URL */ }
  }
  const file = event.dataTransfer?.files?.[0];
  if (file && typeof file.path === 'string' && file.path) {
    const relative = (file.webkitRelativePath || '').replace(/\\/g, '/');
    let full = file.path.replace(/\\/g, '/');
    if (relative && full.endsWith(relative)) {
      return full.slice(0, -relative.length).replace(/\/$/, '') || full;
    }
    const depth = relative.split('/').filter(Boolean).length;
    for (let i = 0; i < depth; i += 1) full = full.replace(/\/[^/]+$/, '');
    return full;
  }
  return '';
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function folderIcon(platform) {
  const windows = platform === 'windows';
  return h('span', {
    class: `folder-icon folder-icon-${windows ? 'windows' : 'mac'}`,
    'aria-hidden': 'true',
    html: windows
      ? '<svg viewBox="0 0 72 56" focusable="false"><path fill="#f8d775" d="M6 14h22l5 6h33a5 5 0 0 1 5 5v23a5 5 0 0 1-5 5H6a5 5 0 0 1-5-5V19a5 5 0 0 1 5-5z"/><path fill="#ffcd4a" d="M1 26h70v17a5 5 0 0 1-5 5H6a5 5 0 0 1-5-5V26z"/></svg>'
      : '<svg viewBox="0 0 72 56" focusable="false"><path fill="#5ac8f5" d="M8 12h20l4 5h32a6 6 0 0 1 6 6v25a6 6 0 0 1-6 6H8a6 6 0 0 1-6-6V18a6 6 0 0 1 6-6z"/><path fill="#8edaf8" d="M2 24h68v18a6 6 0 0 1-6 6H8a6 6 0 0 1-6-6V24z" opacity=".55"/></svg>',
  });
}

/* --------------------------------------------------------------- processing */

function yieldToPaint() {
  // The folder upload and FormData walk can block the main thread. Two frames
  // lets the loading state appear before that work starts.
  return new Promise((resolve) => {
    requestAnimationFrame(() => requestAnimationFrame(resolve));
  });
}

function presentProcessing(main, topbar, navigate) {
  let started = Date.now();
  let stopped = false;
  let watching = false;

  topbar.replaceChildren(
    h('button', { class: 'btn btn-quiet', onClick: () => navigate('/') }, '‹ Spaces'),
    h('div', { class: 'title-block' }, h('span', { class: 't', id: 'proc-name' }, 'Processing')),
    h('div', { class: 'spacer' }));

  const orb = h('div', { class: 'orb', 'aria-hidden': 'true' });
  const count = h('p', { class: 'step-count' });
  const head = h('h1', { class: 'proc-head' }, 'Preparing');
  const sub = h('p', { class: 'proc-sub muted' });
  const time = h('p', { class: 'elapsed' });
  const list = h('ol', { class: 'phase-list' });
  const failure = h('div', { class: 'notice hidden', role: 'alert' });

  main.replaceChildren(h('div', { class: 'processing' },
    orb, count, head, sub, list, time, failure));
  paintPending();
  announce('Processing started');

  const tick = setInterval(() => {
    if (!stopped) time.textContent = `${elapsed((Date.now() - started) / 1000)} elapsed`;
  }, 1000);

  function paintPending() {
    list.replaceChildren();
    PHASES.forEach((phase, index) => {
      const state = index === 0 ? 'active' : 'pending';
      list.append(h('li', { class: 'phase', dataset: { state } },
        h('span', { class: 'glyph', 'aria-hidden': 'true' }, index === 0 ? '→' : '·'),
        h('span', {}, phase.title)));
    });
    count.textContent = `Step 1 of ${PHASES.length}`;
    head.textContent = PHASES[0].title;
    sub.textContent = PHASES[0].blurb;
  }

  async function watch(scanId) {
    if (watching || stopped) return;
    watching = true;
    while (!stopped) {
      let scan;
      try {
        scan = await api.get(`/api/scans/${scanId}/status`);
      } catch (error) {
        stopped = true;
        failure.classList.remove('hidden');
        failure.replaceChildren(
          h('h3', {}, 'Lost contact with the local service'),
          h('p', { class: 'muted' }, error.message));
        break;
      }
      const name = document.getElementById('proc-name');
      if (name) name.textContent = scan.label || 'Processing';
      paint(scan);

      if (scan.status === 'complete') {
        stopped = true;
        announce('Processing complete');
        navigate(`/scans/${scanId}`, true);
        break;
      }
      if (scan.status === 'failed') { stopped = true; showFailure(scan); break; }
      await new Promise((resolve) => setTimeout(resolve, 900));
    }
    clearInterval(tick);
  }

  function paint(scan) {
    const byName = Object.fromEntries(scan.stages.map((s) => [s.name, s]));
    let activeIndex = PHASES.length - 1;

    list.replaceChildren();
    PHASES.forEach((phase, index) => {
      const stages = phase.stages.map((name) => byName[name]).filter(Boolean);
      const anyFailed = stages.some((s) => s.state === 'failed');
      const allDone = stages.length > 0 && stages.every(
        (s) => ['complete', 'skipped', 'not_applicable'].includes(s.state));
      const anyRunning = stages.some((s) => s.state === 'running');
      let state = 'pending';
      if (anyFailed) state = 'failed';
      else if (allDone) state = 'done';
      else if (anyRunning) state = 'active';

      if ((state === 'active' || state === 'pending') && activeIndex === PHASES.length - 1
          && !allDone) {
        activeIndex = Math.min(activeIndex, index);
      }
      const glyph = { done: '✓', active: '→', failed: '×' }[state] || '·';
      list.append(h('li', { class: 'phase', dataset: { state } },
        h('span', { class: 'glyph', 'aria-hidden': 'true' }, glyph),
        h('span', {}, phase.title)));
    });

    const current = PHASES[Math.min(activeIndex, PHASES.length - 1)];
    count.textContent = `Step ${Math.min(activeIndex + 1, PHASES.length)} of ${PHASES.length}`;
    head.textContent = current.title;
    sub.textContent = current.blurb;
  }

  function showFailure(scan) {
    orb.classList.add('hidden');
    head.textContent = 'We couldn’t process this capture';
    sub.textContent = humanFailure(scan);
    count.textContent = '';
    time.textContent = '';
    failure.classList.remove('hidden');
    remount(failure,
      h('h3', {}, 'What happened'),
      h('p', { class: 'muted' }, humanFailure(scan)),
      h('div', { style: 'display:flex;gap:8px;flex-wrap:wrap' },
        h('button', { class: 'btn btn-primary', onClick: () => navigate('/scans/new') },
          'Try another capture'),
        h('button', { class: 'btn btn-secondary', onClick: () => navigate('/') }, 'Back to spaces')),
      h('details', { class: 'disclosure' },
        h('summary', {}, 'View processing details'),
        h('div', { class: 'body mono' },
          scan.failureClass ? h('div', {}, scan.failureClass) : null,
          h('div', {}, scan.error || 'No further detail was recorded.'))));
    announce('Processing failed');
  }

  function failStart(error) {
    const message = error instanceof ApiError ? error.message : String(error);
    showFailure({ error: message, failureClass: 'INTAKE_FAILURE' });
    head.textContent = 'That capture could not be added';
    sub.textContent = humanFailure({ error: message, failureClass: 'INTAKE_FAILURE' });
  }

  function humanFailure(scan) {
    const detail = scan.error || '';
    if (scan.failureClass === 'INTAKE_FAILURE' || /too many fields/i.test(detail)) {
      if (/too many fields|too many files/i.test(detail)) {
        return 'This export has too many files to send as a loose folder. Zip the '
          + 'Stray folder and choose that zip instead.';
      }
      if (/failed to fetch/i.test(detail)) {
        return 'The local service stopped answering while the folder was being '
          + 'copied. Confirm it is running, or send a zip of the capture.';
      }
      return 'The folder could not be copied onto this machine, so processing '
        + 'did not start.';
    }
    if (scan.failureClass === 'CONNECTOR_FAILURE') {
      if (scan.error && /does not exist/i.test(scan.error)) {
        return 'That folder could not be found on this machine, so no room could '
          + 'be built from it.';
      }
      return 'The colour, depth and camera-pose records in this capture could not be '
        + 'read or aligned, so no room could be built from it.';
    }
    if (scan.failureClass === 'GEOMETRY_GENERALIZATION_FAILURE') {
      return 'The capture was read successfully, but not enough room structure could '
        + 'be recovered from it to build a spatial model.';
    }
    return 'Something went wrong after the room was built. The details below say where.';
  }

  return {
    isStopped: () => stopped,
    stop() { stopped = true; clearInterval(tick); },
    watch,
    showFailure,
    failStart,
  };
}

export async function renderProcessing(main, topbar, navigate, scanId) {
  const view = presentProcessing(main, topbar, navigate);
  view.watch(scanId);
  return () => view.stop();
}
