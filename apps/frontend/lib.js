/* Shared helpers: API access, formatting, and small DOM builders.
 * Every value rendered by this app comes from the API. Nothing here invents a
 * measurement, a status, or a classification. */

export const api = {
  async get(path) {
    const r = await fetch(path);
    if (!r.ok) throw new ApiError(await detail(r), r.status);
    return r.json();
  },
  async text(path) {
    const r = await fetch(path);
    if (!r.ok) throw new ApiError(await detail(r), r.status);
    return r.text();
  },
  async post(path, body) {
    const r = await fetch(path, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (!r.ok) throw new ApiError(await detail(r), r.status);
    return r.json();
  },
  async postForm(path, form) {
    const r = await fetch(path, { method: 'POST', body: form });
    if (!r.ok) throw new ApiError(await detail(r), r.status);
    return r.json();
  },
  async patch(path, body) {
    const r = await fetch(path, {
      method: 'PATCH',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new ApiError(await detail(r), r.status);
    return r.json();
  },
  async delete(path) {
    const r = await fetch(path, { method: 'DELETE' });
    if (!r.ok) throw new ApiError(await detail(r), r.status);
    if (r.status === 204) return null;
    const text = await r.text();
    return text ? JSON.parse(text) : null;
  },
};

export class ApiError extends Error {
  constructor(message, status) { super(message); this.status = status; }
}

async function detail(response) {
  try {
    const body = await response.json();
    return body.detail || response.statusText;
  } catch { return response.statusText; }
}

/* ------------------------------------------------------------- formatting */

export function metres(value, digits = 2) {
  return value === null || value === undefined ? '—' : `${value.toFixed(digits)} m`;
}
export function squareMetres(value, digits = 2) {
  return value === null || value === undefined ? '—' : `${value.toFixed(digits)} m²`;
}
export function dimensions(summary) {
  if (summary?.length_m == null || summary?.width_m == null) return null;
  return `${summary.length_m.toFixed(2)} × ${summary.width_m.toFixed(2)} m`;
}
export function relativeDate(iso) {
  if (!iso) return '';
  const then = new Date(iso);
  const days = Math.floor((Date.now() - then.getTime()) / 86400000);
  if (days === 0) return 'Today';
  if (days === 1) return 'Yesterday';
  if (days < 7) return `${days} days ago`;
  return then.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}
export function elapsed(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

/* Human words for machine states. Raw identifiers stay in diagnostics only. */
export function statusWord(scan) {
  if (scan.status === 'failed') return 'Failed';
  if (scan.status === 'processing') return 'Processing';
  if (scan.status === 'created') return 'Ready to process';
  if (scan.summary?.needsReviewCount > 0) return 'Needs review';
  return 'Complete';
}
export function statusTone(scan) {
  if (scan.status === 'failed') return 'failed';
  if (scan.status === 'processing') return 'processing';
  if (scan.summary?.needsReviewCount > 0) return 'review';
  return 'complete';
}
export function classificationWord(classification) {
  return {
    public_development_fixture: 'Development sample',
    final_private_capture: 'Final capture',
    baseline_fallback: 'Baseline fallback',
  }[classification] || classification;
}
export function observationWord(state) {
  return {
    directly_observed: 'Directly observed',
    partially_observed: 'Partially observed',
    inferred: 'Inferred',
    unresolved: 'Unresolved',
  }[state] || state;
}
export function evidenceQualityWord(label) {
  return { high: 'High', medium: 'Medium', low: 'Low', unresolved: 'Unresolved' }[label]
    || label;
}
export function conditionWord(value) {
  return String(value || '').replace(/_/g, ' ').replace(/^\w/, (c) => c.toUpperCase());
}
export function surfaceName(id) {
  const match = /^([a-z]+)-0*(\d+)$/.exec(id || '');
  if (!match) return id;
  const noun = match[1].charAt(0).toUpperCase() + match[1].slice(1);
  return `${noun} ${match[2].padStart(3, '0')}`;
}

/* ------------------------------------------------------------------- DOM */

export function h(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') node.className = value;
    else if (key === 'html') node.innerHTML = value;
    else if (key === 'text') node.textContent = value;
    else if (key.startsWith('on') && typeof value === 'function') {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (key === 'dataset') Object.assign(node.dataset, value);
    else node.setAttribute(key, value === true ? '' : String(value));
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

/* Element.append() and replaceChildren() stringify null and undefined, so a
 * conditional child renders as the literal text "null". h() already filters its
 * children; these give the same guarantee when mounting onto an existing node. */
function usable(children) {
  return children.flat(Infinity)
    .filter((child) => child !== null && child !== undefined && child !== false)
    .map((child) => (child instanceof Node ? child : document.createTextNode(String(child))));
}

export function mount(parent, ...children) {
  parent.append(...usable(children));
  return parent;
}

export function remount(parent, ...children) {
  parent.replaceChildren(...usable(children));
  return parent;
}

export function badge(text, tone, plain = false) {
  return h('span', { class: `badge badge-${tone}${plain ? ' badge-plain' : ''}` }, text);
}

export function stateTag(state) {
  return h('span', { class: 'state-tag', dataset: { state } },
    h('span', { class: 'line', 'aria-hidden': 'true' }),
    observationWord(state));
}

export function metric(key, value, small = false) {
  return h('div', { class: 'metric' },
    h('span', { class: 'k' }, key),
    h('span', { class: `v${small ? ' sm' : ''}` }, value));
}

export function kv(key, value) {
  return h('div', { class: 'kv-row' },
    h('span', { class: 'k' }, key),
    h('span', { class: 'v' }, value));
}

export function disclosure(summaryText, ...body) {
  return h('details', { class: 'disclosure' },
    h('summary', {}, summaryText),
    h('div', { class: 'body' }, ...body));
}

export function announce(message) {
  const live = document.getElementById('live');
  if (live) live.textContent = message;
}

export function hostPlatform() {
  const platform = navigator.userAgentData?.platform || navigator.platform || '';
  const ua = navigator.userAgent || '';
  if (/Win/i.test(platform) || /Windows/i.test(ua)) return 'windows';
  if (/Mac|iPhone|iPad/i.test(platform) || /Mac OS|iPhone|iPad/i.test(ua)) return 'mac';
  return 'other';
}
