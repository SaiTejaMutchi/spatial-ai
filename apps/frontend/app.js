/* Router and entry point.
 *
 * Four screens, real URLs, browser history. A completed space reopens from its
 * saved artifacts; it is never reprocessed just because someone navigated to it.
 */

import { announce, hostPlatform } from './lib.js';

document.documentElement.dataset.platform = hostPlatform();
import { renderAddCapture, renderLibrary, renderProcessing } from './screens.js?v=19';
import { renderResult } from './result.js?v=18';

const main = document.getElementById('main');
const topbar = document.getElementById('topbar');
let teardown = null;

const ROUTES = [
  [/^\/$/, async () => renderLibrary(main, topbar, navigate)],
  [/^\/scans\/new$/, async () => renderAddCapture(main, topbar, navigate)],
  [/^\/scans\/([^/]+)\/processing$/,
    async (id) => renderProcessing(main, topbar, navigate, id)],
  [/^\/scans\/([^/]+)$/, async (id) => renderResult(main, topbar, navigate, id)],
];

export function navigate(path, replace = false) {
  if (replace) history.replaceState({}, '', path);
  else history.pushState({}, '', path);
  render();
}

async function render() {
  if (teardown) { teardown(); teardown = null; }
  const path = location.pathname || '/';

  for (const [pattern, handler] of ROUTES) {
    const match = pattern.exec(path);
    if (!match) continue;
    main.replaceChildren();
    try {
      teardown = await handler(...match.slice(1)) || null;
    } catch (error) {
      main.replaceChildren(errorScreen(error));
    }
    main.focus({ preventScroll: true });
    return;
  }

  // Unknown path: go home rather than showing a dead end.
  navigate('/', true);
}

function errorScreen(error) {
  const page = document.createElement('div');
  page.className = 'page';
  page.innerHTML = `
    <div class="notice" role="alert">
      <h3>Something went wrong</h3>
      <p class="muted"></p>
      <p><a class="btn btn-secondary" href="/">Back to spaces</a></p>
    </div>`;
  page.querySelector('p.muted').textContent = error?.message || String(error);
  announce('An error occurred');
  return page;
}

// Intercept in-app links so navigation stays client-side.
document.addEventListener('click', (event) => {
  const anchor = event.target.closest('a[href^="/"]');
  if (!anchor || anchor.hasAttribute('download') || anchor.target) return;
  event.preventDefault();
  navigate(anchor.getAttribute('href'));
});

window.addEventListener('popstate', render);
render();
