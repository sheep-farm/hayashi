document.addEventListener('DOMContentLoaded', () => {
  // Mobile menu toggle
  const menuToggle = document.querySelector('.menu-toggle');
  const mainNav = document.querySelector('.main-nav');

  if (menuToggle && mainNav) {
    menuToggle.addEventListener('click', () => {
      const isOpen = mainNav.classList.toggle('open');
      menuToggle.setAttribute('aria-expanded', String(isOpen));
    });
  }

  // Copy buttons for install snippets
  document.querySelectorAll('.install-copy-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const installBox = btn.closest('.plugin-install');
      const codeEl = installBox ? installBox.querySelector('code') : null;
      const text = codeEl ? codeEl.textContent : (btn.getAttribute('data-copy') || '');
      navigator.clipboard.writeText(text).then(() => {
        btn.classList.add('copied');
        const svg = btn.innerHTML;
        btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';
        setTimeout(() => {
          btn.innerHTML = svg;
          btn.classList.remove('copied');
        }, 1500);
      });
    });
  });

  // Hayashi syntax highlighting
  highlightHayashi();

  // Fetch latest release version from GitHub
  fetchLatestVersion();
});

function fetchLatestVersion() {
  const el = document.getElementById('latest-version');
  if (!el) return;

  fetch('https://api.github.com/repos/sheep-farm/hayashi/releases/latest')
    .then((resp) => resp.json())
    .then((data) => {
      if (!data || !data.tag_name) return;

      el.textContent = data.tag_name;

      const assets = data.assets || [];
      const platformAssets = {
        linux: assets.find((a) => a.name.includes('x86_64-unknown-linux-gnu')),
        macos: assets.find((a) => a.name.includes('apple-darwin')),
        windows: assets.find((a) => a.name.includes('x86_64-pc-windows-msvc')),
      };

      const mainLink = document.getElementById('latest-release-link');
      const userPlatform = detectPlatform();
      const userAsset = platformAssets[userPlatform];
      if (mainLink && userAsset && userAsset.browser_download_url) {
        mainLink.href = userAsset.browser_download_url;
        mainLink.textContent = `Download for ${capitalize(userPlatform)}`;
      }

      document.querySelectorAll('.release-platform-link').forEach((link) => {
        const platform = link.getAttribute('data-platform');
        const asset = platformAssets[platform];
        if (asset && asset.browser_download_url) {
          link.href = asset.browser_download_url;
        }
      });
    })
    .catch(() => {
      // Keep the static fallback version and links if the request fails
    });
}

function detectPlatform() {
  const os = navigator.platform.toLowerCase();
  const ua = navigator.userAgent.toLowerCase();
  if (os.includes('win') || ua.includes('windows')) return 'windows';
  if (os.includes('mac') || ua.includes('macintosh') || ua.includes('mac os')) return 'macos';
  return 'linux';
}

function capitalize(s) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function highlightHayashi() {
  const keywords = new Set([
    // control flow and core
    'let', 'if', 'else', 'for', 'in', 'while', 'fn', 'return', 'match', 'import', 'export',
    'true', 'false', 'null', 'as',
    // estimators
    'ols', 'reg', 'logit', 'probit', 'poisson', 'nbreg', 'tobit', 'qreg', 'iv',
    'fe', 're', 'ab', 'sysgmm', 'pcse', 'xtgls', 'heckman', 'cox',
    'lasso', 'ridge', 'elasticnet', 'garch', 'arima', 'var', 'vecm', 'did', 'gmm',
    // data and post-estimation
    'load', 'save', 'predict', 'margins', 'test', 'bootstrap', 'estat', 'nlcom', 'esttab',
    'summarize', 'tabulate', 'ttest', 'correlate', 'list', 'describe',
    'generate', 'replace', 'drop', 'keep', 'dropna', 'encode', 'winsor',
    'filter', 'mutate', 'select', 'arrange', 'group_by',
    // CLI subcommands
    'hay', 'install', 'update', 'remove', 'list', 'validate', 'dist-update',
  ]);

  document.querySelectorAll('code.language-hay').forEach((code) => {
    // Skip if already highlighted
    if (code.dataset.highlighted) return;
    code.dataset.highlighted = 'true';

    const text = code.textContent;
    const highlighted = text
      .split(/("(?:[^"\\]|\\.)*")|(\/\/[^\n]*)|(\b\d+(?:\.\d+)?\b)|(~>|\|>|~|=>)|([a-zA-Z_](?:[a-zA-Z0-9_-]*[a-zA-Z0-9_])?)|(\s+)|(.)/g)
      .filter((part) => part !== undefined)
      .map((part) => {
        if (!part) return '';
        // String
        if (part[0] === '"') {
          return `<span class="token-string">${escapeHtml(part)}</span>`;
        }
        // Comment
        if (part.startsWith('//')) {
          return `<span class="token-comment">${escapeHtml(part)}</span>`;
        }
        // Number
        if (/^\d+(?:\.\d+)?$/.test(part)) {
          return `<span class="token-number">${escapeHtml(part)}</span>`;
        }
        // Operator
        if (/^(~>|\|>|~|=>)$/.test(part)) {
          return `<span class="token-operator">${escapeHtml(part)}</span>`;
        }
        // Identifier / keyword
        if (/^[a-zA-Z_][a-zA-Z0-9_-]*$/.test(part)) {
          if (keywords.has(part)) {
            return `<span class="token-keyword">${escapeHtml(part)}</span>`;
          }
        }
        return escapeHtml(part);
      })
      .join('');

    code.innerHTML = highlighted;
  });
}

function escapeHtml(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}
