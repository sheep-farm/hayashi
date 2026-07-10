// Hayashi Playground — loads the wasm module and wires up the editor.

const EXAMPLES = {
  ols: `// OLS regression with dict-based dataset
let d = {"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.0, 4.0, 5.0, 4.0, 5.0]}
let df = dataframe(d)

ols(y ~ x, df)

// Summary statistics
summarize(df)`,

  logit: `// Logit model with dict-based dataset
let d = {
  "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
  "y": [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
let df = dataframe(d)

logit(y ~ x, df)`,

  summarize: `// Descriptive statistics and correlation
let d = {
  "wage":  [1000.0, 1200.0, 1500.0, 1800.0, 2200.0, 2800.0, 3500.0],
  "educ":  [8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
  "exper": [1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 15.0]
}
let df = dataframe(d)

summarize(df)
correlate(df, wage, educ, exper)`,

  pipes: `// Pipes and closures
let d = {"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.0, 4.0, 5.0, 4.0, 5.0]}
let df = dataframe(d)

// Pipe into summarize
df |> summarize()

// Closures and pipe chains on lists
let r = [1, 2, 3] |> map(|x| x * 10)
display r[0]
display r[1]

let s = [3, 1, 2] |> sort |> reverse
display s[0]`,

  fstring: `// F-strings and format specifiers
let ticker = "AAPL"
let price = 178.50
display f"{ticker}: \${price}"

let pi = 3.14159
display f"{pi:.2f}"

let a = 1
let b = 2
display f"{a} + {b} = {a + b}"

// Scientific notation
display f"{0.00123:.2e}"`,

  rolling: `// Rolling regression and time series
let d = {
  "Y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
  "X": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
  "date": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
}
let df = dataframe(d)

tsset df date

// Rolling OLS with window=5
let roll = rolling(Y ~ X, df, window=5)
print(roll)

// Generate lagged returns
generate df ret = (X - L.X) / L.X
summarize(df)`,
};

let wasmModule = null;
let isLoaded = false;
let isLoading = false;

const runBtn = document.getElementById('run-btn');
const clearBtn = document.getElementById('clear-btn');
const codeInput = document.getElementById('code-input');
const codeHighlight = document.getElementById('code-highlight').querySelector('code');
const codeHighlightPre = document.getElementById('code-highlight');
const outputArea = document.getElementById('output-area').querySelector('code');
const runStatus = document.getElementById('run-status');

// ── Syntax highlighting ──────────────────────────────────────────────

const HAY_KEYWORDS = new Set([
  'let', 'const', 'if', 'else', 'for', 'in', 'while', 'fn', 'return', 'match',
  'import', 'export', 'as', 'true', 'false', 'null', 'break', 'continue',
  'input', 'end', 'load', 'save', 'generate', 'replace', 'drop', 'keep',
  'predict', 'summarize', 'tabulate', 'correlate', 'describe', 'list',
  'count', 'tsset', 'xtset', 'quietly', 'try', 'catch', 'display', 'print',
]);

const HAY_FUNCS = new Set([
  'ols', 'reg', 'logit', 'probit', 'poisson', 'nbreg', 'tobit', 'qreg',
  'iv', 'fe', 're', 'ab', 'sysgmm', 'pcse', 'xtgls', 'heckman', 'cox',
  'lasso', 'ridge', 'elasticnet', 'garch', 'arima', 'var', 'vecm',
  'did', 'gmm', 'bootstrap', 'estat', 'nlcom', 'esttab', 'eststo',
  'rolling', 'lag', 'lead', 'diff', 'pct_change', 'mean', 'sd', 'min',
  'max', 'sum', 'median', 'rowmean', 'rowsum', 'cumsum', 'rank',
  'map', 'filter', 'select', 'arrange', 'mutate', 'group_by',
  'dropna', 'fillna', 'ffill', 'bfill', 'interpolate', 'winsor',
  'encode', 'collapse', 'merge', 'tidy', 'glance', 'centile', 'ci',
  'ttest', 'codebook', 'duplicates', 'format', 'assert', 'capture',
  'graph', 'hist', 'scatter', 'line', 'dataframe', 'sort', 'reverse',
  'len', 'dict_merge', 'dict_set',
]);

function highlightHayashi(text) {
  // Tokenize: strings, f-strings, comments, numbers, operators, identifiers
  const regex = /(f"(?:[^"\\]|\\.)*"|"(?:[^"\\]|\\.)*"|\/\/[^\n]*|\b\d+(?:\.\d+)?\b|~>|=>|\|>|~|[a-zA-Z_]\w*|\s+|.)/gs;
  let result = '';
  let m;
  while ((m = regex.exec(text)) !== null) {
    const tok = m[0];
    if (!tok) continue;
    if (tok[0] === 'f' && tok[1] === '"') {
      result += '<span class="token-string">' + escapeHtml(tok) + '</span>';
    } else if (tok[0] === '"') {
      result += '<span class="token-string">' + escapeHtml(tok) + '</span>';
    } else if (tok.startsWith('//')) {
      result += '<span class="token-comment">' + escapeHtml(tok) + '</span>';
    } else if (/^\d/.test(tok)) {
      result += '<span class="token-number">' + escapeHtml(tok) + '</span>';
    } else if (/^(~>|=>|\|>|~)$/.test(tok)) {
      result += '<span class="token-operator">' + escapeHtml(tok) + '</span>';
    } else if (/^[a-zA-Z_]/.test(tok)) {
      if (HAY_KEYWORDS.has(tok)) {
        result += '<span class="token-keyword">' + escapeHtml(tok) + '</span>';
      } else if (HAY_FUNCS.has(tok)) {
        result += '<span class="token-func">' + escapeHtml(tok) + '</span>';
      } else {
        result += escapeHtml(tok);
      }
    } else {
      result += escapeHtml(tok);
    }
  }
  return result;
}

function updateHighlight() {
  codeHighlight.innerHTML = highlightHayashi(codeInput.value);
}

// Sync scroll: textarea → highlight overlay
codeInput.addEventListener('scroll', () => {
  codeHighlightPre.scrollTop = codeInput.scrollTop;
  codeHighlightPre.scrollLeft = codeInput.scrollLeft;
});

// Update highlight on input
codeInput.addEventListener('input', updateHighlight);

// Buffer for capturing output via JS callback
let outputBuffer = '';

// Load wasm module on page load
let wasmRun = null;
let wasmSetCallback = null;

async function loadWasm() {
  if (isLoaded || isLoading) return;
  isLoading = true;

  outputArea.innerHTML = '<span class="output-placeholder">Loading Hayashi WebAssembly… (~6 MB)</span>';

  try {
    const init = await import('../wasm/hayashi_lang.js');
    // Fetch wasm with cache-busting to avoid stale module
    const wasmResp = await fetch('../wasm/hayashi_lang_bg.wasm?v=' + Date.now());
    const wasmBytes = await wasmResp.arrayBuffer();
    await init.default({ module_or_path: wasmBytes });
    // Named exports live on the module object, not on init.default() return
    wasmRun = init.run_hayashi;
    wasmSetCallback = init.set_print_callback;
    // Set up the print callback — Rust's println! routes here via print_output
    wasmSetCallback((text) => {
      outputBuffer += text;
    });
    isLoaded = true;
    isLoading = false;
    runBtn.disabled = false;
    outputArea.innerHTML = '<span class="output-placeholder">// Press Run to execute code (Ctrl+Enter)</span>';
  } catch (err) {
    isLoading = false;
    outputArea.innerHTML = `<span class="output-error">Failed to load WebAssembly: ${escapeHtml(err.message)}</span>`;
  }
}

function runCode() {
  if (!isLoaded) return;

  const code = codeInput.value.trim();
  if (!code) return;

  runStatus.textContent = 'running…';
  runStatus.className = 'playground-status running';
  outputBuffer = '';

  let errorMsg = '';

  try {
    errorMsg = wasmRun(code);
  } catch (err) {
    errorMsg = String(err);
  }

  if (errorMsg) {
    const output = outputBuffer;
    outputArea.innerHTML = '';
    if (output) {
      outputArea.appendChild(document.createTextNode(output + '\n'));
    }
    const errNode = document.createElement('span');
    errNode.className = 'output-error';
    errNode.textContent = errorMsg;
    outputArea.appendChild(errNode);
    runStatus.textContent = 'error';
    runStatus.className = 'playground-status error';
  } else {
    outputArea.textContent = outputBuffer || '(no output)';
    runStatus.textContent = 'ok';
    runStatus.className = 'playground-status ok';
  }
}

function clearOutput() {
  outputArea.innerHTML = '<span class="output-placeholder">// Press Run to execute code</span>';
  runStatus.textContent = '';
  runStatus.className = 'playground-status';
}

function loadExample(name) {
  const code = EXAMPLES[name];
  if (code) {
    codeInput.value = code;
    updateHighlight();
    outputArea.innerHTML = '<span class="output-placeholder">// Press Run to execute code</span>';
    runStatus.textContent = '';
    runStatus.className = 'playground-status';
  }
}

function escapeHtml(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

// Event listeners
runBtn.addEventListener('click', runCode);
clearBtn.addEventListener('click', clearOutput);

document.querySelectorAll('.example-card').forEach((card) => {
  card.addEventListener('click', () => {
    loadExample(card.dataset.example);
  });
});

// Ctrl/Cmd+Enter to run
codeInput.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault();
    runCode();
  }
});

// Tab key inserts spaces
codeInput.addEventListener('keydown', (e) => {
  if (e.key === 'Tab') {
    e.preventDefault();
    const start = codeInput.selectionStart;
    const end = codeInput.selectionEnd;
    codeInput.value = codeInput.value.substring(0, start) + '  ' + codeInput.value.substring(end);
    codeInput.selectionStart = codeInput.selectionEnd = start + 2;
  }
});

// Start loading wasm immediately
loadWasm();
