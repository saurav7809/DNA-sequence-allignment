/* ═══════════════════════════════════════════════════════════
   DNA Alignment Dashboard  —  JavaScript
   ═══════════════════════════════════════════════════════════ */

const API = 'http://localhost:5000/api';

// ── State ──────────────────────────────────────────────────
let currentMatrix   = null;
let currentDiagonals = null;
let animTimer       = null;
let animDiagIdx     = 0;
let chartTime       = null;
let chartSpeedup    = null;

// ── DNA Base Colors ────────────────────────────────────────
const BASE_COLOR = { A:'#48bb78', C:'#63b3ed', G:'#f6e05e', T:'#fc8181' };

function baseSpan(ch) {
  const c = BASE_COLOR[ch] || '#e2e8f0';
  return `<span style="color:${c}">${ch}</span>`;
}

// ── Input colorizer ────────────────────────────────────────
function colorizeInput(input) {
  const id = input.id;
  const val = input.value.toUpperCase();
  input.value = val;
  const preview = document.getElementById(id + '-colored');
  const lenEl   = document.getElementById(id === 'seq1' ? 'len1' : 'len2');
  preview.innerHTML = [...val].map(baseSpan).join('');
  lenEl.textContent = `Length: ${val.length}`;
}

// ── On load ────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  colorizeInput(document.getElementById('seq1'));
  colorizeInput(document.getElementById('seq2'));

  // Size chip toggles
  document.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => chip.classList.toggle('active'));
  });
});

// ── Generate random sequence ───────────────────────────────
async function generateRandom(num) {
  const len = Math.floor(Math.random() * 15) + 6;
  const res = await fetch(`${API}/random`, {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({length: len})
  });
  const d = await res.json();
  const input = document.getElementById(`seq${num}`);
  input.value = num === 1 ? d.seq1 : d.seq2;
  colorizeInput(input);
}

// ── Run Alignment ──────────────────────────────────────────
async function runAlignment() {
  const seq1     = document.getElementById('seq1').value.trim().toUpperCase();
  const seq2     = document.getElementById('seq2').value.trim().toUpperCase();
  const match    = parseInt(document.getElementById('scoreMatch').value);
  const mismatch = parseInt(document.getElementById('scoreMismatch').value);
  const gap      = parseInt(document.getElementById('scoreGap').value);

  if (!seq1 || !seq2) { showError('Please enter both sequences.'); return; }
  if (!/^[ACGT]+$/.test(seq1) || !/^[ACGT]+$/.test(seq2)) {
    showError('Sequences may only contain A, C, G, T.'); return; }

  setAlignLoading(true);
  hideError();

  try {
    const res = await fetch(`${API}/align`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({seq1, seq2, match, mismatch, gap})
    });
    const data = await res.json();
    if (!res.ok) { showError(data.error || 'Alignment failed.'); return; }
    showResults(data);
    buildMatrix(data);
  } catch(e) {
    showError('Cannot reach server. Make sure server.py is running on port 5000.');
  } finally {
    setAlignLoading(false);
  }
}

function setAlignLoading(on) {
  document.getElementById('btnAlignText').style.display = on ? 'none' : '';
  document.getElementById('btnSpinner').style.display   = on ? '' : 'none';
  document.getElementById('btnAlign').disabled = on;
}

function showError(msg) {
  const box = document.getElementById('errorBox');
  box.textContent = '⚠️ ' + msg;
  box.style.display = '';
}
function hideError() { document.getElementById('errorBox').style.display = 'none'; }

// ── Show Results ───────────────────────────────────────────
function showResults(data) {
  document.getElementById('sec-results').style.display = '';
  document.getElementById('cpuScore').textContent = data.cpu_score;
  document.getElementById('gpuScore').textContent = data.wf_score;
  document.getElementById('cpuTime').textContent  = data.cpu_ms + ' ms';
  document.getElementById('gpuTime').textContent  = data.wf_ms  + ' ms';
  document.getElementById('speedupVal').textContent = data.speedup + '×';

  const card = document.getElementById('validationCard');
  const icon = document.getElementById('valIcon');
  const text = document.getElementById('valText');

  if (data.validation) {
    card.classList.remove('fail');
    icon.textContent = '✓';
    text.textContent = 'MATCH';
  } else {
    card.classList.add('fail');
    icon.textContent = '✗';
    text.textContent = 'MISMATCH';
  }

  document.getElementById('scoringInfo').innerHTML =
    `<span>Match:</span> +${data.match} &nbsp;|&nbsp; <span>Mismatch:</span> ${data.mismatch} &nbsp;|&nbsp; <span>Gap:</span> ${data.gap}`;

  // Scroll to results
  document.getElementById('sec-results').scrollIntoView({behavior:'smooth', block:'nearest'});
}

// ── Build DP Matrix ────────────────────────────────────────
function buildMatrix(data) {
  currentMatrix    = data.matrix;
  currentDiagonals = data.diagonals;
  animDiagIdx      = 0;

  const seq1 = data.seq1;
  const seq2 = data.seq2;
  const m    = seq1.length;
  const n    = seq2.length;

  // Build table
  let html = '<table class="matrix-table"><thead><tr><th></th><th class="col-header">—</th>';
  for (let j = 0; j < n; j++) {
    html += `<th class="col-header"><span class="base-${seq2[j]}">${seq2[j]}</span></th>`;
  }
  html += '</tr></thead><tbody>';

  for (let i = 0; i <= m; i++) {
    const rowLabel = i === 0 ? '—' : `<span class="base-${seq1[i-1]}">${seq1[i-1]}</span>`;
    html += `<tr><td class="row-header">${rowLabel}</td>`;
    for (let j = 0; j <= n; j++) {
      const val = currentMatrix[i][j];
      const isFinal = (i === m && j === n);
      const isInit  = (i === 0 || j === 0);
      let cls = isInit ? 'cell-init' : '';
      if (isFinal) cls = 'cell-final';
      html += `<td class="matrix-cell ${cls}" id="cell-${i}-${j}" data-val="${val}">${val}</td>`;
    }
    html += '</tr>';
  }
  html += '</tbody></table>';

  document.getElementById('matrixWrap').innerHTML = html;
  document.getElementById('sec-matrix').style.display = '';
  document.getElementById('diagInfo').textContent = 'Press ▶ Play to animate the wavefront sweep';
  document.getElementById('btnPlay').disabled = false;

  document.getElementById('sec-matrix').scrollIntoView({behavior:'smooth', block:'nearest'});
}

// ── Animation ──────────────────────────────────────────────
function playAnimation() {
  if (!currentDiagonals) return;

  // Clear previous animation
  clearInterval(animTimer);
  resetMatrix();

  const btn = document.getElementById('btnPlay');
  btn.disabled = true;
  animDiagIdx = 0;

  function step() {
    // Clear previous active
    document.querySelectorAll('.cell-active').forEach(el => {
      el.classList.remove('cell-active');
      const i = parseInt(el.id.split('-')[1]);
      const j = parseInt(el.id.split('-')[2]);
      const isFinal = (i === currentMatrix.length-1 && j === currentMatrix[0].length-1);
      el.classList.add(isFinal ? 'cell-final' : 'cell-done');
      el.textContent = el.dataset.val;
    });

    if (animDiagIdx >= currentDiagonals.length) {
      // Final cell
      const m = currentMatrix.length-1, n = currentMatrix[0].length-1;
      const fc = document.getElementById(`cell-${m}-${n}`);
      if (fc) { fc.classList.remove('cell-done'); fc.classList.add('cell-final'); }
      document.getElementById('diagInfo').textContent =
        `✓ Complete! All ${currentDiagonals.length} anti-diagonals processed. Final score = ${currentMatrix[m][n]}`;
      btn.disabled = false;
      return;
    }

    const diag = currentDiagonals[animDiagIdx];
    const d    = animDiagIdx + 2;  // diagonal index (i+j)
    document.getElementById('diagInfo').textContent =
      `Anti-diagonal ${d} (${diag.length} cell${diag.length>1?'s':''} computed in parallel) → CUDA kernel launch #${animDiagIdx+1}`;

    diag.forEach(([i, j]) => {
      const cell = document.getElementById(`cell-${i}-${j}`);
      if (cell) {
        cell.classList.remove('cell-init','cell-done');
        cell.classList.add('cell-active');
        cell.textContent = currentMatrix[i][j];
      }
    });

    animDiagIdx++;
    const delay = 850 - parseInt(document.getElementById('animSpeed').value);
    animTimer = setTimeout(step, Math.max(50, delay));
  }

  step();
}

function resetMatrix() {
  clearTimeout(animTimer);
  animDiagIdx = 0;
  document.querySelectorAll('.matrix-cell').forEach(el => {
    const i = parseInt(el.id.split('-')[1]);
    const j = parseInt(el.id.split('-')[2]);
    el.classList.remove('cell-active','cell-done','cell-final');
    const isInit = (i === 0 || j === 0);
    el.classList.toggle('cell-init', isInit);
    el.textContent = el.dataset.val;
  });
  document.getElementById('diagInfo').textContent = 'Press ▶ Play to animate the wavefront sweep';
  document.getElementById('btnPlay').disabled = false;
}

// ── Benchmark ──────────────────────────────────────────────
async function runBenchmark() {
  const sizes = [...document.querySelectorAll('.chip.active')]
    .map(c => parseInt(c.dataset.size)).sort((a,b) => a-b);
  if (!sizes.length) { alert('Select at least one size.'); return; }

  const trials  = parseInt(document.getElementById('trials').value);
  const match   = parseInt(document.getElementById('scoreMatch').value);
  const mismatch= parseInt(document.getElementById('scoreMismatch').value);
  const gap     = parseInt(document.getElementById('scoreGap').value);

  setBenchLoading(true);

  try {
    const res = await fetch(`${API}/benchmark`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({sizes, trials, match, mismatch, gap})
    });
    const data = await res.json();
    showBenchResults(data.results);
  } catch(e) {
    alert('Cannot reach server. Make sure server.py is running.');
  } finally {
    setBenchLoading(false);
  }
}

function setBenchLoading(on) {
  document.getElementById('btnBenchText').style.display = on ? 'none' : '';
  document.getElementById('benchSpinner').style.display = on ? '' : 'none';
  document.getElementById('btnBench').disabled = on;
}

function showBenchResults(results) {
  document.getElementById('benchResults').style.display = '';

  const labels = results.map(r => r.size + ' bp');
  const cpuTimes = results.map(r => r.cpu_ms);
  const wfTimes  = results.map(r => r.wf_ms);
  const speedups = results.map(r => r.speedup);

  const GRID = 'rgba(99,179,237,0.08)';
  const opts  = (label) => ({
    responsive: true,
    plugins: { legend: { labels: { color: '#94a3b8', font:{size:12} } } },
    scales: {
      x: { ticks:{color:'#94a3b8'}, grid:{color:GRID} },
      y: { ticks:{color:'#94a3b8'}, grid:{color:GRID},
           title:{display:true, text:label, color:'#94a3b8', font:{size:12}} }
    }
  });

  // Time chart
  if (chartTime) chartTime.destroy();
  chartTime = new Chart(document.getElementById('chartTime'), {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label:'CPU Sequential', data:cpuTimes, backgroundColor:'rgba(252,129,129,0.7)', borderColor:'#fc8181', borderWidth:1.5, borderRadius:5 },
        { label:'Wavefront',      data:wfTimes,  backgroundColor:'rgba(72,187,120,0.7)',  borderColor:'#48bb78', borderWidth:1.5, borderRadius:5 }
      ]
    },
    options: opts('Time (ms)')
  });

  // Speedup chart
  if (chartSpeedup) chartSpeedup.destroy();
  chartSpeedup = new Chart(document.getElementById('chartSpeedup'), {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label:'Speedup', data:speedups, borderColor:'#f6e05e', backgroundColor:'rgba(246,224,94,0.12)',
          pointBackgroundColor:'#f6e05e', pointRadius:6, pointHoverRadius:8,
          borderWidth:2.5, fill:true, tension:0.35 }
      ]
    },
    options: {
      ...opts('Speedup (×)'),
      plugins: {
        ...opts('').plugins,
        annotation: {}
      }
    }
  });

  // Table
  const tbody = document.getElementById('benchTableBody');
  tbody.innerHTML = results.map(r => `
    <tr>
      <td>${r.size} bp</td>
      <td>${r.cpu_ms.toFixed(2)} ms</td>
      <td>${r.wf_ms.toFixed(2)} ms</td>
      <td style="color:${r.speedup>=1?'#48bb78':'#fc8181'}">${r.speedup.toFixed(3)}×</td>
      <td><span class="${r.validation?'pass-badge':'fail-badge'}">${r.validation?'PASS':'FAIL'}</span></td>
    </tr>`).join('');

  document.getElementById('benchResults').scrollIntoView({behavior:'smooth', block:'nearest'});
}
