/* ═══════════════════════════════════════════════════════════════════
   ACT Project Site — main.js
   Chart.js charts + nav highlight + copy button + video handling
   ═══════════════════════════════════════════════════════════════════ */

'use strict';

/* ── Colour palette (shared across charts) ─────────────────────── */
const CAR_COLOR   = 'rgba(59, 130, 246, 0.85)';   // blue
const PEN_COLOR   = 'rgba(249, 115, 22, 0.85)';   // orange
const CAR_BORDER  = 'rgb(37, 99, 235)';
const PEN_BORDER  = 'rgb(234, 88, 12)';
const YOLO_COLOR  = 'rgba(168, 85, 247, 0.8)';    // purple
const YOLO_BORDER = 'rgb(147, 51, 234)';

const chartDefaults = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { position: 'top', labels: { font: { family: 'Inter', size: 12 }, padding: 14 } },
    tooltip: { callbacks: {
      label: ctx => ` ${ctx.dataset.label}: ${(ctx.parsed.y * 100).toFixed(1)}%`
    }}
  },
  scales: {
    x: {
      grid: { display: false },
      ticks: { font: { family: 'Inter', size: 11 } }
    },
    y: {
      min: 0, max: 1.05,
      ticks: {
        font: { family: 'Inter', size: 11 },
        callback: v => `${(v * 100).toFixed(0)}%`,
        stepSize: 0.2
      },
      grid: { color: 'rgba(0,0,0,.05)' }
    }
  }
};

function mergeDeep(target, source) {
  const out = Object.assign({}, target);
  for (const key of Object.keys(source)) {
    if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
      out[key] = mergeDeep(target[key] || {}, source[key]);
    } else {
      out[key] = source[key];
    }
  }
  return out;
}

function barOpts(extra = {}) { return mergeDeep(chartDefaults, extra); }

/* ── 1. Dataset Size Chart (scatter + line) ────────────────────── */
(function() {
  const ctx = document.getElementById('chartDatasetSize');
  if (!ctx) return;
  new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [
        {
          label: 'Car Pick-and-Place',
          data: [{x: 25, y: 0.354}, {x: 50, y: 0.583}, {x: 100, y: 0.854}],
          borderColor: CAR_BORDER, backgroundColor: CAR_COLOR,
          pointRadius: 6, pointHoverRadius: 8, borderWidth: 2.5, tension: 0, fill: false
        },
        {
          label: 'Pick Pen',
          data: [{x: 25, y: 0.100}, {x: 50, y: 0.500}, {x: 100, y: 0.550}],
          borderColor: PEN_BORDER, backgroundColor: PEN_COLOR,
          pointRadius: 6, pointHoverRadius: 8, borderWidth: 2.5, tension: 0, fill: false
        }
      ]
    },
    options: mergeDeep(chartDefaults, {
      plugins: {
        title: { display: true, text: 'Success Rate vs. Dataset Size', font: { family: 'Inter', size: 13, weight: '600' } }
      },
      scales: {
        x: {
          type: 'logarithmic', min: 18, max: 140,
          afterBuildTicks: scale => { scale.ticks = [{value:25},{value:50},{value:100}]; },
          ticks: { callback: v => v + '%' },
          title: { display: true, text: 'Dataset fraction', font: { family: 'Inter', size: 11 }, color: '#6b7280' }
        }
      }
    })
  });
})();

/* ── 2. Model Size Chart ────────────────────────────────────────── */
(function() {
  const ctx = document.getElementById('chartModelSize');
  if (!ctx) return;
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Small (4,1)', 'Medium (6,2)', 'Large (8,4)'],
      datasets: [
        { label: 'Car Pick-and-Place', data: [0.583, 0.625, 0.354],
          backgroundColor: CAR_COLOR, borderColor: CAR_BORDER, borderWidth: 2, borderRadius: 5 },
        { label: 'Pick Pen', data: [0.500, 0.600, 0.100],
          backgroundColor: PEN_COLOR, borderColor: PEN_BORDER, borderWidth: 2, borderRadius: 5 }
      ]
    },
    options: barOpts({ plugins: { title: { display: true, text: 'Success Rate vs. Model Size', font: { family: 'Inter', size: 13, weight: '600' } } } })
  });
})();

/* ── 3. Action Chunking Chart ───────────────────────────────────── */
(function() {
  const ctx = document.getElementById('chartChunking');
  if (!ctx) return;
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['K=25', 'K=50', 'K=100', 'K=150'],
      datasets: [
        { label: 'Car (no TE)', data: [0.500, 0.708, 0.854, 0.708],
          backgroundColor: CAR_COLOR, borderColor: CAR_BORDER, borderWidth: 2, borderRadius: 5 },
        { label: 'Pen (no TE)', data: [0.325, 0.550, 0.600, 0.225],
          backgroundColor: PEN_COLOR, borderColor: PEN_BORDER, borderWidth: 2, borderRadius: 5 }
      ]
    },
    options: barOpts({ plugins: { title: { display: true, text: 'Effect of Action Chunk Size K (no TE)', font: { family: 'Inter', size: 13, weight: '600' } } } })
  });
})();

/* ── 4. Temporal Ensembling Chart ───────────────────────────────── */
(function() {
  const ctx = document.getElementById('chartTE');
  if (!ctx) return;
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['K=25, no TE', 'K=100, no TE\n(best)', 'TE (λ=0.01)'],
      datasets: [
        { label: 'Car', data: [0.500, 0.854, 0.688],
          backgroundColor: CAR_COLOR, borderColor: CAR_BORDER, borderWidth: 2, borderRadius: 5 },
        { label: 'Pen', data: [0.325, 0.600, 0.625],
          backgroundColor: PEN_COLOR, borderColor: PEN_BORDER, borderWidth: 2, borderRadius: 5 }
      ]
    },
    options: barOpts({ plugins: { title: { display: true, text: 'Temporal Ensembling vs. Best Baselines', font: { family: 'Inter', size: 13, weight: '600' } } } })
  });
})();

/* ── 5. Distribution Shift Chart ────────────────────────────────── */
(function() {
  const ctx = document.getElementById('chartOOD');
  if (!ctx) return;
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['50k — In-Sample', '50k — Out-of-Sample', '100k — In-Sample', '100k — Out-of-Sample'],
      datasets: [
        { label: 'Car Success Rate',
          data: [0.604, 0.194, 0.854, 0.194],
          backgroundColor: [CAR_COLOR, 'rgba(220,38,38,.75)', CAR_COLOR, 'rgba(220,38,38,.75)'],
          borderColor:      [CAR_BORDER, 'rgb(185,28,28)',     CAR_BORDER, 'rgb(185,28,28)'],
          borderWidth: 2, borderRadius: 5 }
      ]
    },
    options: barOpts({
      plugins: {
        title: { display: true, text: 'In-Sample vs. Out-of-Sample (Car Task)', font: { family: 'Inter', size: 13, weight: '600' } },
        legend: { display: false }
      }
    })
  });
})();

/* ── 7. YOLO vs Baseline Results ────────────────────────────────── */
(function() {
  const ctx = document.getElementById('chartYoloResults');
  if (!ctx) return;
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Car In-Sample', 'Car Out-of-Sample', 'Pen In-Sample', 'Pen Out-of-Sample'],
      datasets: [
        { label: 'Baseline (50 ep)', data: [0.583, null, 0.500, null],
          backgroundColor: CAR_COLOR, borderColor: CAR_BORDER, borderWidth: 2, borderRadius: 5 },
        { label: 'Baseline (96 ep)', data: [0.854, 0.194, null, null],
          backgroundColor: 'rgba(16, 185, 129, 0.8)', borderColor: 'rgb(5,150,105)', borderWidth: 2, borderRadius: 5 },
        { label: 'YOLO-Augmented (50 ep)', data: [0.563, 0.167, 0.175, 0.000],
          backgroundColor: YOLO_COLOR, borderColor: YOLO_BORDER, borderWidth: 2, borderRadius: 5 }
      ]
    },
    options: barOpts({
      plugins: {
        title: { display: true, text: 'YOLO Augmentation vs. Baselines', font: { family: 'Inter', size: 13, weight: '600' } },
        tooltip: { callbacks: {
          label: ctx => ctx.parsed.y === null ? ' N/A' : ` ${ctx.dataset.label}: ${(ctx.parsed.y * 100).toFixed(1)}%`
        }}
      }
    })
  });
})();

/* ── Sticky nav highlight on scroll ────────────────────────────── */
(function() {
  const sections = document.querySelectorAll('section[id], header[id]');
  const navLinks = document.querySelectorAll('.nav-links a');
  if (!sections.length || !navLinks.length) return;

  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        navLinks.forEach(a => a.classList.remove('active'));
        const active = document.querySelector(`.nav-links a[href="#${entry.target.id}"]`);
        if (active) active.classList.add('active');
      }
    });
  }, { rootMargin: '-40% 0px -55% 0px' });

  sections.forEach(s => observer.observe(s));
})();

/* ── Copy BibTeX ────────────────────────────────────────────────── */
function copyBibtex() {
  const text = document.getElementById('bibtex-text')?.innerText;
  if (!text) return;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.querySelector('.copy-btn');
    if (!btn) return;
    const original = btn.innerHTML;
    btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg> Copied!';
    btn.style.background = '#059669';
    btn.style.color = '#fff';
    setTimeout(() => { btn.innerHTML = original; btn.style.background = ''; btn.style.color = ''; }, 2000);
  });
}

/* ── Auto-replace placeholders when actual files exist ─────────── */
(function() {
  // Video placeholders: replace with <video> if the file is present
  document.querySelectorAll('.video-placeholder').forEach(el => {
    const videoFile = el.dataset.video;
    if (!videoFile) return;
    const src = `assets/video/${videoFile}`;
    // Try loading; if 404 the onerror keeps placeholder
    const video = document.createElement('video');
    video.src = src; video.controls = true; video.muted = true;
    video.setAttribute('playsinline', '');
    video.style.borderRadius = '12px'; video.style.width = '100%';
    video.onloadedmetadata = () => {
      el.replaceWith(video);
    };
  });

  // Figure placeholders: replace with <img> if file is present
  document.querySelectorAll('.fig-placeholder[data-fig]').forEach(el => {
    const figFile = el.dataset.fig;
    if (!figFile) return;
    // Try png first, then jpg
    ['png', 'jpg', 'jpeg'].forEach(ext => {
      const img = new Image();
      img.src = `assets/img/${figFile}.${ext}`;
      img.onload = () => {
        img.style.borderRadius = '12px'; img.style.width = '100%';
        img.alt = el.querySelector('.fig-placeholder-label')?.textContent || figFile;
        el.replaceWith(img);
      };
    });
  });
})();
