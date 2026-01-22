const $ = (sel) => document.querySelector(sel);

let project = null;
let profile = null;
let chart = null;
let currentCandidate = null;
let facecamRect = null;
let lastBatchSelectionIds = [];
let calibrating = false;
let isStudioMode = false;
let currentTab = 'edit';

// Publish state
let publishAccounts = [];
let publishExports = [];
let selectedAccountIds = new Set();
let selectedExportIds = new Set();
let publishJobsSSE = null;
let publishJobs = new Map();

const jobs = new Map();

function fmtTime(sec) {
  sec = Math.max(0, Number(sec) || 0);
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = (sec % 60);
  const ss = s.toFixed(2).padStart(5, '0');
  if (h > 0) return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${ss}`;
  return `${String(m).padStart(2,'0')}:${ss}`;
}

function clamp(val, lo, hi) {
  return Math.max(lo, Math.min(hi, val));
}

async function apiGet(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path} -> ${r.status}`);
  return await r.json();
}

async function apiJson(method, path, body) {
  const r = await fetch(path, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) {
    const txt = await r.text();
    throw new Error(`${method} ${path} -> ${r.status} ${txt}`);
  }
  return await r.json();
}

// =========================================================================
// VIEW SWITCHING
// =========================================================================

function showHomeView() {
  isStudioMode = false;
  $('#homeView').style.display = 'block';
  $('#studioView').style.display = 'none';
  loadRecentProjects();
  loadRecentDownloads();
  renderHomeJobs();
}

function showStudioView() {
  isStudioMode = true;
  $('#homeView').style.display = 'none';
  $('#studioView').style.display = 'block';
  // Reload video element src to refresh
  const v = $('#video');
  if (v) {
    v.src = '/video?' + Date.now();
    v.load();
  }
}

// =========================================================================
// HOME VIEW FUNCTIONS
// =========================================================================

async function loadRecentProjects() {
  const container = $('#recentProjects');
  container.innerHTML = '<div class="small">Loading...</div>';

  try {
    const res = await apiGet('/api/home/recent_projects');
    const projects = res.projects || [];

    if (projects.length === 0) {
      container.innerHTML = '<div class="small">No recent projects found.</div>';
      return;
    }

    container.innerHTML = '';
    for (const p of projects) {
      const el = document.createElement('div');
      el.className = 'item';
      const dur = fmtTime(p.duration_seconds || 0);
      const sels = p.selections_count || 0;
      const exps = p.exports_count || 0;
      el.innerHTML = `
        <div class="title">${p.video_name || 'Unknown video'}</div>
        <div class="meta">Duration: ${dur} • Selections: ${sels} • Exports: ${exps}</div>
        <div class="meta small" style="opacity:0.7;word-break:break-all">${p.video_path || ''}</div>
        <div class="actions" style="margin-top:8px">
          <button class="primary">Open project</button>
        </div>
      `;
      const btn = el.querySelector('button');
      btn.onclick = () => openProjectByPath(p.video_path);
      container.appendChild(el);
    }
  } catch (e) {
    container.innerHTML = `<div class="small">Error loading projects: ${e.message}</div>`;
  }
}

async function openProjectByPath(videoPath) {
  if (!videoPath) {
    alert('Please enter a video path.');
    return;
  }

  try {
    const res = await apiJson('POST', '/api/home/open_video', { video_path: videoPath });
    if (res.active && res.project) {
      project = res.project;
      showStudioView();
      await initStudioView();
    }
  } catch (e) {
    alert(`Failed to open video: ${e.message}`);
  }
}

async function openVideoDialog() {
  try {
    const res = await apiJson('POST', '/api/home/open_dialog', {});
    if (res.video_path) {
      await openProjectByPath(res.video_path);
    } else if (res.error === 'not_windows') {
      alert('Native file dialog is only available on Windows. Please use the path input instead.');
    }
    // If null, user cancelled - do nothing
  } catch (e) {
    alert(`Failed to open dialog: ${e.message}`);
  }
}

async function closeProject() {
  try {
    await apiJson('POST', '/api/home/close_project', {});
    project = null;
    showHomeView();
  } catch (e) {
    alert(`Failed to close project: ${e.message}`);
  }
}

function wireHomeUI() {
  const btnOpenDialog = $('#btnOpenDialog');
  const btnOpenByPath = $('#btnOpenByPath');
  const videoPathInput = $('#videoPathInput');

  if (btnOpenDialog) {
    btnOpenDialog.onclick = openVideoDialog;
  }

  if (btnOpenByPath && videoPathInput) {
    btnOpenByPath.onclick = () => openProjectByPath(videoPathInput.value.trim());
    videoPathInput.onkeydown = (e) => {
      if (e.key === 'Enter') {
        openProjectByPath(videoPathInput.value.trim());
      }
    };
  }

  const btnBackToHome = $('#btnBackToHome');
  if (btnBackToHome) {
    btnBackToHome.onclick = closeProject;
  }

  // URL Download UI
  const btnDownloadUrl = $('#btnDownloadUrl');
  const btnProbeUrl = $('#btnProbeUrl');
  const urlInput = $('#urlInput');
  const btnPasteUrl = $('#btnPasteUrl');

  if (btnDownloadUrl && urlInput) {
    btnDownloadUrl.onclick = () => startUrlDownload(urlInput.value.trim());
    urlInput.onkeydown = (e) => {
      if (e.key === 'Enter') {
        startUrlDownload(urlInput.value.trim());
      }
    };
    // Auto-probe on URL change (debounced)
    let probeTimeout = null;
    urlInput.oninput = () => {
      if (probeTimeout) clearTimeout(probeTimeout);
      probeTimeout = setTimeout(() => {
        const url = urlInput.value.trim();
        if (url && (url.startsWith('http://') || url.startsWith('https://'))) {
          probeUrl(url, false);  // Quick heuristic probe
        } else {
          hideProbeBadge();
        }
      }, 300);
    };
  }

  if (btnProbeUrl && urlInput) {
    btnProbeUrl.onclick = () => probeUrl(urlInput.value.trim(), true);  // Full yt-dlp probe
  }

  if (btnPasteUrl && urlInput) {
    btnPasteUrl.onclick = async () => {
      try {
        const text = await navigator.clipboard.readText();
        urlInput.value = text;
        // Trigger auto-probe
        if (text && (text.startsWith('http://') || text.startsWith('https://'))) {
          probeUrl(text, false);
        }
      } catch (e) {
        alert('Failed to read clipboard. Please paste manually.');
      }
    };
  }
}

// =========================================================================
// URL DOWNLOAD FUNCTIONS
// =========================================================================

const homeJobs = new Map();
let lastProbeResult = null;

function hideProbeBadge() {
  const badge = $('#probeBadge');
  if (badge) badge.style.display = 'none';
  lastProbeResult = null;
}

function showProbeBadge(text, notes) {
  const badge = $('#probeBadge');
  const badgeText = $('#probeBadgeText');
  const helpText = $('#dlHelpText');

  if (badge && badgeText) {
    badgeText.textContent = text;
    badge.style.display = 'block';
  }
  if (helpText && notes) {
    helpText.textContent = notes;
  }
}

async function probeUrl(url, useYtdlp = false) {
  if (!url) {
    hideProbeBadge();
    return;
  }

  if (!url.startsWith('http://') && !url.startsWith('https://')) {
    hideProbeBadge();
    return;
  }

  try {
    const res = await apiJson('POST', '/api/ingest/probe', {
      url,
      use_ytdlp: useYtdlp,
    });

    lastProbeResult = res;

    // Build badge text
    let badgeText = res.display_badge || `Detected: ${res.policy?.display_name || 'Unknown'}`;

    // Add concurrency info for HLS sites
    if (res.policy?.supports_fragment_concurrency) {
      badgeText += ` — Auto concurrency ${res.policy.default_concurrency}`;
    }

    // Add title if available from yt-dlp probe
    if (res.title && useYtdlp) {
      badgeText += `\n📺 ${res.title}`;
      if (res.duration_seconds > 0) {
        const mins = Math.floor(res.duration_seconds / 60);
        const secs = Math.floor(res.duration_seconds % 60);
        badgeText += ` (${mins}:${secs.toString().padStart(2, '0')})`;
      }
    }

    showProbeBadge(badgeText, res.policy?.notes);

  } catch (e) {
    hideProbeBadge();
  }
}

async function startUrlDownload(url) {
  if (!url) {
    alert('Please enter a URL.');
    return;
  }

  if (!url.startsWith('http://') && !url.startsWith('https://')) {
    alert('Please enter a valid URL (starting with http:// or https://).');
    return;
  }

  const options = {
    no_playlist: $('#dlNoPlaylist')?.checked ?? true,
    create_preview: $('#dlCreatePreview')?.checked ?? true,
    speed_mode: $('#dlSpeedMode')?.value ?? 'auto',
    quality_cap: $('#dlQualityCap')?.value ?? 'source',
  };

  try {
    const res = await apiJson('POST', '/api/ingest/url', {
      url,
      options,
      auto_open: true,
    });

    // Clear input and badge
    $('#urlInput').value = '';
    hideProbeBadge();

    // Watch job progress
    watchHomeJob(res.job_id);
    renderHomeJobs();

  } catch (e) {
    alert(`Failed to start download: ${e.message}`);
  }
}

function watchHomeJob(jobId) {
  const es = new EventSource(`/api/jobs/${jobId}/events`);
  es.onmessage = (ev) => {
    try {
      const payload = JSON.parse(ev.data);
      if (payload.type === 'job_update' || payload.type === 'job_created') {
        const j = payload.job;
        homeJobs.set(j.id, j);
        renderHomeJobs();

        // If download succeeded and auto-opened, switch to Studio
        if (j.kind === 'download_url' && j.status === 'succeeded' && j.result?.auto_opened) {
          project = j.result.project;
          showStudioView();
          initStudioView();
        }
      }
    } catch (e) {
      console.warn('bad job payload', e);
    }
  };
}

function renderHomeJobs() {
  const container = $('#homeJobs');
  if (!container) return;

  if (homeJobs.size === 0) {
    container.innerHTML = '<div class="small">No active jobs.</div>';
    return;
  }

  container.innerHTML = '';
  const sorted = Array.from(homeJobs.values()).sort((a, b) =>
    a.created_at < b.created_at ? 1 : -1
  );

  for (const job of sorted) {
    const el = document.createElement('div');
    el.className = 'item';
    const pct = Math.round((job.progress || 0) * 100);

    let statusClass = '';
    if (job.status === 'running') statusClass = 'status-badge running';
    else if (job.status === 'succeeded') statusClass = 'status-badge succeeded';
    else if (job.status === 'failed') statusClass = 'status-badge failed';
    else statusClass = 'status-badge queued';

    let resultInfo = '';
    if (job.status === 'succeeded' && job.result?.video_path) {
      const filename = job.result.video_path.split(/[/\\]/).pop();
      resultInfo = `<div class="small" style="margin-top:4px">Downloaded: ${filename}</div>`;
    }

    let errorInfo = '';
    if (job.status === 'failed') {
      errorInfo = `<div class="small" style="color:var(--danger);margin-top:4px">${job.message}</div>`;
    }

    el.innerHTML = `
      <div class="title">
        <span class="${statusClass}">${job.status}</span>
        ${job.kind === 'download_url' ? 'URL Download' : job.kind}
      </div>
      <div class="meta">${job.message || ''}</div>
      ${job.status === 'running' ? `
        <div class="progress" style="margin-top:8px"><div style="width:${pct}%"></div></div>
        <div class="small" style="margin-top:4px">${pct}%</div>
      ` : ''}
      ${resultInfo}
      ${errorInfo}
    `;

    container.appendChild(el);
  }
}

async function loadRecentDownloads() {
  const container = $('#recentDownloads');
  if (!container) return;

  container.innerHTML = '<div class="small">Loading...</div>';

  try {
    const res = await apiGet('/api/ingest/recent_downloads');
    const downloads = res.downloads || [];

    if (downloads.length === 0) {
      container.innerHTML = '<div class="small">No recent downloads.</div>';
      return;
    }

    container.innerHTML = '';
    for (const d of downloads) {
      const el = document.createElement('div');
      el.className = 'item';
      const dur = fmtTime(d.duration_seconds || 0);
      const sizeMB = ((d.size_bytes || 0) / 1024 / 1024).toFixed(1);
      el.innerHTML = `
        <div class="title">${d.title || d.filename}</div>
        <div class="meta">${dur} • ${sizeMB} MB • ${d.extractor || 'local'}</div>
        <div class="meta small" style="opacity:0.7;word-break:break-all">${d.path}</div>
        <div class="actions" style="margin-top:8px">
          <button class="primary">Open project</button>
        </div>
      `;
      const btn = el.querySelector('button');
      btn.onclick = () => openProjectByPath(d.path);
      container.appendChild(el);
    }
  } catch (e) {
    container.innerHTML = `<div class="small">Error: ${e.message}</div>`;
  }
}

function setBuilder(startS, endS, title = '', template = '') {
  $('#startS').value = (Number(startS) || 0).toFixed(2);
  $('#endS').value = (Number(endS) || 0).toFixed(2);
  if (title !== undefined) $('#title').value = title || '';
  if (template) $('#template').value = template;
}

function getBuilder() {
  return {
    start_s: Number($('#startS').value),
    end_s: Number($('#endS').value),
    title: $('#title').value || '',
    template: $('#template').value,
  };
}

function renderProjectInfo() {
  if (!project) return;
  const v = project.video || {};
  $('#projectInfo').textContent = `Project ${project.project_id} • ${fmtTime(v.duration_seconds)} • ${v.path}`;
}

function renderCandidates() {
  const container = $('#candidates');
  container.innerHTML = '';

  const highlights = project?.analysis?.highlights;
  const audio = project?.analysis?.audio;
  const candidates = highlights?.candidates || audio?.candidates || [];
  if (candidates.length === 0) {
    container.innerHTML = `<div class="small">No candidates yet. Click “Analyze highlights”.</div>`;
    return;
  }

  for (const c of candidates) {
    const el = document.createElement('div');
    el.className = 'item';
    const breakdown = c.breakdown ? ` • audio ${Number(c.breakdown.audio).toFixed(2)} / motion ${Number(c.breakdown.motion).toFixed(2)} / chat ${Number(c.breakdown.chat).toFixed(2)}` : '';
    el.innerHTML = `
      <div class="title">#${c.rank} • score ${c.score.toFixed(2)} <span class="badge">peak ${fmtTime(c.peak_time_s)}</span></div>
      <div class="meta">Clip: ${fmtTime(c.start_s)} → ${fmtTime(c.end_s)} (${(c.end_s - c.start_s).toFixed(1)}s)${breakdown}</div>
      <div class="actions">
        <button class="primary">Load</button>
        <button>Seek peak</button>
      </div>
    `;
    const [btnLoad, btnPeak] = el.querySelectorAll('button');
    btnLoad.onclick = () => {
      currentCandidate = c;
      setBuilder(c.start_s, c.end_s, '', $('#template').value);
      const v = $('#video');
      v.currentTime = c.start_s;
      v.play();
    };
    btnPeak.onclick = () => {
      const v = $('#video');
      v.currentTime = c.peak_time_s;
      v.play();
    };
    container.appendChild(el);
  }
}

function renderSelections() {
  const container = $('#selections');
  container.innerHTML = '';

  const sels = project?.selections || [];
  if (sels.length === 0) {
    container.innerHTML = `<div class="small">No selections saved yet.</div>`;
    return;
  }

  for (const s of sels) {
    const dur = Number(s.end_s) - Number(s.start_s);
    const el = document.createElement('div');
    el.className = 'item';
    el.innerHTML = `
      <div class="title">${(s.title && s.title.trim()) ? s.title : ('Selection ' + s.id.slice(0, 8))}</div>
      <div class="meta">${fmtTime(s.start_s)} → ${fmtTime(s.end_s)} (${dur.toFixed(1)}s) • template: <span class="badge">${s.template || 'vertical_blur'}</span></div>
      <div class="actions">
        <button class="primary">Load</button>
        <button>Export</button>
        <button class="danger">Delete</button>
      </div>
    `;
    const [btnLoad, btnExport, btnDelete] = el.querySelectorAll('button');
    btnLoad.onclick = () => {
      setBuilder(s.start_s, s.end_s, s.title || '', s.template || 'vertical_blur');
      const v = $('#video');
      v.currentTime = Number(s.start_s);
      v.play();
    };
    btnDelete.onclick = async () => {
      if (!confirm('Delete this selection?')) return;
      project = await apiJson('DELETE', `/api/selections/${s.id}`, null);
      renderSelections();
    };
    btnExport.onclick = async () => {
      await startExportJob(s.id);
    };
    container.appendChild(el);
  }
}

function renderJobs() {
  const container = $('#jobs');
  container.innerHTML = '';
  if (jobs.size === 0) {
    container.innerHTML = `<div class="small">No jobs yet.</div>`;
    return;
  }

  for (const job of Array.from(jobs.values()).sort((a,b) => (a.created_at < b.created_at ? 1 : -1))) {
    const el = document.createElement('div');
    el.className = 'item';
    const pct = Math.round((job.progress || 0) * 100);
    el.innerHTML = `
      <div class="title">${job.kind} • <span class="badge">${job.status}</span></div>
      <div class="meta">${job.message || ''}${job.result?.output ? `<br/>Output: ${job.result.output}` : ''}</div>
      <div class="progress" style="margin-top:10px"><div style="width:${pct}%"></div></div>
      <div class="meta" style="margin-top:6px">${pct}%</div>
    `;
    container.appendChild(el);
  }
}

function watchJob(jobId) {
  const es = new EventSource(`/api/jobs/${jobId}/events`);
  es.onmessage = (ev) => {
    try {
      const payload = JSON.parse(ev.data);
      if (payload.type === 'job_update' || payload.type === 'job_created') {
        const j = payload.job;
        jobs.set(j.id, j);
        renderJobs();
        if ((j.kind === 'analyze_audio' || j.kind === 'analyze_highlights' || j.kind === 'analyze_speech') && j.status === 'succeeded') {
          // refresh project and timeline to show reranked candidates
          refreshProject();
        }
        if (j.kind === 'export' && j.status === 'succeeded') {
          // refresh project to show exports list eventually
          refreshProject();
        }
        if (j.kind === 'download_chat' && j.status === 'succeeded') {
          // refresh chat status after download completes
          loadChatStatus();
        }
      }
    } catch (e) {
      console.warn('bad job payload', e);
    }
  };
  es.onerror = () => {
    // We'll let the server keepalive; if it breaks, close.
    // (Often happens when job finishes and server closes the stream.)
  };
}

async function refreshTimeline() {
  try {
    let t = null;
    try {
      t = await apiGet('/api/highlights/timeline');
    } catch (_) {
      t = await apiGet('/api/audio/timeline');
    }
    if (!t || !t.ok) return;

    const hop = t.hop_seconds;
    const xs = t.indices.map(i => i * hop);
    const ys = t.scores;

    const ctx = $('#chart').getContext('2d');
    if (chart) chart.destroy();
    chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: xs,
        datasets: [{
          label: 'Interest score',
          data: ys,
          borderWidth: 1,
          pointRadius: 0,
          tension: 0.15,
        }]
      },
      options: {
        responsive: true,
        animation: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: (items) => `t = ${fmtTime(items[0].label)}`,
              label: (item) => `score ${Number(item.raw).toFixed(2)}`
            }
          }
        },
        scales: {
          x: {
            ticks: {
              callback: (val) => {
                const sec = xs[val];
                return sec !== undefined ? fmtTime(sec) : '';
              },
              maxTicksLimit: 8,
            }
          },
          y: {
            ticks: { maxTicksLimit: 5 }
          }
        },
        onClick: (_, elements) => {
          if (!elements || elements.length === 0) return;
          const idx = elements[0].index;
          const sec = xs[idx];
          if (sec === undefined) return;
          const v = $('#video');
          v.currentTime = sec;
          v.play();
        }
      }
    });

  } catch (e) {
    // timeline might not exist yet
  }
}

async function refreshProject() {
  try {
    const res = await apiGet('/api/project');
    if (res.active && res.project) {
      project = res.project;
    }
  } catch (e) {
    console.warn('Failed to refresh project:', e);
  }
  renderProjectInfo();
  renderCandidates();
  renderSelections();
  await refreshTimeline();
}

function updateFacecamStatus() {
  const status = $('#facecamStatus');
  if (!facecamRect) {
    status.textContent = 'No facecam calibration yet.';
    return;
  }
  status.textContent = `Facecam: x=${facecamRect.x.toFixed(3)}, y=${facecamRect.y.toFixed(3)}, w=${facecamRect.w.toFixed(3)}, h=${facecamRect.h.toFixed(3)}`;
}

function setupFacecamCanvas() {
  const canvas = $('#facecamCanvas');
  const v = $('#video');
  const ctx = canvas.getContext('2d');

  function resizeCanvas() {
    canvas.width = v.clientWidth;
    canvas.height = v.clientHeight;
    drawRect();
  }

  function drawRect(tempRect = null) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const rect = tempRect || facecamRect;
    if (!rect) return;
    const x = rect.x * canvas.width;
    const y = rect.y * canvas.height;
    const w = rect.w * canvas.width;
    const h = rect.h * canvas.height;
    ctx.strokeStyle = 'rgba(79, 140, 255, 0.95)';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
  }

  let dragging = false;
  let start = null;

  function normRectFromPoints(p1, p2) {
    const x = clamp(Math.min(p1.x, p2.x) / canvas.width, 0, 1);
    const y = clamp(Math.min(p1.y, p2.y) / canvas.height, 0, 1);
    const w = clamp(Math.abs(p2.x - p1.x) / canvas.width, 0, 1);
    const h = clamp(Math.abs(p2.y - p1.y) / canvas.height, 0, 1);
    return { x, y, w, h };
  }

  canvas.onmousedown = (ev) => {
    if (!calibrating) return;
    dragging = true;
    start = { x: ev.offsetX, y: ev.offsetY };
  };

  canvas.onmousemove = (ev) => {
    if (!calibrating || !dragging || !start) return;
    const temp = normRectFromPoints(start, { x: ev.offsetX, y: ev.offsetY });
    drawRect(temp);
  };

  canvas.onmouseup = (ev) => {
    if (!calibrating || !dragging || !start) return;
    dragging = false;
    const rect = normRectFromPoints(start, { x: ev.offsetX, y: ev.offsetY });
    if (rect.w > 0.01 && rect.h > 0.01) {
      facecamRect = rect;
      updateFacecamStatus();
      $('#btnSaveFacecam').disabled = false;
      drawRect();
    }
  };

  window.addEventListener('resize', resizeCanvas);
  v.addEventListener('loadedmetadata', resizeCanvas);
  resizeCanvas();

  return { resizeCanvas, drawRect };
}

async function startAnalyzeJob() {
  $('#analysisStatus').textContent = 'Starting analysis...';
  const res = await apiJson('POST', '/api/analyze/highlights', {});
  const jobId = res.job_id;
  watchJob(jobId);
  $('#analysisStatus').textContent = `Analysis running (job ${jobId.slice(0,8)})...`;
}

async function startAnalyzeAudioJob() {
  $('#analysisStatus').textContent = 'Starting audio analysis...';
  const res = await apiJson('POST', '/api/analyze/audio', {});
  const jobId = res.job_id;
  watchJob(jobId);
  $('#analysisStatus').textContent = `Audio analysis running (job ${jobId.slice(0,8)})...`;
}

async function startAnalyzeSpeechJob() {
  $('#analysisStatus').textContent = 'Starting speech analysis (Whisper transcription)...';
  const res = await apiJson('POST', '/api/analyze/speech', {});
  const jobId = res.job_id;
  watchJob(jobId);
  $('#analysisStatus').textContent = `Speech analysis running (job ${jobId.slice(0,8)})...`;
}

async function startExportJob(selectionId) {
  const exp = {
    width: Number($('#expW').value),
    height: Number($('#expH').value),
    fps: Number($('#expFps').value),
    crf: Number($('#expCrf').value),
    preset: $('#expPreset').value,
    template: $('#template').value,
    normalize_audio: $('#normalizeAudio').checked,
  };

  const withCaptions = $('#withCaptions').checked;

  const res = await apiJson('POST', '/api/export', {
    selection_id: selectionId,
    export: exp,
    with_captions: withCaptions,
  });

  const jobId = res.job_id;
  watchJob(jobId);
}

function wireUI() {
  const v = $('#video');
  const canvas = $('#facecamCanvas');
  const canvasApi = setupFacecamCanvas();

  v.addEventListener('timeupdate', () => {
    $('#timeReadout').textContent = `Now: ${fmtTime(v.currentTime)} / ${fmtTime(v.duration || 0)}`;
  });

  $('#btnSetStart').onclick = () => {
    const b = getBuilder();
    setBuilder(v.currentTime, b.end_s, b.title, b.template);
  };

  $('#btnSetEnd').onclick = () => {
    const b = getBuilder();
    setBuilder(b.start_s, v.currentTime, b.title, b.template);
  };

  $('#btnSnapCandidate').onclick = () => {
    if (!currentCandidate) return;
    setBuilder(currentCandidate.start_s, currentCandidate.end_s, $('#title').value, $('#template').value);
  };

  $('#btnNudgeBack').onclick = () => {
    const b = getBuilder();
    setBuilder(b.start_s - 0.5, b.end_s - 0.5, b.title, b.template);
  };

  $('#btnNudgeFwd').onclick = () => {
    const b = getBuilder();
    setBuilder(b.start_s + 0.5, b.end_s + 0.5, b.title, b.template);
  };

  $('#btnSaveSelection').onclick = async () => {
    const b = getBuilder();
    if (!(b.end_s > b.start_s)) {
      alert('End must be greater than Start');
      return;
    }
    project = await apiJson('POST', '/api/selections', {
      start_s: b.start_s,
      end_s: b.end_s,
      title: b.title,
      template: b.template,
    });
    renderSelections();
  };

  $('#btnAnalyzeHighlights').onclick = async () => {
    try {
      await startAnalyzeJob();
    } catch (e) {
      alert(`Analysis failed: ${e}`);
    }
  };
  $('#btnAnalyzeAudio').onclick = async () => {
    try {
      await startAnalyzeAudioJob();
    } catch (e) {
      alert(`Audio analysis failed: ${e}`);
    }
  };
  $('#btnAnalyzeSpeech').onclick = async () => {
    try {
      await startAnalyzeSpeechJob();
    } catch (e) {
      alert(`Speech analysis failed: ${e}`);
    }
  };

  $('#btnCalibrateFacecam').onclick = async () => {
    calibrating = !calibrating;
    canvas.style.display = calibrating ? 'block' : 'none';
    $('#btnCalibrateFacecam').textContent = calibrating ? 'Cancel calibration' : 'Calibrate facecam';
    if (!calibrating) {
      canvasApi.drawRect();
    }
  };

  $('#btnSaveFacecam').onclick = async () => {
    if (!facecamRect) return;
    try {
      await apiJson('POST', '/api/layout/facecam', facecamRect);
      calibrating = false;
      canvas.style.display = 'none';
      $('#btnCalibrateFacecam').textContent = 'Calibrate facecam';
      $('#btnSaveFacecam').disabled = true;
      canvasApi.drawRect();
      updateFacecamStatus();
    } catch (e) {
      alert(`Failed to save facecam: ${e}`);
    }
  };

  $('#btnCreateSelections').onclick = async () => {
    const top = Number($('#batchTopN').value || 10);
    const template = $('#batchTemplate').value;
    try {
      const res = await apiJson('POST', '/api/selections/from_candidates', { top, template });
      project = res.project;
      lastBatchSelectionIds = res.created_ids || [];
      renderSelections();
      alert(`Created ${lastBatchSelectionIds.length} selections.`);
    } catch (e) {
      alert(`Failed to create selections: ${e}`);
    }
  };

  $('#btnBatchExport').onclick = async () => {
    try {
      const exp = {
        width: Number($('#expW').value),
        height: Number($('#expH').value),
        fps: Number($('#expFps').value),
        crf: Number($('#expCrf').value),
        preset: $('#expPreset').value,
        template: $('#batchTemplate').value,
        normalize_audio: $('#normalizeAudio').checked,
      };
      const withCaptions = $('#withCaptions').checked;
      const res = await apiJson('POST', '/api/export/batch', {
        selection_ids: lastBatchSelectionIds,
        export: exp,
        with_captions: withCaptions,
      });
      watchJob(res.job_id);
    } catch (e) {
      alert(`Batch export failed: ${e}`);
    }
  };

  // Keyboard shortcuts
  window.addEventListener('keydown', (ev) => {
    const tag = (ev.target && ev.target.tagName) ? ev.target.tagName.toLowerCase() : '';
    const isTyping = (tag === 'input' || tag === 'textarea' || tag === 'select');
    if (isTyping) return;

    if (ev.code === 'Space') {
      ev.preventDefault();
      if (v.paused) v.play(); else v.pause();
    }
    if (ev.key === 'i' || ev.key === 'I') {
      ev.preventDefault();
      const b = getBuilder();
      setBuilder(v.currentTime, b.end_s, b.title, b.template);
    }
    if (ev.key === 'o' || ev.key === 'O') {
      ev.preventDefault();
      const b = getBuilder();
      setBuilder(b.start_s, v.currentTime, b.title, b.template);
    }
    if (ev.key === '[') {
      ev.preventDefault();
      const b = getBuilder();
      setBuilder(b.start_s - 0.5, b.end_s - 0.5, b.title, b.template);
    }
    if (ev.key === ']') {
      ev.preventDefault();
      const b = getBuilder();
      setBuilder(b.start_s + 0.5, b.end_s + 0.5, b.title, b.template);
    }
  });
}

// =========================================================================
// TAB SWITCHING
// =========================================================================

function switchTab(tabName) {
  currentTab = tabName;

  // Update tab buttons
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === tabName);
  });

  // Show/hide tab content
  const editContent = $('#editTabContent');
  const publishContent = $('#publishTabContent');

  if (tabName === 'edit') {
    editContent.style.display = 'grid';
    publishContent.style.display = 'none';
  } else if (tabName === 'publish') {
    editContent.style.display = 'none';
    publishContent.style.display = 'grid';
    // Load publish data when switching to tab
    loadPublishData();
    startPublishJobsSSE();
  }
}

function wireTabUI() {
  const tabEdit = $('#tabEdit');
  const tabPublish = $('#tabPublish');

  if (tabEdit) tabEdit.onclick = () => switchTab('edit');
  if (tabPublish) tabPublish.onclick = () => switchTab('publish');
}

// =========================================================================
// PUBLISH TAB FUNCTIONS
// =========================================================================

async function loadPublishData() {
  await Promise.all([
    loadPublishAccounts(),
    loadPublishExports(),
    loadPublishJobs(),
  ]);
  updatePublishSelectionInfo();
}

async function loadPublishAccounts() {
  const container = $('#publishAccounts');
  const noAccountsCard = $('#noAccountsCard');
  container.innerHTML = '<div class="small">Loading accounts...</div>';

  try {
    const res = await apiGet('/api/publisher/accounts');
    publishAccounts = res.accounts || [];

    if (publishAccounts.length === 0) {
      container.innerHTML = '';
      noAccountsCard.style.display = 'block';
      return;
    }

    noAccountsCard.style.display = 'none';
    renderPublishAccounts();
  } catch (e) {
    container.innerHTML = `<div class="small">Error: ${e.message}</div>`;
  }
}

function renderPublishAccounts() {
  const container = $('#publishAccounts');
  const platformFilter = $('#publishPlatformFilter').value;
  const searchTerm = $('#publishAccountSearch').value.toLowerCase();

  let filtered = publishAccounts;
  if (platformFilter) {
    filtered = filtered.filter(a => a.platform === platformFilter);
  }
  if (searchTerm) {
    filtered = filtered.filter(a =>
      a.label.toLowerCase().includes(searchTerm) ||
      a.platform.toLowerCase().includes(searchTerm)
    );
  }

  if (filtered.length === 0) {
    container.innerHTML = '<div class="small">No matching accounts.</div>';
    return;
  }

  container.innerHTML = '';
  for (const account of filtered) {
    const el = document.createElement('div');
    el.className = 'selectable-item' + (selectedAccountIds.has(account.id) ? ' selected' : '');
    el.innerHTML = `
      <span class="checkbox"></span>
      <span class="title">${account.label}</span>
      <span class="badge" style="margin-left:8px">${account.platform}</span>
    `;
    el.onclick = () => toggleAccountSelection(account.id);
    container.appendChild(el);
  }
}

function toggleAccountSelection(accountId) {
  if (selectedAccountIds.has(accountId)) {
    selectedAccountIds.delete(accountId);
  } else {
    selectedAccountIds.add(accountId);
  }
  renderPublishAccounts();
  updatePublishSelectionInfo();
}

async function loadPublishExports() {
  const container = $('#publishExports');
  container.innerHTML = '<div class="small">Loading exports...</div>';

  try {
    const res = await apiGet('/api/publisher/exports');
    publishExports = res.exports || [];

    if (publishExports.length === 0) {
      container.innerHTML = '<div class="small">No exports yet. Export clips from the Edit tab first.</div>';
      return;
    }

    renderPublishExports();
  } catch (e) {
    container.innerHTML = `<div class="small">Error: ${e.message}</div>`;
  }
}

function renderPublishExports() {
  const container = $('#publishExports');

  if (publishExports.length === 0) {
    container.innerHTML = '<div class="small">No exports yet.</div>';
    return;
  }

  container.innerHTML = '';
  for (const exp of publishExports) {
    const el = document.createElement('div');
    el.className = 'selectable-item' + (selectedExportIds.has(exp.export_id) ? ' selected' : '');
    const dur = fmtTime(exp.duration_seconds || 0);
    const sizeMB = ((exp.file_size_bytes || 0) / 1024 / 1024).toFixed(1);
    el.innerHTML = `
      <span class="checkbox"></span>
      <div class="title">${exp.mp4_filename}</div>
      <div class="meta">${dur} • ${sizeMB} MB • ${exp.template || 'unknown template'}</div>
      <div class="meta small" style="margin-top:4px">${exp.title || '(no title)'}</div>
    `;
    el.onclick = () => toggleExportSelection(exp.export_id);
    container.appendChild(el);
  }
}

function toggleExportSelection(exportId) {
  if (selectedExportIds.has(exportId)) {
    selectedExportIds.delete(exportId);
  } else {
    selectedExportIds.add(exportId);
  }
  renderPublishExports();
  updatePublishSelectionInfo();
}

function updatePublishSelectionInfo() {
  const info = $('#publishSelectionInfo');
  const numAccounts = selectedAccountIds.size;
  const numExports = selectedExportIds.size;

  if (numAccounts === 0 && numExports === 0) {
    info.textContent = 'Select exports and accounts above';
  } else if (numAccounts === 0) {
    info.textContent = `${numExports} export(s) selected — select accounts`;
  } else if (numExports === 0) {
    info.textContent = `${numAccounts} account(s) selected — select exports`;
  } else {
    const totalJobs = numAccounts * numExports;
    info.textContent = `${numExports} export(s) × ${numAccounts} account(s) = ${totalJobs} job(s)`;
  }
}

async function queuePublish() {
  if (selectedAccountIds.size === 0) {
    alert('Please select at least one account.');
    return;
  }
  if (selectedExportIds.size === 0) {
    alert('Please select at least one export.');
    return;
  }

  const options = {
    privacy: $('#publishPrivacy').value,
  };

  const titleOverride = $('#publishTitleOverride').value.trim();
  if (titleOverride) options.title_override = titleOverride;

  const descOverride = $('#publishDescOverride').value.trim();
  if (descOverride) options.description_override = descOverride;

  const hashtags = $('#publishHashtags').value.trim();
  if (hashtags) options.hashtags_append = hashtags;

  const stagger = parseInt($('#publishStagger').value, 10) || 0;

  try {
    const res = await apiJson('POST', '/api/publisher/queue_batch', {
      account_ids: Array.from(selectedAccountIds),
      export_ids: Array.from(selectedExportIds),
      options,
      stagger_seconds: stagger,
    });

    alert(`Queued ${res.total} publish job(s)!`);

    // Clear selections
    selectedAccountIds.clear();
    selectedExportIds.clear();
    renderPublishAccounts();
    renderPublishExports();
    updatePublishSelectionInfo();

    // Refresh jobs
    await loadPublishJobs();
  } catch (e) {
    alert(`Failed to queue publish: ${e.message}`);
  }
}

async function loadPublishJobs() {
  const container = $('#publishJobs');

  try {
    const res = await apiGet('/api/publisher/jobs?project_only=true&limit=50');
    const jobsList = res.jobs || [];

    publishJobs.clear();
    for (const j of jobsList) {
      publishJobs.set(j.id, j);
    }

    renderPublishJobs();
  } catch (e) {
    container.innerHTML = `<div class="small">Error: ${e.message}</div>`;
  }
}

function renderPublishJobs() {
  const container = $('#publishJobs');

  if (publishJobs.size === 0) {
    container.innerHTML = '<div class="small">No publish jobs yet.</div>';
    return;
  }

  container.innerHTML = '';
  const sorted = Array.from(publishJobs.values()).sort((a, b) =>
    a.created_at < b.created_at ? 1 : -1
  );

  for (const job of sorted) {
    const el = document.createElement('div');
    el.className = 'item';

    const pct = Math.round((job.progress || 0) * 100);
    const account = publishAccounts.find(a => a.id === job.account_id);
    const accountLabel = account ? account.label : job.account_id.slice(0, 8);
    const fileName = job.file_path.split(/[/\\]/).pop();

    let statusClass = job.status;
    let actions = '';

    if (job.status === 'failed' || job.status === 'canceled') {
      actions = `<button class="retry-btn" data-job-id="${job.id}">Retry</button>`;
    } else if (job.status === 'queued' || job.status === 'running') {
      actions = `<button class="cancel-btn danger" data-job-id="${job.id}">Cancel</button>`;
    }

    let resultInfo = '';
    if (job.status === 'succeeded' && job.remote_url) {
      resultInfo = `<a href="${job.remote_url}" target="_blank" class="small">Open on ${job.platform}</a>`;
    } else if (job.status === 'succeeded' && job.remote_id) {
      resultInfo = `<span class="small">ID: ${job.remote_id}</span>`;
    }

    let errorInfo = '';
    if (job.last_error) {
      errorInfo = `<div class="small" style="color:var(--danger);margin-top:4px">${job.last_error}</div>`;
    }

    el.innerHTML = `
      <div class="title">
        <span class="status-badge ${statusClass}">${job.status}</span>
        ${fileName} → ${accountLabel}
      </div>
      <div class="meta">${job.platform} • attempts: ${job.attempts}</div>
      ${job.status === 'running' ? `
        <div class="progress" style="margin-top:8px"><div style="width:${pct}%"></div></div>
        <div class="small" style="margin-top:4px">${pct}%</div>
      ` : ''}
      ${resultInfo ? `<div style="margin-top:6px">${resultInfo}</div>` : ''}
      ${errorInfo}
      ${actions ? `<div class="actions" style="margin-top:8px">${actions}</div>` : ''}
    `;

    // Wire up buttons
    const retryBtn = el.querySelector('.retry-btn');
    if (retryBtn) {
      retryBtn.onclick = (e) => {
        e.stopPropagation();
        retryPublishJob(job.id);
      };
    }

    const cancelBtn = el.querySelector('.cancel-btn');
    if (cancelBtn) {
      cancelBtn.onclick = (e) => {
        e.stopPropagation();
        cancelPublishJob(job.id);
      };
    }

    container.appendChild(el);
  }
}

async function retryPublishJob(jobId) {
  try {
    const res = await apiJson('POST', `/api/publisher/jobs/${jobId}/retry`, {});
    publishJobs.set(res.job.id, res.job);
    renderPublishJobs();
  } catch (e) {
    alert(`Failed to retry job: ${e.message}`);
  }
}

async function cancelPublishJob(jobId) {
  if (!confirm('Cancel this job?')) return;
  try {
    const res = await apiJson('POST', `/api/publisher/jobs/${jobId}/cancel`, {});
    publishJobs.set(res.job.id, res.job);
    renderPublishJobs();
  } catch (e) {
    alert(`Failed to cancel job: ${e.message}`);
  }
}

function startPublishJobsSSE() {
  // Close existing SSE if any
  if (publishJobsSSE) {
    publishJobsSSE.close();
  }

  publishJobsSSE = new EventSource('/api/publisher/jobs/stream');

  publishJobsSSE.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data);
      if (data.type === 'jobs_update' && data.jobs) {
        for (const job of data.jobs) {
          publishJobs.set(job.id, job);
        }
        renderPublishJobs();
      }
    } catch (e) {
      console.warn('Failed to parse SSE:', e);
    }
  };

  publishJobsSSE.onerror = () => {
    // Reconnect after a delay
    setTimeout(() => {
      if (currentTab === 'publish') {
        startPublishJobsSSE();
      }
    }, 3000);
  };
}

function wirePublishUI() {
  // Account filters
  const platformFilter = $('#publishPlatformFilter');
  const searchInput = $('#publishAccountSearch');

  if (platformFilter) platformFilter.onchange = renderPublishAccounts;
  if (searchInput) searchInput.oninput = renderPublishAccounts;

  // Copy command button
  const copyBtn = $('#btnCopyAccountCmd');
  if (copyBtn) {
    copyBtn.onclick = () => {
      const cmd = 'vp accounts add youtube --client-secrets "path/to/client_secret.json" --label "My Channel"';
      navigator.clipboard.writeText(cmd).then(() => {
        copyBtn.textContent = 'Copied!';
        setTimeout(() => { copyBtn.textContent = 'Copy command'; }, 2000);
      });
    };
  }

  // Queue publish button
  const queueBtn = $('#btnQueuePublish');
  if (queueBtn) queueBtn.onclick = queuePublish;
}

// =========================================================================
// CHAT REPLAY FUNCTIONS
// =========================================================================

let chatStatus = null;
let chatAutoScroll = true;
let chatSyncInterval = null;

async function loadChatStatus() {
  const panel = $('#chatPanel');
  if (!panel) return;

  try {
    chatStatus = await apiGet('/api/chat/status');
    updateChatUI();
  } catch (e) {
    chatStatus = null;
    updateChatUI();
  }
}

function updateChatUI() {
  const panel = $('#chatPanel');
  const statusEl = $('#chatStatus');
  const sourceUrlInput = $('#chatSourceUrl');
  const offsetInput = $('#chatOffsetMs');
  const messagesEl = $('#chatMessages');
  const downloadBtn = $('#btnDownloadChat');
  const clearBtn = $('#btnClearChat');
  const offsetControls = $('#chatOffsetControls');

  if (!panel) return;

  if (!chatStatus || !chatStatus.available) {
    statusEl.textContent = 'No chat replay available.';
    statusEl.className = 'small';
    if (messagesEl) messagesEl.innerHTML = '';
    if (offsetControls) offsetControls.style.display = 'none';
    if (downloadBtn) downloadBtn.disabled = false;
    if (clearBtn) clearBtn.style.display = 'none';

    // Pre-fill source URL from project if available
    if (sourceUrlInput && chatStatus?.source_url) {
      sourceUrlInput.value = chatStatus.source_url;
    }
    return;
  }

  statusEl.textContent = `Chat: ${chatStatus.message_count.toLocaleString()} messages`;
  statusEl.className = 'small success';
  if (offsetControls) offsetControls.style.display = 'flex';
  if (offsetInput) offsetInput.value = chatStatus.sync_offset_ms || 0;
  if (sourceUrlInput && chatStatus.source_url) {
    sourceUrlInput.value = chatStatus.source_url;
  }
  if (downloadBtn) downloadBtn.disabled = false;
  if (clearBtn) clearBtn.style.display = 'inline-block';

  // Start syncing chat
  startChatSync();
}

function startChatSync() {
  if (chatSyncInterval) {
    clearInterval(chatSyncInterval);
  }
  chatSyncInterval = setInterval(syncChatMessages, 500);
}

function stopChatSync() {
  if (chatSyncInterval) {
    clearInterval(chatSyncInterval);
    chatSyncInterval = null;
  }
}

async function syncChatMessages() {
  const v = $('#video');
  const messagesEl = $('#chatMessages');
  if (!v || !messagesEl || !chatStatus?.available) return;

  const currentTimeMs = Math.floor(v.currentTime * 1000);
  const windowMs = 10000; // Show 10 seconds of chat around current time
  const startMs = Math.max(0, currentTimeMs - windowMs / 2);
  const endMs = currentTimeMs + windowMs / 2;

  try {
    const res = await apiGet(`/api/chat/messages?start_ms=${startMs}&end_ms=${endMs}&limit=100`);
    if (!res.ok) return;

    renderChatMessages(res.messages, currentTimeMs);
  } catch (e) {
    // Silently fail
  }
}

function renderChatMessages(messages, currentTimeMs) {
  const messagesEl = $('#chatMessages');
  if (!messagesEl) return;

  if (!messages || messages.length === 0) {
    messagesEl.innerHTML = '<div class="small" style="opacity:0.5">No messages in this time range</div>';
    return;
  }

  const html = messages.map(m => {
    const offsetMs = chatStatus?.sync_offset_ms || 0;
    const msgTimeMs = m.t_ms - offsetMs;
    const timeSec = msgTimeMs / 1000;
    const isNear = Math.abs(msgTimeMs - currentTimeMs) < 2000;
    const nearClass = isNear ? 'chat-msg-near' : '';
    const timeStr = fmtTime(timeSec);
    
    return `
      <div class="chat-msg ${nearClass}" data-time="${timeSec}">
        <span class="chat-time" title="Click to seek">${timeStr}</span>
        <span class="chat-author">${escapeHtml(m.author || 'anon')}</span>
        <span class="chat-text">${escapeHtml(m.text || '')}</span>
      </div>
    `;
  }).join('');

  messagesEl.innerHTML = html;

  // Add click handlers for seeking
  messagesEl.querySelectorAll('.chat-msg').forEach(el => {
    el.onclick = () => {
      const t = parseFloat(el.dataset.time);
      if (!isNaN(t)) {
        const v = $('#video');
        if (v) v.currentTime = t;
      }
    };
  });

  // Auto-scroll to current time
  if (chatAutoScroll) {
    const nearEl = messagesEl.querySelector('.chat-msg-near');
    if (nearEl) {
      nearEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

async function downloadChat() {
  const sourceUrl = $('#chatSourceUrl')?.value?.trim();

  if (!sourceUrl) {
    alert('Please enter a source URL (e.g., Twitch VOD or YouTube URL)');
    return;
  }

  if (!sourceUrl.startsWith('http://') && !sourceUrl.startsWith('https://')) {
    alert('Please enter a valid URL starting with http:// or https://');
    return;
  }

  try {
    const res = await apiJson('POST', '/api/chat/download', { source_url: sourceUrl });
    watchJob(res.job_id);
    $('#chatStatus').textContent = 'Downloading chat...';
  } catch (e) {
    alert(`Failed to start chat download: ${e.message}`);
  }
}

async function clearChat() {
  if (!confirm('Clear all chat data for this project?')) return;

  try {
    await apiJson('POST', '/api/chat/clear', {});
    chatStatus = null;
    stopChatSync();
    updateChatUI();
    $('#chatMessages').innerHTML = '';
  } catch (e) {
    alert(`Failed to clear chat: ${e.message}`);
  }
}

async function setChatOffset() {
  const offsetInput = $('#chatOffsetMs');
  if (!offsetInput) return;

  const offset = parseInt(offsetInput.value, 10) || 0;

  try {
    await apiJson('POST', '/api/chat/set_offset', { sync_offset_ms: offset });
    if (chatStatus) chatStatus.sync_offset_ms = offset;
    syncChatMessages(); // Refresh immediately
  } catch (e) {
    alert(`Failed to set offset: ${e.message}`);
  }
}

function wireChatUI() {
  const downloadBtn = $('#btnDownloadChat');
  const clearBtn = $('#btnClearChat');
  const offsetInput = $('#chatOffsetMs');
  const autoScrollChk = $('#chatAutoScroll');
  const nudgeBackBtn = $('#chatNudgeBack');
  const nudgeFwdBtn = $('#chatNudgeFwd');

  if (downloadBtn) downloadBtn.onclick = downloadChat;
  if (clearBtn) clearBtn.onclick = clearChat;

  if (offsetInput) {
    offsetInput.onchange = setChatOffset;
    offsetInput.onkeydown = (e) => {
      if (e.key === 'Enter') setChatOffset();
    };
  }

  if (autoScrollChk) {
    autoScrollChk.onchange = () => {
      chatAutoScroll = autoScrollChk.checked;
    };
  }

  if (nudgeBackBtn) {
    nudgeBackBtn.onclick = () => {
      const offsetInput = $('#chatOffsetMs');
      if (offsetInput) {
        offsetInput.value = (parseInt(offsetInput.value, 10) || 0) - 1000;
        setChatOffset();
      }
    };
  }

  if (nudgeFwdBtn) {
    nudgeFwdBtn.onclick = () => {
      const offsetInput = $('#chatOffsetMs');
      if (offsetInput) {
        offsetInput.value = (parseInt(offsetInput.value, 10) || 0) + 1000;
        setChatOffset();
      }
    };
  }
}

async function main() {
  try {
    profile = await apiGet('/api/profile');
  } catch (_) {}

  // Wire up home UI first
  wireHomeUI();
  wireTabUI();
  wirePublishUI();

  // Check if there's an active project
  try {
    const res = await apiGet('/api/project');
    if (res.active && res.project) {
      project = res.project;
      showStudioView();
      await initStudioView();
    } else {
      showHomeView();
    }
  } catch (e) {
    console.error('Failed to check project status:', e);
    showHomeView();
  }
}

async function initStudioView() {
  renderProjectInfo();
  renderCandidates();
  renderSelections();
  renderJobs();
  await refreshTimeline();
  await loadChatStatus();

  try {
    const layout = await apiGet('/api/layout');
    if (layout?.facecam) {
      facecamRect = layout.facecam;
      updateFacecamStatus();
    }
  } catch (_) {}

  // Set default builder from first candidate if exists
  const cands = project?.analysis?.highlights?.candidates || project?.analysis?.audio?.candidates || [];
  if (cands.length > 0) {
    currentCandidate = cands[0];
    setBuilder(cands[0].start_s, cands[0].end_s);
  } else {
    setBuilder(0, 10);
  }

  // Apply defaults from profile for export widgets
  if (profile?.export) {
    $('#expW').value = profile.export.width ?? 1080;
    $('#expH').value = profile.export.height ?? 1920;
    $('#expFps').value = profile.export.fps ?? 30;
    $('#expCrf').value = profile.export.crf ?? 20;
    $('#expPreset').value = profile.export.preset ?? 'veryfast';
    $('#normalizeAudio').checked = !!profile.export.normalize_audio;
    $('#template').value = profile.export.template ?? 'vertical_blur';
    $('#batchTemplate').value = profile.export.template ?? 'vertical_blur';
  }
  if (profile?.captions) {
    $('#withCaptions').checked = !!profile.captions.enabled;
  }

  wireUI();
  wireChatUI();
  updateFacecamStatus();
}

main();
