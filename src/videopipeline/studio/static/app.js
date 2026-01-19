const $ = (sel) => document.querySelector(sel);

let project = null;
let profile = null;
let chart = null;
let currentCandidate = null;
let facecamRect = null;
let lastBatchSelectionIds = [];
let calibrating = false;

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
        if ((j.kind === 'analyze_audio' || j.kind === 'analyze_highlights') && j.status === 'succeeded') {
          // refresh project and timeline
          refreshProject();
        }
        if (j.kind === 'export' && j.status === 'succeeded') {
          // refresh project to show exports list eventually
          refreshProject();
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
  project = await apiGet('/api/project');
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

async function main() {
  try {
    profile = await apiGet('/api/profile');
  } catch (_) {}

  await refreshProject();

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
  updateFacecamStatus();
}

main();
