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
  loadRecentVideos();
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

// Track selected videos
const selectedVideos = new Set();

function updateVideoSelectionToolbar() {
  const toolbar = $('#videoSelectionToolbar');
  const countEl = $('#videoSelectionCount');
  const btnDeleteProjects = $('#btnDeleteSelectedProjects');
  
  if (!toolbar) return;
  
  const count = selectedVideos.size;
  if (count > 0) {
    toolbar.style.display = 'flex';
    countEl.textContent = `${count} selected`;
    
    // Check if any selected videos have projects
    const hasProjects = Array.from(selectedVideos).some(path => {
      const checkbox = document.querySelector(`.video-checkbox[data-path="${CSS.escape(path)}"]`);
      return checkbox && checkbox.dataset.hasProject === 'true';
    });
    btnDeleteProjects.style.display = hasProjects ? 'inline-block' : 'none';
  } else {
    toolbar.style.display = 'none';
  }
}

function setupVideoSelectionToolbar() {
  const btnSelectAll = $('#btnSelectAll');
  const btnDeselectAll = $('#btnDeselectAll');
  const btnDeleteProjects = $('#btnDeleteSelectedProjects');
  const btnDeleteVideos = $('#btnDeleteSelectedVideos');
  
  if (btnSelectAll) {
    btnSelectAll.onclick = () => {
      document.querySelectorAll('.video-checkbox').forEach(cb => {
        cb.checked = true;
        selectedVideos.add(cb.dataset.path);
      });
      updateVideoSelectionToolbar();
    };
  }
  
  if (btnDeselectAll) {
    btnDeselectAll.onclick = () => {
      document.querySelectorAll('.video-checkbox').forEach(cb => {
        cb.checked = false;
      });
      selectedVideos.clear();
      updateVideoSelectionToolbar();
    };
  }
  
  if (btnDeleteProjects) {
    btnDeleteProjects.onclick = async () => {
      const paths = Array.from(selectedVideos);
      const projectPaths = paths.filter(path => {
        const cb = document.querySelector(`.video-checkbox[data-path="${CSS.escape(path)}"]`);
        return cb && cb.dataset.hasProject === 'true';
      });
      
      if (projectPaths.length === 0) {
        alert('No projects to delete.');
        return;
      }
      
      if (!confirm(`Delete ${projectPaths.length} project(s)?\n\nThis will remove analysis data, selections, and exports. The video files will NOT be deleted.`)) {
        return;
      }
      
      for (const path of projectPaths) {
        const cb = document.querySelector(`.video-checkbox[data-path="${CSS.escape(path)}"]`);
        if (cb && cb.dataset.projectId) {
          try {
            await apiJson('DELETE', `/api/home/project/${cb.dataset.projectId}`, null);
          } catch (err) {
            console.error(`Failed to delete project for ${path}:`, err);
          }
        }
      }
      selectedVideos.clear();
      loadRecentVideos();
    };
  }
  
  if (btnDeleteVideos) {
    btnDeleteVideos.onclick = async () => {
      const paths = Array.from(selectedVideos);
      if (paths.length === 0) {
        alert('No videos selected.');
        return;
      }
      
      const hasProjects = paths.some(path => {
        const cb = document.querySelector(`.video-checkbox[data-path="${CSS.escape(path)}"]`);
        return cb && cb.dataset.hasProject === 'true';
      });
      
      let msg = `Delete ${paths.length} video(s)?\n\nThis will permanently delete the video files.`;
      if (hasProjects) {
        msg += `\n\nAssociated projects (analysis data, selections, exports) will also be deleted.`;
      }
      
      if (!confirm(msg)) return;
      
      for (const path of paths) {
        try {
          await apiJson('DELETE', '/api/home/video', { video_path: path, delete_project: true });
        } catch (err) {
          console.error(`Failed to delete video ${path}:`, err);
        }
      }
      selectedVideos.clear();
      loadRecentVideos();
    };
  }
}

async function loadRecentVideos() {
  const container = $('#recentVideos');
  if (!container) return;
  container.innerHTML = '<div class="small">Loading...</div>';
  
  // Clear selection when reloading
  selectedVideos.clear();
  updateVideoSelectionToolbar();

  try {
    const res = await apiGet('/api/home/videos');
    const videos = res.videos || [];

    if (videos.length === 0) {
      container.innerHTML = '<div class="small">No videos found. Download or open a video to get started.</div>';
      return;
    }

    container.innerHTML = '';
    for (const v of videos) {
      const el = document.createElement('div');
      el.className = 'item' + (v.favorite ? ' favorite' : '');
      const dur = fmtTime(v.duration_seconds || 0);
      const sizeMB = ((v.size_bytes || 0) / 1024 / 1024).toFixed(1);
      
      // Build status badges
      let badges = '';
      if (v.favorite) {
        badges += `<span class="badge badge-favorite">★ Favorite</span> `;
      }
      if (v.has_project) {
        badges += `<span class="badge" style="background:#22c55e;color:#000">Project</span> `;
        if (v.selections_count > 0) badges += `<span class="badge">${v.selections_count} sel</span> `;
        if (v.exports_count > 0) badges += `<span class="badge">${v.exports_count} exp</span> `;
      }
      if (v.extractor) {
        badges += `<span class="badge" style="background:#6366f1">${v.extractor}</span> `;
      }
      
      el.innerHTML = `
        <div class="item-header">
          <input type="checkbox" class="video-checkbox" data-path="${v.path.replace(/"/g, '&quot;')}" data-has-project="${v.has_project}" data-project-id="${v.project_id || ''}" style="margin-right:8px;cursor:pointer" />
          <div class="title" style="flex:1">${v.title || v.filename}</div>
          <button class="btn-favorite ${v.favorite ? 'active' : ''}" title="${v.favorite ? 'Remove from favorites' : 'Add to favorites'}" data-path="${v.path.replace(/"/g, '&quot;')}">
            ${v.favorite ? '★' : '☆'}
          </button>
        </div>
        <div class="meta" style="margin-left:24px">${dur} • ${sizeMB} MB ${badges}</div>
        <div class="meta small" style="opacity:0.7;word-break:break-all;margin-left:24px">${v.path}</div>
        <div class="actions" style="margin-top:8px;display:flex;gap:8px;flex-wrap:wrap;margin-left:24px">
          <button class="primary btn-open">Open</button>
          ${v.has_project ? `<button class="btn-delete-project" style="background:#ef4444" data-project-id="${v.project_id}">Delete Project</button>` : ''}
          <button class="btn-delete-video" style="background:#dc2626" data-path="${v.path.replace(/"/g, '&quot;')}" data-has-project="${v.has_project}">Delete Video</button>
        </div>
      `;
      
      // Checkbox for multi-select
      const checkbox = el.querySelector('.video-checkbox');
      checkbox.onclick = (e) => {
        e.stopPropagation();
        if (checkbox.checked) {
          selectedVideos.add(v.path);
        } else {
          selectedVideos.delete(v.path);
        }
        updateVideoSelectionToolbar();
      };
      
      const btnOpen = el.querySelector('.btn-open');
      btnOpen.onclick = () => openProjectByPath(v.path);
      
      // Favorite button
      const btnFavorite = el.querySelector('.btn-favorite');
      if (btnFavorite) {
        btnFavorite.onclick = async (e) => {
          e.stopPropagation();
          try {
            await apiJson('POST', '/api/home/favorite', { video_path: v.path });
            loadRecentVideos(); // Refresh list
          } catch (err) {
            alert(`Failed to toggle favorite: ${err.message}`);
          }
        };
      }
      
      // Delete project button
      const btnDeleteProject = el.querySelector('.btn-delete-project');
      if (btnDeleteProject) {
        btnDeleteProject.onclick = async (e) => {
          e.stopPropagation();
          const projectId = btnDeleteProject.dataset.projectId;
          if (confirm(`Delete project for "${v.title || v.filename}"?\n\nThis will remove analysis data, selections, and exports. The video file will NOT be deleted.`)) {
            try {
              await apiJson('DELETE', `/api/home/project/${projectId}`, null);
              loadRecentVideos(); // Refresh list
            } catch (err) {
              alert(`Failed to delete project: ${err.message}`);
            }
          }
        };
      }
      
      // Delete video button
      const btnDeleteVideo = el.querySelector('.btn-delete-video');
      if (btnDeleteVideo) {
        btnDeleteVideo.onclick = async (e) => {
          e.stopPropagation();
          const hasProject = btnDeleteVideo.dataset.hasProject === 'true';
          let msg = `Delete video "${v.title || v.filename}"?\n\nThis will permanently delete the video file.`;
          if (hasProject) {
            msg += `\n\nThe associated project (analysis data, selections, exports) will also be deleted.`;
          }
          if (confirm(msg)) {
            try {
              await apiJson('DELETE', '/api/home/video', { video_path: v.path, delete_project: true });
              loadRecentVideos(); // Refresh list
            } catch (err) {
              alert(`Failed to delete video: ${err.message}`);
            }
          }
        };
      }
      
      container.appendChild(el);
    }
    
    // Setup toolbar handlers after videos are loaded
    setupVideoSelectionToolbar();
  } catch (e) {
    container.innerHTML = `<div class="small">Error loading videos: ${e.message}</div>`;
  }
}

// Keep old function names for backwards compatibility
async function loadRecentProjects() {
  return loadRecentVideos();
}

async function loadRecentDownloads() {
  // Deprecated - now merged into loadRecentVideos
  return;
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
        
        // Close event source when job is finished
        if (j.status === 'succeeded' || j.status === 'failed' || j.status === 'cancelled') {
          es.close();
        }
      }
    } catch (e) {
      console.warn('bad job payload', e);
    }
  };
}

async function cancelJob(jobId) {
  if (!confirm('Cancel this download? Any downloaded data will be removed.')) {
    return;
  }
  try {
    await apiJson('POST', `/api/jobs/${jobId}/cancel`, {});
  } catch (e) {
    alert(`Failed to cancel: ${e.message}`);
  }
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
    else if (job.status === 'cancelled') statusClass = 'status-badge canceled';
    else statusClass = 'status-badge queued';

    let resultInfo = '';
    if (job.status === 'succeeded' && job.result?.video_path) {
      const filename = job.result.video_path.split(/[/\\]/).pop();
      resultInfo = `<div class="small" style="margin-top:4px">Downloaded: ${filename}</div>`;
      
      // Show chat status
      const chat = job.result?.chat;
      if (chat) {
        if (chat.imported) {
          resultInfo += `<div class="small" style="color:#22c55e;margin-top:2px">✓ Chat imported</div>`;
        } else if (chat.import_error) {
          resultInfo += `<div class="small" style="color:#f59e0b;margin-top:2px">⚠ Chat import failed: ${chat.import_error}</div>`;
        } else if (chat.status === 'failed') {
          resultInfo += `<div class="small" style="color:#f59e0b;margin-top:2px">⚠ Chat download failed: ${chat.message}</div>`;
        } else if (chat.status === 'skipped') {
          resultInfo += `<div class="small" style="color:#888;margin-top:2px">○ ${chat.message}</div>`;
        }
      }
    }
    
    // Show chat status while running
    let chatStatusInfo = '';
    if (job.status === 'running' && job.result?.chat_status) {
      const cs = job.result.chat_status;
      const chatMsg = job.result.chat_message || '';
      const chatPct = job.result.chat_progress;
      if (cs === 'downloading') {
        // Show detailed chat progress if available
        let chatDetail = chatMsg || 'Downloading...';
        if (chatPct !== undefined && chatPct > 0) {
          const chatPctDisplay = Math.round(chatPct * 100);
          chatStatusInfo = `<div class="small" style="color:#6366f1;margin-top:2px">📥 Chat: ${chatDetail}</div>`;
        } else {
          chatStatusInfo = `<div class="small" style="color:#6366f1;margin-top:2px">📥 Chat: ${chatDetail}</div>`;
        }
      } else if (cs === 'importing') {
        chatStatusInfo = `<div class="small" style="color:#a855f7;margin-top:2px">⚙️ Importing chat: ${chatMsg || 'Processing...'}</div>`;
      } else if (cs === 'pending') {
        chatStatusInfo = `<div class="small" style="color:#888;margin-top:2px">○ Chat: pending</div>`;
      }
    }

    let errorInfo = '';
    if (job.status === 'failed') {
      errorInfo = `<div class="small" style="color:var(--danger);margin-top:4px">${job.message}</div>`;
    }

    el.innerHTML = `
      <div class="title">
        <span class="${statusClass}">${job.status}</span>
        ${job.kind === 'download_url' ? 'URL Download' : job.kind}
        ${job.status === 'running' ? `<button class="btn btn-small btn-danger" style="margin-left:auto;padding:2px 8px;font-size:11px" onclick="cancelJob('${job.id}')">Cancel</button>` : ''}
      </div>
      <div class="meta">${job.message || ''}</div>
      ${chatStatusInfo}
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
  const director = project?.analysis?.director;
  const candidates = highlights?.candidates || audio?.candidates || [];
  if (candidates.length === 0) {
    container.innerHTML = `<div class="small">No candidates yet. Click "Analyze highlights".</div>`;
    return;
  }

  for (const c of candidates) {
    const el = document.createElement('div');
    el.className = 'item';
    let breakdown = '';
    if (c.breakdown) {
      const audio = Number(c.breakdown.audio).toFixed(2);
      const motion = Number(c.breakdown.motion).toFixed(2);
      const chat = Number(c.breakdown.chat).toFixed(2);
      const audioEvents = Number(c.breakdown.audio_events || 0).toFixed(2);
      breakdown = ` • audio ${audio} / motion ${motion}`;
      if (Number(c.breakdown.chat) !== 0) breakdown += ` / chat ${chat}`;
      if (Number(c.breakdown.audio_events || 0) !== 0) breakdown += ` / events ${audioEvents}`;
    }
    
    // Get AI metadata if available
    const aiResult = director?.results?.find(r => r.rank === c.rank);
    const aiTitle = aiResult?.title || '';
    const aiHook = aiResult?.hook || '';
    const aiVariant = aiResult?.chosen_variant || '';
    const hasAI = !!aiResult;
    
    let aiHtml = '';
    if (hasAI) {
      const aiTags = (aiResult?.tags || []).slice(0, 5);
      const tagsHtml = aiTags.length > 0 ? `<div class="ai-tags">${aiTags.map(t => `<span class="ai-tag">#${escapeHtml(t)}</span>`).join('')}</div>` : '';
      aiHtml = `
        <div class="candidate-ai-section">
          <span class="ai-badge">AI Generated</span>
          <div class="ai-title">${escapeHtml(aiTitle)}</div>
          ${aiHook ? `<div class="ai-hook">"${escapeHtml(aiHook)}"</div>` : ''}
          ${tagsHtml}
          <div class="small" style="margin-top:4px;">Best variant: <span class="badge">${aiVariant}</span></div>
        </div>
      `;
    }
    
    el.innerHTML = `
      <div class="title">#${c.rank} • score ${c.score.toFixed(2)} <span class="badge">peak ${fmtTime(c.peak_time_s)}</span></div>
      <div class="meta">Clip: ${fmtTime(c.start_s)} → ${fmtTime(c.end_s)} (${(c.end_s - c.start_s).toFixed(1)}s)${breakdown}</div>
      ${aiHtml}
      <div class="actions">
        <button class="primary">Load</button>
        <button>Seek peak</button>
        ${hasAI ? '<button class="apply-ai">Apply AI</button>' : ''}
        ${hasAI ? '<button class="variants">Variants</button>' : ''}
      </div>
    `;
    const btnLoad = el.querySelector('button.primary');
    const btnPeak = el.querySelectorAll('button')[1];
    const btnApplyAI = el.querySelector('.apply-ai');
    const btnVariants = el.querySelector('.variants');
    
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
    if (btnApplyAI && aiResult) {
      btnApplyAI.onclick = () => {
        // Apply AI-suggested variant times and title
        const variant = aiResult.chosen_variant_data;
        if (variant) {
          setBuilder(variant.start_s, variant.end_s, aiTitle, $('#template').value);
          const v = $('#video');
          v.currentTime = variant.start_s;
        } else {
          setBuilder(c.start_s, c.end_s, aiTitle, $('#template').value);
        }
      };
    }
    if (btnVariants && aiResult) {
      btnVariants.onclick = () => showVariantsModal(c.rank, aiResult);
    }
    container.appendChild(el);
  }
}

// Modal for showing clip variants
function showVariantsModal(rank, aiResult) {
  // Remove existing modal if any
  const existing = document.querySelector('.variants-modal-overlay');
  if (existing) existing.remove();
  
  // Fetch variants from API
  apiGet(`/api/clip_variants/${rank}`).then(data => {
    const variants = data.variants || [];
    if (variants.length === 0) {
      alert('No variants available for this candidate.');
      return;
    }
    
    const overlay = document.createElement('div');
    overlay.className = 'variants-modal-overlay';
    overlay.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.8);z-index:1000;display:flex;align-items:center;justify-content:center;';
    
    let variantsHtml = variants.map((v, i) => {
      const isChosen = v.variant_id === aiResult?.chosen_variant;
      return `
        <div class="variant-item ${isChosen ? 'chosen' : ''}" data-idx="${i}" style="padding:12px;margin:8px 0;background:${isChosen ? 'rgba(79,140,255,0.2)' : 'rgba(255,255,255,0.05)'};border-radius:6px;cursor:pointer;">
          <div style="font-weight:600;">${v.variant_id} ${isChosen ? '✓ AI choice' : ''}</div>
          <div class="small">${fmtTime(v.start_s)} → ${fmtTime(v.end_s)} (${v.duration_s.toFixed(1)}s)</div>
          <div class="small" style="margin-top:4px;">Strategy: ${v.strategy} • Cut in: ${v.cut_in_reason} • Cut out: ${v.cut_out_reason}</div>
        </div>
      `;
    }).join('');
    
    const modal = document.createElement('div');
    modal.style.cssText = 'background:var(--bg);padding:24px;border-radius:8px;max-width:500px;max-height:80vh;overflow-y:auto;';
    modal.innerHTML = `
      <h3 style="margin:0 0 16px 0;">Clip Variants for #${rank}</h3>
      ${variantsHtml}
      <div style="margin-top:16px;text-align:right;">
        <button class="close-modal">Close</button>
      </div>
    `;
    
    overlay.appendChild(modal);
    document.body.appendChild(overlay);
    
    // Wire up click handlers
    modal.querySelectorAll('.variant-item').forEach((el, i) => {
      el.onclick = () => {
        const v = variants[i];
        setBuilder(v.start_s, v.end_s, aiResult?.title || '', $('#template').value);
        const video = $('#video');
        video.currentTime = v.start_s;
        overlay.remove();
      };
    });
    
    modal.querySelector('.close-modal').onclick = () => overlay.remove();
    overlay.onclick = (e) => { if (e.target === overlay) overlay.remove(); };
  }).catch(e => {
    alert(`Failed to load variants: ${e.message}`);
  });
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

  // Helper to format seconds nicely, or handle "cached" marker
  const formatTime = (seconds) => {
    if (seconds === "cached") return 'cached';
    if (seconds == null) return '';
    if (typeof seconds !== 'number') return String(seconds);
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}m ${secs.toFixed(0)}s`;
  };
  
  // Helper to compute elapsed time from a timestamp
  const getElapsed = (startedAt) => {
    if (!startedAt) return null;
    return (Date.now() / 1000) - startedAt;
  };

  for (const job of Array.from(jobs.values()).sort((a,b) => (a.created_at < b.created_at ? 1 : -1))) {
    const el = document.createElement('div');
    el.className = 'item';
    const pct = Math.round((job.progress || 0) * 100);
    
    // Build elapsed time display (total time)
    let totalTimeHtml = '';
    if (job.elapsed_seconds != null) {
      totalTimeHtml = `<span style="color:#4f8cff;font-weight:600">${formatTime(job.elapsed_seconds)}</span>`;
    }
    
    // Build detailed status for analyze_full jobs
    let detailHtml = '';
    if (job.kind === 'analyze_full' && job.result) {
      const r = job.result;
      const stageTimes = r.stage_times || {};
      const currentStage = r.current_stage || {};
      
      if (r.stage === 1 && (r.pending || r.completed || r.failed)) {
        // Stage 1 detailed view - parallel tasks
        const taskTimes = r.task_times || {};
        const completed = r.completed || [];
        const failed = r.failed || [];
        const pending = r.pending || [];
        
        // In parallel stage 1, pending tasks are actually running in parallel
        // So we show them as "running" not "pending"
        const runningTasks = pending; // These are running in parallel
        
        // Build a clearer display
        let tasksHtml = '<div style="margin-top:6px;font-size:11px;line-height:1.6">';
        
        // Completed tasks - distinguish cached from computed
        for (const t of completed) {
          const time = taskTimes[t];
          const isCached = time === "cached";
          if (isCached) {
            tasksHtml += `<div style="color:#888">✓ ${t} <span style="font-style:italic">(cached)</span></div>`;
          } else {
            const timeStr = time != null ? formatTime(time) : '';
            tasksHtml += `<div style="color:#22c55e">✓ ${t} <span style="color:#888">${timeStr}</span></div>`;
          }
        }
        
        // Failed tasks
        for (const t of failed) {
          const time = taskTimes[t] != null ? formatTime(taskTimes[t]) : '';
          tasksHtml += `<div style="color:#ef4444">✗ ${t} <span style="color:#888">${time}</span></div>`;
        }
        
        // Running tasks (pending in parallel mode means running)
        const taskProgressInfo = r.task_progress || {};
        for (const t of runningTasks) {
          const taskStartedAt = r.task_start_times?.[t];
          const elapsed = getElapsed(taskStartedAt);
          const elapsedStr = elapsed != null ? formatTime(elapsed) : '';
          
          // Check if we have detailed progress for this task
          const progress = taskProgressInfo[t];
          let statusStr = '';
          if (progress && progress.message) {
            statusStr = progress.message;
            if (elapsedStr) statusStr += ` (${elapsedStr})`;
          } else {
            statusStr = `running${elapsedStr ? ` ${elapsedStr}` : '...'}`;
          }
          
          tasksHtml += `<div style="color:#fbbf24">▸ ${t} <span style="color:#888;font-style:italic">${statusStr}</span></div>`;
          
          // Show mini progress bar for tasks with progress
          if (progress && progress.progress > 0 && progress.progress < 1) {
            const pct = Math.round(progress.progress * 100);
            tasksHtml += `<div style="margin-left:16px;margin-top:2px;margin-bottom:4px">
              <div style="background:#333;border-radius:2px;height:4px;width:100%;max-width:200px">
                <div style="background:#fbbf24;height:100%;width:${pct}%;border-radius:2px;transition:width 0.3s"></div>
              </div>
            </div>`;
          }
        }
        
        tasksHtml += '</div>';
        detailHtml = tasksHtml;
        
      } else if (r.completed_stages) {
        // Final result with stage timing
        const candidateCount = r.candidates_count || 0;
        const errorCount = (r.errors || []).length;
        detailHtml = `<div class="meta" style="margin-top:4px;font-size:11px">`;
        detailHtml += `Stages: ${r.completed_stages.join(', ')}`;
        if (candidateCount > 0) detailHtml += ` | <strong>${candidateCount} candidates</strong>`;
        if (errorCount > 0) detailHtml += ` | <span style="color:#f59e0b">${errorCount} errors</span>`;
        detailHtml += `</div>`;
        
        // Add stage timing breakdown - separate cached from computed
        if (Object.keys(stageTimes).length > 0) {
          const computedParts = [];
          const cachedStages = [];
          for (const [stage, time] of Object.entries(stageTimes)) {
            if (time === "cached") {
              cachedStages.push(stage);
            } else {
              computedParts.push(`${stage}: ${formatTime(time)}`);
            }
          }
          let timingHtml = '';
          if (computedParts.length > 0) {
            timingHtml += `<div class="meta" style="margin-top:2px;font-size:10px;color:#22c55e">⏱ ${computedParts.join(' | ')}</div>`;
          }
          if (cachedStages.length > 0) {
            timingHtml += `<div class="meta" style="margin-top:2px;font-size:10px;color:#888">📦 Cached: ${cachedStages.join(', ')}</div>`;
          }
          detailHtml += timingHtml;
        }
        
        if (r.errors && r.errors.length > 0) {
          detailHtml += `<div class="meta" style="margin-top:2px;font-size:10px;color:#ef4444">${r.errors.join('<br/>')}</div>`;
        }
      } else if (Object.keys(stageTimes).length > 0 || currentStage.name) {
        // In-progress after Stage 1 - sequential stages
        let tasksHtml = '<div style="margin-top:6px;font-size:11px;line-height:1.6">';
        
        // Show completed stages - distinguish cached from computed
        for (const [stage, time] of Object.entries(stageTimes)) {
          if (time === "cached") {
            tasksHtml += `<div style="color:#888">✓ ${stage} <span style="font-style:italic">(cached)</span></div>`;
          } else {
            tasksHtml += `<div style="color:#22c55e">✓ ${stage} <span style="color:#888">${formatTime(time)}</span></div>`;
          }
        }
        
        // Show currently running stage with elapsed time and progress
        if (currentStage.name) {
          const elapsed = getElapsed(currentStage.started_at);
          const elapsedStr = elapsed != null ? formatTime(elapsed) : '';
          
          // Check for detailed progress for this stage
          const taskProgressInfo = r.task_progress || {};
          const progress = taskProgressInfo[currentStage.name];
          let statusStr = '';
          if (progress && progress.message) {
            statusStr = progress.message;
            if (elapsedStr) statusStr += ` (${elapsedStr})`;
          } else {
            statusStr = `running${elapsedStr ? ` ${elapsedStr}` : '...'}`;
          }
          
          tasksHtml += `<div style="color:#fbbf24">▸ ${currentStage.name} <span style="color:#888;font-style:italic">${statusStr}</span></div>`;
          
          // Show mini progress bar for stages with progress
          if (progress && progress.progress > 0 && progress.progress < 1) {
            const pct = Math.round(progress.progress * 100);
            tasksHtml += `<div style="margin-left:16px;margin-top:2px;margin-bottom:4px">
              <div style="background:#333;border-radius:2px;height:4px;width:100%;max-width:200px">
                <div style="background:#fbbf24;height:100%;width:${pct}%;border-radius:2px;transition:width 0.3s"></div>
              </div>
            </div>`;
          }
        }
        
        tasksHtml += '</div>';
        detailHtml = tasksHtml;
      }
    }
    
    // Add cancel button for running jobs
    const cancelBtnHtml = (job.status === 'running' || job.status === 'queued') 
      ? `<button class="danger cancel-job-btn" style="padding:4px 12px;font-size:11px" data-job-id="${job.id}">Cancel</button>` 
      : '';
    
    el.innerHTML = `
      <div class="title">${job.kind} • <span class="badge">${job.status}</span></div>
      <div class="meta" style="display:flex;justify-content:space-between;align-items:center">
        <span>${job.message || ''}</span>
        ${totalTimeHtml ? `<span style="margin-left:8px">Total: ${totalTimeHtml}</span>` : ''}
      </div>
      ${job.result?.output ? `<div class="meta">Output: ${job.result.output}</div>` : ''}
      ${detailHtml}
      <div class="progress" style="margin-top:10px"><div style="width:${pct}%"></div></div>
      <div class="meta" style="margin-top:6px;display:flex;justify-content:space-between;align-items:center">
        <span>${pct}%</span>
        ${cancelBtnHtml}
      </div>
    `;
    
    // Add cancel button handler
    const cancelBtn = el.querySelector('.cancel-job-btn');
    if (cancelBtn) {
      cancelBtn.onclick = async (e) => {
        e.stopPropagation();
        if (!confirm('Cancel this job?')) return;
        try {
          await apiJson('POST', `/api/jobs/${job.id}/cancel`, {});
        } catch (err) {
          alert(`Failed to cancel: ${err.message}`);
        }
      };
    }
    
    container.appendChild(el);
  }
}

// Timer for live elapsed time updates on running jobs
let jobsTimerInterval = null;

function startJobsTimer() {
  if (jobsTimerInterval) return; // Already running
  jobsTimerInterval = setInterval(() => {
    // Check if there are any running jobs
    const hasRunning = Array.from(jobs.values()).some(j => j.status === 'running');
    if (hasRunning) {
      renderJobs();
    } else {
      stopJobsTimer();
    }
  }, 1000); // Update every second
}

function stopJobsTimer() {
  if (jobsTimerInterval) {
    clearInterval(jobsTimerInterval);
    jobsTimerInterval = null;
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
        
        // Start/stop timer based on job status
        if (j.status === 'running') {
          startJobsTimer();
        }
        
        if ((j.kind === 'analyze_audio' || j.kind === 'analyze_highlights' || j.kind === 'analyze_speech' || j.kind === 'analyze_context_titles' || j.kind === 'analyze_full') && j.status === 'succeeded') {
          // refresh project and timeline to show reranked candidates
          refreshProject();
        }
        if (j.kind === 'export' && j.status === 'succeeded') {
          // refresh project to show exports list eventually
          refreshProject();
        }
        if ((j.kind === 'download_chat' || j.kind === 'download_url') && j.status === 'succeeded') {
          // refresh chat status after download completes
          loadChatStatus();
          // Also refresh videos list in home view if visible
          if ($('#recentVideos')) {
            loadRecentVideos();
          }
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

async function startAnalyzeAudioEventsJob() {
  $('#analysisStatus').textContent = 'Starting audio events analysis (laughter/cheer/shout)...';
  const res = await apiJson('POST', '/api/analyze/audio_events', {});
  const jobId = res.job_id;
  watchJob(jobId);
  $('#analysisStatus').textContent = `Audio events analysis running (job ${jobId.slice(0,8)})...`;
}

async function startAnalyzeFullJob() {
  $('#analysisStatus').textContent = 'Starting full parallel analysis (DAG)...';
  const motionMode = $('#motionWeightMode')?.value || 'low';
  const whisperBackend = $('#whisperBackend')?.value || 'auto';
  const res = await apiJson('POST', '/api/analyze/full', {
    highlights: {
      motion_weight_mode: motionMode
    },
    speech: {
      backend: whisperBackend,
      strict: whisperBackend !== 'auto'  // Strict mode when explicitly choosing a backend
    }
  });
  const jobId = res.job_id;
  watchJob(jobId);
  $('#analysisStatus').textContent = `Full analysis running (job ${jobId.slice(0,8)})...`;
}

async function startAnalyzeSpeechJob() {
  $('#analysisStatus').textContent = 'Starting speech analysis (Whisper transcription)...';
  const whisperBackend = $('#whisperBackend')?.value || 'auto';
  const res = await apiJson('POST', '/api/analyze/speech', {
    speech: {
      backend: whisperBackend,
      strict: whisperBackend !== 'auto'  // Strict mode when explicitly choosing a backend
    }
  });
  const jobId = res.job_id;
  watchJob(jobId);
  $('#analysisStatus').textContent = `Speech analysis running (job ${jobId.slice(0,8)})...`;
}

async function startAnalyzeContextJob() {
  $('#analysisStatus').textContent = 'Starting context + titles analysis (AI)...';
  const res = await apiJson('POST', '/api/analyze/context_titles', {});
  const jobId = res.job_id;
  watchJob(jobId);
  $('#analysisStatus').textContent = `Context analysis running (job ${jobId.slice(0,8)})...`;
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
  $('#btnAnalyzeAudioEvents').onclick = async () => {
    try {
      await startAnalyzeAudioEventsJob();
    } catch (e) {
      alert(`Audio events analysis failed: ${e}`);
    }
  };
  $('#btnAnalyzeFull').onclick = async () => {
    try {
      await startAnalyzeFullJob();
    } catch (e) {
      alert(`Full analysis failed: ${e}`);
    }
  };
  $('#btnAnalyzeSpeech').onclick = async () => {
    try {
      await startAnalyzeSpeechJob();
    } catch (e) {
      alert(`Speech analysis failed: ${e}`);
    }
  };
  $('#btnAnalyzeContext').onclick = async () => {
    try {
      await startAnalyzeContextJob();
    } catch (e) {
      alert(`Context analysis failed: ${e}`);
    }
  };

  $('#btnResetAnalysis').onclick = async () => {
    const keepChat = confirm('Keep chat data?\n\nClick OK to keep chat data, or Cancel to delete everything including chat.');
    const keepTranscript = confirm('Keep transcript?\n\nClick OK to keep the transcript (Whisper output), or Cancel to delete it too.');
    
    if (!confirm(`Reset all analysis data?\n\nThis will delete:\n- Audio/motion/highlight analysis\n- Scene detection\n- Speech features\n- Audio events\n- Clip variants\n- AI director results\n${keepChat ? '(Keeping chat data)' : '- Chat data'}\n${keepTranscript ? '(Keeping transcript)' : '- Transcript'}\n\nSelections and exports will be preserved.`)) {
      return;
    }
    
    try {
      const res = await apiJson('POST', '/api/project/reset_analysis', {
        keep_chat: keepChat,
        keep_transcript: keepTranscript,
      });
      alert(`Analysis reset complete!\n\nDeleted ${res.deleted_files.length} files.`);
      refreshProject();
      refreshTimeline();
    } catch (e) {
      alert(`Failed to reset analysis: ${e.message}`);
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

  // Auto-scroll to current time (within chat container only, not the page)
  if (chatAutoScroll) {
    const nearEl = messagesEl.querySelector('.chat-msg-near');
    if (nearEl) {
      // Calculate scroll position to center the element within the container
      const containerHeight = messagesEl.clientHeight;
      const elementTop = nearEl.offsetTop - messagesEl.offsetTop;
      const elementHeight = nearEl.clientHeight;
      const scrollTarget = elementTop - (containerHeight / 2) + (elementHeight / 2);
      messagesEl.scrollTop = Math.max(0, scrollTarget);
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
  
  // Update whisper backend dropdown with availability info
  updateWhisperBackendOptions();
}

async function updateWhisperBackendOptions() {
  try {
    const info = await apiGet('/api/system/info');
    const backends = info.transcription?.backends || {};
    const select = $('#whisperBackend');
    if (!select) return;
    
    // Update option labels with availability
    for (const opt of select.options) {
      if (opt.value === 'whispercpp') {
        const available = backends.whispercpp;
        const gpu = backends.whispercpp_gpu;
        opt.textContent = `whisper.cpp${available ? (gpu ? ' (GPU)' : ' ✓') : ' ✗'}`;
        opt.disabled = !available;
      } else if (opt.value === 'faster_whisper') {
        const available = backends.faster_whisper;
        const gpu = backends.faster_whisper_gpu;
        opt.textContent = `faster-whisper${available ? (gpu ? ' (CUDA)' : ' ✓') : ' ✗'}`;
        opt.disabled = !available;
      } else if (opt.value === 'auto') {
        opt.textContent = `Auto (${info.transcription?.recommended || 'best available'})`;
      }
    }
  } catch (e) {
    console.warn('Could not fetch backend info:', e);
  }
}

main();
