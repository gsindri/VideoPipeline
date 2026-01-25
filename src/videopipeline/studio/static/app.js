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

// =========================================================================
// UI UPDATE THROTTLING (Issue 1: Prevent flickering)
// =========================================================================

// Throttle renderHomeJobs to max ~8 Hz to prevent flickering
let homeJobsRenderPending = false;
let homeJobsLastRender = 0;
const HOME_JOBS_RENDER_INTERVAL = 125; // ms (8 Hz)

function throttledRenderHomeJobs() {
  const now = Date.now();
  if (now - homeJobsLastRender >= HOME_JOBS_RENDER_INTERVAL) {
    homeJobsLastRender = now;
    _doRenderHomeJobs();
  } else if (!homeJobsRenderPending) {
    homeJobsRenderPending = true;
    setTimeout(() => {
      homeJobsRenderPending = false;
      homeJobsLastRender = Date.now();
      _doRenderHomeJobs();
    }, HOME_JOBS_RENDER_INTERVAL - (now - homeJobsLastRender));
  }
}

// =========================================================================
// COLLAPSIBLE PANELS & UI PREFERENCES
// =========================================================================

// Load collapsed panels from localStorage
function getCollapsedPanels() {
  try {
    return JSON.parse(localStorage.getItem('vp_collapsed_panels') || '[]');
  } catch { return []; }
}

function saveCollapsedPanels(panels) {
  localStorage.setItem('vp_collapsed_panels', JSON.stringify(panels));
}

function togglePanelCollapse(panel) {
  const panelId = panel.dataset.panel;
  if (!panelId) return;
  
  const collapsed = getCollapsedPanels();
  const isCollapsed = panel.classList.toggle('collapsed');
  
  if (isCollapsed) {
    if (!collapsed.includes(panelId)) collapsed.push(panelId);
  } else {
    const idx = collapsed.indexOf(panelId);
    if (idx > -1) collapsed.splice(idx, 1);
  }
  saveCollapsedPanels(collapsed);
}

function initCollapsiblePanels() {
  const collapsed = getCollapsedPanels();
  
  document.querySelectorAll('.panel[data-panel]').forEach(panel => {
    const panelId = panel.dataset.panel;
    const h2 = panel.querySelector('h2');
    
    // Restore collapsed state
    if (collapsed.includes(panelId)) {
      panel.classList.add('collapsed');
    }
    
    // Click on h2 toggles collapse
    if (h2) {
      h2.addEventListener('click', (e) => {
        // Don't collapse if clicking on a button/input inside h2
        if (e.target.tagName === 'BUTTON' || e.target.tagName === 'INPUT') return;
        togglePanelCollapse(panel);
      });
    }
  });
}

// Analysis button visibility preferences
function getVisibleAnalysisButtons() {
  try {
    return JSON.parse(localStorage.getItem('vp_visible_analysis_btns') || '{}');
  } catch { return {}; }
}

function saveVisibleAnalysisButtons(btns) {
  localStorage.setItem('vp_visible_analysis_btns', JSON.stringify(btns));
}

function updateAnalysisButtonVisibility() {
  const prefs = getVisibleAnalysisButtons();
  
  const mapping = {
    'showBtnHighlights': 'btnAnalyzeHighlights',
    'showBtnAudio': 'btnAnalyzeAudio',
    'showBtnAudioEvents': 'btnAnalyzeAudioEvents',
    'showBtnSpeech': 'btnAnalyzeSpeech',
    'showBtnContext': 'btnAnalyzeContext',
    'showBtnReset': 'btnResetAnalysis',
  };
  
  for (const [checkboxId, btnId] of Object.entries(mapping)) {
    const checkbox = $(`#${checkboxId}`);
    const btn = $(`#${btnId}`);
    if (checkbox && btn) {
      const visible = !!prefs[checkboxId];
      checkbox.checked = visible;
      btn.classList.toggle('visible', visible);
    }
  }
}

function initAnalysisButtonToggles() {
  const dropdown = $('#analysisToolsDropdown');
  const toggleBtn = $('#btnAnalysisTools');
  
  if (!dropdown || !toggleBtn) return;
  
  // Toggle dropdown
  toggleBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    dropdown.style.display = dropdown.style.display === 'none' ? 'block' : 'none';
  });
  
  // Close dropdown when clicking outside
  document.addEventListener('click', () => {
    dropdown.style.display = 'none';
  });
  
  // Prevent dropdown from closing when clicking inside
  dropdown.addEventListener('click', (e) => {
    e.stopPropagation();
  });
  
  // Wire up checkboxes
  const mapping = {
    'showBtnHighlights': 'btnAnalyzeHighlights',
    'showBtnAudio': 'btnAnalyzeAudio',
    'showBtnAudioEvents': 'btnAnalyzeAudioEvents',
    'showBtnSpeech': 'btnAnalyzeSpeech',
    'showBtnContext': 'btnAnalyzeContext',
    'showBtnReset': 'btnResetAnalysis',
  };
  
  for (const [checkboxId, btnId] of Object.entries(mapping)) {
    const checkbox = $(`#${checkboxId}`);
    if (checkbox) {
      checkbox.addEventListener('change', () => {
        const prefs = getVisibleAnalysisButtons();
        prefs[checkboxId] = checkbox.checked;
        saveVisibleAnalysisButtons(prefs);
        
        const btn = $(`#${btnId}`);
        if (btn) {
          btn.classList.toggle('visible', checkbox.checked);
        }
      });
    }
  }
  
  // Load initial state
  updateAnalysisButtonVisibility();
}

// =========================================================================
// UTILITY FUNCTIONS
// =========================================================================

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
    throttledRenderHomeJobs();

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
        throttledRenderHomeJobs();

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

// Alias for initial render (non-throttled)
function renderHomeJobs() {
  _doRenderHomeJobs();
}

function _doRenderHomeJobs() {
  const container = $('#homeJobs');
  if (!container) return;

  if (homeJobs.size === 0) {
    container.innerHTML = '<div class="small">No active jobs.</div>';
    return;
  }

  // Build job cards - update in place if possible to prevent flicker
  const sorted = Array.from(homeJobs.values()).sort((a, b) =>
    a.created_at < b.created_at ? 1 : -1
  );

  // Track existing cards by job ID
  const existingCards = new Map();
  container.querySelectorAll('.item[data-job-id]').forEach(el => {
    existingCards.set(el.dataset.jobId, el);
  });

  // Remove cards for jobs that no longer exist
  existingCards.forEach((el, id) => {
    if (!homeJobs.has(id)) {
      el.remove();
    }
  });

  for (const job of sorted) {
    let el = existingCards.get(job.id);
    if (!el) {
      el = document.createElement('div');
      el.className = 'item';
      el.dataset.jobId = job.id;
      container.appendChild(el);
    }
    
    updateJobCard(el, job);
  }
}

function updateJobCard(el, job) {
  const pct = Math.round((job.progress || 0) * 100);
  const jobId = job.id;

  let statusClass = '';
  if (job.status === 'running') statusClass = 'status-badge running';
  else if (job.status === 'succeeded') statusClass = 'status-badge succeeded';
  else if (job.status === 'failed') statusClass = 'status-badge failed';
  else if (job.status === 'cancelled') statusClass = 'status-badge canceled';
  else statusClass = 'status-badge queued';

  // Check if we need to create the initial structure or just update values
  const existingMeta = el.querySelector('.meta');
  
  if (!existingMeta) {
    // First render - create the full structure with data attributes for updates
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
        
        // Show transcript status
        const transcript = job.result?.transcript;
        if (transcript) {
          if (transcript.status === 'complete') {
            resultInfo += `<div class="small" style="color:#22c55e;margin-top:2px">✓ Transcript ready (${transcript.segment_count} segments)</div>`;
          } else if (transcript.status === 'failed') {
            resultInfo += `<div class="small" style="color:#f59e0b;margin-top:2px">⚠ Early transcript failed (will run during analysis)</div>`;
          }
        }
    }

    let errorInfo = '';
    if (job.status === 'failed') {
      errorInfo = `<div class="small error-info" style="color:var(--danger);margin-top:4px">${job.message}</div>`;
    }

    el.innerHTML = `
      <div class="title">
        <span class="${statusClass}" data-role="status">${job.status}</span>
        ${job.kind === 'download_url' ? 'URL Download' : job.kind}
        ${job.status === 'running' ? `<button class="btn btn-small btn-danger" style="margin-left:auto;padding:2px 8px;font-size:11px" onclick="cancelJob('${jobId}')">Cancel</button>` : ''}
      </div>
      <div class="meta" data-role="meta">${job.message || ''}</div>
      <div class="parallel-tasks" data-role="tasks" style="min-height:80px;margin-top:8px"></div>
      <div class="progress-section" data-role="progress-section" style="margin-top:8px">
        <div class="progress"><div data-role="progress-bar" style="width:${pct}%;transition:width 0.2s"></div></div>
        <div class="small" data-role="progress-pct" style="margin-top:4px">${pct}%</div>
      </div>
      <div class="result-info" data-role="result">${resultInfo}</div>
      ${errorInfo}
    `;
    
    // Hide progress section if not running
    if (job.status !== 'running') {
      const progressSection = el.querySelector('[data-role="progress-section"]');
      if (progressSection) progressSection.style.display = 'none';
    }
  } else {
    // Update existing elements in place (no innerHTML replacement to prevent flicker)
    
    // Update meta text
    const metaEl = el.querySelector('[data-role="meta"]');
    if (metaEl && metaEl.textContent !== (job.message || '')) {
      metaEl.textContent = job.message || '';
    }
    
    // Update progress bar and percentage
    const progressBar = el.querySelector('[data-role="progress-bar"]');
    const progressPct = el.querySelector('[data-role="progress-pct"]');
    const progressSection = el.querySelector('[data-role="progress-section"]');
    
    if (job.status === 'running') {
      if (progressSection) progressSection.style.display = '';
      if (progressBar) progressBar.style.width = `${pct}%`;
      if (progressPct && progressPct.textContent !== `${pct}%`) {
        progressPct.textContent = `${pct}%`;
      }
    } else {
      if (progressSection) progressSection.style.display = 'none';
    }
    
    // Update status badge
    const statusEl = el.querySelector('[data-role="status"]');
    if (statusEl) {
      statusEl.className = statusClass;
      if (statusEl.textContent !== job.status) {
        statusEl.textContent = job.status;
      }
    }
  }
  
  // Always update parallel tasks section for running jobs (use innerHTML here but scoped to tasks div)
  if (job.status === 'running') {
    const tasksEl = el.querySelector('[data-role="tasks"]');
    if (tasksEl) {
      updateParallelTasks(tasksEl, job);
    }
  }
}

// Update parallel task rows (chat + transcript) in-place to prevent flickering
function updateParallelTasks(tasksEl, job) {
  const cs = job.result?.chat_status;
  const chatPct = job.result?.chat_progress;
  const chatMsgCount = job.result?.chat_messages_count || 0;
  const chatMsg = job.result?.chat_message || '';
  
  const ts = job.result?.transcript_status;
  const transPct = job.result?.transcript_progress || 0;
  const audioPct = job.result?.audio_progress || 0;
  const audioTotal = job.result?.audio_total_bytes;
  const audioSpeed = job.result?.audio_speed;
  
  // Get or create chat row
  let chatRow = tasksEl.querySelector('[data-task="chat"]');
  if (!chatRow) {
    chatRow = document.createElement('div');
    chatRow.className = 'task-row';
    chatRow.dataset.task = 'chat';
    chatRow.style.cssText = 'padding:6px 8px;border-radius:6px;margin-bottom:6px';
    chatRow.innerHTML = `
      <div class="small" style="display:flex;align-items:center;gap:6px">
        <span data-role="icon">💬</span>
        <span style="flex:1" data-role="label">Chat</span>
        <span data-role="pct"></span>
      </div>
      <div class="progress" style="margin-top:4px;height:3px"><div data-role="bar" style="width:0%;transition:width 0.2s"></div></div>
      <div class="small" data-role="detail" style="margin-top:2px;color:#888;font-size:10px"></div>
    `;
    tasksEl.appendChild(chatRow);
  }
  
  // Get or create transcript row
  let transRow = tasksEl.querySelector('[data-task="transcript"]');
  if (!transRow) {
    transRow = document.createElement('div');
    transRow.className = 'task-row';
    transRow.dataset.task = 'transcript';
    transRow.style.cssText = 'padding:6px 8px;border-radius:6px;margin-bottom:6px';
    transRow.innerHTML = `
      <div class="small" style="display:flex;align-items:center;gap:6px">
        <span data-role="icon">🎙️</span>
        <span style="flex:1" data-role="label">Transcript</span>
        <span data-role="pct"></span>
      </div>
      <div class="progress" style="margin-top:4px;height:3px"><div data-role="bar" style="width:0%;transition:width 0.2s"></div></div>
      <div class="small" data-role="detail" style="margin-top:2px;color:#888;font-size:10px"></div>
    `;
    tasksEl.appendChild(transRow);
  }
  
  // Update chat row based on state
  const chatIcon = chatRow.querySelector('[data-role="icon"]');
  const chatLabel = chatRow.querySelector('[data-role="label"]');
  const chatPctEl = chatRow.querySelector('[data-role="pct"]');
  const chatBar = chatRow.querySelector('[data-role="bar"]');
  const chatDetail = chatRow.querySelector('[data-role="detail"]');
  const chatProgress = chatRow.querySelector('.progress');
  
  if (cs === 'downloading') {
    chatRow.style.background = 'rgba(99,102,241,0.1)';
    chatRow.querySelector('.small').style.color = '#6366f1';
    chatIcon.textContent = '💬';
    chatLabel.textContent = 'Chat';
    const chatPctDisplay = chatPct !== undefined ? Math.round(chatPct * 100) : 0;
    chatPctEl.textContent = `${chatPctDisplay}%`;
    chatBar.style.width = `${chatPctDisplay}%`;
    chatBar.style.background = '#6366f1';
    chatProgress.style.display = '';
    chatDetail.style.display = '';
    chatDetail.textContent = chatMsgCount > 0 ? `${chatMsgCount.toLocaleString()} messages` : (chatMsg || 'Starting...');
  } else if (cs === 'importing' || cs === 'ai_learning') {
    chatRow.style.background = 'rgba(168,85,247,0.1)';
    chatRow.querySelector('.small').style.color = '#a855f7';
    chatIcon.textContent = '⚙️';
    chatLabel.textContent = cs === 'ai_learning' ? 'Learning emotes...' : 'Importing...';
    chatPctEl.textContent = '';
    chatProgress.style.display = 'none';
    chatDetail.style.display = 'none';
  } else if (cs === 'complete' || cs === 'success') {
    chatRow.style.background = 'rgba(34,197,94,0.1)';
    chatRow.querySelector('.small').style.color = '#22c55e';
    chatIcon.textContent = '✓';
    chatLabel.textContent = 'Chat ready';
    chatPctEl.textContent = '';
    chatProgress.style.display = 'none';
    chatDetail.style.display = 'none';
  } else if (cs === 'failed') {
    chatRow.style.background = 'rgba(239,68,68,0.1)';
    chatRow.querySelector('.small').style.color = '#ef4444';
    chatIcon.textContent = '⚠️';
    chatLabel.textContent = 'Chat failed';
    chatPctEl.textContent = '';
    chatProgress.style.display = 'none';
    chatDetail.style.display = 'none';
  } else {
    // Pending or unknown state
    chatRow.style.background = 'rgba(100,100,100,0.1)';
    chatRow.querySelector('.small').style.color = '#888';
    chatIcon.textContent = '💬';
    chatLabel.textContent = 'Chat waiting...';
    chatPctEl.textContent = '';
    chatProgress.style.display = 'none';
    chatDetail.style.display = 'none';
  }
  
  // Update transcript row based on state
  const transIcon = transRow.querySelector('[data-role="icon"]');
  const transLabel = transRow.querySelector('[data-role="label"]');
  const transPctEl = transRow.querySelector('[data-role="pct"]');
  const transBar = transRow.querySelector('[data-role="bar"]');
  const transDetail = transRow.querySelector('[data-role="detail"]');
  const transProgress = transRow.querySelector('.progress');
  
  if (ts === 'downloading_audio' || ts === 'audio_ready') {
    transRow.style.background = 'rgba(34,197,94,0.1)';
    transRow.querySelector('.small').style.color = '#22c55e';
    const isReady = ts === 'audio_ready';
    transIcon.textContent = '🎙️';
    transLabel.textContent = isReady ? 'Preparing transcript...' : 'Audio';
    const audioPctDisplay = Math.round(audioPct * 100);
    transPctEl.textContent = isReady ? '' : `${audioPctDisplay}%`;
    transBar.style.width = isReady ? '100%' : `${audioPctDisplay}%`;
    transBar.style.background = '#22c55e';
    transProgress.style.display = '';
    // Audio detail
    let audioDetail = '';
    if (audioTotal && !isReady) {
      const sizeMB = (audioTotal / 1024 / 1024).toFixed(1);
      audioDetail = `${sizeMB} MB`;
    }
    if (audioSpeed && !isReady) {
      const speedMBs = (audioSpeed / 1024 / 1024).toFixed(1);
      audioDetail += audioDetail ? ` @ ${speedMBs} MB/s` : `${speedMBs} MB/s`;
    }
    transDetail.style.display = audioDetail ? '' : 'none';
    transDetail.textContent = audioDetail;
  } else if (ts === 'transcribing') {
    transRow.style.background = 'rgba(34,197,94,0.1)';
    transRow.querySelector('.small').style.color = '#22c55e';
    transIcon.textContent = '🎙️';
    transLabel.textContent = 'Transcribing';
    const transPctDisplay = Math.round(transPct * 100);
    transPctEl.textContent = `${transPctDisplay}%`;
    transBar.style.width = `${transPctDisplay}%`;
    transBar.style.background = '#22c55e';
    transProgress.style.display = '';
    transDetail.style.display = 'none';
  } else if (ts === 'complete') {
    transRow.style.background = 'rgba(34,197,94,0.1)';
    transRow.querySelector('.small').style.color = '#22c55e';
    transIcon.textContent = '✓';
    transLabel.textContent = 'Transcript ready';
    transPctEl.textContent = '';
    transProgress.style.display = 'none';
    transDetail.style.display = 'none';
  } else if (ts === 'failed' || ts === 'audio_failed') {
    transRow.style.background = 'rgba(239,68,68,0.1)';
    transRow.querySelector('.small').style.color = '#ef4444';
    transIcon.textContent = '⚠️';
    transLabel.textContent = 'Transcript failed';
    transPctEl.textContent = '';
    transProgress.style.display = 'none';
    transDetail.style.display = 'none';
  } else if (ts === 'disabled') {
    transRow.style.background = 'rgba(100,100,100,0.1)';
    transRow.querySelector('.small').style.color = '#888';
    transIcon.textContent = '🎙️';
    transLabel.textContent = 'Transcript disabled';
    transPctEl.textContent = '';
    transProgress.style.display = 'none';
    transDetail.style.display = 'none';
  } else if (ts === 'deferred') {
    transRow.style.background = 'rgba(100,100,100,0.1)';
    transRow.querySelector('.small').style.color = '#888';
    transIcon.textContent = '🎙️';
    transLabel.textContent = 'Transcript deferred';
    transPctEl.textContent = '';
    transProgress.style.display = 'none';
    transDetail.style.display = 'none';
  } else {
    // Pending or unknown state
    transRow.style.background = 'rgba(100,100,100,0.1)';
    transRow.querySelector('.small').style.color = '#888';
    transIcon.textContent = '🎙️';
    transLabel.textContent = 'Transcript waiting...';
    transPctEl.textContent = '';
    transProgress.style.display = 'none';
    transDetail.style.display = 'none';
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

// =========================================================================
// ANALYSIS PIPELINE STATUS (Issue 2: Unified status panel)
// =========================================================================

function renderPipelineStatus() {
  const container = $('#pipelineStatus');
  if (!container || !project) return;

  const analysis = project.analysis || {};
  
  // Helper to format elapsed time for pipeline stages
  const formatPipelineTime = (seconds) => {
    if (seconds == null) return '';
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    const mins = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    return `${mins}m ${secs}s`;
  };

  // Helper: treat either generated_at or created_at as "complete"
  const hasRun = (obj) => !!(obj && (obj.generated_at || obj.created_at));
  
  // Define all pipeline stages with their detection logic
  // Order reflects actual dependency/execution order in analyze_full:
  //   Stage 1 (parallel): transcript, audio, motion, audio_events, chat, reaction_audio
  //   Stage 1.5: scenes (depends on motion)
  //   Stage 1.6: speech_features (depends on transcript)
  //   Stage 2: highlights (combines ALL signals incl. speech + reaction, with optional LLM semantic scoring)
  //   Stage 3: enrich (adds hook_text and quote_text to candidates)
  //   Stage 4: chapters, boundaries, clip_variants, director
  const stages = [
    // === Stage 1: Parallel input analysis ===
    {
      id: 'transcript',
      name: 'Transcription',
      icon: '🎙️',
      check: () => {
        const t = analysis.transcript;
        if (t?.segment_count > 0) {
          const backend = t.backend_used || 'auto';
          const gpu = t.gpu_used ? ' (GPU)' : '';
          const lang = t.detected_language ? ` [${t.detected_language}]` : '';
          const elapsed = t.elapsed_seconds;
          const timeStr = elapsed ? ` • ${formatPipelineTime(elapsed)}` : '';
          return { state: 'done', detail: `${t.segment_count} segments • ${backend}${gpu}${lang}${timeStr}` };
        }
        return { state: 'pending', detail: 'Not yet run' };
      }
    },
    {
      id: 'audio',
      name: 'Audio RMS',
      icon: '🔊',
      check: () => {
        const a = analysis.audio;
        if (hasRun(a)) {
          const peakCount = a.candidates?.length || 0;
          const timeStr = a.elapsed_seconds ? ` • ${formatPipelineTime(a.elapsed_seconds)}` : '';
          return { state: 'done', detail: `${peakCount} peaks found${timeStr}` };
        }
        return { state: 'pending', detail: 'Not yet run' };
      }
    },
    {
      id: 'motion',
      name: 'Motion Analysis',
      icon: '🎬',
      check: () => {
        const m = analysis.motion;
        if (hasRun(m)) {
          const timeStr = m.elapsed_seconds ? ` • ${formatPipelineTime(m.elapsed_seconds)}` : '';
          return { state: 'done', detail: `Analyzed${timeStr}` };
        }
        const motionMode = $('#motionWeightMode')?.value || 'off';
        if (motionMode === 'off') {
          return { state: 'skipped', detail: 'Disabled (Motion=Off)' };
        }
        return { state: 'pending', detail: 'Not yet run' };
      }
    },
    {
      id: 'audio_events',
      name: 'Audio Events',
      icon: '🎉',
      check: () => {
        const ae = analysis.audio_events;
        if (hasRun(ae)) {
          const candidateCount = ae.candidates?.length || 0;
          const timeStr = ae.elapsed_seconds ? ` • ${formatPipelineTime(ae.elapsed_seconds)}` : '';
          return { state: 'done', detail: `${candidateCount} candidates${timeStr}` };
        }
        return { state: 'pending', detail: 'Not yet run' };
      }
    },
    {
      id: 'chat_features',
      name: 'Chat Features',
      icon: '💬',
      check: () => {
        const cf = analysis.chat;
        if (hasRun(cf)) {
          const source = cf.laugh_source || 'unknown';
          const isLLM = source.startsWith('llm');
          const tokenCount = cf.laugh_tokens_count || 0;
          const llmCount = cf.llm_learned_count || 0;
          const timeStr = cf.elapsed_seconds ? ` • ${formatPipelineTime(cf.elapsed_seconds)}` : '';
          if (isLLM) {
            return { state: 'done', detail: `LLM: ${tokenCount} emotes (${llmCount} AI-learned)${timeStr}` };
          } else {
            return { state: 'partial', detail: `Seeds only: ${tokenCount} emotes (LLM not used)${timeStr}` };
          }
        }
        // Check if chat is available (try multiple sources)
        const hasChat = project.chat_ai_status?.has_chat || chatStatus?.available || project.chat?.available;
        if (!hasChat) {
          return { state: 'skipped', detail: 'No chat data' };
        }
        return { state: 'pending', detail: 'Not yet run' };
      }
    },
    // === Stage 2: Signal fusion (with optional LLM semantic scoring) ===
    {
      id: 'highlights',
      name: 'Highlight Fusion',
      icon: '⭐',
      check: () => {
        const h = analysis.highlights;
        if (hasRun(h)) {
          const count = h.candidates?.length || 0;
          const llmUsed = h.signals_used?.llm_semantic ? ' + LLM' : '';
          const speechUsed = h.signals_used?.speech ? ' + speech' : '';
          const reactionUsed = h.signals_used?.reaction ? ' + reaction' : '';
          const extras = speechUsed + reactionUsed + llmUsed;
          const timeStr = h.elapsed_seconds ? ` • ${formatPipelineTime(h.elapsed_seconds)}` : '';
          return { state: 'done', detail: `${count} candidates${extras}${timeStr}` };
        }
        return { state: 'pending', detail: 'Not yet run' };
      }
    },
    // === Stage 3: Enrichment (hook/quote text) ===
    {
      id: 'speech_features',
      name: 'Speech Features',
      icon: '🗣️',
      check: () => {
        // Backend stores speech features in analysis.speech (with created_at)
        const sf = analysis.speech;
        if (hasRun(sf)) {
          const timeStr = sf.elapsed_seconds ? ` • ${formatPipelineTime(sf.elapsed_seconds)}` : '';
          return { state: 'done', detail: `Extracted from transcript${timeStr}` };
        }
        if (!analysis.transcript?.segment_count) {
          return { state: 'skipped', detail: 'Requires transcript' };
        }
        return { state: 'pending', detail: 'Not yet run' };
      }
    },
    {
      id: 'reaction_audio',
      name: 'Reaction Audio',
      icon: '🎭',
      check: () => {
        const ra = analysis.reaction_audio;
        if (hasRun(ra)) {
          const timeStr = ra.elapsed_seconds ? ` • ${formatPipelineTime(ra.elapsed_seconds)}` : '';
          return { state: 'done', detail: `Acoustic reaction cues${timeStr}` };
        }
        return { state: 'pending', detail: 'Not yet run' };
      }
    },
    {
      id: 'enrich',
      name: 'Enrich Candidates',
      icon: '📝',
      check: () => {
        const h = analysis.highlights;
        // Backend sets enriched_at when enrichment completes
        if (h?.enriched_at) {
          const timeStr = h.enrich_elapsed_seconds ? ` • ${formatPipelineTime(h.enrich_elapsed_seconds)}` : '';
          return { state: 'done', detail: `Hook & quote text extracted${timeStr}` };
        }
        // Check dependencies
        if (!hasRun(analysis.highlights)) {
          return { state: 'skipped', detail: 'Requires highlights' };
        }
        if (!hasRun(analysis.transcript)) {
          return { state: 'skipped', detail: 'Requires transcript' };
        }
        return { state: 'pending', detail: 'Not yet run' };
      }
    },
    // === Stage 4: Context analysis & AI ===
    {
      id: 'chapters',
      name: 'Semantic Chapters',
      icon: '📖',
      check: () => {
        const ch = analysis.chapters;
        if (hasRun(ch)) {
          const count = ch.chapter_count || 0;
          const timeStr = ch.elapsed_seconds ? ` • ${formatPipelineTime(ch.elapsed_seconds)}` : '';
          return { state: 'done', detail: `${count} chapters detected${timeStr}` };
        }
        // Check if chapters are enabled in profile
        const chaptersEnabled = profile?.context?.chapters?.enabled !== false;
        if (!chaptersEnabled) {
          return { state: 'skipped', detail: 'Disabled in profile' };
        }
        if (!analysis.transcript?.segment_count) {
          return { state: 'skipped', detail: 'Requires transcript' };
        }
        return { state: 'pending', detail: 'Not yet run' };
      }
    },
    {
      id: 'boundaries',
      name: 'Context Boundaries',
      icon: '📍',
      check: () => {
        const b = analysis.boundaries;
        if (hasRun(b)) {
          const timeStr = b.elapsed_seconds ? ` • ${formatPipelineTime(b.elapsed_seconds)}` : '';
          return { state: 'done', detail: `Scene boundaries computed${timeStr}` };
        }
        return { state: 'pending', detail: 'Not yet run' };
      }
    },
    {
      id: 'clip_variants',
      name: 'Clip Variants',
      icon: '🎞️',
      check: () => {
        const cv = analysis.clip_variants;
        if (hasRun(cv)) {
          const count = cv.variants_count || 0;
          const timeStr = cv.elapsed_seconds ? ` • ${formatPipelineTime(cv.elapsed_seconds)}` : '';
          return { state: 'done', detail: `${count} variants generated${timeStr}` };
        }
        if (!hasRun(analysis.boundaries)) {
          return { state: 'skipped', detail: 'Requires boundaries' };
        }
        return { state: 'pending', detail: 'Not yet run' };
      }
    },
    {
      id: 'director',
      name: 'AI Director',
      icon: '🤖',
      check: () => {
        const d = analysis.ai_director;
        if (hasRun(d)) {
          const count = d.candidate_count || 0;
          const llm = d.llm_available ? 'LLM' : 'heuristic';
          const timeStr = d.elapsed_seconds ? ` • ${formatPipelineTime(d.elapsed_seconds)}` : '';
          return { state: 'done', detail: `${count} clips analyzed • ${llm}${timeStr}` };
        }
        // Check if AI is disabled
        const aiEnabled = profile?.ai?.director?.enabled !== false;
        if (!aiEnabled) {
          return { state: 'skipped', detail: 'Disabled in profile (ai.director.enabled=false)' };
        }
        if (!hasRun(analysis.clip_variants)) {
          return { state: 'skipped', detail: 'Requires clip variants' };
        }
        return { state: 'pending', detail: 'Not yet run' };
      }
    }
  ];

  // Count states
  let doneCount = 0, pendingCount = 0, skippedCount = 0, partialCount = 0;
  
  // Build HTML
  let html = '<div style="display:grid;gap:6px">';
  
  for (const stage of stages) {
    const result = stage.check();
    
    let stateIcon, stateColor, bgColor;
    switch (result.state) {
      case 'done':
        stateIcon = '✓';
        stateColor = '#22c55e';
        bgColor = 'rgba(34,197,94,0.1)';
        doneCount++;
        break;
      case 'partial':
        stateIcon = '⚠';
        stateColor = '#eab308';
        bgColor = 'rgba(234,179,8,0.1)';
        partialCount++;
        break;
      case 'skipped':
        stateIcon = '○';
        stateColor = '#666';
        bgColor = 'rgba(100,100,100,0.1)';
        skippedCount++;
        break;
      case 'running':
        stateIcon = '⏳';
        stateColor = '#3b82f6';
        bgColor = 'rgba(59,130,246,0.1)';
        break;
      case 'failed':
        stateIcon = '✗';
        stateColor = '#ef4444';
        bgColor = 'rgba(239,68,68,0.1)';
        break;
      default:
        stateIcon = '·';
        stateColor = '#555';
        bgColor = 'transparent';
        pendingCount++;
    }
    
    html += `
      <div style="display:flex;align-items:center;gap:8px;padding:6px 10px;background:${bgColor};border-radius:6px;border-left:3px solid ${stateColor}">
        <span style="font-size:14px">${stage.icon}</span>
        <span style="flex:1;color:${stateColor};font-weight:500">${stage.name}</span>
        <span style="color:${stateColor};font-size:11px">${stateIcon}</span>
      </div>
      <div style="margin-left:32px;margin-top:-4px;margin-bottom:4px;color:#888;font-size:10px">${result.detail}</div>
    `;
  }
  
  html += '</div>';
  
  // Summary line
  const total = stages.length;
  const summaryColor = doneCount === total ? '#22c55e' : (doneCount > 0 ? '#eab308' : '#666');
  html = `
    <div style="margin-bottom:12px;padding:8px;background:#1a1a2e;border-radius:6px;display:flex;justify-content:space-between;align-items:center">
      <span style="color:${summaryColor};font-weight:600">Pipeline: ${doneCount}/${total} complete</span>
      <span style="color:#888;font-size:11px">${skippedCount} skipped • ${partialCount} partial • ${pendingCount} pending</span>
    </div>
  ` + html;
  
  container.innerHTML = html;
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
      const speech = Number(c.breakdown.speech || 0).toFixed(2);
      const reaction = Number(c.breakdown.reaction || 0).toFixed(2);
      breakdown = ` • audio ${audio} / motion ${motion}`;
      if (Number(c.breakdown.chat) !== 0) breakdown += ` / chat ${chat}`;
      if (Number(c.breakdown.audio_events || 0) !== 0) breakdown += ` / events ${audioEvents}`;
      if (Number(c.breakdown.speech || 0) !== 0) breakdown += ` / speech ${speech}`;
      if (Number(c.breakdown.reaction || 0) !== 0) breakdown += ` / reaction ${reaction}`;
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
    
    // Warning badge if boundary graph wasn't used (may start mid-sentence)
    const boundaryWarning = c.used_boundary_graph === false ? 
      '<span class="badge" style="background:#854d0e;color:#fef08a;margin-left:6px" title="May start mid-sentence (no boundary data)">⚠ rough cut</span>' : '';
    
    el.innerHTML = `
      <div class="title">#${c.rank} • score ${c.score.toFixed(2)} <span class="badge">peak ${fmtTime(c.peak_time_s)}</span>${boundaryWarning}</div>
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

  // Helper to format seconds nicely, or handle "cached"/"skipped" markers
  const formatTime = (seconds) => {
    if (seconds === "cached") return 'cached';
    if (seconds === "skipped") return 'skipped';
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
        
        // Completed tasks - distinguish cached/skipped from computed
        for (const t of completed) {
          const time = taskTimes[t];
          const isCached = time === "cached";
          const isSkipped = time === "skipped";
          if (isSkipped) {
            tasksHtml += `<div style="color:#888">○ ${t} <span style="font-style:italic">(skipped - no data)</span></div>`;
          } else if (isCached) {
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
        
        // Summary line
        detailHtml = `<div class="meta" style="margin-top:4px;font-size:11px">`;
        if (candidateCount > 0) detailHtml += `<strong>${candidateCount} candidates</strong>`;
        if (errorCount > 0) detailHtml += ` | <span style="color:#f59e0b">${errorCount} errors</span>`;
        detailHtml += `</div>`;
        
        // Add stage timing breakdown as a detailed list
        if (Object.keys(stageTimes).length > 0) {
          const computedStages = [];
          const cachedStages = [];
          const skippedStages = [];
          for (const [stage, time] of Object.entries(stageTimes)) {
            if (time === "skipped") {
              skippedStages.push(stage);
            } else if (time === "cached") {
              cachedStages.push(stage);
            } else {
              computedStages.push({ stage, time });
            }
          }
          
          let timingHtml = '<div style="margin-top:6px;font-size:11px;line-height:1.6">';
          
          // Show computed stages with times
          for (const { stage, time } of computedStages) {
            timingHtml += `<div style="color:#22c55e">✓ ${stage} <span style="color:#888">${formatTime(time)}</span></div>`;
          }
          
          // Show cached stages
          for (const stage of cachedStages) {
            timingHtml += `<div style="color:#888">✓ ${stage} <span style="font-style:italic">(cached)</span></div>`;
          }
          
          // Show skipped stages
          for (const stage of skippedStages) {
            timingHtml += `<div style="color:#888">○ ${stage} <span style="font-style:italic">(skipped - no data)</span></div>`;
          }
          
          timingHtml += '</div>';
          detailHtml += timingHtml;
        }
        
        if (r.errors && r.errors.length > 0) {
          detailHtml += `<div class="meta" style="margin-top:2px;font-size:10px;color:#ef4444">${r.errors.join('<br/>')}</div>`;
        }
      } else if (Object.keys(stageTimes).length > 0 || currentStage.name) {
        // In-progress after Stage 1 - sequential stages
        let tasksHtml = '<div style="margin-top:6px;font-size:11px;line-height:1.6">';
        
        // Show completed stages - distinguish cached/skipped from computed
        for (const [stage, time] of Object.entries(stageTimes)) {
          if (time === "skipped") {
            tasksHtml += `<div style="color:#888">○ ${stage} <span style="font-style:italic">(skipped - no data)</span></div>`;
          } else if (time === "cached") {
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
  es.onmessage = async (ev) => {
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
        
        if ((j.kind === 'analyze_audio' || j.kind === 'analyze_highlights' || j.kind === 'analyze_speech' || j.kind === 'analyze_context_titles' || j.kind === 'analyze_full') && (j.status === 'succeeded' || j.status === 'failed')) {
          // refresh project and timeline to show enriched candidates (even on partial failure)
          refreshProject();
          // refresh chat status to update AI Director status
          loadChatStatus();
        }
        if (j.kind === 'export' && j.status === 'succeeded') {
          // refresh project to show exports list eventually
          refreshProject();
        }
        if ((j.kind === 'download_chat' || j.kind === 'download_url') && j.status === 'succeeded') {
          // refresh chat status after download completes
          await refreshProject();  // refresh project first to get updated chat_ai_status
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
  renderPipelineStatus();
  renderCandidates();
  renderSelections();
  await refreshTimeline();
  // Also refresh chat UI to show AI analysis status
  updateChatUI();
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
      loadChatStatus();  // Refresh AI Analysis Status panel (director, chat emotes)
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

  // Build AI status indicator (from chatStatus.ai_status, not project)
  let aiStatusHtml = '';
  const aiStatus = chatStatus?.ai_status;
  const directorStatus = chatStatus?.director_status;
  
  if (aiStatus?.has_chat || directorStatus?.analyzed) {
    aiStatusHtml = '<div id="aiStatusPanel" style="margin-top:12px;padding:10px;background:#1a1a2e;border-radius:8px;border:1px solid #333">';
    aiStatusHtml += '<div style="font-weight:600;color:#e0e0e0;margin-bottom:8px;font-size:13px">🤖 AI Analysis Status</div>';
    
    // Chat Emotes section
    if (aiStatus?.has_chat) {
      if (aiStatus.ai_analyzed) {
        const llmCount = aiStatus.llm_learned_count || 0;
        const tokenCount = aiStatus.laugh_tokens_count || 0;
        const newlyLearnedCount = aiStatus.newly_learned_count || 0;
        const loadedFromGlobal = aiStatus.loaded_from_global || 0;
        const newlyLearnedTokens = aiStatus.newly_learned_tokens || [];
        const llmLearned = aiStatus.llm_learned_tokens || [];
        
        // Show newly learned tokens if any, otherwise show all LLM-learned
        const displayTokens = newlyLearnedTokens.length > 0 ? newlyLearnedTokens : llmLearned;
        const tokenDisplay = displayTokens.slice(0, 6).join(', ');
        const tokenMore = displayTokens.length > 6 ? ` +${displayTokens.length - 6}` : '';
        
        // Build description based on what happened
        let description = `${tokenCount} emotes`;
        if (loadedFromGlobal > 0 && newlyLearnedCount === 0) {
          description += ` (${loadedFromGlobal} from channel history)`;
        } else if (newlyLearnedCount > 0) {
          description += ` (${newlyLearnedCount} NEW this session)`;
        } else if (llmCount > 0) {
          description += ` (${llmCount} AI-discovered)`;
        }
        
        // Color code: green if new tokens learned, blue if using cached
        const hasNew = newlyLearnedCount > 0;
        const borderColor = hasNew ? '#22c55e' : '#3b82f6';
        const bgColor = hasNew ? 'rgba(34,197,94,0.1)' : 'rgba(59,130,246,0.1)';
        const textColor = hasNew ? '#22c55e' : '#3b82f6';
        const tokenColor = hasNew ? '#a855f7' : '#60a5fa';
        
        aiStatusHtml += `
          <div style="padding:8px;background:${bgColor};border-radius:6px;border-left:3px solid ${borderColor};margin-bottom:8px">
            <div style="display:flex;justify-content:space-between;align-items:center">
              <div>
                <div style="color:${textColor};font-weight:500;font-size:12px">✓ Chat Emotes (LLM)</div>
                <div style="color:#aaa;font-size:11px;margin-top:2px">${description}</div>
                ${displayTokens.length > 0 ? `<div style="color:${tokenColor};font-size:10px;margin-top:4px">${tokenDisplay}${tokenMore}</div>` : ''}
              </div>
              <button id="btnRelearnAI" style="padding:4px 8px;font-size:11px;background:#333;border:1px solid #555;color:#aaa;border-radius:4px;cursor:pointer">Re-learn</button>
            </div>
          </div>`;
      } else {
        const skipReason = aiStatus.llm_skip_reason || 'LLM not used';
        aiStatusHtml += `
          <div style="padding:8px;background:rgba(234,179,8,0.1);border-radius:6px;border-left:3px solid #eab308;margin-bottom:8px">
            <div style="display:flex;justify-content:space-between;align-items:center">
              <div>
                <div style="color:#eab308;font-weight:500;font-size:12px">⚠ Chat Emotes (Seeds Only)</div>
                <div style="color:#888;font-size:11px;margin-top:2px">${skipReason}</div>
              </div>
              <button id="btnRelearnAI" style="padding:4px 8px;font-size:11px;background:#4a3f00;border:1px solid #eab308;color:#eab308;border-radius:4px;cursor:pointer">Learn with AI</button>
            </div>
          </div>`;
      }
    }
    
    // Director section
    if (directorStatus?.analyzed) {
      aiStatusHtml += `
        <div style="padding:8px;background:rgba(59,130,246,0.1);border-radius:6px;border-left:3px solid #3b82f6">
          <div style="color:#3b82f6;font-weight:500;font-size:12px">✓ AI Director</div>
          <div style="color:#aaa;font-size:11px;margin-top:2px">${directorStatus.candidates_count} clips analyzed • ${directorStatus.model}</div>
        </div>`;
    } else {
      aiStatusHtml += `
        <div style="padding:8px;background:rgba(100,100,100,0.1);border-radius:6px;border-left:3px solid #666">
          <div style="color:#888;font-weight:500;font-size:12px">○ AI Director</div>
          <div style="color:#666;font-size:11px;margin-top:2px">Run Analyze (Full) to generate AI clip metadata</div>
        </div>`;
    }
    
    aiStatusHtml += '</div>';
  }

  if (!chatStatus || !chatStatus.available) {
    statusEl.innerHTML = 'No chat replay available.' + aiStatusHtml;
    statusEl.className = 'small';
    if (messagesEl) messagesEl.innerHTML = '';
    if (offsetControls) offsetControls.style.display = 'none';
    if (downloadBtn) downloadBtn.disabled = false;
    if (clearBtn) clearBtn.style.display = 'none';

    // Pre-fill source URL from project if available
    if (sourceUrlInput && chatStatus?.source_url) {
      sourceUrlInput.value = chatStatus.source_url;
    }
    // Wire up re-learn button if present
    wireRelearnButton();
    return;
  }

  statusEl.innerHTML = `Chat: ${chatStatus.message_count.toLocaleString()} messages` + aiStatusHtml;
  statusEl.className = 'small success';
  // Wire up re-learn button
  wireRelearnButton();
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
    await loadChatStatus();  // Refresh AI Analysis Status panel
  } catch (e) {
    alert(`Failed to clear chat: ${e.message}`);
  }
}

function wireRelearnButton() {
  const btn = $('#btnRelearnAI');
  if (!btn) return;
  
  btn.onclick = async () => {
    if (!confirm('This will clear the cached emote data and require re-analysis with Analyze (Full) to learn channel-specific emotes using AI.\n\nContinue?')) {
      return;
    }
    
    btn.disabled = true;
    btn.textContent = 'Clearing...';
    
    try {
      await apiJson('POST', '/api/chat/relearn_ai', {});
      await loadChatStatus();
      alert('Chat AI cache cleared. Run "Analyze (Full)" to re-learn emotes with AI.');
    } catch (e) {
      alert(`Failed to clear AI cache: ${e.message}`);
    } finally {
      btn.disabled = false;
      btn.textContent = 'Re-learn';
    }
  };
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

  // Initialize collapsible panels
  initCollapsiblePanels();
  initAnalysisButtonToggles();

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
  renderPipelineStatus();
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
    
    // Update option labels with availability (unavailable shown with ✗)
    for (const opt of select.options) {
      if (opt.value === 'openai_whisper') {
        const available = backends.openai_whisper;
        const gpu = backends.openai_whisper_gpu;
        if (!available) {
          opt.textContent = 'openai-whisper ✗';
          opt.disabled = true;
        } else {
          opt.textContent = gpu ? 'openai-whisper (GPU)' : 'openai-whisper';
          opt.disabled = false;
        }
      } else if (opt.value === 'whispercpp') {
        const available = backends.whispercpp;
        const gpu = backends.whispercpp_gpu;
        if (!available) {
          opt.textContent = 'whisper.cpp ✗';
          opt.disabled = true;
        } else {
          opt.textContent = gpu ? 'whisper.cpp (GPU)' : 'whisper.cpp';
          opt.disabled = false;
        }
      } else if (opt.value === 'faster_whisper') {
        const available = backends.faster_whisper;
        const gpu = backends.faster_whisper_gpu;
        if (!available) {
          opt.textContent = 'faster-whisper ✗';
          opt.disabled = true;
        } else {
          opt.textContent = gpu ? 'faster-whisper (CUDA)' : 'faster-whisper';
          opt.disabled = false;
        }
      }
    }
  } catch (e) {
    console.warn('Could not fetch backend info:', e);
  }
}

main();
