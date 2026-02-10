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

function readOptionalIntInput(el, { min = 1 } = {}) {
  if (!el) return null;
  const raw = String(el.value ?? '').trim();
  if (!raw) return null;
  const n = parseInt(raw, 10);
  if (!Number.isFinite(n)) return null;
  if (min != null && n < min) return null;
  return n;
}

function setOptionalNumberInput(el, value) {
  if (!el) return;
  if (value == null || value === '') {
    el.value = '';
    return;
  }
  const n = Number(value);
  el.value = Number.isFinite(n) ? String(Math.trunc(n)) : '';
}

function updateDiarizationControls() {
  const diarizeEnabled = $('#diarizeEnabled')?.checked ?? false;
  const minEl = $('#diarizeMinSpeakers');
  const maxEl = $('#diarizeMaxSpeakers');
  if (minEl) minEl.disabled = !diarizeEnabled;
  if (maxEl) maxEl.disabled = !diarizeEnabled;
}

function applyAnalysisDefaultsFromProfile() {
  const speech = profile?.analysis?.speech || {};
  const diar = profile?.analysis?.diarization || {};
  const diarize = speech.diarize;
  if ($('#diarizeEnabled') && diarize !== undefined) $('#diarizeEnabled').checked = diarize === true;
  if ($('#dlDiarizeEnabled') && diarize !== undefined) $('#dlDiarizeEnabled').checked = diarize === true;

  const merged = { ...speech, ...diar };
  const minSpk = merged.diarize_min_speakers ?? merged.min_speakers ?? null;
  const maxSpk = merged.diarize_max_speakers ?? merged.max_speakers ?? null;
  setOptionalNumberInput($('#diarizeMinSpeakers'), minSpk);
  setOptionalNumberInput($('#diarizeMaxSpeakers'), maxSpk);
  updateDiarizationControls();
}

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

  // Some panels contain canvases that need a rerender once visible again.
  if (!isCollapsed && panelId === 'chat_timeline') {
    requestAnimationFrame(() => {
      renderChatMiniTimeline();
      updateChatTimelinePlayhead();
    });
  }
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
  // Use centisecond rounding to avoid displaying "01:60.00" due to float rounding.
  const totalCs = Math.max(0, Math.round((Number(sec) || 0) * 100));
  const cs = totalCs % 100;
  const totalSeconds = Math.floor(totalCs / 100);

  const seconds = totalSeconds % 60;
  const totalMinutes = Math.floor(totalSeconds / 60);
  const minutes = totalMinutes % 60;
  const hours = Math.floor(totalMinutes / 60);

  const ss = `${String(seconds).padStart(2, '0')}.${String(cs).padStart(2, '0')}`;
  if (hours > 0) return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${ss}`;
  return `${String(totalMinutes).padStart(2, '0')}:${ss}`;
}

function clamp(val, lo, hi) {
  return Math.max(lo, Math.min(hi, val));
}

function fmtDuration(seconds, opts = {}) {
  // seconds can be a number OR string markers like "cached"/"skipped"
  if (seconds === "cached") return "cached";
  if (seconds === "skipped") return "skipped";
  if (seconds == null) return "";

  const n = Number(seconds);
  if (!Number.isFinite(n)) return String(seconds);

  const decimalsUnderMinute = Number.isFinite(opts.decimalsUnderMinute) ? Math.max(0, opts.decimalsUnderMinute) : 1;
  const sign = n < 0 ? "-" : "";
  const abs = Math.abs(n);

  const totalMs = Math.round(abs * 1000);
  const totalSeconds = Math.floor(totalMs / 1000);
  const ms = totalMs % 1000;

  const hours = Math.floor(totalSeconds / 3600);
  const minutesTotal = Math.floor(totalSeconds / 60);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const secs = totalSeconds % 60;

  if (hours > 0) return `${sign}${hours}h ${minutes}m ${secs}s`;
  if (totalSeconds >= 60) return `${sign}${minutesTotal}m ${secs}s`;

  if (decimalsUnderMinute === 0) return `${sign}${totalSeconds}s`;

  // Under 60 seconds: include fractional seconds, but never show "60.0s".
  const value = totalSeconds + ms / 1000;
  const factor = 10 ** decimalsUnderMinute;
  const rounded = Math.round(value * factor) / factor;
  if (rounded >= 60) return `${sign}1m 0s`;
  return `${sign}${rounded.toFixed(decimalsUnderMinute)}s`;
}

async function apiGet(path) {
  const r = await apiFetch(path, {}, { retry401: true });
  if (!r.ok) throw new Error(`${path} -> ${r.status}`);
  return await r.json();
}

async function apiJson(method, path, body) {
  const r = await apiFetch(
    path,
    {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: body ? JSON.stringify(body) : undefined,
    },
    { retry401: true },
  );
  if (!r.ok) {
    const txt = await r.text();
    throw new Error(`${method} ${path} -> ${r.status} ${txt}`);
  }
  return await r.json();
}

// =========================================================================
// API TOKEN SUPPORT (VP_API_TOKEN)
// =========================================================================

const VP_API_TOKEN_KEY = 'vp_api_token';

function getApiToken() {
  try {
    return (sessionStorage.getItem(VP_API_TOKEN_KEY) || localStorage.getItem(VP_API_TOKEN_KEY) || '').trim();
  } catch {
    return '';
  }
}

function setApiToken(token, { remember = false } = {}) {
  const t = String(token || '').trim();
  if (!t) return;
  try {
    sessionStorage.removeItem(VP_API_TOKEN_KEY);
    localStorage.removeItem(VP_API_TOKEN_KEY);
    if (remember) localStorage.setItem(VP_API_TOKEN_KEY, t);
    else sessionStorage.setItem(VP_API_TOKEN_KEY, t);
  } catch {}
  updateApiTokenLinks();
}

function forgetApiToken() {
  try {
    sessionStorage.removeItem(VP_API_TOKEN_KEY);
    localStorage.removeItem(VP_API_TOKEN_KEY);
  } catch {}
  updateApiTokenLinks();
}

function apiUrlWithToken(path) {
  const token = getApiToken();
  if (!token) return path;
  const sep = path.includes('?') ? '&' : '?';
  return `${path}${sep}token=${encodeURIComponent(token)}`;
}

function updateApiTokenLinks() {
  // Links opened via navigation (not fetch) can't send Authorization headers.
  const lnk = $('#lnkViewDiarization');
  if (lnk) lnk.href = apiUrlWithToken('/api/task_artifact/diarization');
}

let apiTokenPromptInFlight = null;

function promptForApiToken({ title = 'API Token Required' } = {}) {
  if (apiTokenPromptInFlight) return apiTokenPromptInFlight;

  const modal = $('#apiTokenModal');
  const input = $('#apiTokenInput');
  const remember = $('#apiTokenRemember');
  const btnSave = $('#btnApiTokenSave');
  const btnCancel = $('#btnApiTokenCancel');
  const btnForget = $('#btnApiTokenForget');
  const errorEl = $('#apiTokenError');

  if (!modal || !input || !btnSave || !btnCancel || !btnForget) {
    return Promise.reject(new Error('API token UI is missing from index.html'));
  }

  modal.querySelector('h2').textContent = title;
  if (errorEl) {
    errorEl.style.display = 'none';
    errorEl.textContent = '';
  }
  input.value = '';
  if (remember) remember.checked = false;

  const show = () => { modal.style.display = 'flex'; };
  const hide = () => { modal.style.display = 'none'; };

  apiTokenPromptInFlight = new Promise((resolve, reject) => {
    const cleanup = () => {
      btnSave.onclick = null;
      btnCancel.onclick = null;
      btnForget.onclick = null;
      modal.onclick = null;
      window.removeEventListener('keydown', onKeyDown);
      apiTokenPromptInFlight = null;
    };

    const onKeyDown = (e) => {
      if (e.key === 'Escape') {
        hide();
        cleanup();
        reject(new Error('cancelled'));
      } else if (e.key === 'Enter') {
        btnSave.click();
      }
    };

    btnSave.onclick = () => {
      const t = String(input.value || '').trim();
      if (!t) {
        if (errorEl) {
          errorEl.textContent = 'Token is required.';
          errorEl.style.display = 'block';
        }
        return;
      }
      setApiToken(t, { remember: !!(remember && remember.checked) });
      hide();
      cleanup();
      resolve(t);
    };

    btnCancel.onclick = () => {
      hide();
      cleanup();
      reject(new Error('cancelled'));
    };

    btnForget.onclick = () => {
      forgetApiToken();
      hide();
      cleanup();
      resolve('');
    };

    // Click outside modal content cancels.
    modal.onclick = (e) => {
      if (e.target === modal) {
        hide();
        cleanup();
        reject(new Error('cancelled'));
      }
    };

    window.addEventListener('keydown', onKeyDown);
    show();
    input.focus();
  });

  return apiTokenPromptInFlight;
}

function initApiTokenUi() {
  document.querySelectorAll('[data-action="api-token"]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      try {
        await promptForApiToken({ title: 'API Token' });
      } catch (_) {}
    });
  });
  updateApiTokenLinks();
}

async function apiFetch(path, options = {}, { retry401 = true } = {}) {
  const token = getApiToken();
  const headers = new Headers(options.headers || {});
  if (token) headers.set('Authorization', `Bearer ${token}`);

  const r = await fetch(path, { ...options, headers });
  if (r.status === 401 && retry401) {
    try {
      await promptForApiToken();
    } catch (e) {
      throw new Error('Unauthorized (API token required)');
    }
    return await apiFetch(path, options, { retry401: false });
  }
  return r;
}

// =========================================================================
// ACTIONS HELPER (Quick Tunnel + Actions OpenAPI import URL)
// =========================================================================

let actionsHelperTunnelUrl = '';
let actionsHelperImportUrl = '';

async function copyTextToClipboard(text) {
  const t = String(text || '');
  if (!t) return false;
  try {
    await navigator.clipboard.writeText(t);
    return true;
  } catch (_) {
    // Fallback for older/locked-down contexts.
    try {
      const ta = document.createElement('textarea');
      ta.value = t;
      ta.setAttribute('readonly', '');
      ta.style.position = 'fixed';
      ta.style.left = '-9999px';
      ta.style.top = '0';
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      const ok = document.execCommand('copy');
      document.body.removeChild(ta);
      return ok;
    } catch (_) {
      return false;
    }
  }
}

async function refreshActionsHelper() {
  const tunnelEl = $('#actionsHelperTunnelUrl');
  const importEl = $('#actionsHelperImportUrl');
  const statusEl = $('#actionsHelperStatus');
  const btnCopyTunnel = $('#btnCopyActionsTunnelUrl');
  const btnCopyImport = $('#btnCopyActionsImportUrl');
  const btnRefresh = $('#btnRefreshActionsHelper');

  if (!tunnelEl || !importEl || !statusEl || !btnCopyTunnel || !btnCopyImport) return;

  if (btnRefresh) btnRefresh.disabled = true;
  statusEl.textContent = 'Detecting tunnel URL...';

  try {
    const info = await apiGet('/api/system/actions_helper');
    actionsHelperTunnelUrl = info.tunnel_base_url || '';
    actionsHelperImportUrl = info.openapi_import_url || '';

    tunnelEl.value = actionsHelperTunnelUrl || '';
    importEl.value = info.openapi_import_url_masked || actionsHelperImportUrl || '';

    btnCopyTunnel.disabled = !actionsHelperTunnelUrl;
    btnCopyImport.disabled = !actionsHelperImportUrl;

    if (!actionsHelperTunnelUrl) {
      statusEl.innerHTML = 'Tunnel URL not detected. Run <code>tools\\\\studio-quick-tunnel.bat</code> and refresh.';
    } else if (info.tunnel_base_source === 'cloudflared_log') {
      statusEl.textContent = 'Tunnel URL detected from the latest Cloudflare Quick Tunnel log.';
    } else if (info.tunnel_base_source === 'env') {
      statusEl.textContent = 'Tunnel URL loaded from VP_PUBLIC_BASE_URL / VP_TUNNEL_BASE_URL.';
    } else {
      statusEl.textContent = 'Tunnel URL detected from the current request origin.';
    }
  } catch (e) {
    actionsHelperTunnelUrl = '';
    actionsHelperImportUrl = '';
    tunnelEl.value = '';
    importEl.value = '';
    btnCopyTunnel.disabled = true;
    btnCopyImport.disabled = true;
    statusEl.textContent = `Failed to load Actions helper info: ${e.message}`;
  } finally {
    if (btnRefresh) btnRefresh.disabled = false;
  }
}

function wireActionsHelperUI() {
  const modal = $('#actionsHelperModal');
  const btnClose = $('#btnActionsHelperClose');
  const btnCopyTunnel = $('#btnCopyActionsTunnelUrl');
  const btnCopyImport = $('#btnCopyActionsImportUrl');
  const btnRefresh = $('#btnRefreshActionsHelper');

  document.querySelectorAll('[data-action="actions-helper"]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      if (!modal) return;
      modal.style.display = 'flex';
      await refreshActionsHelper();
    });
  });

  const close = () => {
    if (modal) modal.style.display = 'none';
  };

  if (btnClose) btnClose.onclick = close;

  if (modal) {
    modal.onclick = (e) => {
      if (e.target === modal) close();
    };
    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && modal.style.display !== 'none') close();
    });
  }

  if (btnCopyTunnel) {
    btnCopyTunnel.onclick = async () => {
      const ok = await copyTextToClipboard(actionsHelperTunnelUrl);
      if (!ok) return alert('Failed to copy. Please copy manually.');
      const old = btnCopyTunnel.textContent;
      btnCopyTunnel.textContent = 'Copied!';
      setTimeout(() => { btnCopyTunnel.textContent = old; }, 1200);
    };
  }

  if (btnCopyImport) {
    btnCopyImport.onclick = async () => {
      const ok = await copyTextToClipboard(actionsHelperImportUrl);
      if (!ok) return alert('Failed to copy. Please copy manually.');
      const old = btnCopyImport.textContent;
      btnCopyImport.textContent = 'Copied!';
      setTimeout(() => { btnCopyImport.textContent = old; }, 1200);
    };
  }

  if (btnRefresh) {
    btnRefresh.onclick = refreshActionsHelper;
  }
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
      const hasOrphaned = paths.some(path => {
        const cb = document.querySelector(`.video-checkbox[data-path="${CSS.escape(path)}"]`);
        return cb && cb.dataset.orphaned === 'true';
      });
      
      let msg = `Delete ${paths.length} video(s)?\n\nThis will permanently delete the video files.`;
      if (hasProjects) {
        msg += `\n\nAssociated projects (analysis data, selections, exports) will also be deleted.`;
      }
      if (hasOrphaned) {
        msg += `\n\nSome selected items have no video file; their project folders will be deleted.`;
      }
      
      if (!confirm(msg)) return;
      
      for (const path of paths) {
        try {
          const cb = document.querySelector(`.video-checkbox[data-path="${CSS.escape(path)}"]`);
          const isOrphaned = cb && cb.dataset.orphaned === 'true';
          const projectId = cb?.dataset.projectId || null;
          if (isOrphaned && projectId) {
            await apiJson('DELETE', `/api/home/project/${projectId}`, null);
          } else {
            await apiJson('DELETE', '/api/home/video', { video_path: path, delete_project: true, project_id: projectId });
          }
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
    const includeIncomplete = $('#showNeedsAttention')?.checked ?? false;
    const qs = includeIncomplete ? '?include_incomplete=1' : '';
    const res = await apiGet(`/api/home/videos${qs}`);
    const videos = res.videos || [];

    if (videos.length === 0) {
      container.innerHTML = '<div class="small">No videos found. Download or open a video to get started.</div>';
      return;
    }

    container.innerHTML = '';
    for (const v of videos) {
      const el = document.createElement('div');
      const status = String(v.status || '').toLowerCase();
      const needsAttention = !!(v.orphaned || status === 'download_failed' || status === 'analysis_failed');
      el.className = 'item' + (v.favorite ? ' favorite' : '') + (needsAttention ? ' orphaned' : '');
      const dur = fmtTime(v.duration_seconds || 0);
      const sizeMB = ((v.size_bytes || 0) / 1024 / 1024).toFixed(1);
      
      // Build status badges
      let badges = '';
      if (status === 'downloading') {
        badges += `<span class="badge" style="background:#f59e0b;color:#000">Downloading</span> `;
      } else if (status === 'download_failed') {
        badges += `<span class="badge" style="background:#dc2626;color:#fff">Download failed</span> `;
      } else if (status === 'analyzing') {
        badges += `<span class="badge" style="background:#38bdf8;color:#000">Analyzing</span> `;
      } else if (status === 'analysis_failed') {
        badges += `<span class="badge" style="background:#dc2626;color:#fff">Analysis failed</span> `;
      } else if (status === 'complete') {
        badges += `<span class="badge" style="background:#22c55e;color:#000">Complete</span> `;
      } else if (v.orphaned) {
        badges += `<span class="badge" style="background:#dc2626;color:#fff">Needs attention</span> `;
      }
      if (v.orphaned && v.analysis_count > 0) badges += `<span class="badge">${v.analysis_count} files</span> `;
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
      
      const pathDisplay = v.path || `outputs/projects/${v.project_id}/`;
      const checkboxPath = v.path || v.project_id;
      const errText = String(v.status_error || v.video_error || '').trim();
      const statusLine = v.orphaned
        ? `${v.status || 'Missing video file'}${errText ? ` — ${errText}` : ''}`
        : `${dur} • ${sizeMB} MB`;
      
      el.innerHTML = `
        <div class="item-header">
          <input type="checkbox" class="video-checkbox" data-path="${(checkboxPath || '').replace(/"/g, '&quot;')}" data-has-project="${v.has_project}" data-project-id="${v.project_id || ''}" data-orphaned="${v.orphaned || false}" style="margin-right:8px;cursor:pointer" />
          <div class="title" style="flex:1">${v.title || v.filename}</div>
          ${!v.orphaned ? `<button class="btn-favorite ${v.favorite ? 'active' : ''}" title="${v.favorite ? 'Remove from favorites' : 'Add to favorites'}" data-path="${(v.path || '').replace(/"/g, '&quot;')}">
            ${v.favorite ? '★' : '☆'}
          </button>` : ''}
        </div>
        <div class="meta" style="margin-left:24px" title="${errText.replace(/"/g, '&quot;')}">${statusLine} ${badges}</div>
        <div class="meta small" style="opacity:0.7;word-break:break-all;margin-left:24px">${pathDisplay}</div>
        ${v.url ? `<div class="meta small" style="opacity:0.5;word-break:break-all;margin-left:24px">${v.url}</div>` : ''}
        <div class="actions" style="margin-top:8px;display:flex;gap:8px;flex-wrap:wrap;margin-left:24px">
          ${!v.orphaned ? `<button class="primary btn-open">Open</button>` : ''}
          ${v.orphaned && v.url ? `<button class="primary btn-retry-download" data-url="${String(v.url).replace(/"/g, '&quot;')}">Retry Download</button>` : ''}
          ${v.has_project && !v.orphaned ? `<button class="btn-delete-all" style="background:#dc2626" data-path="${(v.path || '').replace(/"/g, '&quot;')}" data-project-id="${v.project_id}">Delete All</button>` : ''}
          ${v.has_project && !v.orphaned ? `<button class="btn-delete-project" style="background:#ef4444" data-project-id="${v.project_id}">Delete Project Only</button>` : ''}
          ${v.orphaned ? `<button class="btn-delete-project" style="background:#ef4444" data-project-id="${v.project_id}">Delete Project</button>` : ''}
          ${!v.orphaned && !v.has_project ? `<button class="btn-delete-video" style="background:#dc2626" data-path="${(v.path || '').replace(/"/g, '&quot;')}" data-has-project="false">Delete Video</button>` : ''}
        </div>
      `;
      
      // Checkbox for multi-select
      const checkbox = el.querySelector('.video-checkbox');
      checkbox.onclick = (e) => {
        e.stopPropagation();
        if (checkbox.checked) {
          selectedVideos.add(checkboxPath);
        } else {
          selectedVideos.delete(checkboxPath);
        }
        updateVideoSelectionToolbar();
      };
      
      const btnOpen = el.querySelector('.btn-open');
      if (btnOpen) {
        btnOpen.onclick = () => openProjectByPath(v.path);
      }

      const btnRetry = el.querySelector('.btn-retry-download');
      if (btnRetry) {
        btnRetry.onclick = async (e) => {
          e.stopPropagation();
          const url = btnRetry.dataset.url || '';
          if (!url) return;
          await startUrlDownload(url);
        };
      }
      
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
      
      // Delete All button (project + video + downloads + all analysis data)
      const btnDeleteAll = el.querySelector('.btn-delete-all');
      if (btnDeleteAll) {
        btnDeleteAll.onclick = async (e) => {
          e.stopPropagation();
          const msg = `Delete EVERYTHING for "${v.title || v.filename}"?\n\nThis will permanently delete:\n• Project folder (all analysis data, selections, exports)\n• Video file in project\n• Original download and metadata files\n\nThis cannot be undone!`;
          if (confirm(msg)) {
            try {
              await apiJson('DELETE', '/api/home/video_complete', { 
                video_path: v.path, 
                project_id: v.project_id
              });
              loadRecentVideos(); // Refresh list
            } catch (err) {
              alert(`Failed to delete: ${err.message}`);
            }
          }
        };
      }
      
      // Delete project button
      const btnDeleteProject = el.querySelector('.btn-delete-project');
      if (btnDeleteProject) {
        btnDeleteProject.onclick = async (e) => {
          e.stopPropagation();
          const projectId = btnDeleteProject.dataset.projectId;
          const msg = v.orphaned
            ? `Delete project "${v.title || v.filename}"?\n\nThis will remove the project folder (analysis data, selections, exports).`
            : `Delete project for "${v.title || v.filename}"?\n\nThis will remove analysis data, selections, and exports. The video file will NOT be deleted.`;
          if (confirm(msg)) {
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
              // Pass project_id if available for reliable deletion
              await apiJson('DELETE', '/api/home/video', { 
                video_path: v.path, 
                delete_project: true,
                project_id: v.project_id || null
              });
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
  const showNeedsAttention = $('#showNeedsAttention');

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

  if (showNeedsAttention) {
    showNeedsAttention.onchange = () => loadRecentVideos();
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

  // Persist "Verbose logs" for URL downloads
  const dlVerboseLogs = $('#dlVerboseLogs');
  if (dlVerboseLogs) {
    const saved = localStorage.getItem('vp_dl_verbose_logs');
    dlVerboseLogs.checked = saved === null ? true : saved === '1';
    dlVerboseLogs.onchange = () => {
      localStorage.setItem('vp_dl_verbose_logs', dlVerboseLogs.checked ? '1' : '0');
    };
  }

  wireActionsHelperUI();
}

function initTraceTfImportsToggle() {
  const cb = $('#traceTfImports');
  if (!cb) return;
  const saved = localStorage.getItem('vp_trace_tf_imports');
  cb.checked = saved === '1';
  cb.onchange = () => {
    localStorage.setItem('vp_trace_tf_imports', cb.checked ? '1' : '0');
  };
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
    whisper_verbose: $('#dlWhisperVerbose')?.checked ?? true,
    verbose_logs: $('#dlVerboseLogs')?.checked ?? false,
    diarize: $('#dlDiarizeEnabled')?.checked ?? false,
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
  const es = new EventSource(apiUrlWithToken(`/api/jobs/${jobId}/events`));
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
          // If the backend kicked off an auto-upgrade analysis job, watch it in Studio.
          if (j.result?.auto_upgrade_job_id) {
            watchJob(j.result.auto_upgrade_job_id);
          }
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

  // Remove "No active jobs" message if present (it doesn't have data-job-id)
  const noJobsMsg = container.querySelector('.small:not([data-job-id])');
  if (noJobsMsg && !noJobsMsg.closest('.item')) {
    noJobsMsg.remove();
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
  
  // Keep parallel tasks (chat/transcript/audio) updated for URL downloads,
  // including after completion, so the card shows final statuses.
  if (job.kind === 'download_url') {
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
  
  // Get audio analysis status from job result
  const rmsStatus = job.result?.audio_rms_status;
  const rmsPct = job.result?.audio_rms_progress || 0;
  const eventsStatus = job.result?.audio_events_status;
  const eventsPct = job.result?.audio_events_progress || 0;
  
  // Get or create Audio RMS row
  let rmsRow = tasksEl.querySelector('[data-task="audio_rms"]');
  if (!rmsRow) {
    rmsRow = document.createElement('div');
    rmsRow.className = 'task-row';
    rmsRow.dataset.task = 'audio_rms';
    rmsRow.style.cssText = 'padding:6px 8px;border-radius:6px;margin-bottom:6px';
    rmsRow.innerHTML = `
      <div class="small" style="display:flex;align-items:center;gap:6px">
        <span data-role="icon">📊</span>
        <span style="flex:1" data-role="label">Audio RMS</span>
        <span data-role="pct"></span>
      </div>
      <div class="progress" style="margin-top:4px;height:3px"><div data-role="bar" style="width:0%;transition:width 0.2s"></div></div>
    `;
    tasksEl.appendChild(rmsRow);
  }
  
  // Get or create Audio Events row
  let eventsRow = tasksEl.querySelector('[data-task="audio_events"]');
  if (!eventsRow) {
    eventsRow = document.createElement('div');
    eventsRow.className = 'task-row';
    eventsRow.dataset.task = 'audio_events';
    eventsRow.style.cssText = 'padding:6px 8px;border-radius:6px;margin-bottom:6px';
    eventsRow.innerHTML = `
      <div class="small" style="display:flex;align-items:center;gap:6px">
        <span data-role="icon">🎭</span>
        <span style="flex:1" data-role="label">Audio Events</span>
        <span data-role="pct"></span>
      </div>
      <div class="progress" style="margin-top:4px;height:3px"><div data-role="bar" style="width:0%;transition:width 0.2s"></div></div>
    `;
    tasksEl.appendChild(eventsRow);
  }
  
  // Update Audio RMS row
  const rmsIcon = rmsRow.querySelector('[data-role="icon"]');
  const rmsLabel = rmsRow.querySelector('[data-role="label"]');
  const rmsPctEl = rmsRow.querySelector('[data-role="pct"]');
  const rmsBar = rmsRow.querySelector('[data-role="bar"]');
  const rmsProgress = rmsRow.querySelector('.progress');
  
  if (rmsStatus === 'analyzing') {
    rmsRow.style.background = 'rgba(59,130,246,0.1)';
    rmsRow.querySelector('.small').style.color = '#3b82f6';
    rmsIcon.textContent = '📊';
    rmsLabel.textContent = 'Audio RMS';
    const rmsPctDisplay = Math.round(rmsPct * 100);
    rmsPctEl.textContent = `${rmsPctDisplay}%`;
    rmsBar.style.width = `${rmsPctDisplay}%`;
    rmsBar.style.background = '#3b82f6';
    rmsProgress.style.display = '';
  } else if (rmsStatus === 'complete') {
    rmsRow.style.background = 'rgba(34,197,94,0.1)';
    rmsRow.querySelector('.small').style.color = '#22c55e';
    rmsIcon.textContent = '✓';
    rmsLabel.textContent = 'Audio RMS ready';
    rmsPctEl.textContent = '';
    rmsProgress.style.display = 'none';
  } else if (rmsStatus === 'failed') {
    rmsRow.style.background = 'rgba(239,68,68,0.1)';
    rmsRow.querySelector('.small').style.color = '#ef4444';
    rmsIcon.textContent = '⚠️';
    rmsLabel.textContent = 'Audio RMS failed';
    rmsPctEl.textContent = '';
    rmsProgress.style.display = 'none';
  } else {
    // Pending or unknown state - hide until analysis starts
    rmsRow.style.background = 'rgba(100,100,100,0.1)';
    rmsRow.querySelector('.small').style.color = '#888';
    rmsIcon.textContent = '📊';
    rmsLabel.textContent = 'Audio RMS waiting...';
    rmsPctEl.textContent = '';
    rmsProgress.style.display = 'none';
  }
  
  // Update Audio Events row
  const eventsIcon = eventsRow.querySelector('[data-role="icon"]');
  const eventsLabel = eventsRow.querySelector('[data-role="label"]');
  const eventsPctEl = eventsRow.querySelector('[data-role="pct"]');
  const eventsBar = eventsRow.querySelector('[data-role="bar"]');
  const eventsProgress = eventsRow.querySelector('.progress');
  
  if (eventsStatus === 'analyzing') {
    eventsRow.style.background = 'rgba(168,85,247,0.1)';
    eventsRow.querySelector('.small').style.color = '#a855f7';
    eventsIcon.textContent = '🎭';
    eventsLabel.textContent = 'Audio Events';
    const eventsPctDisplay = Math.round(eventsPct * 100);
    eventsPctEl.textContent = `${eventsPctDisplay}%`;
    eventsBar.style.width = `${eventsPctDisplay}%`;
    eventsBar.style.background = '#a855f7';
    eventsProgress.style.display = '';
  } else if (eventsStatus === 'complete') {
    eventsRow.style.background = 'rgba(34,197,94,0.1)';
    eventsRow.querySelector('.small').style.color = '#22c55e';
    eventsIcon.textContent = '✓';
    eventsLabel.textContent = 'Audio Events ready';
    eventsPctEl.textContent = '';
    eventsProgress.style.display = 'none';
  } else if (eventsStatus === 'failed') {
    eventsRow.style.background = 'rgba(239,68,68,0.1)';
    eventsRow.querySelector('.small').style.color = '#ef4444';
    eventsIcon.textContent = '⚠️';
    eventsLabel.textContent = 'Audio Events failed';
    eventsPctEl.textContent = '';
    eventsProgress.style.display = 'none';
  } else {
    // Pending or unknown state - hide until analysis starts
    eventsRow.style.background = 'rgba(100,100,100,0.1)';
    eventsRow.querySelector('.small').style.color = '#888';
    eventsIcon.textContent = '🎭';
    eventsLabel.textContent = 'Audio Events waiting...';
    eventsPctEl.textContent = '';
    eventsProgress.style.display = 'none';
  }
  
  // Get or create Diarization row (only show if diarization is being used)
  const diarStatus = job.result?.diarization_status;
  const diarPct = job.result?.diarization_progress || 0;
  const diarSpeakers = job.result?.diarization_speakers || [];
  
  let diarRow = tasksEl.querySelector('[data-task="diarization"]');
  if (diarStatus && diarStatus !== 'pending') {
    if (!diarRow) {
      diarRow = document.createElement('div');
      diarRow.className = 'task-row';
      diarRow.dataset.task = 'diarization';
      diarRow.style.cssText = 'padding:6px 8px;border-radius:6px;margin-bottom:6px';
      diarRow.innerHTML = `
        <div class="small" style="display:flex;align-items:center;gap:6px">
          <span data-role="icon">🗣️</span>
          <span style="flex:1" data-role="label">Speaker Diarization</span>
          <span data-role="pct"></span>
        </div>
        <div class="progress" style="margin-top:4px;height:3px"><div data-role="bar" style="width:0%;transition:width 0.2s"></div></div>
      `;
      tasksEl.appendChild(diarRow);
    }
    
    const diarIcon = diarRow.querySelector('[data-role="icon"]');
    const diarLabel = diarRow.querySelector('[data-role="label"]');
    const diarPctEl = diarRow.querySelector('[data-role="pct"]');
    const diarBar = diarRow.querySelector('[data-role="bar"]');
    const diarProgress = diarRow.querySelector('.progress');
    
    if (diarStatus === 'analyzing') {
      diarRow.style.background = 'rgba(251,191,36,0.1)';
      diarRow.querySelector('.small').style.color = '#fbbf24';
      diarIcon.textContent = '🗣️';
      diarLabel.textContent = 'Speaker Diarization';
      const diarPctDisplay = Math.round(diarPct * 100);
      diarPctEl.textContent = `${diarPctDisplay}%`;
      diarBar.style.width = `${diarPctDisplay}%`;
      diarBar.style.background = '#fbbf24';
      diarProgress.style.display = '';
    } else if (diarStatus === 'complete') {
      diarRow.style.background = 'rgba(34,197,94,0.1)';
      diarRow.querySelector('.small').style.color = '#22c55e';
      diarIcon.textContent = '✓';
      diarLabel.textContent = `Speakers identified (${diarSpeakers.length})`;
      diarPctEl.textContent = '';
      diarProgress.style.display = 'none';
    } else if (diarStatus === 'failed') {
      diarRow.style.background = 'rgba(239,68,68,0.1)';
      diarRow.querySelector('.small').style.color = '#ef4444';
      diarIcon.textContent = '⚠️';
      diarLabel.textContent = 'Diarization failed';
      diarPctEl.textContent = '';
      diarProgress.style.display = 'none';
    }
  } else if (diarRow) {
    // Remove diarization row if not being used
    diarRow.remove();
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
	          const timeStr = elapsed != null ? ` • ${fmtDuration(elapsed)}` : '';
	          return { state: 'done', detail: `${t.segment_count} segments • ${backend}${gpu}${lang}${timeStr}` };
	        }
	        return { state: 'pending', detail: 'Not yet run' };
	      }
	    },
	    {
	      id: 'diarization',
	      name: 'Speaker Diarization',
	      icon: '👥',
	      check: () => {
	        const d = analysis.diarization;
	        if (hasRun(d)) {
	          const speakers = d.speaker_count || 0;
	          const segments = d.segment_count || 0;
	          const timeStr = d.elapsed_seconds != null ? ` • ${fmtDuration(d.elapsed_seconds)}` : '';
	          
	          // Hardware + batch info (best-effort; older projects won't have these fields)
	          let hw = '';
	          const fp = d.device_fingerprint || '';
	          if (fp) {
	            const parts = String(fp).split(':');
	            const backend = parts[0] || '';
	            let deviceName = '';
	            const torchI = parts.findIndex(p => String(p).startsWith('torch='));
	            if (torchI > 1) deviceName = parts.slice(1, torchI).join(':');
	            else if (parts.length > 1) deviceName = parts.slice(1).join(':');
	            
	            if (backend === 'rocm') hw = 'GPU (ROCm)';
	            else if (backend === 'cuda') hw = 'GPU (CUDA)';
	            else if (backend === 'mps') hw = 'GPU (MPS)';
	            else if (backend === 'cpu') hw = 'CPU';
	            else hw = backend ? backend.toUpperCase() : 'unknown';
	            
	            if (deviceName) hw += ` • ${deviceName}`;
	          }
	          
	          let bs = '';
	          const b = d.batching || {};
	          const segBs = b.segmentation_batch_size ?? null;
	          const embBs = b.embedding_batch_size ?? null;
	          if (segBs != null || embBs != null) {
	            if (segBs != null && embBs != null && segBs === embBs) bs = `bs=${segBs}`;
	            else bs = `bs=${segBs ?? '∅'}/${embBs ?? '∅'}`;
	            if (b.auto_probe_used) bs += b.auto_probe_from_cache ? ' (auto,cached)' : ' (auto)';
	          } else if (b && Object.keys(b).length > 0) {
	            bs = 'bs=default';
	          }

	          const decode = d.used_waveform_input ? 'decode=ffmpeg' : '';
	          
	          const extras = [hw, bs, decode].filter(Boolean).map(s => ` • ${s}`).join('');
	          return { state: 'done', detail: `${speakers} speakers • ${segments} segments${extras}${timeStr}` };
	        }

		        // If diarization isn't enabled, treat as skipped (it is optional).
		        const diarizeEnabled = $('#diarizeEnabled')?.checked ?? (profile?.analysis?.speech?.diarize === true);
		        if (!diarizeEnabled) {
		          return { state: 'skipped', detail: 'Disabled (speech.diarize=false)' };
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
	          const timeStr = a.elapsed_seconds != null ? ` • ${fmtDuration(a.elapsed_seconds)}` : '';
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
	          const timeStr = m.elapsed_seconds != null ? ` • ${fmtDuration(m.elapsed_seconds)}` : '';
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
	          const timeStr = ae.elapsed_seconds != null ? ` • ${fmtDuration(ae.elapsed_seconds)}` : '';
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
	          const timeStr = cf.elapsed_seconds != null ? ` • ${fmtDuration(cf.elapsed_seconds)}` : '';
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
	          const llmScoring = h.signals_used?.llm_semantic ? ' + LLM' : '';
	          const llmFilter = h.signals_used?.llm_filter ? ' (filtered)' : '';
	          const speechUsed = h.signals_used?.speech ? ' + speech' : '';
	          const reactionUsed = h.signals_used?.reaction ? ' + reaction' : '';
	          const extras = speechUsed + reactionUsed + llmScoring + llmFilter;
	          const timeStr = h.elapsed_seconds != null ? ` • ${fmtDuration(h.elapsed_seconds)}` : '';
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
	          const timeStr = sf.elapsed_seconds != null ? ` • ${fmtDuration(sf.elapsed_seconds)}` : '';
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
	          const timeStr = ra.elapsed_seconds != null ? ` • ${fmtDuration(ra.elapsed_seconds)}` : '';
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
	          const timeStr = h.enrich_elapsed_seconds != null ? ` • ${fmtDuration(h.enrich_elapsed_seconds)}` : '';
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
	          const timeStr = ch.elapsed_seconds != null ? ` • ${fmtDuration(ch.elapsed_seconds)}` : '';
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
	          const timeStr = b.elapsed_seconds != null ? ` • ${fmtDuration(b.elapsed_seconds)}` : '';
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
	          const count = cv.candidate_count || 0;
	          const timeStr = cv.elapsed_seconds != null ? ` • ${fmtDuration(cv.elapsed_seconds)}` : '';
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
	          const timeStr = d.elapsed_seconds != null ? ` • ${fmtDuration(d.elapsed_seconds)}` : '';
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
    
    // Only make clickable if task has run (done or partial)
    const isClickable = result.state === 'done' || result.state === 'partial';
    const cursorStyle = isClickable ? 'cursor:pointer;' : '';
    const hoverTitle = isClickable ? 'title="Click for details"' : '';
    
    // Optional quick-actions per task (keep minimal; avoid clutter)
    let actionsHtml = '';
    if (isClickable && stage.id === 'diarization') {
      actionsHtml = `
        <a
          data-role="artifact-download"
          href="/api/task_artifact/diarization?download=1"
          title="Download diarization.json"
          style="padding:2px 6px;border-radius:6px;background:rgba(34,197,94,0.15);color:#22c55e;text-decoration:none;font-size:11px;"
        >⬇</a>
      `;
    } else if (isClickable && stage.id === 'transcript') {
      actionsHtml = `
        <a
          data-role="artifact-download"
          href="/api/task_artifact/transcript?download=1"
          title="Download transcript_full.json"
          style="padding:2px 6px;border-radius:6px;background:rgba(34,197,94,0.15);color:#22c55e;text-decoration:none;font-size:11px;"
        >⬇</a>
      `;
    }
    
    html += `
      <div class="pipeline-stage" data-task-id="${stage.id}" data-clickable="${isClickable}" style="display:flex;align-items:center;gap:8px;padding:6px 10px;background:${bgColor};border-radius:6px;border-left:3px solid ${stateColor};${cursorStyle}" ${hoverTitle}>
        <span style="font-size:14px">${stage.icon}</span>
        <span style="flex:1;color:${stateColor};font-weight:500">${stage.name}</span>
        ${actionsHtml}
        ${isClickable ? '<span style="color:#666;font-size:10px;margin-right:4px">ℹ️</span>' : ''}
        <span style="color:${stateColor};font-size:11px">${stateIcon}</span>
      </div>
      <div style="margin-left:32px;margin-top:-4px;margin-bottom:4px;color:#888;font-size:10px">${result.detail}</div>
    `;
  }
  
  html += '</div>';
  
  // AI Enhancement section - show when highlights exist but LLM wasn't fully used
  const highlights = analysis.highlights;
  const hasHighlights = highlights?.candidates?.length > 0;
  const llmScoringUsed = highlights?.signals_used?.llm_semantic;
  const llmFilterUsed = highlights?.signals_used?.llm_filter;
  const showAISection = hasHighlights && (!llmScoringUsed || !llmFilterUsed);
  
  if (showAISection) {
    html += `
      <div style="margin-top:16px;padding:12px;background:linear-gradient(135deg, rgba(139,92,246,0.15), rgba(59,130,246,0.15));border:1px solid rgba(139,92,246,0.3);border-radius:8px">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
          <span style="font-size:16px">🤖</span>
          <span style="color:#a78bfa;font-weight:600">AI Enhancement</span>
        </div>
        <div style="color:#888;font-size:11px;margin-bottom:10px">
          ${!llmScoringUsed ? 'LLM semantic scoring not applied. ' : ''}
          ${!llmFilterUsed ? 'LLM quality filter not applied.' : ''}
        </div>
        <div style="display:flex;gap:8px;flex-wrap:wrap">
          ${!llmScoringUsed ? `
            <button id="pipelineLlmScoring" style="padding:6px 12px;background:#eab308;color:#000;border:none;border-radius:6px;font-size:12px;font-weight:600;cursor:pointer;display:flex;align-items:center;gap:4px" title="Apply LLM semantic scoring to existing candidates">
              🎯 Apply LLM Scoring
            </button>
          ` : ''}
          ${!llmFilterUsed ? `
            <button id="pipelineLlmFilter" style="padding:6px 12px;background:#ef4444;color:#fff;border:none;border-radius:6px;font-size:12px;font-weight:600;cursor:pointer;display:flex;align-items:center;gap:4px" title="Filter out low-quality candidates using LLM">
              🗑️ Filter Low Quality
            </button>
          ` : ''}
        </div>
      </div>
    `;
  }
  
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
  
  // Wire up AI Enhancement button handlers
  const llmScoringBtn = container.querySelector('#pipelineLlmScoring');
  if (llmScoringBtn) {
    llmScoringBtn.addEventListener('click', async () => {
      llmScoringBtn.disabled = true;
      llmScoringBtn.innerHTML = '⏳ Running...';
      try {
        const data = await apiJson('POST', `/api/analyze/llm_semantic?project_id=${project.id}`, null);
        if (data.job_id) {
          pollJobStatus(data.job_id, 'LLM Scoring', async () => {
            await loadProjectData(project.id);
            renderPipelineStatus();
            renderCandidateList();
          });
        } else if (data.error) {
          alert('Error: ' + data.error);
        }
      } catch (e) {
        alert('Error applying LLM scoring: ' + e.message);
      }
    });
  }
  
  const llmFilterBtn = container.querySelector('#pipelineLlmFilter');
  if (llmFilterBtn) {
    llmFilterBtn.addEventListener('click', async () => {
      const count = highlights?.candidates?.length || 0;
      if (!confirm(`Apply LLM quality filter to ${count} candidates?\n\nThis will score each candidate 1-10 and remove those below quality threshold.`)) return;
      llmFilterBtn.disabled = true;
      llmFilterBtn.innerHTML = '⏳ Filtering...';
      try {
        const data = await apiJson('POST', `/api/analyze/llm_filter?project_id=${project.id}`, null);
        if (data.job_id) {
          pollJobStatus(data.job_id, 'LLM Filter', async () => {
            await loadProjectData(project.id);
            renderPipelineStatus();
            renderCandidateList();
          });
        } else if (data.error) {
          alert('Error: ' + data.error);
        }
      } catch (e) {
        alert('Error applying LLM filter: ' + e.message);
      }
    });
  }
  
  // Wire up click handlers for completed stages
  container.querySelectorAll('.pipeline-stage[data-clickable="true"]').forEach(el => {
    el.addEventListener('click', () => {
      const taskId = el.dataset.taskId;
      showTaskDetailsModal(taskId);
    });
    // Add hover effect
    el.addEventListener('mouseenter', () => el.style.opacity = '0.8');
    el.addEventListener('mouseleave', () => el.style.opacity = '1');
  });

  // Prevent quick-action clicks from also opening the details modal.
  container.querySelectorAll('[data-role="artifact-download"]').forEach(a => {
    a.addEventListener('click', (e) => e.stopPropagation());
  });
}

// Modal for showing task details ("show work" feature)
async function showTaskDetailsModal(taskId) {
  // Remove existing modal if any
  const existing = document.querySelector('.task-details-modal-overlay');
  if (existing) existing.remove();
  
  // Fetch task details from API
  try {
    const data = await apiGet(`/api/task_details/${taskId}`);
    if (!data.ok) {
      alert(`No details available: ${data.reason}`);
      return;
    }
    
    const overlay = document.createElement('div');
    overlay.className = 'task-details-modal-overlay';
    overlay.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.8);z-index:1000;display:flex;align-items:center;justify-content:center;';
    
    const modal = document.createElement('div');
    modal.style.cssText = 'background:var(--bg);padding:24px;border-radius:8px;max-width:600px;max-height:80vh;overflow-y:auto;min-width:400px;';
    
    // Build content based on task type
    let contentHtml = `<h3 style="margin:0 0 16px 0;">📊 ${formatTaskName(taskId)} - Details</h3>`;
    
    // Summary section
    if (data.summary) {
      contentHtml += '<div style="background:rgba(255,255,255,0.05);padding:12px;border-radius:6px;margin-bottom:12px;">';
      contentHtml += '<div style="font-weight:600;margin-bottom:8px;color:#22c55e;">Summary</div>';
      for (const [key, value] of Object.entries(data.summary)) {
        if (value !== null && value !== undefined) {
          const displayKey = key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
          let displayValue = value;
	          if (typeof value === 'object') {
	            displayValue = JSON.stringify(value, null, 2);
	          } else if (typeof value === 'boolean') {
	            displayValue = value ? '✓ Yes' : '✗ No';
	          } else if (key.includes('seconds') && typeof value === 'number') {
	            displayValue = fmtDuration(value, { decimalsUnderMinute: 2 });
	          } else if ((key === 'created_at' || key === 'enriched_at') && typeof value === 'string') {
	            // Format ISO timestamp to readable date/time
	            try {
	              const d = new Date(value);
	              displayValue = d.toLocaleString();
            } catch (e) {
              displayValue = value;
            }
          }
          contentHtml += `<div style="display:flex;justify-content:space-between;margin:4px 0;"><span style="color:#888;">${displayKey}:</span><span style="color:#fff;">${displayValue}</span></div>`;
        }
      }
      contentHtml += '</div>';
    }

    // Artifact links (download/view JSON, etc.)
    if (Array.isArray(data.artifacts) && data.artifacts.length > 0) {
      contentHtml += '<div style="background:rgba(255,255,255,0.05);padding:12px;border-radius:6px;margin-bottom:12px;">';
      contentHtml += '<div style="font-weight:600;margin-bottom:8px;color:#60a5fa;">Artifacts</div>';
      for (const a of data.artifacts) {
        const label = a.label || 'artifact';
        const urlView = a.url_view || '';
        const urlDl = a.url_download || '';
        contentHtml += `
          <div style="display:flex;align-items:center;justify-content:space-between;gap:8px;margin:6px 0;">
            <span style="color:#fff;">${label}</span>
            <span style="display:flex;gap:6px;">
              ${urlView ? `<a href="${urlView}" target="_blank" style="padding:4px 8px;border-radius:6px;background:rgba(96,165,250,0.2);color:#60a5fa;text-decoration:none;font-size:11px;">View</a>` : ''}
              ${urlDl ? `<a href="${urlDl}" style="padding:4px 8px;border-radius:6px;background:rgba(34,197,94,0.2);color:#22c55e;text-decoration:none;font-size:11px;">Download</a>` : ''}
            </span>
          </div>
        `;
      }
      contentHtml += '</div>';
    }
    
    // Source breakdown for boundaries
    if (data.source_breakdown) {
      contentHtml += '<div style="background:rgba(255,255,255,0.05);padding:12px;border-radius:6px;margin-bottom:12px;">';
      contentHtml += '<div style="font-weight:600;margin-bottom:8px;color:#3b82f6;">Boundary Sources Used</div>';
      for (const [source, count] of Object.entries(data.source_breakdown).sort((a,b) => b[1] - a[1])) {
        const barWidth = Math.min(100, (count / Math.max(...Object.values(data.source_breakdown))) * 100);
        contentHtml += `
          <div style="margin:6px 0;">
            <div style="display:flex;justify-content:space-between;margin-bottom:2px;">
              <span style="color:#fff;">${source}</span>
              <span style="color:#888;">${count}</span>
            </div>
            <div style="background:rgba(255,255,255,0.1);border-radius:3px;height:6px;">
              <div style="background:#3b82f6;width:${barWidth}%;height:100%;border-radius:3px;"></div>
            </div>
          </div>
        `;
      }
      contentHtml += '</div>';
    }
    
    // Variant type counts
    if (data.variant_type_counts) {
      contentHtml += '<div style="background:rgba(255,255,255,0.05);padding:12px;border-radius:6px;margin-bottom:12px;">';
      contentHtml += '<div style="font-weight:600;margin-bottom:8px;color:#eab308;">Variants by Type</div>';
      for (const [type, count] of Object.entries(data.variant_type_counts)) {
        contentHtml += `<span style="background:rgba(234,179,8,0.2);color:#eab308;padding:4px 8px;border-radius:4px;margin-right:6px;">${type}: ${count}</span>`;
      }
      contentHtml += '</div>';
    }
    
    // Chapters list
    if (data.chapters && data.chapters.length > 0) {
      contentHtml += '<div style="background:rgba(255,255,255,0.05);padding:12px;border-radius:6px;margin-bottom:12px;">';
      contentHtml += '<div style="font-weight:600;margin-bottom:8px;color:#a855f7;">Chapters</div>';
      for (const ch of data.chapters) {
        contentHtml += `<div style="margin:4px 0;padding:6px;background:rgba(168,85,247,0.1);border-radius:4px;">
          <span style="color:#a855f7;">${fmtTime(ch.start_s)} - ${fmtTime(ch.end_s)}</span>
          <span style="color:#fff;margin-left:8px;">${ch.title}</span>
        </div>`;
      }
      contentHtml += '</div>';
    }
    
    // Sample segments/events/candidates
    const sampleSections = taskId === 'diarization' ? [
      { key: 'speakers_list', title: 'Speakers', color: '#22c55e' },
      { key: 'sample_segments', title: 'Sample Speaker Segments', color: '#22c55e' },
    ] : [
      { key: 'sample_segments', title: 'Sample Transcript Segments', color: '#22c55e' },
      { key: 'top_peaks', title: 'Top Audio Peaks', color: '#f59e0b' },
      { key: 'sample_events', title: 'Sample Audio Events', color: '#ec4899' },
      { key: 'top_candidates', title: 'Top Highlight Candidates', color: '#eab308' },
      { key: 'sample_enriched', title: 'Sample Enriched Candidates', color: '#6366f1' },
      { key: 'sample_results', title: 'Sample AI Director Results', color: '#06b6d4' },
    ];
    
    for (const section of sampleSections) {
      if (data[section.key] && data[section.key].length > 0) {
        contentHtml += `<div style="background:rgba(255,255,255,0.05);padding:12px;border-radius:6px;margin-bottom:12px;">`;
        contentHtml += `<div style="font-weight:600;margin-bottom:8px;color:${section.color};">${section.title}</div>`;
        for (const item of data[section.key]) {
          contentHtml += '<div style="margin:4px 0;padding:6px;background:rgba(255,255,255,0.03);border-radius:4px;font-size:12px;">';
          if (item === null || item === undefined) {
            // skip
          } else if (typeof item !== 'object') {
            contentHtml += `<div style="color:#fff;">${String(item)}</div>`;
          } else {
            for (const [k, v] of Object.entries(item)) {
              if (v !== null && v !== undefined && v !== '') {
                const displayVal = typeof v === 'object' ? JSON.stringify(v) : v;
                contentHtml += `<div><span style="color:#888;">${k}:</span> <span style="color:#fff;">${displayVal}</span></div>`;
              }
            }
          }
          contentHtml += '</div>';
        }
        contentHtml += '</div>';
      }
    }
    
    // Add "Apply LLM Scoring" button for highlights task when llm_semantic is not true
    const showLlmScoringBtn = taskId === 'highlights' && 
      data.summary?.signals_used && 
      !data.summary.signals_used.llm_semantic;
    
    if (showLlmScoringBtn) {
      contentHtml += `
        <div style="background:rgba(234,179,8,0.1);border:1px solid rgba(234,179,8,0.3);padding:12px;border-radius:6px;margin-bottom:12px;">
          <div style="display:flex;align-items:center;gap:12px;">
            <span style="color:#eab308;">⚠️ LLM semantic scoring was not applied</span>
            <button class="run-llm-scoring" style="background:#eab308;color:#000;padding:6px 12px;border:none;border-radius:4px;cursor:pointer;font-weight:600;">
              🤖 Apply LLM Scoring
            </button>
          </div>
          <div style="color:#888;font-size:12px;margin-top:6px;">
            This will use AI to re-rank candidates based on content quality (30% weight).
          </div>
        </div>
      `;
    }
    
    // Add "Apply LLM Filter" button when llm_filter is not true (more aggressive than scoring)
    const showLlmFilterBtn = taskId === 'highlights' && 
      data.summary?.signals_used && 
      !data.summary.signals_used.llm_filter;
    
    if (showLlmFilterBtn) {
      contentHtml += `
        <div style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);padding:12px;border-radius:6px;margin-bottom:12px;">
          <div style="display:flex;align-items:center;gap:12px;">
            <span style="color:#ef4444;">⚠️ LLM quality filter was not applied</span>
            <button class="run-llm-filter" style="background:#ef4444;color:#fff;padding:6px 12px;border:none;border-radius:4px;cursor:pointer;font-weight:600;">
              🗑️ Filter Low Quality
            </button>
          </div>
          <div style="color:#888;font-size:12px;margin-top:6px;">
            This will use AI to REMOVE boring candidates, keeping only genuinely good clips.
          </div>
        </div>
      `;
    }
    
    contentHtml += `<div style="margin-top:16px;text-align:right;"><button class="close-modal">Close</button></div>`;
    
    modal.innerHTML = contentHtml;
    overlay.appendChild(modal);
    document.body.appendChild(overlay);
    
    modal.querySelector('.close-modal').onclick = () => overlay.remove();
    overlay.onclick = (e) => { if (e.target === overlay) overlay.remove(); };
    
    // Handle LLM scoring button click
    const llmBtn = modal.querySelector('.run-llm-scoring');
    if (llmBtn) {
      llmBtn.onclick = async () => {
        llmBtn.disabled = true;
        llmBtn.textContent = '⏳ Running...';
        try {
          const resp = await apiJson('POST', '/api/analyze/llm_semantic', {});
          if (resp.job_id) {
            // Close modal
            overlay.remove();
            // Poll for completion
            pollJobStatus(resp.job_id, () => {
              alert('✅ LLM scoring complete! Refresh to see updated rankings.');
              // Refresh project data to show updated results
              if (typeof refreshProject === 'function') {
                refreshProject();
              }
            }, (err) => {
              alert(`❌ LLM scoring failed: ${err}`);
            });
          }
        } catch (e) {
          llmBtn.disabled = false;
          llmBtn.textContent = '🤖 Apply LLM Scoring';
          alert(`Failed to start LLM scoring: ${e.message}`);
        }
      };
    }
    
    // Handle LLM filter button click
    const filterBtn = modal.querySelector('.run-llm-filter');
    if (filterBtn) {
      filterBtn.onclick = async () => {
        filterBtn.disabled = true;
        filterBtn.textContent = '⏳ Filtering...';
        try {
          const resp = await apiJson('POST', '/api/analyze/llm_filter', {});
          if (resp.job_id) {
            overlay.remove();
            pollJobStatus(resp.job_id, (job) => {
              const result = job.result || {};
              const kept = result.kept || '?';
              const rejected = result.rejected || '?';
              alert(`✅ LLM filter complete!\n\nKept: ${kept} candidates\nRejected: ${rejected} low-quality candidates`);
              if (typeof refreshProject === 'function') {
                refreshProject();
              }
            }, (err) => {
              alert(`❌ LLM filter failed: ${err}`);
            });
          }
        } catch (e) {
          filterBtn.disabled = false;
          filterBtn.textContent = '🗑️ Filter Low Quality';
          alert(`Failed to start LLM filter: ${e.message}`);
        }
      };
    }
    
  } catch (e) {
    alert(`Failed to load task details: ${e.message}`);
  }
}

function formatTaskName(taskId) {
  const names = {
    'transcript': 'Transcription',
    'diarization': 'Speaker Diarization',
    'audio': 'Audio RMS',
    'motion': 'Motion Analysis',
    'audio_events': 'Audio Events',
    'chat_features': 'Chat Features',
    'highlights': 'Highlight Fusion',
    'speech_features': 'Speech Features',
    'reaction_audio': 'Reaction Audio',
    'enrich': 'Enrich Candidates',
    'chapters': 'Semantic Chapters',
    'boundaries': 'Context Boundaries',
    'clip_variants': 'Clip Variants',
    'director': 'AI Director',
  };
  return names[taskId] || taskId;
}

function renderCandidates() {
  const container = $('#candidates');
  container.innerHTML = '';

  const highlights = project?.analysis?.highlights;
  const audio = project?.analysis?.audio;
  const aiDirector = project?.analysis?.ai_director;
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
    
    // Get AI metadata if available (stored directly on candidate by backend)
    const ai = c.ai;
    const aiTitle = ai?.title || '';
    const aiHook = ai?.hook || '';
    const aiVariant = ai?.chosen_variant_id || '';
    const aiReason = ai?.reason || '';
    const aiConfidence = typeof ai?.confidence === 'number' ? ai.confidence : null;
    const aiUsedFallback = !!ai?.used_fallback;
    const hasAI = !!ai && (aiTitle || aiHook || aiVariant || aiReason);
    
    let aiHtml = '';
    if (hasAI) {
      const aiTags = (ai?.hashtags || ai?.tags || []).slice(0, 5);
      const tagsHtml = aiTags.length > 0 ? `<div class="ai-tags">${aiTags.map(t => `<span class="ai-tag">#${escapeHtml(String(t).replace(/^#/, ''))}</span>`).join('')}</div>` : '';
      const modelName = aiDirector?.config?.model_name || aiDirector?.config?.model || '';
      const modelBadge = modelName ? `<span class="badge" style="margin-left:6px;opacity:0.85">${escapeHtml(modelName)}</span>` : '';
      const confBadge = aiConfidence !== null ? `<span class="badge" style="margin-left:6px;opacity:0.85">conf ${(aiConfidence * 100).toFixed(0)}%</span>` : '';
      aiHtml = `
        <div class="candidate-ai-section">
          <span class="ai-badge">AI Generated</span>${modelBadge}${aiUsedFallback ? '<span class="badge" style="margin-left:6px;background:#334155;color:#cbd5e1" title="LLM unavailable; rule-based fallback">fallback</span>' : ''}${confBadge}
          <div class="ai-title">${escapeHtml(aiTitle)}</div>
          ${aiHook ? `<div class="ai-hook">"${escapeHtml(aiHook)}"</div>` : ''}
          ${tagsHtml}
          ${aiReason ? `<div class="small" style="margin-top:4px;opacity:0.9">${escapeHtml(aiReason)}</div>` : ''}
          ${aiVariant ? `<div class="small" style="margin-top:4px;">Best variant: <span class="badge">${escapeHtml(aiVariant)}</span></div>` : ''}
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
      // Update timeline overlays to show new selection
      updateTimelineOverlays();
      // Show AI suggestions panel if AI data is available
      if (hasAI) {
        showAISuggestionsPanel(c.rank);
      } else {
        hideAISuggestionsPanel();
      }
    };
    btnPeak.onclick = () => {
      const v = $('#video');
      v.currentTime = c.peak_time_s;
      v.play();
    };
    if (btnApplyAI && hasAI) {
      btnApplyAI.onclick = () => {
        // Apply AI-suggested variant times and title (fetch variant window)
        const variantId = aiVariant;
        if (!variantId) {
          setBuilder(c.start_s, c.end_s, aiTitle, $('#template').value);
          updateTimelineOverlays();
          return;
        }

        apiGet(`/api/clip_variants/${c.rank}`).then(data => {
          const variants = data.variants || [];
          const chosen = variants.find(v => v.variant_id === variantId) || null;
          const start = chosen ? chosen.start_s : c.start_s;
          const end = chosen ? chosen.end_s : c.end_s;
          setBuilder(start, end, aiTitle, $('#template').value);
          const v = $('#video');
          v.currentTime = start;
          // Update timeline overlays
          updateTimelineOverlays();
        }).catch(() => {
          // Fallback if variants are missing
          setBuilder(c.start_s, c.end_s, aiTitle, $('#template').value);
          updateTimelineOverlays();
        });
      };
    }
    if (btnVariants && hasAI) {
      btnVariants.onclick = () => showVariantsModal(c.rank, ai);
    }
    container.appendChild(el);
  }
}

// Modal for showing clip variants
function showVariantsModal(rank, ai) {
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
      const isChosen = v.variant_id === ai?.chosen_variant_id;
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
        setBuilder(v.start_s, v.end_s, ai?.title || '', $('#template').value);
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

function showResetAnalysisModal() {
  // Remove existing modal if any
  const existing = document.querySelector('.reset-analysis-modal-overlay');
  if (existing) existing.remove();

  const chatCount = project?.chat_ai_status?.message_count;
  const hasChat = project?.chat_ai_status?.has_chat || (Number(chatCount || 0) > 0);
  const transcriptSegCount = project?.analysis?.transcript?.segment_count;
  const hasTranscript = !!project?.analysis?.transcript || Number(transcriptSegCount || 0) > 0;

  const fmtCount = (n) => {
    const v = Number(n);
    if (!Number.isFinite(v)) return '';
    return v.toLocaleString();
  };

  return new Promise((resolve) => {
    const analysis = project?.analysis || {};

    const items = [
      // Inputs
      { key: 'transcript', label: 'Transcript', detail: 'Whisper output', status: hasTranscript ? `${fmtCount(transcriptSegCount)} segments` : 'not detected' },
      { key: 'diarization', label: 'Diarization', detail: 'Speaker turns', status: analysis?.diarization ? 'detected' : 'not detected' },
      { key: 'chat', label: 'Chat', detail: 'Chat DB + features', status: hasChat ? `${fmtCount(chatCount)} messages` : 'not detected' },

      // Core signals + highlights
      { key: 'audio', label: 'Audio', detail: 'Energy/RMS timeline', status: analysis?.audio ? 'detected' : 'not detected' },
      { key: 'vad', label: 'VAD (speech activity)', detail: 'Speech segments + timeline', status: analysis?.vad ? 'detected' : 'not detected' },
      { key: 'motion', label: 'Motion', detail: 'Frame-diff motion scores', status: analysis?.motion ? 'detected' : 'not detected' },
      { key: 'scenes', label: 'Scenes', detail: 'Visual cut boundaries', status: analysis?.scenes ? 'detected' : 'not detected' },
      { key: 'highlights', label: 'Highlights', detail: 'Candidates + scoring', status: analysis?.highlights ? 'detected' : 'not detected' },

      // Post-highlights
      { key: 'silence', label: 'Silence', detail: 'Silence intervals', status: analysis?.silence ? 'detected' : 'not detected' },
      { key: 'sentences', label: 'Sentences', detail: 'Sentence boundaries', status: analysis?.sentences ? 'detected' : 'not detected' },
      { key: 'speech', label: 'Speech features', detail: 'Speech density/rate/etc.', status: analysis?.speech ? 'detected' : 'not detected' },
      { key: 'reaction_audio', label: 'Reaction audio', detail: 'Prosody/arousal timeline', status: analysis?.reaction_audio ? 'detected' : 'not detected' },
      { key: 'audio_events', label: 'Audio events', detail: 'Laughter/cheering/etc.', status: analysis?.audio_events ? 'detected' : 'not detected' },

      // Context + AI
      { key: 'chapters', label: 'Chapters', detail: 'Semantic chapters', status: analysis?.chapters ? 'detected' : 'not detected' },
      { key: 'boundaries', label: 'Boundaries', detail: 'Merged boundary graph', status: analysis?.boundaries ? 'detected' : 'not detected' },
      { key: 'variants', label: 'Clip variants', detail: 'Variant windows + strategies', status: (analysis?.variants || analysis?.clip_variants) ? 'detected' : 'not detected' },
      { key: 'ai_director', label: 'AI director', detail: 'LLM picks/titles/hooks', status: analysis?.ai_director ? 'detected' : 'not detected' },
      { key: 'director', label: 'Director (rules)', detail: 'Heuristic picks', status: analysis?.director ? 'detected' : 'not detected' },

      // Chat extras
      { key: 'chat_boundaries', label: 'Chat boundaries', detail: 'Chat valleys', status: analysis?.chat_boundaries ? 'detected' : 'not detected' },
      { key: 'chat_sync', label: 'Chat sync', detail: 'Chat↔video offset', status: analysis?.chat_sync ? 'detected' : 'not detected' },

      // Internal
      { key: 'analysis_state', label: 'Task cache', detail: 'Freshness metadata', status: '' },
    ];

    const overlay = document.createElement('div');
    overlay.className = 'reset-analysis-modal-overlay';
    overlay.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.8);z-index:1000;display:flex;align-items:center;justify-content:center;';

    const modal = document.createElement('div');
    modal.style.cssText = 'background:var(--bg);padding:24px;border-radius:8px;max-width:620px;max-height:80vh;overflow-y:auto;min-width:420px;border:1px solid var(--border);box-shadow:0 20px 60px rgba(0,0,0,0.5);';

    const itemsHtml = items.map((it) => {
      const status = it.status ? ` • ${it.status}` : '';
      return `
        <label style="display:flex;align-items:flex-start;gap:10px;padding:10px;border-radius:8px;cursor:pointer;border:1px solid rgba(255,255,255,0.06);background:rgba(255,255,255,0.03);">
          <input type="checkbox" data-reset-key="${it.key}" checked style="margin-top:2px" />
          <div style="flex:1;">
            <div style="font-weight:700;">${it.label}</div>
            <div class="small">${it.detail}${status}</div>
          </div>
        </label>
      `;
    }).join('');

    modal.innerHTML = `
      <h3 style="margin:0 0 10px 0;">🗑 Reset analysis</h3>
      <div class="small" style="margin-bottom:14px;">
        Uncheck anything you want to keep. Checked items will be reset (deleted). Selections and exports are preserved.
      </div>

      <div style="display:flex;gap:8px;align-items:center;margin:10px 0 12px 0;">
        <button class="keep-all" style="padding:6px 10px;font-size:12px;">Keep all</button>
        <button class="reset-all" style="padding:6px 10px;font-size:12px;">Reset all</button>
        <span id="resetSummary" class="small" style="margin-left:auto;"></span>
      </div>

      <div style="display:grid;gap:8px;margin-bottom:12px;">
        ${itemsHtml}
      </div>

      <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;margin-top:16px;">
        <div class="small">This cannot be undone.</div>
        <div style="display:flex;gap:8px;">
          <button class="close-modal">Cancel</button>
          <button class="confirm-reset" style="background:#dc2626;border-color:#dc2626;color:#fff;font-weight:700;">Reset</button>
        </div>
      </div>
    `;

    overlay.appendChild(modal);
    document.body.appendChild(overlay);

    const summaryEl = modal.querySelector('#resetSummary');
    const confirmBtn = modal.querySelector('.confirm-reset');
    const resetInputs = Array.from(modal.querySelectorAll('input[type="checkbox"][data-reset-key]'));

    const getResetMap = () => {
      const reset = {};
      for (const el of resetInputs) {
        reset[el.dataset.resetKey] = !!el.checked;
      }
      return reset;
    };

    const updateSummary = () => {
      const reset = getResetMap();
      const resetKeys = Object.entries(reset).filter(([, v]) => v).map(([k]) => k);
      if (summaryEl) summaryEl.textContent = `Resetting: ${resetKeys.length}`;
      if (confirmBtn) confirmBtn.disabled = resetKeys.length === 0;
    };

    const close = (result) => {
      window.removeEventListener('keydown', onKeyDown);
      overlay.remove();
      resolve(result);
    };

    const onKeyDown = (e) => {
      if (e.key === 'Escape') close(null);
    };
    window.addEventListener('keydown', onKeyDown);

    modal.querySelector('.close-modal').onclick = () => close(null);
    modal.querySelector('.confirm-reset').onclick = () => close(getResetMap());
    modal.querySelector('.keep-all').onclick = () => {
      resetInputs.forEach(el => el.checked = false);
      updateSummary();
    };
    modal.querySelector('.reset-all').onclick = () => {
      resetInputs.forEach(el => el.checked = true);
      updateSummary();
    };
    resetInputs.forEach(el => el.addEventListener('change', updateSummary));
    updateSummary();
    overlay.onclick = (e) => { if (e.target === overlay) close(null); };
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
      // Update timeline clip region
      updateClipRegion();
      // Hide AI suggestions panel for saved selections (no AI data)
      hideAISuggestionsPanel();
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

  const formatTime = (seconds) => fmtDuration(seconds);
  
  // Helper to compute elapsed time from a timestamp
  const getElapsed = (startedAt) => {
    const s = Number(startedAt);
    if (!Number.isFinite(s) || s <= 0) return null;
    return (Date.now() / 1000) - s;
  };

  for (const job of Array.from(jobs.values()).sort((a,b) => (a.created_at < b.created_at ? 1 : -1))) {
    const el = document.createElement('div');
    el.className = 'item';
    const pct = Math.round((job.progress || 0) * 100);
    
    // Build elapsed time display (total time)
    let totalTimeHtml = '';
    const isRunning = job.status === 'running';
    const liveElapsed = isRunning ? getElapsed(job.started_at) : null;
    const totalSeconds = liveElapsed != null ? liveElapsed : job.elapsed_seconds;
    if (totalSeconds != null) {
      totalTimeHtml = `<span style="color:#4f8cff;font-weight:600">${formatTime(totalSeconds)}</span>`;
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

// Simple job polling for background tasks (used in modals)
function pollJobStatus(jobId, onComplete, onError) {
  const poll = async () => {
    try {
      const job = await apiGet(`/api/jobs/${jobId}`);
      if (job.status === 'succeeded') {
        if (onComplete) onComplete(job);
        return;
      } else if (job.status === 'failed') {
        if (onError) onError(job.message || 'Job failed');
        return;
      }
      // Still running, poll again
      setTimeout(poll, 1000);
    } catch (e) {
      if (onError) onError(e.message);
    }
  };
  poll();
}

function watchJob(jobId) {
  const es = new EventSource(apiUrlWithToken(`/api/jobs/${jobId}/events`));
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
        
        if ((j.kind === 'analyze_audio' || j.kind === 'analyze_highlights' || j.kind === 'analyze_speech' || j.kind === 'analyze_context_titles' || j.kind === 'analyze_full' || j.kind === 'analyze_upgrade') && (j.status === 'succeeded' || j.status === 'failed')) {
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
          // If backend kicked off an auto-upgrade analysis job, watch it too.
          if (j.kind === 'download_chat' && j.result?.auto_upgrade_job_id) {
            watchJob(j.result.auto_upgrade_job_id);
          }
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

// =========================================================================
// INTERACTIVE TIMELINE - State and utilities
// =========================================================================

// Store timeline data globally for overlay calculations
let timelineData = {
  xs: [],           // time values (seconds)
  ys: [],           // score values
  hop: 0.5,         // hop in seconds
  duration: 0,      // total video duration
  chartArea: null,  // chart area bounds for coordinate mapping
};

// Track clip handles drag state
let clipHandleDrag = {
  active: false,
  handle: null,    // 'start' or 'end'
  startX: 0,
};

// Track segment drag state
let segmentDrag = {
  active: false,
  segment: null,
  handle: null,    // 'left', 'right', or 'body'
  candidateIndex: -1,
  originalStart: 0,
  originalEnd: 0,
  startX: 0,
};

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
    
    // Store timeline data for overlays
    timelineData.xs = xs;
    timelineData.ys = ys;
    timelineData.hop = hop;
    timelineData.duration = xs.length > 0 ? xs[xs.length - 1] : (project?.video?.duration_seconds || 0);

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
          borderColor: 'rgba(79, 140, 255, 0.9)',
          backgroundColor: 'rgba(79, 140, 255, 0.1)',
          fill: true,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: {
          mode: 'index',
          intersect: false,
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: (items) => `t = ${fmtTime(items[0].label)}`,
              label: (item) => `score ${Number(item.raw).toFixed(2)}`
            }
          },
          zoom: {
            pan: {
              enabled: true,
              mode: 'x',
              onPan: () => { updateTimelineOverlays(); updateZoomInfo(); },
              onPanComplete: () => { updateTimelineOverlays(); updateZoomInfo(); },
            },
            zoom: {
              wheel: { enabled: true },
              pinch: { enabled: true },
              mode: 'x',
              onZoom: () => { updateTimelineOverlays(); updateZoomInfo(); },
              onZoomComplete: () => { updateTimelineOverlays(); updateZoomInfo(); },
            },
            limits: {
              x: { min: 0, max: timelineData.duration, minRange: 5 },
            },
          },
        },
        scales: {
          x: {
            type: 'linear',
            min: 0,
            max: xs.length > 0 ? xs[xs.length - 1] : 100,
            ticks: {
              callback: (val) => fmtTime(val),
              maxTicksLimit: 8,
            }
          },
          y: {
            ticks: { maxTicksLimit: 5 },
            beginAtZero: true,
          }
        },
        onClick: (evt, elements) => {
          // Only handle clicks when not dragging
          if (clipHandleDrag.active || segmentDrag.active) return;
          
          // Get click position and convert to time
          const rect = evt.chart.canvas.getBoundingClientRect();
          const x = evt.native.clientX - rect.left;
          const time = pixelToTime(x);
          if (time !== null && time >= 0) {
            const v = $('#video');
            v.currentTime = time;
            v.play();
          }
        }
      }
    });
    
    // Store chart area reference and set up overlays after chart renders
    setTimeout(() => {
      if (chart) {
        timelineData.chartArea = chart.chartArea;
        updateTimelineOverlays();
        updatePlayhead();
        updateZoomInfo();
      }
    }, 50);

  } catch (e) {
    // timeline might not exist yet
  }
}

// =========================================================================
// TIMELINE COORDINATE CONVERSION
// =========================================================================

function timeToPixel(time) {
  if (!chart || !chart.chartArea) return null;
  const { left, right } = chart.chartArea;
  const scale = chart.scales.x;
  if (!scale) return null;
  
  const minTime = scale.min;
  const maxTime = scale.max;
  if (maxTime <= minTime) return null;
  
  const fraction = (time - minTime) / (maxTime - minTime);
  return left + fraction * (right - left);
}

function pixelToTime(pixel) {
  if (!chart || !chart.chartArea) return null;
  const { left, right } = chart.chartArea;
  const scale = chart.scales.x;
  if (!scale) return null;
  
  const minTime = scale.min;
  const maxTime = scale.max;
  
  const fraction = (pixel - left) / (right - left);
  return minTime + fraction * (maxTime - minTime);
}

// =========================================================================
// CANDIDATE SEGMENT OVERLAYS
// =========================================================================

function getCandidates() {
  const highlights = project?.analysis?.highlights;
  const audio = project?.analysis?.audio;
  return highlights?.candidates || audio?.candidates || [];
}

function getSegmentScoreClass(score) {
  // Normalize score to determine color class
  const candidates = getCandidates();
  if (candidates.length === 0) return 'score-medium';
  
  const scores = candidates.map(c => c.score);
  const maxScore = Math.max(...scores);
  const minScore = Math.min(...scores);
  const range = maxScore - minScore || 1;
  const normalized = (score - minScore) / range;
  
  if (normalized >= 0.66) return 'score-high';
  if (normalized >= 0.33) return 'score-medium';
  return 'score-low';
}

function buildSegmentTooltip(candidate) {
  const breakdown = candidate.breakdown || {};
  const reasons = [];
  
  // Build reason text based on breakdown
  if (breakdown.audio > 0.3) reasons.push('high audio energy');
  if (breakdown.motion > 0.3) reasons.push('high motion');
  if (breakdown.chat > 0.3) reasons.push('chat spike');
  if (breakdown.audio_events > 0.3) reasons.push('audio event (laughter/cheer)');
  if (breakdown.speech > 0.3) reasons.push('speech emphasis');
  if (breakdown.reaction > 0.3) reasons.push('reaction audio');
  
  const reasonText = reasons.length > 0 ? reasons.join(' + ') : 'combined signals';
  
  // Build breakdown bars HTML
  const breakdownItems = [
    { label: 'Audio', value: breakdown.audio || 0, color: '#3b82f6' },
    { label: 'Motion', value: breakdown.motion || 0, color: '#8b5cf6' },
    { label: 'Chat', value: breakdown.chat || 0, color: '#22c55e' },
    { label: 'Events', value: breakdown.audio_events || 0, color: '#eab308' },
    { label: 'Speech', value: breakdown.speech || 0, color: '#ec4899' },
    { label: 'Reaction', value: breakdown.reaction || 0, color: '#f97316' },
  ].filter(item => item.value > 0.01);
  
  const breakdownHtml = breakdownItems.map(item => `
    <div class="breakdown-item">
      <span>${item.label}</span>
      <div class="breakdown-bar">
        <div class="breakdown-bar-fill" style="width:${Math.min(100, item.value * 100)}%;background:${item.color}"></div>
      </div>
    </div>
  `).join('');
  
  return `
    <div class="tooltip-title">#${candidate.rank} • Score: ${candidate.score.toFixed(2)}</div>
    <div class="tooltip-reason">${reasonText}</div>
    <div style="margin-top:4px;font-size:10px;color:#9aa4b2">
      ${fmtTime(candidate.start_s)} → ${fmtTime(candidate.end_s)} (${(candidate.end_s - candidate.start_s).toFixed(1)}s)
    </div>
    ${breakdownHtml ? `<div class="tooltip-breakdown">${breakdownHtml}</div>` : ''}
  `;
}

function renderCandidateSegments() {
  const container = $('#timelineSegments');
  if (!container) return;
  
  // Check if segments should be shown
  const showSegments = $('#showCandidateSegments')?.checked ?? true;
  if (!showSegments || !chart || !chart.chartArea) {
    container.innerHTML = '';
    return;
  }
  
  const candidates = getCandidates();
  if (candidates.length === 0) {
    container.innerHTML = '';
    return;
  }
  
  // Clear and rebuild segments
  container.innerHTML = '';
  
  candidates.forEach((candidate, index) => {
    const startPx = timeToPixel(candidate.start_s);
    const endPx = timeToPixel(candidate.end_s);
    
    // Skip if segment is not in view
    if (startPx === null || endPx === null) return;
    if (endPx < chart.chartArea.left || startPx > chart.chartArea.right) return;
    
    // Clamp to visible area
    const left = Math.max(startPx, chart.chartArea.left);
    const right = Math.min(endPx, chart.chartArea.right);
    const width = right - left;
    
    if (width < 2) return; // Skip if too narrow
    
    const segment = document.createElement('div');
    segment.className = `timeline-segment ${getSegmentScoreClass(candidate.score)}`;
    segment.dataset.index = index;
    segment.style.left = `${left}px`;
    segment.style.width = `${width}px`;
    
    // Check if this is the currently selected candidate
    if (currentCandidate && currentCandidate.rank === candidate.rank) {
      segment.classList.add('selected');
    }
    
    // Add drag handles
    segment.innerHTML = `
      <div class="segment-handle left"></div>
      <div class="segment-handle right"></div>
    `;
    
    // Tooltip on hover
    segment.addEventListener('mouseenter', (e) => {
      if (segmentDrag.active) return;
      showSegmentTooltip(segment, candidate);
    });
    
    segment.addEventListener('mouseleave', () => {
      hideSegmentTooltip();
    });
    
    // Click to load candidate
    segment.addEventListener('click', (e) => {
      if (segmentDrag.active) return;
      if (e.target.classList.contains('segment-handle')) return;
      
      currentCandidate = candidate;
      setBuilder(candidate.start_s, candidate.end_s, '', $('#template').value);
      const v = $('#video');
      v.currentTime = candidate.start_s;
      v.play();
      
      // Update selection visual
      container.querySelectorAll('.timeline-segment').forEach(s => s.classList.remove('selected'));
      segment.classList.add('selected');
      
      // Update clip region
      updateClipRegion();
    });
    
    // Drag handles for resizing segments
    const leftHandle = segment.querySelector('.segment-handle.left');
    const rightHandle = segment.querySelector('.segment-handle.right');
    
    leftHandle.addEventListener('mousedown', (e) => {
      e.stopPropagation();
      startSegmentDrag(e, segment, 'left', index, candidate);
    });
    
    rightHandle.addEventListener('mousedown', (e) => {
      e.stopPropagation();
      startSegmentDrag(e, segment, 'right', index, candidate);
    });
    
    container.appendChild(segment);
  });
}

let tooltipElement = null;

function showSegmentTooltip(segment, candidate) {
  hideSegmentTooltip();
  
  tooltipElement = document.createElement('div');
  tooltipElement.className = 'segment-tooltip';
  tooltipElement.innerHTML = buildSegmentTooltip(candidate);
  
  segment.appendChild(tooltipElement);
}

function hideSegmentTooltip() {
  if (tooltipElement) {
    tooltipElement.remove();
    tooltipElement = null;
  }
}

// =========================================================================
// SEGMENT DRAG/RESIZE
// =========================================================================

function startSegmentDrag(e, segment, handle, index, candidate) {
  segmentDrag = {
    active: true,
    segment,
    handle,
    candidateIndex: index,
    originalStart: candidate.start_s,
    originalEnd: candidate.end_s,
    startX: e.clientX,
  };
  
  hideSegmentTooltip();
  document.addEventListener('mousemove', onSegmentDrag);
  document.addEventListener('mouseup', onSegmentDragEnd);
  document.body.style.cursor = 'ew-resize';
  document.body.style.userSelect = 'none';
}

function onSegmentDrag(e) {
  if (!segmentDrag.active) return;
  
  const deltaX = e.clientX - segmentDrag.startX;
  const rect = $('#chart').getBoundingClientRect();
  
  // Convert delta to time
  const scale = chart?.scales?.x;
  if (!scale) return;
  
  const { left, right } = chart.chartArea;
  const pxPerSec = (right - left) / (scale.max - scale.min);
  const deltaTime = deltaX / pxPerSec;
  
  const candidates = getCandidates();
  const candidate = candidates[segmentDrag.candidateIndex];
  if (!candidate) return;
  
  let newStart = segmentDrag.originalStart;
  let newEnd = segmentDrag.originalEnd;
  
  if (segmentDrag.handle === 'left') {
    newStart = Math.max(0, segmentDrag.originalStart + deltaTime);
    newStart = Math.min(newStart, newEnd - 1); // Ensure minimum 1 second
  } else if (segmentDrag.handle === 'right') {
    newEnd = Math.min(timelineData.duration, segmentDrag.originalEnd + deltaTime);
    newEnd = Math.max(newEnd, newStart + 1); // Ensure minimum 1 second
  }
  
  // Update segment visual position
  const startPx = timeToPixel(newStart);
  const endPx = timeToPixel(newEnd);
  if (startPx !== null && endPx !== null) {
    segmentDrag.segment.style.left = `${Math.max(startPx, chart.chartArea.left)}px`;
    segmentDrag.segment.style.width = `${Math.min(endPx, chart.chartArea.right) - Math.max(startPx, chart.chartArea.left)}px`;
  }
  
  // Update clip builder in real-time
  if (currentCandidate && currentCandidate.rank === candidate.rank) {
    setBuilder(newStart, newEnd, $('#title').value, $('#template').value);
    updateClipRegion();
  }
  
  // Store the new values for use on drag end
  segmentDrag.newStart = newStart;
  segmentDrag.newEnd = newEnd;
}

function onSegmentDragEnd(e) {
  if (!segmentDrag.active) return;
  
  const candidates = getCandidates();
  const candidate = candidates[segmentDrag.candidateIndex];
  
  if (candidate && (segmentDrag.newStart !== undefined || segmentDrag.newEnd !== undefined)) {
    const newStart = segmentDrag.newStart ?? segmentDrag.originalStart;
    const newEnd = segmentDrag.newEnd ?? segmentDrag.originalEnd;
    
    // Update the candidate in local data
    candidate.start_s = newStart;
    candidate.end_s = newEnd;
    
    // If this is the current candidate, update the builder
    if (currentCandidate && currentCandidate.rank === candidate.rank) {
      currentCandidate.start_s = newStart;
      currentCandidate.end_s = newEnd;
      setBuilder(newStart, newEnd, $('#title').value, $('#template').value);
    }
  }
  
  segmentDrag = { active: false, segment: null, handle: null, candidateIndex: -1, originalStart: 0, originalEnd: 0, startX: 0 };
  document.removeEventListener('mousemove', onSegmentDrag);
  document.removeEventListener('mouseup', onSegmentDragEnd);
  document.body.style.cursor = '';
  document.body.style.userSelect = '';
  
  updateTimelineOverlays();
}

// =========================================================================
// CLIP REGION OVERLAY
// =========================================================================

function updateClipRegion() {
  const clipRegion = $('#clipRegion');
  const showClip = $('#showClipHandles')?.checked ?? true;
  
  if (!clipRegion || !showClip || !chart || !chart.chartArea) {
    if (clipRegion) clipRegion.style.display = 'none';
    return;
  }
  
  const startS = Number($('#startS').value) || 0;
  const endS = Number($('#endS').value) || 0;
  
  if (endS <= startS) {
    clipRegion.style.display = 'none';
    return;
  }
  
  const startPx = timeToPixel(startS);
  const endPx = timeToPixel(endS);
  
  if (startPx === null || endPx === null) {
    clipRegion.style.display = 'none';
    return;
  }
  
  // Clamp to visible area
  const left = Math.max(startPx, chart.chartArea.left);
  const right = Math.min(endPx, chart.chartArea.right);
  
  if (right <= left) {
    clipRegion.style.display = 'none';
    return;
  }
  
  clipRegion.style.display = 'block';
  clipRegion.style.left = `${left}px`;
  clipRegion.style.width = `${right - left}px`;
  
  // Position handles
  const startHandle = $('#clipHandleStart');
  const endHandle = $('#clipHandleEnd');
  
  if (startHandle) {
    startHandle.style.left = `${startPx - chart.chartArea.left}px`;
    startHandle.style.display = startPx >= chart.chartArea.left ? 'block' : 'none';
  }
  
  if (endHandle) {
    endHandle.style.left = `${endPx - chart.chartArea.left - 12}px`;
    endHandle.style.display = endPx <= chart.chartArea.right ? 'block' : 'none';
  }
}

// =========================================================================
// CLIP HANDLE DRAG
// =========================================================================

function initClipHandleDrag() {
  const startHandle = $('#clipHandleStart');
  const endHandle = $('#clipHandleEnd');
  
  if (startHandle) {
    startHandle.addEventListener('mousedown', (e) => {
      e.stopPropagation();
      startClipHandleDrag(e, 'start');
    });
  }
  
  if (endHandle) {
    endHandle.addEventListener('mousedown', (e) => {
      e.stopPropagation();
      startClipHandleDrag(e, 'end');
    });
  }
}

function startClipHandleDrag(e, handle) {
  clipHandleDrag = {
    active: true,
    handle,
    startX: e.clientX,
  };
  
  document.addEventListener('mousemove', onClipHandleDrag);
  document.addEventListener('mouseup', onClipHandleDragEnd);
  document.body.style.cursor = 'ew-resize';
  document.body.style.userSelect = 'none';
}

function onClipHandleDrag(e) {
  if (!clipHandleDrag.active || !chart) return;
  
  const rect = $('#chart').getBoundingClientRect();
  const x = e.clientX - rect.left;
  const time = pixelToTime(x);
  
  if (time === null) return;
  
  const clampedTime = Math.max(0, Math.min(time, timelineData.duration));
  
  if (clipHandleDrag.handle === 'start') {
    const endS = Number($('#endS').value) || 0;
    if (clampedTime < endS - 0.5) {
      $('#startS').value = clampedTime.toFixed(2);
    }
  } else {
    const startS = Number($('#startS').value) || 0;
    if (clampedTime > startS + 0.5) {
      $('#endS').value = clampedTime.toFixed(2);
    }
  }
  
  updateClipRegion();
}

function onClipHandleDragEnd() {
  clipHandleDrag = { active: false, handle: null, startX: 0 };
  document.removeEventListener('mousemove', onClipHandleDrag);
  document.removeEventListener('mouseup', onClipHandleDragEnd);
  document.body.style.cursor = '';
  document.body.style.userSelect = '';
}

// =========================================================================
// PLAYHEAD INDICATOR
// =========================================================================

function updatePlayhead() {
  const playhead = $('#timelinePlayhead');
  if (!playhead || !chart || !chart.chartArea) {
    if (playhead) playhead.style.display = 'none';
    return;
  }
  
  const video = $('#video');
  if (!video) return;
  
  const currentTime = video.currentTime;
  const px = timeToPixel(currentTime);
  
  if (px === null || px < chart.chartArea.left || px > chart.chartArea.right) {
    playhead.style.display = 'none';
    return;
  }
  
  playhead.style.display = 'block';
  playhead.style.left = `${px}px`;
}

// =========================================================================
// ZOOM CONTROLS
// =========================================================================

function initZoomControls() {
  const btnZoomIn = $('#btnZoomIn');
  const btnZoomOut = $('#btnZoomOut');
  const btnResetZoom = $('#btnResetZoom');
  
  if (btnZoomIn) {
    btnZoomIn.onclick = () => {
      if (chart) {
        chart.zoom(1.5);
        updateTimelineOverlays();
        updateZoomInfo();
      }
    };
  }
  
  if (btnZoomOut) {
    btnZoomOut.onclick = () => {
      if (chart) {
        chart.zoom(0.67);
        updateTimelineOverlays();
        updateZoomInfo();
      }
    };
  }
  
  if (btnResetZoom) {
    btnResetZoom.onclick = () => {
      if (chart) {
        chart.resetZoom();
        updateTimelineOverlays();
        updateZoomInfo();
      }
    };
  }
  
  // Toggle checkboxes
  const showSegments = $('#showCandidateSegments');
  const showClip = $('#showClipHandles');
  
  if (showSegments) {
    showSegments.onchange = () => updateTimelineOverlays();
  }
  
  if (showClip) {
    showClip.onchange = () => updateClipRegion();
  }
}

function updateZoomInfo() {
  const zoomInfo = $('#zoomInfo');
  if (!zoomInfo || !chart || !chart.scales?.x) return;
  
  const scale = chart.scales.x;
  const visibleRange = scale.max - scale.min;
  const totalRange = timelineData.duration;
  
  if (totalRange <= 0) return;
  
  const zoomPercent = Math.round((totalRange / visibleRange) * 100);
  const viewStart = fmtTime(scale.min);
  const viewEnd = fmtTime(scale.max);
  
  if (zoomPercent <= 100) {
    zoomInfo.textContent = '100% • Drag to pan, scroll to zoom';
  } else {
    zoomInfo.textContent = `${zoomPercent}% • Viewing ${viewStart} - ${viewEnd}`;
  }
}

// =========================================================================
// UPDATE ALL TIMELINE OVERLAYS
// =========================================================================

function updateTimelineOverlays() {
  renderCandidateSegments();
  updateClipRegion();
  updatePlayhead();
}

// =========================================================================
// TIMELINE INITIALIZATION
// =========================================================================

function initInteractiveTimeline() {
  initZoomControls();
  initClipHandleDrag();
  
  // Update playhead on video timeupdate
  const video = $('#video');
  if (video) {
    video.addEventListener('timeupdate', () => {
      updatePlayhead();
    });
  }
  
  // Update clip region when builder inputs change
  const startInput = $('#startS');
  const endInput = $('#endS');
  
  if (startInput) {
    startInput.addEventListener('input', () => updateClipRegion());
    startInput.addEventListener('change', () => updateClipRegion());
  }
  
  if (endInput) {
    endInput.addEventListener('input', () => updateClipRegion());
    endInput.addEventListener('change', () => updateClipRegion());
  }
  
  // Handle window resize
  window.addEventListener('resize', () => {
    setTimeout(() => {
      if (chart) {
        timelineData.chartArea = chart.chartArea;
        updateTimelineOverlays();
      }
    }, 100);
  });
}

// =========================================================================
// END INTERACTIVE TIMELINE
// =========================================================================

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
  // Update timeline overlays after refresh
  updateTimelineOverlays();
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
  const contentType = $('#contentType')?.value || 'gaming';
  const res = await apiJson('POST', '/api/analyze/highlights', {
    highlights: {
      content_type: contentType
    }
  });
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
  const contentType = $('#contentType')?.value || 'gaming';
  const whisperBackend = $('#whisperBackend')?.value || 'auto';
  const whisperVerbose = $('#whisperVerbose')?.checked || false;
  const diarizeEnabled = $('#diarizeEnabled')?.checked || false;
  const diarizeMinSpeakers = readOptionalIntInput($('#diarizeMinSpeakers'));
  const diarizeMaxSpeakers = readOptionalIntInput($('#diarizeMaxSpeakers'));
  const traceTfImports = $('#traceTfImports')?.checked || false;
  
  const speechCfg = {
    backend: whisperBackend,
    strict: whisperBackend !== 'auto',  // Strict mode when explicitly choosing a backend
    verbose: whisperVerbose,
    diarize: diarizeEnabled
  };
  if (diarizeEnabled) {
    if (diarizeMinSpeakers != null) speechCfg.diarize_min_speakers = diarizeMinSpeakers;
    if (diarizeMaxSpeakers != null) speechCfg.diarize_max_speakers = diarizeMaxSpeakers;
  }
  const res = await apiJson('POST', '/api/analyze/full', {
    highlights: {
      motion_weight_mode: motionMode,
      content_type: contentType
    },
    speech: speechCfg,
    debug: {
      trace_tf_imports: traceTfImports
    }
  });
  const jobId = res.job_id;
  watchJob(jobId);
  $('#analysisStatus').textContent = `Full analysis running (job ${jobId.slice(0,8)})...`;
}

async function startAnalyzeSpeechJob() {
  $('#analysisStatus').textContent = 'Starting speech analysis (Whisper transcription)...';
  const whisperBackend = $('#whisperBackend')?.value || 'auto';
  const whisperVerbose = $('#whisperVerbose')?.checked || false;
  const diarizeEnabled = $('#diarizeEnabled')?.checked || false;
  const diarizeMinSpeakers = readOptionalIntInput($('#diarizeMinSpeakers'));
  const diarizeMaxSpeakers = readOptionalIntInput($('#diarizeMaxSpeakers'));
  
  const speechCfg = {
    backend: whisperBackend,
    strict: whisperBackend !== 'auto',  // Strict mode when explicitly choosing a backend
    verbose: whisperVerbose,
    diarize: diarizeEnabled
  };
  if (diarizeEnabled) {
    if (diarizeMinSpeakers != null) speechCfg.diarize_min_speakers = diarizeMinSpeakers;
    if (diarizeMaxSpeakers != null) speechCfg.diarize_max_speakers = diarizeMaxSpeakers;
  }
  const res = await apiJson('POST', '/api/analyze/speech', {
    speech: speechCfg
  });
  const jobId = res.job_id;
  watchJob(jobId);
  $('#analysisStatus').textContent = `Speech analysis running (job ${jobId.slice(0,8)})...`;
}

async function startAnalyzeContextJob() {
  $('#analysisStatus').textContent = 'Starting context + titles analysis (AI)...';
  const traceTfImports = $('#traceTfImports')?.checked || false;
  const res = await apiJson('POST', '/api/analyze/context_titles', {
    debug: {
      trace_tf_imports: traceTfImports
    }
  });
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

  const diarizeCb = $('#diarizeEnabled');
  if (diarizeCb) diarizeCb.onchange = updateDiarizationControls;
  updateDiarizationControls();

  v.addEventListener('timeupdate', () => {
    $('#timeReadout').textContent = `Now: ${fmtTime(v.currentTime)} / ${fmtTime(v.duration || 0)}`;
  });

  $('#btnSetStart').onclick = () => {
    const b = getBuilder();
    setBuilder(v.currentTime, b.end_s, b.title, b.template);
    updateClipRegion();
  };

  $('#btnSetEnd').onclick = () => {
    const b = getBuilder();
    setBuilder(b.start_s, v.currentTime, b.title, b.template);
    updateClipRegion();
  };

  $('#btnSnapCandidate').onclick = () => {
    if (!currentCandidate) return;
    setBuilder(currentCandidate.start_s, currentCandidate.end_s, $('#title').value, $('#template').value);
    updateClipRegion();
  };

  $('#btnNudgeBack').onclick = () => {
    const b = getBuilder();
    setBuilder(b.start_s - 0.5, b.end_s - 0.5, b.title, b.template);
    updateClipRegion();
  };

  $('#btnNudgeFwd').onclick = () => {
    const b = getBuilder();
    setBuilder(b.start_s + 0.5, b.end_s + 0.5, b.title, b.template);
    updateClipRegion();
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
    const reset = await showResetAnalysisModal();
    if (!reset) return;

    const resetKeys = Object.entries(reset).filter(([, v]) => v).map(([k]) => k);
    if (resetKeys.length === 0) return;

    try {
      const res = await apiJson('POST', '/api/project/reset_analysis', {
        reset,
      });
      const deleted = Array.isArray(res.deleted_files) ? res.deleted_files.length : 0;
      alert(`Analysis reset complete!\n\nDeleted ${deleted} files.\nReset: ${resetKeys.join(', ')}`);
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

  publishJobsSSE = new EventSource(apiUrlWithToken('/api/publisher/jobs/stream'));

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
  const timelinePanel = $('#chatTimelinePanel');
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

  // Get references to new UI elements
  const miniTimelineWrap = $('#chatMiniTimelineWrap');
  const filterControls = $('#chatFilterControls');

  if (!chatStatus || !chatStatus.available) {
    if (timelinePanel) timelinePanel.style.display = 'none';
    statusEl.innerHTML = 'No chat replay available.' + aiStatusHtml;
    statusEl.className = 'small';
    if (messagesEl) messagesEl.innerHTML = '';
    if (offsetControls) offsetControls.style.display = 'none';
    if (downloadBtn) downloadBtn.disabled = false;
    if (clearBtn) clearBtn.style.display = 'none';
    if (miniTimelineWrap) miniTimelineWrap.style.display = 'none';
    if (filterControls) filterControls.style.display = 'none';

    // Pre-fill source URL from project if available
    if (sourceUrlInput && chatStatus?.source_url) {
      sourceUrlInput.value = chatStatus.source_url;
    }
    // Wire up re-learn button if present
    wireRelearnButton();
    return;
  }

  if (timelinePanel) timelinePanel.style.display = 'block';
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
  
  // Show filter controls
  if (filterControls) filterControls.style.display = 'flex';
  
  // Load and show chat mini-timeline
  loadChatTimeline();

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
    updateChatFilterBadge(0);
    return;
  }

  // Apply filtering
  let filteredMessages = messages;
  let matchCount = 0;
  
  if (chatFilterState.preset) {
    filteredMessages = messages.map(m => {
      const matches = matchesChatFilter(m);
      if (matches) matchCount++;
      return { ...m, _matches: matches };
    });
    
    // If highlight only is off, filter to only matching messages
    if (chatFilterState.highlightOnly) {
      // Keep all messages but mark matches
    } else if (chatFilterState.preset) {
      filteredMessages = filteredMessages.filter(m => m._matches);
    }
  }
  
  updateChatFilterBadge(matchCount);

  if (filteredMessages.length === 0) {
    messagesEl.innerHTML = '<div class="small" style="opacity:0.5">No messages match the filter</div>';
    return;
  }

  const html = filteredMessages.map(m => {
    const offsetMs = chatStatus?.sync_offset_ms || 0;
    const msgTimeMs = m.t_ms - offsetMs;
    const timeSec = msgTimeMs / 1000;
    const isNear = Math.abs(msgTimeMs - currentTimeMs) < 2000;
    const nearClass = isNear ? 'chat-msg-near' : '';
    const matchClass = m._matches ? 'filter-match' : '';
    const timeStr = fmtTime(timeSec);
    
    // Highlight filter matches in text
    const displayText = chatFilterState.preset ? highlightFilterMatches(m.text || '') : escapeHtml(m.text || '');
    
    return `
      <div class="chat-msg ${nearClass} ${matchClass}" data-time="${timeSec}">
        <span class="chat-time" title="Click to seek">${timeStr}</span>
        <span class="chat-author">${escapeHtml(m.author || 'anon')}</span>
        <span class="chat-text">${displayText}</span>
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

// =========================================================================
// CHAT MINI-TIMELINE
// =========================================================================

let chatTimelineData = null;
let chatTimelineOverviewData = null;
let chatTimelineView = { startS: 0, endS: null };
let chatTimelineRequestId = 0;
let chatTimelineDebounce = null;
let chatTimelineOverviewRequestId = 0;
let chatMiniTimelineInitialized = false;
let chatMiniTimelineDrag = {
  active: false,
  startX: 0,
  startStartS: 0,
  startEndS: 0,
  moved: false,
};

function isChatTimelineData(obj) {
  return (
    obj &&
    Array.isArray(obj.scores) &&
    Number.isFinite(Number(obj.hopSeconds)) &&
    Number.isFinite(Number(obj.startS ?? 0))
  );
}

function chatTimelineDataCoversWindow(data, startS, endS) {
  if (!isChatTimelineData(data)) return false;
  const dataStart = Number(data.startS ?? 0);
  const hop = Number(data.hopSeconds) || 0;
  const n = Array.isArray(data.scores) ? data.scores.length : 0;
  const computedEnd = (hop > 0 && n > 0) ? (dataStart + (n - 1) * hop) : dataStart;
  const dataEnd = Number.isFinite(Number(data.endS)) ? Number(data.endS) : computedEnd;
  return startS >= dataStart - 1e-6 && endS <= dataEnd + 1e-6;
}

function getChatTimelineDurationSeconds() {
  const video = $('#video');
  const d = (video && isFinite(video.duration) && video.duration > 0) ? video.duration : 0;
  return d > 0 ? d : (chatTimelineData?.durationSeconds || chatTimelineOverviewData?.durationSeconds || 0);
}

function getChatTimelineViewWindow() {
  const durationS = getChatTimelineDurationSeconds();
  let startS = Number(chatTimelineView?.startS ?? 0);
  let endS = chatTimelineView?.endS;
  if (endS == null || !isFinite(endS)) {
    endS = durationS > 0 ? durationS : (startS + 60.0);
  } else {
    endS = Number(endS);
  }
  if (durationS > 0) {
    startS = Math.max(0, Math.min(startS, durationS));
    endS = Math.max(startS + 0.1, Math.min(endS, durationS));
  } else {
    endS = Math.max(startS + 0.1, endS);
  }
  return { startS, endS, durationS };
}

function resetChatTimelineView() {
  chatTimelineView.startS = 0;
  const d = getChatTimelineDurationSeconds();
  chatTimelineView.endS = d > 0 ? d : null;
}

function scheduleChatTimelineFetch(delayMs = 120) {
  if (chatTimelineDebounce) clearTimeout(chatTimelineDebounce);
  chatTimelineDebounce = setTimeout(() => loadChatTimeline(), delayMs);
}

async function loadChatTimelineOverview() {
  try {
    const params = new URLSearchParams();
    params.set('start_s', '0');
    params.set('max_points', '2000');

    const reqId = ++chatTimelineOverviewRequestId;
    const res = await apiGet(`/api/chat/timeline?${params.toString()}`);
    if (reqId !== chatTimelineOverviewRequestId) return;
    if (!res.ok) return;

    chatTimelineOverviewData = {
      hopSeconds: res.hop_seconds,
      indices: res.indices,
      scores: res.scores,
      durationSeconds: res.duration_seconds || 0,
      startS: res.start_s ?? 0,
      endS: res.end_s ?? null,
      syncOffsetMs: res.sync_offset_ms || 0,
    };
    renderChatMiniTimeline();
    updateChatTimelinePlayhead();
  } catch (e) {
    // Ignore; overview is only an optimization for smoother panning previews.
  }
}

function maybeRefreshChatTimelineOverview() {
  if (!isChatTimelineData(chatTimelineData)) return;
  const curOffset = Number(chatTimelineData.syncOffsetMs || 0);
  const curDur = Number(chatTimelineData.durationSeconds || 0);
  const hasOverview = isChatTimelineData(chatTimelineOverviewData);
  const overviewOffset = hasOverview ? Number(chatTimelineOverviewData.syncOffsetMs || 0) : null;
  const overviewDur = hasOverview ? Number(chatTimelineOverviewData.durationSeconds || 0) : null;

  const needsRefresh =
    !hasOverview ||
    (overviewOffset != null && overviewOffset !== curOffset) ||
    (overviewDur != null && curDur > 0 && Math.abs(overviewDur - curDur) > 0.25);

  if (!needsRefresh) return;
  if (hasOverview && overviewOffset != null && overviewOffset !== curOffset) {
    chatTimelineOverviewData = null;
  }
  loadChatTimelineOverview();
}

async function loadChatTimeline() {
  const wrap = $('#chatMiniTimelineWrap');
  try {
    const { startS, endS } = getChatTimelineViewWindow();
    const params = new URLSearchParams();
    params.set('start_s', startS.toFixed(3));
    // Only send end_s once user has zoomed/panned; otherwise let backend pick full duration.
    if (chatTimelineView?.endS != null) {
      params.set('end_s', endS.toFixed(3));
    }
    params.set('max_points', '2000');

    const reqId = ++chatTimelineRequestId;
    const res = await apiGet(`/api/chat/timeline?${params.toString()}`);
    if (reqId !== chatTimelineRequestId) return;

    if (res.ok) {
      chatTimelineData = {
        hopSeconds: res.hop_seconds,
        indices: res.indices,
        scores: res.scores,
        durationSeconds: res.duration_seconds || 0,
        startS: res.start_s ?? 0,
        endS: res.end_s ?? null,
        syncOffsetMs: res.sync_offset_ms || 0,
      };
      const durationS = Number(res.duration_seconds || 0);
      const isFullDuration =
        typeof res.start_s === 'number' &&
        Math.abs(res.start_s) < 1e-6 &&
        typeof res.end_s === 'number' &&
        durationS > 0 &&
        Math.abs(res.end_s - durationS) < 1e-3;
      if (isFullDuration) chatTimelineOverviewData = { ...chatTimelineData };
      // Adopt the server-clamped view window.
      if (typeof res.start_s === 'number') chatTimelineView.startS = res.start_s;
      if (typeof res.end_s === 'number') chatTimelineView.endS = res.end_s;
      if (wrap) wrap.style.display = 'block';
      renderChatMiniTimeline();
      updateChatTimelinePlayhead();
    } else {
      chatTimelineData = null;
      if (wrap) wrap.style.display = 'none';
    }
  } catch (e) {
    // No chat timeline available
    chatTimelineData = null;
    if (wrap) wrap.style.display = 'none';
  }
}

function getChatTimelineRenderData() {
  const { startS, endS } = getChatTimelineViewWindow();
  if (chatTimelineDataCoversWindow(chatTimelineData, startS, endS)) return chatTimelineData;
  if (chatTimelineDataCoversWindow(chatTimelineOverviewData, startS, endS)) return chatTimelineOverviewData;
  return isChatTimelineData(chatTimelineData) ? chatTimelineData : chatTimelineOverviewData;
}

function getChatTimelineScoresForView(data, viewStartS, viewEndS) {
  if (!isChatTimelineData(data)) return [];
  const scores = data.scores || [];
  if (!scores.length) return [];

  const hop = Number(data.hopSeconds) || 0;
  if (!(hop > 0)) return scores;

  const dataStart = Number(data.startS ?? 0);
  let i0 = Math.floor((viewStartS - dataStart) / hop);
  let i1 = Math.floor((viewEndS - dataStart) / hop) + 1;

  if (!Number.isFinite(i0)) i0 = 0;
  if (!Number.isFinite(i1)) i1 = scores.length;

  i0 = Math.max(0, Math.min(i0, scores.length - 1));
  i1 = Math.max(i0 + 1, Math.min(i1, scores.length));
  return scores.slice(i0, i1);
}

function renderChatMiniTimeline(data) {
  const canvas = $('#chatMiniTimeline');
  const wrap = $('#chatMiniTimelineWrap');
  if (!canvas || !wrap) return;

  const renderData = isChatTimelineData(data) ? data : getChatTimelineRenderData();
  if (!isChatTimelineData(renderData)) return;
  
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  
  const width = wrap.clientWidth;
  const height = wrap.clientHeight;
  if (!width || !height) return;

  // Set canvas size
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  
  // Clear
  ctx.fillStyle = '#0a0c0f';
  ctx.fillRect(0, 0, width, height);
  
  const { startS, endS } = getChatTimelineViewWindow();
  const scores = chatTimelineDataCoversWindow(renderData, startS, endS) ?
    getChatTimelineScoresForView(renderData, startS, endS) :
    (renderData.scores || []);
  if (scores.length === 0) return;
  
  let maxScore = 0.01;
  for (let i = 0; i < scores.length; i++) {
    const v = Number(scores[i]);
    if (Number.isFinite(v) && v > maxScore) maxScore = v;
  }
  const barWidth = Math.max(1, width / scores.length);
  
  // Draw bars
  scores.forEach((score, i) => {
    const normalizedScore = score / maxScore;
    const barHeight = normalizedScore * (height - 4);
    const x = (i / scores.length) * width;
    
    // Color gradient based on intensity
    const intensity = Math.min(1, normalizedScore * 1.5);
    const r = Math.floor(34 + intensity * (234 - 34));
    const g = Math.floor(197 + intensity * (179 - 197));
    const b = Math.floor(94 + intensity * (8 - 94));
    
    ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${0.4 + intensity * 0.5})`;
    ctx.fillRect(x, height - barHeight - 2, barWidth + 0.5, barHeight);
  });
}

function updateChatTimelinePlayhead() {
  const playhead = $('#chatTimelinePlayhead');
  const wrap = $('#chatMiniTimelineWrap');
  const video = $('#video');
  
  if (!playhead || !wrap || !video || !chatTimelineData) return;

  const { startS, endS } = getChatTimelineViewWindow();
  const span = endS - startS;
  if (span <= 0) return;

  const x = ((video.currentTime - startS) / span) * wrap.clientWidth;
  const clamped = Math.max(0, Math.min(wrap.clientWidth, x));

  playhead.style.left = `${clamped}px`;
  playhead.style.opacity = (video.currentTime < startS || video.currentTime > endS) ? '0.35' : '1.0';
}

function initChatMiniTimeline() {
  const wrap = $('#chatMiniTimelineWrap');
  const video = $('#video');

  if (chatMiniTimelineInitialized) return;
  chatMiniTimelineInitialized = true;
  
  if (wrap) {
    // Zoom with wheel, centered around cursor.
    wrap.addEventListener('wheel', (e) => {
      if (!chatTimelineData) return;
      const d = getChatTimelineDurationSeconds();
      if (!d || d <= 0) return;

      e.preventDefault();

      const rect = wrap.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const frac = rect.width > 0 ? (x / rect.width) : 0.5;

      const { startS, endS } = getChatTimelineViewWindow();
      const span = Math.max(0.1, endS - startS);
      const center = startS + frac * span;

      // Smooth exponential zoom.
      const zoom = Math.exp(e.deltaY * 0.0015); // deltaY>0 => zoom out
      const minSpan = 5.0;
      const maxSpan = Math.max(minSpan, d);
      let newSpan = span * zoom;
      newSpan = Math.max(minSpan, Math.min(maxSpan, newSpan));

      let newStart = center - frac * newSpan;
      let newEnd = newStart + newSpan;

      // Clamp to [0, d]
      if (newStart < 0) {
        newEnd -= newStart;
        newStart = 0;
      }
      if (newEnd > d) {
        const over = newEnd - d;
        newStart = Math.max(0, newStart - over);
        newEnd = d;
      }

      chatTimelineView.startS = newStart;
      chatTimelineView.endS = newEnd;
      scheduleChatTimelineFetch(80);
      updateChatTimelinePlayhead();
      renderChatMiniTimeline();
    }, { passive: false });

    // Pan with drag.
    wrap.addEventListener('mousedown', (e) => {
      if (!chatTimelineData) return;
      if (e.button !== 0) return; // left mouse only
      e.preventDefault();
      wrap.classList.add('dragging');
      maybeRefreshChatTimelineOverview();
      const { startS, endS } = getChatTimelineViewWindow();
      chatMiniTimelineDrag.active = true;
      chatMiniTimelineDrag.startX = e.clientX;
      chatMiniTimelineDrag.startStartS = startS;
      chatMiniTimelineDrag.startEndS = endS;
      chatMiniTimelineDrag.moved = false;
    });

    window.addEventListener('mousemove', (e) => {
      if (!chatMiniTimelineDrag.active) return;
      const d = getChatTimelineDurationSeconds();
      if (!d || d <= 0) return;
      const rect = wrap.getBoundingClientRect();
      if (!rect.width) return;

      const dx = e.clientX - chatMiniTimelineDrag.startX;
      if (Math.abs(dx) > 3) chatMiniTimelineDrag.moved = true;

      const span = chatMiniTimelineDrag.startEndS - chatMiniTimelineDrag.startStartS;
      const dt = -(dx / rect.width) * span;

      let newStart = chatMiniTimelineDrag.startStartS + dt;
      let newEnd = chatMiniTimelineDrag.startEndS + dt;

      if (newStart < 0) {
        newEnd -= newStart;
        newStart = 0;
      }
      if (newEnd > d) {
        const over = newEnd - d;
        newStart = Math.max(0, newStart - over);
        newEnd = d;
      }

      chatTimelineView.startS = newStart;
      chatTimelineView.endS = newEnd;
      renderChatMiniTimeline();
      updateChatTimelinePlayhead();
    });

    window.addEventListener('mouseup', () => {
      if (!chatMiniTimelineDrag.active) return;
      chatMiniTimelineDrag.active = false;
      wrap.classList.remove('dragging');
      if (chatMiniTimelineDrag.moved) {
        scheduleChatTimelineFetch(0);
      }
    });

    // Click-to-seek (in the current view window).
    wrap.addEventListener('click', (e) => {
      if (!video || !chatTimelineData) return;
      if (chatMiniTimelineDrag.moved) {
        chatMiniTimelineDrag.moved = false;
        return;
      }
      const rect = wrap.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const frac = rect.width > 0 ? (x / rect.width) : 0;
      const { startS, endS, durationS } = getChatTimelineViewWindow();
      const t = startS + frac * (endS - startS);
      const clamped = durationS > 0 ? Math.max(0, Math.min(durationS, t)) : Math.max(0, t);
      video.currentTime = clamped;
    });

    // Reset zoom on double-click.
    wrap.addEventListener('dblclick', () => {
      resetChatTimelineView();
      loadChatTimeline();
    });
  }
  
  if (video) {
    video.addEventListener('timeupdate', updateChatTimelinePlayhead);
  }
  
  // Handle resize
  window.addEventListener('resize', renderChatMiniTimeline);
}

// =========================================================================
// CHAT FILTERING
// =========================================================================

const CHAT_FILTER_PRESETS = {
  laughter: ['lol', 'lmao', 'lmfao', 'rofl', 'haha', 'hahaha', 'lul', 'kekw', 'omegalul', '😂', '🤣', 'xd'],
  excitement: ['pog', 'poggers', 'pogchamp', 'lets go', 'let\'s go', 'hype', 'yooo', 'omg', 'wow', '!', '🔥', '🎉', 'w', 'dub'],
  questions: ['?', 'what', 'how', 'why', 'when', 'where', 'who'],
  emotes: [],  // Will match any emote pattern
};

let chatFilterState = {
  preset: '',
  keywords: [],
  highlightOnly: false,
  matchCount: 0,
};

function applyChatFilter() {
  const preset = $('#chatFilterPreset')?.value || '';
  const customKeywords = $('#chatFilterKeywords')?.value || '';
  const highlightOnly = $('#chatFilterHighlightOnly')?.checked || false;
  
  chatFilterState.preset = preset;
  chatFilterState.highlightOnly = highlightOnly;
  
  if (preset === 'custom' && customKeywords.trim()) {
    chatFilterState.keywords = customKeywords.split(',').map(k => k.trim().toLowerCase()).filter(k => k);
  } else if (preset && CHAT_FILTER_PRESETS[preset]) {
    chatFilterState.keywords = CHAT_FILTER_PRESETS[preset];
  } else {
    chatFilterState.keywords = [];
  }
  
  // Show/hide custom input
  const customWrap = $('#chatCustomFilterWrap');
  if (customWrap) {
    customWrap.style.display = preset === 'custom' ? 'flex' : 'none';
  }
  
  // Trigger re-render
  syncChatMessages();
}

function matchesChatFilter(message) {
  if (!chatFilterState.preset || chatFilterState.keywords.length === 0) {
    if (chatFilterState.preset === 'emotes') {
      // Match emote patterns (words starting with : or containing only caps)
      const text = message.text || '';
      return /:[a-zA-Z0-9_]+:|^[A-Z]{2,}$/.test(text) || text.includes('KEKW') || text.includes('POG');
    }
    return false;
  }
  
  const text = (message.text || '').toLowerCase();
  return chatFilterState.keywords.some(kw => text.includes(kw));
}

function highlightFilterMatches(text) {
  if (!chatFilterState.preset || chatFilterState.keywords.length === 0) {
    return escapeHtml(text);
  }
  
  let result = escapeHtml(text);
  
  // Highlight matching keywords
  chatFilterState.keywords.forEach(kw => {
    const regex = new RegExp(`(${escapeRegex(kw)})`, 'gi');
    result = result.replace(regex, '<span class="keyword-highlight">$1</span>');
  });
  
  return result;
}

function escapeRegex(str) {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function initChatFilters() {
  const presetSelect = $('#chatFilterPreset');
  const keywordsInput = $('#chatFilterKeywords');
  const highlightOnlyChk = $('#chatFilterHighlightOnly');
  const filterBadge = $('#chatFilterCount');
  
  if (presetSelect) {
    presetSelect.onchange = applyChatFilter;
  }
  
  if (keywordsInput) {
    keywordsInput.onchange = applyChatFilter;
    keywordsInput.onkeydown = (e) => {
      if (e.key === 'Enter') applyChatFilter();
    };
  }
  
  if (highlightOnlyChk) {
    highlightOnlyChk.onchange = applyChatFilter;
  }
  
  // Clear filter button in badge
  if (filterBadge) {
    const clearBtn = filterBadge.querySelector('.remove-filter');
    if (clearBtn) {
      clearBtn.onclick = () => {
        if (presetSelect) presetSelect.value = '';
        if (keywordsInput) keywordsInput.value = '';
        if (highlightOnlyChk) highlightOnlyChk.checked = false;
        applyChatFilter();
      };
    }
  }
}

function updateChatFilterBadge(matchCount) {
  const badge = $('#chatFilterCount');
  if (!badge) return;
  
  chatFilterState.matchCount = matchCount;
  
  if (chatFilterState.preset && matchCount >= 0) {
    badge.style.display = 'inline-flex';
    badge.querySelector('.count').textContent = matchCount;
  } else {
    badge.style.display = 'none';
  }
}

// =========================================================================
// AI SUGGESTIONS PANEL
// =========================================================================

let aiSuggestionsState = {
  candidateRank: null,
  titles: [],
  hooks: [],
  hashtags: [],
  selectedTitleIdx: 0,
  selectedHookIdx: 0,
  selectedHashtags: new Set(),
  model: '',
};

function showAISuggestionsPanel(candidateRank) {
  const highlights = project?.analysis?.highlights;
  const audio = project?.analysis?.audio;
  const candidates = highlights?.candidates || audio?.candidates || [];
  const cand = candidates.find(c => c.rank === candidateRank);
  const ai = cand?.ai;
  if (!ai) {
    hideAISuggestionsPanel();
    return;
  }
  
  const panel = $('#aiSuggestionsPanel');
  if (!panel) return;
  
  // Build multiple title/hook options
  // Primary suggestion from AI director
  const titles = [ai.title].filter(Boolean);
  const hooks = [ai.hook].filter(Boolean);
  
  // Add variations based on the original
  if (ai.title) {
    // Generate some variations
    titles.push(ai.title.toUpperCase());
    if (ai.title.length > 30) {
      titles.push(ai.title.slice(0, 30) + '...');
    }
    // Add a question variant if it's not already a question
    if (!ai.title.endsWith('?') && !ai.title.includes('?')) {
      titles.push('Did this really happen? ' + ai.title);
    }
  }
  
  if (ai.hook) {
    hooks.push(ai.hook.toUpperCase());
    hooks.push('🔥 ' + ai.hook);
    hooks.push(ai.hook + ' 👀');
  }
  
  // Add hashtags
  const hashtags = ai.hashtags || ai.tags || [];
  const modelName = project?.analysis?.ai_director?.config?.model_name || project?.analysis?.ai_director?.config?.model || '';
  
  aiSuggestionsState = {
    candidateRank,
    titles: [...new Set(titles)].filter(t => t), // Remove duplicates and empty
    hooks: [...new Set(hooks)].filter(h => h),
    hashtags,
    selectedTitleIdx: 0,
    selectedHookIdx: 0,
    selectedHashtags: new Set(hashtags.slice(0, 5)),
    model: modelName || 'AI',
  };
  
  renderAISuggestionsPanel();
  panel.style.display = 'block';
}

function hideAISuggestionsPanel() {
  const panel = $('#aiSuggestionsPanel');
  if (panel) panel.style.display = 'none';
}

function renderAISuggestionsPanel() {
  const titleOptions = $('#aiTitleOptions');
  const hookOptions = $('#aiHookOptions');
  const hashtagsContainer = $('#aiHashtags');
  const modelBadge = $('#aiModelBadge');
  
  if (modelBadge) {
    modelBadge.textContent = aiSuggestionsState.model;
  }
  
  // Render title options
  if (titleOptions) {
    titleOptions.innerHTML = aiSuggestionsState.titles.map((title, idx) => `
      <div class="ai-suggestion-option ${idx === aiSuggestionsState.selectedTitleIdx ? 'selected' : ''}" data-idx="${idx}" data-type="title">
        <input type="radio" name="aiTitle" ${idx === aiSuggestionsState.selectedTitleIdx ? 'checked' : ''} />
        <input type="text" class="ai-suggestion-text editable" value="${escapeHtml(title)}" data-idx="${idx}" />
      </div>
    `).join('');
    
    // Wire up events
    titleOptions.querySelectorAll('.ai-suggestion-option').forEach(opt => {
      opt.onclick = (e) => {
        if (e.target.classList.contains('editable')) return;
        const idx = parseInt(opt.dataset.idx);
        aiSuggestionsState.selectedTitleIdx = idx;
        renderAISuggestionsPanel();
      };
    });
    
    titleOptions.querySelectorAll('.editable').forEach(input => {
      input.oninput = (e) => {
        const idx = parseInt(input.dataset.idx);
        aiSuggestionsState.titles[idx] = input.value;
      };
      input.onfocus = () => {
        const idx = parseInt(input.dataset.idx);
        aiSuggestionsState.selectedTitleIdx = idx;
        renderAISuggestionsPanel();
      };
    });
  }
  
  // Render hook options
  if (hookOptions) {
    hookOptions.innerHTML = aiSuggestionsState.hooks.map((hook, idx) => `
      <div class="ai-suggestion-option ${idx === aiSuggestionsState.selectedHookIdx ? 'selected' : ''}" data-idx="${idx}" data-type="hook">
        <input type="radio" name="aiHook" ${idx === aiSuggestionsState.selectedHookIdx ? 'checked' : ''} />
        <input type="text" class="ai-suggestion-text editable" value="${escapeHtml(hook)}" data-idx="${idx}" />
      </div>
    `).join('');
    
    hookOptions.querySelectorAll('.ai-suggestion-option').forEach(opt => {
      opt.onclick = (e) => {
        if (e.target.classList.contains('editable')) return;
        const idx = parseInt(opt.dataset.idx);
        aiSuggestionsState.selectedHookIdx = idx;
        renderAISuggestionsPanel();
      };
    });
    
    hookOptions.querySelectorAll('.editable').forEach(input => {
      input.oninput = (e) => {
        const idx = parseInt(input.dataset.idx);
        aiSuggestionsState.hooks[idx] = input.value;
      };
      input.onfocus = () => {
        const idx = parseInt(input.dataset.idx);
        aiSuggestionsState.selectedHookIdx = idx;
        renderAISuggestionsPanel();
      };
    });
  }
  
  // Render hashtags
  if (hashtagsContainer) {
    hashtagsContainer.innerHTML = aiSuggestionsState.hashtags.map((tag, idx) => `
      <div class="ai-tag-chip ${aiSuggestionsState.selectedHashtags.has(tag) ? 'selected' : ''}" data-tag="${escapeHtml(tag)}">
        #${escapeHtml(tag)}
      </div>
    `).join('');
    
    hashtagsContainer.querySelectorAll('.ai-tag-chip').forEach(chip => {
      chip.onclick = () => {
        const tag = chip.dataset.tag;
        if (aiSuggestionsState.selectedHashtags.has(tag)) {
          aiSuggestionsState.selectedHashtags.delete(tag);
        } else {
          aiSuggestionsState.selectedHashtags.add(tag);
        }
        chip.classList.toggle('selected');
      };
    });
  }
}

function applyAISuggestions() {
  const selectedTitle = aiSuggestionsState.titles[aiSuggestionsState.selectedTitleIdx] || '';
  const selectedHook = aiSuggestionsState.hooks[aiSuggestionsState.selectedHookIdx] || '';
  const selectedTags = Array.from(aiSuggestionsState.selectedHashtags);
  
  // Apply title to clip builder
  const titleInput = $('#title');
  if (titleInput && selectedTitle) {
    // Combine title with hashtags
    const tagsStr = selectedTags.length > 0 ? ' ' + selectedTags.map(t => '#' + t).join(' ') : '';
    titleInput.value = selectedTitle + tagsStr;
  }
  
  // Store hook for export (could add to notes field or a dedicated hook field)
  // For now, we'll show a confirmation
  const message = `Applied:\n• Title: ${selectedTitle}\n• Hook: ${selectedHook}\n• Tags: ${selectedTags.join(', ')}`;
  console.log('AI Suggestions applied:', { title: selectedTitle, hook: selectedHook, tags: selectedTags });
}

function initAISuggestionsPanel() {
  const applyBtn = $('#btnApplyAISuggestions');
  const regenBtn = $('#btnRegenerateAI');
  
  if (applyBtn) {
    applyBtn.onclick = applyAISuggestions;
  }
  
  if (regenBtn) {
    regenBtn.onclick = async () => {
      if (!aiSuggestionsState.candidateRank) return;
      
      // Show loading state
      regenBtn.disabled = true;
      regenBtn.textContent = '⏳ Generating...';
      
      try {
        // Could call an API to regenerate suggestions
        // For now, just shuffle existing options
        aiSuggestionsState.titles = aiSuggestionsState.titles.sort(() => Math.random() - 0.5);
        aiSuggestionsState.hooks = aiSuggestionsState.hooks.sort(() => Math.random() - 0.5);
        renderAISuggestionsPanel();
      } finally {
        regenBtn.disabled = false;
        regenBtn.textContent = '↻ Regenerate';
      }
    };
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
    if (chatTimelineData) chatTimelineData.syncOffsetMs = offset;
    syncChatMessages(); // Refresh immediately
    // Refresh timeline so spikes align with video time.
    scheduleChatTimelineFetch(0);
    // Refresh overview used for smooth panning previews.
    chatTimelineOverviewData = null;
    loadChatTimelineOverview();
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

  // Initialize chat mini-timeline
  initChatMiniTimeline();
  
  // Initialize chat filters
  initChatFilters();
  
  // Initialize AI suggestions panel
  initAISuggestionsPanel();
}

async function main() {
  initApiTokenUi();
  try {
    profile = await apiGet('/api/profile');
  } catch (_) {}
  applyAnalysisDefaultsFromProfile();

  // Initialize collapsible panels
  initCollapsiblePanels();
  initAnalysisButtonToggles();

  // Wire up home UI first
  wireHomeUI();
  wireTabUI();
  wirePublishUI();
  initTraceTfImportsToggle();

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
  // Reset chat timeline state when switching projects
  chatTimelineData = null;
  chatTimelineOverviewData = null;
  resetChatTimelineView();

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
  
  // Initialize interactive timeline features (zoom, pan, segment overlays, clip handles)
  initInteractiveTimeline();
  
  // Update whisper backend dropdown with availability info
  updateWhisperBackendOptions();
  
  // GPU memory status
  updateGpuMemoryStatus();
  const btnClearGpu = $('#btnClearGpuMemory');
  if (btnClearGpu) {
    btnClearGpu.onclick = clearGpuMemory;
  }
}

async function updateGpuMemoryStatus() {
  const statusEl = $('#gpuMemoryStatus');
  if (!statusEl) return;
  
  try {
    // Fetch both GPU and LLM status in parallel
    const [gpuData, llmData] = await Promise.all([
      apiGet('/api/system/gpu'),
      apiGet('/api/system/llm').catch(() => null)
    ]);
    
    let lines = [];
    
    // GPU info
    if (!gpuData.available) {
      lines.push('No GPU available');
    } else {
      const mem = gpuData.memory;
      if (mem) {
        const allocated = mem.allocated_gb.toFixed(2);
        const reserved = mem.reserved_gb.toFixed(2);
        lines.push(`${gpuData.device_name}`);
        lines.push(`PyTorch: ${allocated} GB allocated`);
      } else {
        lines.push(gpuData.device_name || 'GPU available');
      }
    }
    
    // LLM server info
    if (llmData && llmData.running) {
      const model = llmData.model || 'unknown';
      const mem = llmData.memory_mb ? ` (~${(llmData.memory_mb / 1024).toFixed(1)} GB)` : '';
      lines.push(`LLM: ${model}${mem}`);
    }
    
    statusEl.innerHTML = lines.join('<br>');
    
    // Color based on memory usage
    const hasHighUsage = (gpuData.memory?.reserved_gb > 1) || (llmData?.running);
    statusEl.style.color = hasHighUsage ? '#fbbf24' : '#4ade80';
  } catch (e) {
    statusEl.textContent = 'Error checking GPU';
    statusEl.style.color = '#ef4444';
  }
}

async function clearGpuMemory() {
  const btn = $('#btnClearGpuMemory');
  if (!btn) return;
  
  const origText = btn.textContent;
  btn.textContent = 'Clearing...';
  btn.disabled = true;
  
  let messages = [];
  
  try {
    // First, stop the LLM server if running
    try {
      const llmResult = await apiJson('POST', '/api/system/llm/stop');
      if (llmResult.was_running && llmResult.success) {
        messages.push('LLM stopped');
        // Give it a moment to release memory
        await new Promise(r => setTimeout(r, 500));
      }
    } catch (e) {
      // LLM stop failed, continue anyway
    }
    
    // Then clear PyTorch GPU cache
    const data = await apiJson('POST', '/api/system/gpu/clear');
    if (data.success) {
      const freed = data.freed_gb || 0;
      if (freed > 0.01) {
        messages.push(`${freed.toFixed(2)} GB freed`);
      }
    }
    
    if (messages.length > 0) {
      btn.textContent = `✓ ${messages.join(', ')}`;
    } else {
      btn.textContent = '✓ Cleared';
    }
    
    setTimeout(() => {
      btn.textContent = origText;
      btn.disabled = false;
      updateGpuMemoryStatus();
    }, 2500);
  } catch (e) {
    btn.textContent = 'Error';
    setTimeout(() => {
      btn.textContent = origText;
      btn.disabled = false;
    }, 2000);
  }
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
