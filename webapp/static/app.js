const state = {
  result: null,
  selectedFile: null,
  waveform: [],
  animationId: null,
  recorder: null,
};

const els = {
  form: document.getElementById("uploadForm"),
  audioInput: document.getElementById("audioInput"),
  titleInput: document.getElementById("titleInput"),
  fileName: document.getElementById("fileName"),
  dropZone: document.getElementById("dropZone"),
  analyzeBtn: document.getElementById("analyzeBtn"),
  statusLine: document.getElementById("statusLine"),
  exampleSelect: document.getElementById("exampleSelect"),
  loadExampleBtn: document.getElementById("loadExampleBtn"),
  durationStat: document.getElementById("durationStat"),
  tempoStat: document.getElementById("tempoStat"),
  barsStat: document.getElementById("barsStat"),
  actionsStat: document.getElementById("actionsStat"),
  pipelineBadge: document.getElementById("pipelineBadge"),
  methodLine: document.getElementById("methodLine"),
  processSteps: document.getElementById("processSteps"),
  exportJsonBtn: document.getElementById("exportJsonBtn"),
  exportMdBtn: document.getElementById("exportMdBtn"),
  recordBtn: document.getElementById("recordBtn"),
  canvas: document.getElementById("videoCanvas"),
  audio: document.getElementById("audioPlayer"),
  timelineBody: document.getElementById("timelineBody"),
  currentAction: document.getElementById("currentAction"),
};

function resolveApiBase() {
  const params = new URLSearchParams(window.location.search);
  const fromQuery = params.get("api");
  if (fromQuery) {
    const normalized = fromQuery.replace(/\/$/, "");
    localStorage.setItem("yestiger_api_base", normalized);
    return normalized;
  }
  return (window.YESTIGER_API_BASE || localStorage.getItem("yestiger_api_base") || "").replace(/\/$/, "");
}

const API_BASE = resolveApiBase();

function apiUrl(path) {
  return `${API_BASE}${path}`;
}

async function fetchJson(url) {
  const response = await fetch(url);
  const data = await response.json();
  if (!response.ok || data.error) {
    throw new Error(data.message || data.error || `Request failed: ${response.status}`);
  }
  return data;
}

const roleColors = {
  keepspace: "#6b7280",
  rhythmcall: "#1d8f74",
  mix: "#c65347",
  underground_gei: "#7a4fa3",
};

const musicColors = {
  intro: "#2f6fb2",
  verse: "#1d8f74",
  pre_chorus: "#b7791f",
  pre_chorus_build: "#d97706",
  chorus: "#c65347",
  post_chorus: "#9f5f2a",
  bridge: "#7a4fa3",
  instrumental_break: "#0f766e",
  interlude: "#0f766e",
  solo: "#8b5cf6",
  outro: "#475467",
  end: "#334155",
  unknown: "#64748b",
};

function setStatus(text) {
  els.statusLine.textContent = text;
}

function fmtTime(seconds) {
  const safe = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(safe / 60);
  const secs = safe - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${secs.toFixed(2).padStart(5, "0")}`;
}

function downloadText(filename, text, type) {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function editableResult() {
  if (!state.result) return null;
  return JSON.parse(JSON.stringify(state.result));
}

function markdownFromTimeline(result) {
  const title = result?.song?.title || "YesTiger Callbook";
  const lines = [
    `# ${title}`,
    "",
    "| Time | Music | Struct | Role | Action | Bars | Risk | Text |",
    "|---:|---|---|---|---|---:|---|---|",
  ];
  for (const action of result.timeline || []) {
    lines.push(
      `| ${fmtTime(action.start)}-${fmtTime(action.end)} | ${action.music_label || "-"} | ${action.struct_label || "-"} | ${action.role || "-"} | ${action.display_name || "-"} | ${action.bar_count ?? "-"} | ${action.risk || "-"} | ${action.typical_text || "-"} |`
    );
  }
  return `${lines.join("\n")}\n`;
}

function actionAt(time) {
  const timeline = state.result?.timeline || [];
  return timeline.find((item) => time >= Number(item.start) && time < Number(item.end)) || null;
}

function nextAction(time) {
  const timeline = state.result?.timeline || [];
  return timeline.find((item) => Number(item.start) > time) || null;
}

function musicSegmentAt(time) {
  const segments = state.result?.music_segments || state.result?.segments || [];
  return segments.find((item) => time >= Number(item.start) && time < Number(item.end)) || null;
}

function renderStats() {
  const song = state.result?.song || {};
  els.durationStat.textContent = song.duration ? fmtTime(song.duration) : "-";
  els.tempoStat.textContent = song.tempo ? `${Math.round(song.tempo)} BPM` : "-";
  els.barsStat.textContent = song.bar_count ?? "-";
  els.actionsStat.textContent = (state.result?.timeline || []).length || "-";
}

function renderProcess() {
  const process = state.result?.signal_process || {};
  const method = state.result?.method || {};
  const status = process.status || state.result?.pipeline_status || "idle";
  els.pipelineBadge.textContent = status;
  els.pipelineBadge.dataset.status = status;
  els.methodLine.textContent = method.structure
    ? `${method.structure} -> ${method.actions || "actions"}`
    : "No analysis loaded";
  els.processSteps.innerHTML = "";

  const counts = [];
  if (process.rows != null) counts.push(`${process.rows} bars`);
  if (process.music_segments != null) counts.push(`${process.music_segments} music segments`);
  if (process.call_spans != null) counts.push(`${process.call_spans} call spans`);
  if (counts.length) {
    const item = document.createElement("li");
    item.innerHTML = `<strong>Output grid</strong><span>${counts.join(" / ")}</span>`;
    els.processSteps.appendChild(item);
  }

  for (const step of process.steps || []) {
    const item = document.createElement("li");
    item.innerHTML = `<strong>${escapeHtml(step.name || "")}</strong><span>${escapeHtml(step.detail || "")}</span>`;
    els.processSteps.appendChild(item);
  }

  if (process.fallback_reason) {
    const item = document.createElement("li");
    item.className = "fallback-note";
    item.innerHTML = `<strong>Fallback reason</strong><span>${escapeHtml(process.fallback_reason)}</span>`;
    els.processSteps.appendChild(item);
  }
}

function renderTimeline() {
  const timeline = state.result?.timeline || [];
  els.timelineBody.innerHTML = "";
  timeline.forEach((action, index) => {
    const tr = document.createElement("tr");
    tr.dataset.index = String(index);
    const riskClass = action.risk === "high" ? "risk-high" : action.risk === "medium" ? "risk-medium" : "";
    tr.innerHTML = `
      <td>${fmtTime(action.start)}-${fmtTime(action.end)}</td>
      <td><span class="music-pill music-${action.music_label || "unknown"}">${escapeHtml(action.music_label || "-")}</span></td>
      <td><span class="struct-pill">${escapeHtml(action.struct_label || "-")}</span></td>
      <td><span class="role-pill role-${action.role || "keepspace"}">${action.role || "-"}</span></td>
      <td><input class="editable-action" value="${escapeAttr(action.display_name || "")}" data-field="display_name" /></td>
      <td>${action.bar_count ?? "-"}</td>
      <td class="${riskClass}">${action.risk || "low"}</td>
      <td>${escapeHtml(action.typical_text || "")}</td>
    `;
    els.timelineBody.appendChild(tr);
  });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function escapeAttr(value) {
  return escapeHtml(value).replaceAll("'", "&#39;");
}

function updateActiveRow() {
  const time = els.audio.currentTime || 0;
  const active = actionAt(time);
  const rows = els.timelineBody.querySelectorAll("tr");
  rows.forEach((row) => row.classList.remove("is-active"));
  if (active) {
    const index = (state.result.timeline || []).indexOf(active);
    const row = els.timelineBody.querySelector(`tr[data-index="${index}"]`);
    if (row) row.classList.add("is-active");
    els.currentAction.textContent = `${active.music_label || "-"} | ${active.role || "-"} | ${active.display_name} | ${fmtTime(active.start)}-${fmtTime(active.end)}`;
  } else {
    els.currentAction.textContent = state.result ? "Keep Space" : "No action loaded";
  }
}

async function buildWaveformFromUrl(url) {
  try {
    const response = await fetch(url);
    const buffer = await response.arrayBuffer();
    await buildWaveform(buffer);
  } catch (error) {
    state.waveform = [];
  }
}

async function buildWaveform(buffer) {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) {
    state.waveform = [];
    return;
  }
  const context = new AudioContextClass();
  const audioBuffer = await context.decodeAudioData(buffer.slice(0));
  const data = audioBuffer.getChannelData(0);
  const buckets = 240;
  const step = Math.max(1, Math.floor(data.length / buckets));
  const values = [];
  for (let i = 0; i < buckets; i += 1) {
    let sum = 0;
    const start = i * step;
    const end = Math.min(data.length, start + step);
    for (let j = start; j < end; j += 1) sum += Math.abs(data[j]);
    values.push(sum / Math.max(1, end - start));
  }
  const max = Math.max(...values, 0.0001);
  state.waveform = values.map((value) => value / max);
  if (context.close) context.close();
}

function drawCanvas() {
  const canvas = els.canvas;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const result = state.result;
  const time = els.audio.currentTime || 0;
  const duration = Number(result?.song?.duration || els.audio.duration || 1);
  const current = actionAt(time);
  const upcoming = nextAction(time);
  const currentMusic = musicSegmentAt(time);

  ctx.fillStyle = "#111827";
  ctx.fillRect(0, 0, width, height);

  const role = current?.role || "keepspace";
  ctx.fillStyle = roleColors[role] || "#6b7280";
  ctx.fillRect(0, 0, width, 14);

  ctx.fillStyle = "#f8fafc";
  ctx.font = "700 44px Segoe UI, sans-serif";
  ctx.fillText(result?.song?.title || "YesTiger", 48, 78);

  ctx.fillStyle = "#cbd5e1";
  ctx.font = "28px Segoe UI, sans-serif";
  const musicText = currentMusic
    ? `Music: ${currentMusic.music_label || "-"} | Struct: ${currentMusic.struct_label || "-"}`
    : "Music: -";
  ctx.fillText(musicText, 52, 124);

  ctx.fillStyle = "#f8fafc";
  ctx.font = "700 68px Segoe UI, sans-serif";
  const actionText = current?.display_name || "Keep Space";
  ctx.fillText(actionText, 48, 218);

  ctx.font = "28px Segoe UI, sans-serif";
  ctx.fillStyle = "#cbd5e1";
  const meta = current ? `${fmtTime(current.start)}-${fmtTime(current.end)} | ${current.bar_count ?? "-"} bars | ${current.role || "-"} | ${current.risk || "low"}` : fmtTime(time);
  ctx.fillText(meta, 52, 270);

  if (upcoming) {
    ctx.fillStyle = "#94a3b8";
    ctx.font = "26px Segoe UI, sans-serif";
    ctx.fillText(`Next: ${upcoming.music_label || "-"} / ${upcoming.display_name} @ ${fmtTime(upcoming.start)}`, 52, 326);
  }

  drawMusicBands(ctx, width, height, duration, time);
  drawWaveform(ctx, width, height, duration, time);
  drawRoleBands(ctx, width, height, duration, time);

  updateActiveRow();
  state.animationId = requestAnimationFrame(drawCanvas);
}

function drawMusicBands(ctx, width, height, duration, time) {
  const segments = state.result?.music_segments || state.result?.segments || [];
  const left = 52;
  const top = height - 226;
  const w = width - 104;
  const h = 24;
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(left, top, w, h);
  segments.forEach((segment) => {
    const label = segment.music_label || "unknown";
    const x = left + (Number(segment.start) / duration) * w;
    const endX = left + (Number(segment.end) / duration) * w;
    ctx.fillStyle = musicColors[label] || musicColors.unknown;
    ctx.fillRect(x, top, Math.max(2, endX - x), h);
  });
  ctx.fillStyle = "#cbd5e1";
  ctx.font = "18px Segoe UI, sans-serif";
  ctx.fillText("music structure", left, top - 8);
  const progressX = left + (time / Math.max(0.001, duration)) * w;
  ctx.fillStyle = "#f8fafc";
  ctx.fillRect(progressX, top - 6, 3, h + 12);
}

function drawWaveform(ctx, width, height, duration, time) {
  const values = state.waveform;
  const left = 52;
  const top = height - 178;
  const w = width - 104;
  const h = 86;
  ctx.strokeStyle = "#334155";
  ctx.strokeRect(left, top, w, h);
  if (!values.length) return;
  ctx.fillStyle = "#3b82a8";
  values.forEach((value, index) => {
    const x = left + (index / values.length) * w;
    const barH = value * h;
    ctx.fillRect(x, top + (h - barH) / 2, Math.max(2, w / values.length - 1), barH);
  });
  const progressX = left + (time / Math.max(0.001, duration)) * w;
  ctx.fillStyle = "#f8fafc";
  ctx.fillRect(progressX, top - 8, 3, h + 16);
}

function drawRoleBands(ctx, width, height, duration, time) {
  const spans = state.result?.call_spans || [];
  const left = 52;
  const top = height - 62;
  const w = width - 104;
  const h = 18;
  spans.forEach((span) => {
    const x = left + (Number(span.start) / duration) * w;
    const endX = left + (Number(span.end) / duration) * w;
    ctx.fillStyle = roleColors[span.call_role] || "#6b7280";
    ctx.fillRect(x, top, Math.max(2, endX - x), h);
  });
  ctx.fillStyle = "#cbd5e1";
  ctx.font = "18px Segoe UI, sans-serif";
  ctx.fillText("call role", left, top - 8);
  ctx.fillStyle = "#f8fafc";
  const progressX = left + (time / Math.max(0.001, duration)) * w;
  ctx.fillRect(progressX, top - 8, 3, h + 16);
}

function setResult(result, audioUrl) {
  state.result = result;
  renderStats();
  renderProcess();
  renderTimeline();
  els.exportJsonBtn.disabled = false;
  els.exportMdBtn.disabled = false;
  els.recordBtn.disabled = false;
  if (audioUrl) {
    els.audio.src = audioUrl;
    buildWaveformFromUrl(audioUrl);
  }
  if (state.animationId) cancelAnimationFrame(state.animationId);
  drawCanvas();
}

async function loadExamples() {
  let data;
  try {
    data = await fetchJson(apiUrl("/api/songs"));
  } catch (_error) {
    data = await fetchJson("/examples/index.json");
    setStatus("Static examples ready");
  }
  for (const song of data.songs || []) {
    const option = document.createElement("option");
    option.value = song.song_id;
    option.textContent = song.title;
    els.exampleSelect.appendChild(option);
  }
}

async function analyzeUpload(event) {
  event.preventDefault();
  const file = state.selectedFile || els.audioInput.files[0];
  if (!file) {
    setStatus("No audio selected");
    return;
  }
  const form = new FormData();
  form.append("audio", file);
  form.append("title", els.titleInput.value || file.name.replace(/\.[^.]+$/, ""));
  els.analyzeBtn.disabled = true;
  setStatus("Analyzing...");
  try {
    const response = await fetch(apiUrl("/api/analyze"), { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.message || data.error || "analysis failed");
    setResult(data, data.audio_url);
    setStatus("Analysis ready");
  } catch (error) {
    const backendHint = API_BASE || window.location.protocol.startsWith("http")
      ? "Check the backend log for details."
      : "Upload analysis needs a YesTiger backend API.";
    setStatus(`Error: ${error.message}. ${backendHint}`);
  } finally {
    els.analyzeBtn.disabled = false;
  }
}

async function loadExample() {
  const songId = els.exampleSelect.value;
  if (!songId) return;
  setStatus("Loading example...");
  try {
    const data = await fetchJson(apiUrl(`/api/examples/${songId}`));
    setResult(data, data.audio_url);
    setStatus("Example ready");
  } catch (_error) {
    try {
      const data = await fetchJson(`/examples/${songId}.json`);
      setResult(data, data.audio_url);
      setStatus("Static example ready");
    } catch (error) {
      setStatus(`Error: ${error.message}`);
    }
  }
}

function bindEvents() {
  els.form.addEventListener("submit", analyzeUpload);
  els.loadExampleBtn.addEventListener("click", loadExample);

  els.audioInput.addEventListener("change", async () => {
    const file = els.audioInput.files[0];
    if (!file) return;
    state.selectedFile = file;
    els.fileName.textContent = file.name;
    if (!els.titleInput.value) els.titleInput.value = file.name.replace(/\.[^.]+$/, "");
    const url = URL.createObjectURL(file);
    els.audio.src = url;
    await buildWaveform(await file.arrayBuffer());
  });

  ["dragenter", "dragover"].forEach((name) => {
    els.dropZone.addEventListener(name, (event) => {
      event.preventDefault();
      els.dropZone.classList.add("is-dragging");
    });
  });
  ["dragleave", "drop"].forEach((name) => {
    els.dropZone.addEventListener(name, (event) => {
      event.preventDefault();
      els.dropZone.classList.remove("is-dragging");
    });
  });
  els.dropZone.addEventListener("drop", async (event) => {
    const file = event.dataTransfer.files[0];
    if (!file) return;
    state.selectedFile = file;
    els.fileName.textContent = file.name;
    els.titleInput.value = file.name.replace(/\.[^.]+$/, "");
    const url = URL.createObjectURL(file);
    els.audio.src = url;
    await buildWaveform(await file.arrayBuffer());
  });

  els.timelineBody.addEventListener("input", (event) => {
    const input = event.target.closest("[data-field]");
    if (!input || !state.result) return;
    const row = input.closest("tr");
    const index = Number(row.dataset.index);
    const field = input.dataset.field;
    state.result.timeline[index][field] = input.value;
  });

  els.exportJsonBtn.addEventListener("click", () => {
    const result = editableResult();
    if (!result) return;
    downloadText(`${result.song.song_id || "yetiger"}.timeline.json`, JSON.stringify(result, null, 2), "application/json");
  });

  els.exportMdBtn.addEventListener("click", () => {
    const result = editableResult();
    if (!result) return;
    downloadText(`${result.song.song_id || "yetiger"}.callbook.md`, markdownFromTimeline(result), "text/markdown");
  });

  els.recordBtn.addEventListener("click", recordWebm);
}

function recordWebm() {
  if (!state.result || !window.MediaRecorder || !els.canvas.captureStream) {
    setStatus("Recording unavailable");
    return;
  }
  const canvasStream = els.canvas.captureStream(30);
  const audioCapture = els.audio.captureStream || els.audio.mozCaptureStream;
  if (audioCapture) {
    const audioStream = audioCapture.call(els.audio);
    audioStream.getAudioTracks().forEach((track) => canvasStream.addTrack(track));
  }
  const chunks = [];
  let recorder;
  try {
    recorder = new MediaRecorder(canvasStream, { mimeType: "video/webm" });
  } catch (_error) {
    recorder = new MediaRecorder(canvasStream);
  }
  recorder.ondataavailable = (event) => {
    if (event.data.size) chunks.push(event.data);
  };
  recorder.onstop = () => {
    const blob = new Blob(chunks, { type: "video/webm" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${state.result.song.song_id || "yetiger"}.webm`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    setStatus("Recording saved");
  };
  els.audio.currentTime = 0;
  recorder.start();
  setStatus("Recording...");
  els.audio.play();
  const stop = () => {
    if (recorder.state !== "inactive") recorder.stop();
    els.audio.removeEventListener("ended", stop);
  };
  els.audio.addEventListener("ended", stop);
}

bindEvents();
loadExamples();
drawCanvas();
