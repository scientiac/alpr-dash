// Dashboard behaviour. Plain JavaScript, no libraries; talks to the JSON API in server.py.
"use strict";

const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const img = p => p ? "/images/" + p.split("/").map(encodeURIComponent).join("/") : "";
const qs = o => new URLSearchParams(Object.entries(o).filter(([, v]) => v !== "" && v != null)).toString();
const num = n => Number(n || 0).toLocaleString();
const cap = s => s ? s[0].toUpperCase() + s.slice(1) : "";

async function api(path, opts = {}) {
  if (opts.json !== undefined) {
    opts = {...opts, method: opts.method || "POST", body: JSON.stringify(opts.json),
            headers: {"Content-Type": "application/json"}};
  }
  const r = await fetch(path, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || `${r.status} ${r.statusText}`);
  return data;
}

// ---------- shared bits ----------
const TYPES = ["car", "van", "motorcycle", "bus", "truck", "unknown"];
const ICON = {
  recognized: '<path d="M20 6 9 17l-5-5"/>',
  review: '<path d="M12 9v4"/><path d="M12 17h.01"/><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>',
  unrecognized: '<circle cx="12" cy="12" r="9"/><path d="m15 9-6 6M9 9l6 6"/>',
};
const STATUS_LABEL = {recognized: "Recognized", review: "Requires review", unrecognized: "Unrecognized"};
const icon = (name, size = 12) => `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICON[name]}</svg>`;
const statusBadge = s => `<span class="badge ${esc(s)}">${icon(s)}${STATUS_LABEL[s] || esc(s)}</span>`;
const watchBadge = w => w ? `<span class="badge ${w}">${w === "block" ? "Blacklist" : "Whitelist"}</span>` : "";
const swatch = t => `<span class="swatch" style="background:var(--s-${TYPES.includes(t) ? t : "unknown"})"></span>`;

function videoTime(s) {
  if (s == null) return "–";
  const m = Math.floor(s / 60);
  return `${m}:${(s - m * 60).toFixed(1).padStart(4, "0")}`;
}

function when(t) {
  if (!t) return "";
  const [d, time] = t.split(" ");
  const today = new Date().toLocaleDateString("sv");     // YYYY-MM-DD
  return d === today ? time : `${d} ${time.slice(0, 5)}`;
}

function vehicleText(r) {
  return [r.colour, r.make].filter(Boolean).join(" ");
}

function plateCell(r) {
  return r.plate ? `<span class="plate">${esc(r.plate)}</span> ${watchBadge(r.watch)}`
                 : `<span class="muted">Not readable</span>`;
}

function toast(html, kind = "", onclick = null, ms = 8000) {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.innerHTML = `<div>${html}</div><button aria-label="Dismiss">&times;</button>`;
  el.querySelector("button").onclick = e => { e.stopPropagation(); el.remove(); };
  if (onclick) { el.style.cursor = "pointer"; el.onclick = () => { onclick(); el.remove(); }; }
  $("toasts").prepend(el);
  if (ms) setTimeout(() => el.remove(), ms);
}

// ---------- theme ----------
const THEMES = ["system", "light", "dark"];
const THEME_ICON = {
  system: '<rect x="3" y="4" width="18" height="12" rx="2"/><path d="M8 20h8M12 16v4"/>',
  light: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  dark: '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>',
};
function setTheme(name) {
  if (name === "system") { delete document.documentElement.dataset.theme; localStorage.removeItem("theme"); }
  else { document.documentElement.dataset.theme = name; localStorage.setItem("theme", name); }
  $("theme").innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${THEME_ICON[name]}</svg>`;
  $("theme").title = `Theme: ${name}`;
}
$("theme").onclick = () => {
  const cur = localStorage.getItem("theme") || "system";
  setTheme(THEMES[(THEMES.indexOf(cur) + 1) % THEMES.length]);
};
setTheme(localStorage.getItem("theme") || "system");

// ---------- sections (one page; the tabs jump and follow the scroll) ----------
const pages = {};
const SECTIONS = ["overview", "live", "upload", "records", "analytics", "watchlist"];

function go(id) {
  $(id).scrollIntoView({block: "start"});
  history.replaceState(null, "", "#" + id);
}

function measureTop() {
  document.documentElement.style.setProperty("--top", document.querySelector(".topbar").offsetHeight + "px");
}

function spy() {
  const top = document.querySelector(".topbar").offsetHeight + 60;
  let current = SECTIONS[0];
  for (const id of SECTIONS) if ($(id).getBoundingClientRect().top <= top) current = id;
  if (innerHeight + scrollY >= document.body.scrollHeight - 4) current = SECTIONS[SECTIONS.length - 1];
  document.querySelectorAll("nav a").forEach(a =>
    a.hash === "#" + current ? a.setAttribute("aria-current", "true") : a.removeAttribute("aria-current"));
}
addEventListener("scroll", spy, {passive: true});
addEventListener("resize", () => { measureTop(); spy(); });

function refreshAll() {
  for (const p of Object.values(pages)) p.tick?.(true);
}

document.addEventListener("click", e => {
  const b = e.target.closest("[data-go]");
  if (b && b.dataset.go === "review") { setRecordFilter({status: "review"}); pages.records.load(); go("records"); }
});

// ---------- polling ----------
let state = null, lastId = null, alertId = null;

async function poll() {
  try {
    state = await api("/api/state");
    $("model-dot").className = "dot " + (state.error ? "err" : state.ready ? "on" : "");
    $("model-text").textContent = state.error ? "Models failed to load" : state.ready
      ? (state.running.length ? `${state.running.length} running` : "Ready") : "Loading models…";
    if (state.error && !poll.shownError) { poll.shownError = true; toast(esc(state.error), "error", null, 0); }
    if (lastId === null) { lastId = alertId = state.latest_id; }
    const fresh = state.latest_id !== lastId;
    if (fresh) { lastId = state.latest_id; checkAlerts(); }
    for (const p of Object.values(pages)) p.tick?.(fresh);
    spy();
  } catch (e) {
    $("model-dot").className = "dot err";
    $("model-text").textContent = "Server not reachable";
  }
}

async function checkAlerts() {
  const {records} = await api("/api/records?" + qs({watch: "block", limit: 10}));
  for (const r of records.filter(r => r.id > alertId).reverse()) {
    toast(`<strong>Blacklisted plate</strong> <span class="plate">${esc(r.plate)}</span><br>
           <span class="muted small">${esc(r.source)} · ${videoTime(r.video_time)}${r.watch_note ? " · " + esc(r.watch_note) : ""}</span>`,
          "block", () => openDetail(r.id), 0);
  }
  alertId = Math.max(alertId, ...records.map(r => r.id));
}

// ---------- tooltip ----------
const tip = $("tip");
function showTip(e, html) {
  tip.innerHTML = html;
  tip.classList.remove("hidden");
  const w = tip.offsetWidth, h = tip.offsetHeight;
  let x = e.clientX + 14, y = e.clientY + 14;
  if (x + w > innerWidth - 8) x = e.clientX - w - 14;
  if (y + h > innerHeight - 8) y = e.clientY - h - 14;
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
const hideTip = () => tip.classList.add("hidden");

// ---------- category bars (types) ----------
function typeBars(el, types) {
  const entries = TYPES.map(t => [t, types[t] || 0]).filter(([, n]) => n);
  const max = Math.max(1, ...entries.map(([, n]) => n));
  el.innerHTML = entries.length ? entries.map(([t, n]) =>
    `<div class="row" data-tip="${cap(t)}: ${num(n)}"><span>${cap(t)}</span>
      <div class="track"><div class="fill" style="width:${n / max * 100}%;background:var(--s-${t})"></div></div>
      <span class="n">${num(n)}</span></div>`).join("")
    : `<span class="muted small">No vehicles yet</span>`;
}

// =====================================================================
// Overview
// =====================================================================
pages.overview = {
  enter() { this.tick(true); },
  async tick(fresh) {
    if (!state) return;
    const s = state.stats;
    $("o-total").textContent = num(s.total);
    $("o-unique").textContent = `${num(s.unique_plates)} different plates`;
    $("o-rec-label").innerHTML = `${icon("recognized")} Recognized`;
    $("o-rev-label").innerHTML = `${icon("review")} Requires review`;
    $("o-unr-label").innerHTML = `${icon("unrecognized")} Unreadable`;
    $("o-rec").textContent = num(s.recognized);
    $("o-rec-sub").textContent = s.total ? `${Math.round(100 * s.recognized / s.total)}% of records` : "";
    $("o-rev").textContent = num(s.review);
    $("o-unr").textContent = num(s.unrecognized);
    $("o-dir").textContent = `${num(s.incoming)} in · ${num(s.outgoing)} out`;
    typeBars($("o-types"), s.types);
    $("o-running").innerHTML = state.running.length ? state.running.map(j => `
      <div class="grid" style="gap:6px;margin-bottom:10px"><div class="small"><strong>${esc(j.name)}</strong>
        <span class="muted">${j.kind === "camera" ? "live camera" : j.kind === "test" ? esc(j.error || "test loop")
          : j.progress != null ? Math.round(j.progress * 100) + "%" : ""} · ${(j.live?.fps || 0).toFixed(0)} fps</span></div>
        ${j.kind === "camera" || j.kind === "test" ? "" : `<div class="progress"><div style="width:${(j.progress || 0) * 100}%"></div></div>`}</div>`).join("")
      : `<span class="muted small">Nothing is being processed. Upload a video or start a camera.</span>`;
  },
};

// =====================================================================
// Upload and processing results
// =====================================================================
let uploadCfg = {max_upload_mb: 2048, video_exts: [".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"]};
api("/api/config").then(c => {
  uploadCfg = c;
  $("u-hint").textContent = `${c.video_exts.map(e => e.slice(1).toUpperCase()).join(", ")} · up to ${num(c.max_upload_mb)} MB`;
  $("u-file").accept = c.video_exts.join(",");
}).catch(() => {});

const uploadQueue = [];
let uploading = false;

function addFiles(files) {
  $("u-error").textContent = "";
  for (const f of files) {
    const ext = "." + f.name.split(".").pop().toLowerCase();
    if (!uploadCfg.video_exts.includes(ext)) { $("u-error").textContent = `${f.name}: not a supported video type`; continue; }
    if (f.size > uploadCfg.max_upload_mb * 1048576) { $("u-error").textContent = `${f.name}: larger than ${uploadCfg.max_upload_mb} MB`; continue; }
    if (!f.size) { $("u-error").textContent = `${f.name}: the file is empty`; continue; }
    uploadQueue.push(f);
  }
  nextUpload();
}

function nextUpload() {
  if (uploading || !uploadQueue.length) return;
  uploading = true;
  const f = uploadQueue.shift();
  $("u-progress").classList.remove("hidden");
  $("u-name").textContent = f.name;
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/upload?" + qs({name: f.name}));
  xhr.upload.onprogress = e => {
    const p = e.lengthComputable ? e.loaded / e.total : 0;
    $("u-bar").style.width = `${p * 100}%`;
    $("u-pct").textContent = p < 1 ? `uploading ${Math.round(p * 100)}%` : "checking the video…";
  };
  xhr.onloadend = () => {
    let data = {};
    try { data = JSON.parse(xhr.responseText); } catch (e) {}
    if (xhr.status === 201) toast(`<strong>${esc(f.name)}</strong> uploaded and queued`, "ok", null, 4000);
    else $("u-error").textContent = `${f.name}: ${data.error || "upload failed"}`;
    uploading = false;
    $("u-progress").classList.add("hidden");
    $("u-bar").style.width = "0";
    pages.upload.load();
    nextUpload();
  };
  xhr.send(f);
}

$("u-file").onchange = e => { addFiles(e.target.files); e.target.value = ""; };
const drop = $("drop");
drop.ondragover = e => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = e => { e.preventDefault(); drop.classList.remove("over"); addFiles(e.dataTransfer.files); };

const JOB_STATUS = {queued: "Queued", processing: "Processing", done: "Done", failed: "Failed", stopped: "Stopped"};

pages.upload = {
  enter() { this.load(); },
  tick() { this.load(); },
  async load() {
    const jobs = await api("/api/jobs");
    $("jobs").innerHTML = jobs.length ? jobs.map(j => {
      const live = state?.running.find(r => r.id === j.id);
      const pct = Math.round((j.progress || 0) * 100);
      const active = j.status === "processing" || j.status === "queued";
      return `<tr>
        <td><div><strong>${esc(j.name)}</strong></div>
            <div class="muted small">${j.kind === "camera" ? "camera · " + esc(j.path) : j.kind === "test" ? "test loop" : `#${j.id}`} · ${esc(when(j.created))}</div>
            ${j.error ? `<div class="${j.status === "failed" ? "error-text" : "muted small"}">${esc(j.error)}</div>` : ""}</td>
        <td><span class="badge ${j.status}">${JOB_STATUS[j.status] || esc(j.status)}</span></td>
        <td>${j.kind === "camera" || j.kind === "test" ? '<span class="muted small">live</span>'
              : `<div class="progress"><div style="width:${pct}%"></div></div><div class="muted small">${pct}%</div>`}</td>
        <td class="num">${j.fps ? `${(live?.live?.fps ?? j.fps).toFixed(1)} fps` : "–"}</td>
        <td class="num">${j.duration ? videoTime(j.duration) : "–"}</td>
        <td class="num">${num(j.records)}</td>
        <td class="nowrap" style="text-align:right">
          ${j.output ? `<button data-act="watch" data-id="${j.id}">Annotated video</button>` : ""}
          ${live ? `<button data-act="live" data-id="${j.id}">Live</button>` : ""}
          ${j.records ? `<button data-act="records" data-id="${j.id}">Records</button>` : ""}
          ${active ? `<button data-act="cancel" data-id="${j.id}">Cancel</button>`
                   : `<button class="icon danger" data-act="delete" data-id="${j.id}" title="Remove job and videos (records stay)" aria-label="Remove">&times;</button>`}
        </td></tr>`;
    }).join("") : `<tr><td colspan="7" class="empty">No videos yet</td></tr>`;
    this.jobs = jobs;
  },
};

$("jobs").onclick = async e => {
  const b = e.target.closest("button[data-act]");
  if (!b) return;
  const id = +b.dataset.id, job = pages.upload.jobs.find(j => j.id === id);
  try {
    if (b.dataset.act === "watch") openPlayer(job);
    else if (b.dataset.act === "live") { liveJob = id; pages.live.tick(); go("live"); }
    else if (b.dataset.act === "records") { setRecordFilter({job: String(id)}); pages.records.load(); go("records"); }
    else if (b.dataset.act === "cancel") { await api(`/api/jobs/${id}/cancel`, {method: "POST"}); pages.upload.load(); }
    else if (b.dataset.act === "delete" && confirm(`Remove "${job.name}" and its videos? Its records are kept.`)) {
      await api(`/api/jobs/${id}`, {method: "DELETE"}); pages.upload.load();
    }
  } catch (err) { toast(esc(err.message), "error"); }
};

function openPlayer(job) {
  $("p-title").textContent = job.name;
  $("p-video").src = `/videos/${encodeURIComponent(job.output)}`;
  $("player").showModal();
}
$("p-close").onclick = () => $("player").close();
$("player").onclose = () => { $("p-video").pause(); $("p-video").removeAttribute("src"); $("p-video").load(); };

// =====================================================================
// Traffic records
// =====================================================================
const recForm = $("filters");
let recOffset = 0, recJob = "";
const PAGE = 50;

recForm.querySelector("[name=type]").innerHTML += TYPES.map(t => `<option value="${t === "unknown" ? "none" : t}">${cap(t)}</option>`).join("");

function recordFilters() {
  const f = {};
  for (const el of recForm.querySelectorAll("[name]")) if (el.value) f[el.name] = el.value;
  if (recJob) f.job = recJob;
  return f;
}

function setRecordFilter(f) {
  for (const el of recForm.querySelectorAll("[name]")) el.value = el.name === "sort" ? "newest" : "";
  recJob = f.job || "";
  for (const [k, v] of Object.entries(f)) { const el = recForm.querySelector(`[name=${k}]`); if (el) el.value = v; }
  recOffset = 0;
}

async function loadSources(select) {
  const cur = select.value;
  const list = await api("/api/sources");
  select.innerHTML = `<option value="">All sources</option>` + list.map(s => `<option>${esc(s)}</option>`).join("");
  select.value = list.includes(cur) ? cur : "";
}

pages.records = {
  enter() { loadSources(recForm.querySelector("[name=source]")); this.load(); },
  tick(fresh) { if (fresh && recOffset === 0) this.load(); },
  async load() {
    const f = recordFilters();
    $("export").href = "/api/export.csv?" + qs(f);
    let chip = $("job-chip");
    if (recJob && !chip) {
      recForm.insertAdjacentHTML("beforeend", `<button type="button" id="job-chip" class="badge" title="Show all jobs">Job #${esc(recJob)} &times;</button>`);
      $("job-chip").onclick = () => { recJob = ""; $("job-chip").remove(); recOffset = 0; this.load(); };
    } else if (!recJob && chip) chip.remove();
    const {records, total} = await api("/api/records?" + qs({...f, limit: PAGE, offset: recOffset}));
    this.records = records;
    $("rows").innerHTML = records.length ? records.map((r, i) => `
      <tr class="click" data-i="${i}">
        <td class="nowrap"><div>${esc(r.time)}</div><div class="muted small">${esc(r.source)}</div></td>
        <td>${plateCell(r)}</td>
        <td class="nowrap"><div>${swatch(r.vehicle_type)} ${esc(cap(r.vehicle_type || "unknown"))}${r.track_id ? ` <span class="muted small">#${r.track_id}</span>` : ""}</div>
          <div class="muted small">${esc(vehicleText(r))}${r.direction ? ` · ${r.direction === "in" ? "incoming" : "outgoing"}` : ""}</div></td>
        <td class="num">${videoTime(r.video_time)}</td>
        <td class="num">${r.plate ? `${r.confidence}%` : "–"}</td>
        <td>${r.plate_image ? `<img class="thumb" src="${img(r.plate_image)}" alt="Plate ${esc(r.plate)}" loading="lazy">` : '<span class="muted small">–</span>'}</td>
        <td>${r.vehicle_image ? `<img class="thumb vehicle" src="${img(r.vehicle_image)}" alt="Vehicle" loading="lazy">` : ""}</td>
        <td class="nowrap">${statusBadge(r.status)}${r.reviewed ? ' <span class="muted small" title="Checked by a person">checked</span>' : ""}</td>
      </tr>`).join("")
      : `<tr><td colspan="8" class="empty">${Object.keys(f).length > 1 || f.q ? "No records match these filters" : "No records yet"}</td></tr>`;
    $("r-count").textContent = total ? `${num(recOffset + 1)}–${num(recOffset + records.length)} of ${num(total)}` : "0 records";
    $("r-prev").disabled = recOffset === 0;
    $("r-next").disabled = recOffset + PAGE >= total;
  },
};

let typing;
recForm.oninput = e => {
  recOffset = 0;
  clearTimeout(typing);
  typing = setTimeout(() => pages.records.load(), e.target.name === "q" ? 250 : 0);
};
recForm.querySelector("[name=q]").onkeydown = e => { if (e.key === "Escape") { e.target.value = ""; recForm.oninput(e); } };
$("f-clear").onclick = () => { setRecordFilter({}); pages.records.load(); };
$("r-prev").onclick = () => { recOffset = Math.max(0, recOffset - PAGE); pages.records.load(); };
$("r-next").onclick = () => { recOffset += PAGE; pages.records.load(); };
$("rows").onclick = e => {
  const tr = e.target.closest("tr[data-i]");
  if (tr) openDetail(pages.records.records[tr.dataset.i].id);
};

// ---------- record detail and manual review ----------
let detail = null;

async function openDetail(id) {
  try { detail = await api(`/api/records/${id}`); } catch (e) { toast(esc(e.message), "error"); return; }
  const r = detail;
  $("d-title").innerHTML = `${r.plate ? `<span class="plate">${esc(r.plate)}</span>` : "Unreadable plate"} ${statusBadge(r.status)} ${watchBadge(r.watch)}`;
  $("d-vehicle").src = img(r.vehicle_image);
  $("d-vehicle-link").href = img(r.vehicle_image);
  $("d-plate-link").classList.toggle("hidden", !r.plate_image);
  $("d-plate").src = img(r.plate_image);
  $("d-plate-link").href = img(r.plate_image);
  const rows = [
    ["Record", `#${r.id}`], ["Date and time", r.time], ["Video time", videoTime(r.video_time)],
    ["Source", r.source], ["Vehicle type", cap(r.vehicle_type || "unknown")],
    ["Tracking ID", r.track_id ? `#${r.track_id}` : "–"], ["Colour", cap(r.colour) || "–"],
    ["Make", r.make || "–"], ["Direction", r.direction === "in" ? "Incoming" : r.direction === "out" ? "Outgoing" : "–"],
    ["Confidence", r.plate ? `${r.confidence}% (${r.readings} readings)` : "–"],
    ["Checked", r.reviewed ? "Yes, by a person" : "No"],
  ];
  if (r.watch_note) rows.push(["Watchlist note", r.watch_note]);
  $("d-info").innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("");
  const form = $("d-form");
  form.plate.value = r.plate;
  form.status.value = r.status;
  $("d-block").disabled = $("d-allow").disabled = !r.plate;
  if (!$("detail").open) $("detail").showModal();
}

$("d-close").onclick = () => $("detail").close();
$("detail").onclick = e => { if (e.target === $("detail")) $("detail").close(); };
$("d-form").onsubmit = async e => {
  e.preventDefault();
  const form = e.target;
  try {
    await api(`/api/records/${detail.id}`, {json: {plate: form.plate.value, status: form.status.value}});
    toast("Review saved", "ok", null, 3000);
    $("detail").close();
    refreshAll();
    pages.records.load();
  } catch (err) { toast(esc(err.message), "error"); }
};
async function listPlate(kind) {
  try {
    await api("/api/watchlist", {json: {plate: detail.plate, kind}});
    toast(`${esc(detail.plate)} added to the ${kind === "block" ? "blacklist" : "whitelist"}`, "ok", null, 3000);
    openDetail(detail.id);
    pages.records.load();
    pages.watchlist.load();
  } catch (err) { toast(esc(err.message), "error"); }
}
$("d-block").onclick = () => listPlate("block");
$("d-allow").onclick = () => listPlate("allow");
$("d-delete").onclick = async () => {
  if (!confirm("Delete this record and its photos?")) return;
  try {
    await api(`/api/records/${detail.id}`, {method: "DELETE"});
    $("detail").close();
    refreshAll();
    pages.records.load();
  } catch (err) { toast(esc(err.message), "error"); }
};

// =====================================================================
// Analytics
// =====================================================================
const anaForm = $("a-filters");

function anaFilters() {
  const f = {};
  for (const el of anaForm.querySelectorAll("[name]")) if (el.value) f[el.name] = el.value;
  return f;
}

pages.analytics = {
  enter() { loadSources(anaForm.querySelector("[name=source]")); this.load(); },
  tick(fresh) { if (fresh) this.load(); },
  async load() {
    const f = anaFilters();
    const [s, trend] = await Promise.all([api("/api/stats?" + qs(f)), api("/api/trend?" + qs(f))]);
    $("a-total").textContent = `${num(s.total)} records`;
    typeBars($("a-types"), s.types);
    statusBar(s);
    $("a-dir").innerHTML = `<dt>Incoming</dt><dd>${num(s.incoming)}</dd><dt>Outgoing</dt><dd>${num(s.outgoing)}</dd>
      <dt>Direction unknown</dt><dd>${num(s.total - s.incoming - s.outgoing)}</dd>
      <dt>Blacklist hits</dt><dd>${num(s.blocked)}</dd><dt>Whitelist hits</dt><dd>${num(s.allowed)}</dd>`;
    this.trend = trend;
    trendChart(trend);
  },
};
anaForm.oninput = () => pages.analytics.load();
$("a-clear").onclick = () => { anaForm.querySelectorAll("[name]").forEach(el => el.value = ""); pages.analytics.load(); };
window.addEventListener("resize", () => { if (pages.analytics.trend) trendChart(pages.analytics.trend); });

// Status: one whole split into parts (tall segmented bar), status colours with icon + label
function statusBar(s) {
  const parts = [["recognized", s.recognized, "var(--good)", ""], ["review", s.review, "var(--warn)", "warn"],
                 ["unrecognized", s.unrecognized, "var(--bad)", ""]];
  const total = s.total || 0;
  const bar = $("a-status");
  bar.setAttribute("aria-label", parts.map(([k, n]) => `${STATUS_LABEL[k]} ${n}`).join(", "));
  bar.innerHTML = total ? parts.filter(([, n]) => n).map(([k, n, c, cls]) => {
    const pct = Math.round(100 * n / total);
    return `<div class="${cls}" style="flex:${n};background:${c}" data-tip="${STATUS_LABEL[k]}: ${num(n)} (${pct}%)">${pct >= 12 ? pct + "%" : ""}</div>`;
  }).join("") : `<div style="flex:1;background:var(--subtle)"></div>`;
  $("a-status-legend").innerHTML = parts.map(([k, n]) =>
    `<span>${statusBadge(k)} ${num(n)}</span>`).join("");
}

// Traffic over time: stacked columns by vehicle type
function niceMax(v) {
  if (v <= 5) return Math.max(1, v);
  const p = 10 ** Math.floor(Math.log10(v)), f = v / p;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p;
}

function trendChart(trend) {
  const svg = $("a-trend");
  const pts = trend.points;
  const types = TYPES.filter(t => pts.some(p => p.counts[t]));
  $("a-legend").innerHTML = types.map(t => `<span>${swatch(t)}${cap(t)}</span>`).join("");
  $("a-bucket").textContent = pts.length ? `per ${trend.bucket === "5min" ? "5 minutes" : trend.bucket}` : "";
  const W = svg.clientWidth || 800, H = 240, L = 36, R = 8, T = 8, B = 28;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  if (!pts.length) {
    svg.innerHTML = `<text x="${W / 2}" y="${H / 2}" text-anchor="middle">No records in this range</text>`;
    $("a-table").innerHTML = "";
    return;
  }
  const totals = pts.map(p => types.reduce((a, t) => a + (p.counts[t] || 0), 0));
  const max = niceMax(Math.max(...totals));
  const ticks = max <= 5 ? max : 4;
  const pw = W - L - R, ph = H - T - B, band = pw / pts.length;
  const bw = Math.max(2, Math.min(24, band * 0.7));
  const y = v => T + ph - (v / max) * ph;
  const label = t => trend.bucket === "day" ? t.slice(5) : t.slice(11);
  const every = Math.ceil(pts.length / Math.max(1, Math.floor(pw / 70)));
  let out = `<g class="grid">`;
  for (let i = 0; i <= ticks; i++) {
    const v = Math.round(max * i / ticks);
    out += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${num(v)}</text>`;
  }
  out += `</g><line class="axis" x1="${L}" x2="${W - R}" y1="${T + ph}" y2="${T + ph}"/>`;
  pts.forEach((p, i) => {
    const x = L + i * band + (band - bw) / 2;
    let acc = 0, segs = "";
    const shown = types.filter(t => p.counts[t]);
    shown.forEach((t, k) => {
      const n = p.counts[t];
      const y1 = y(acc + n), y0 = y(acc);
      const gap = k > 0 ? 2 : 0;                      // 2px surface gap between stacked parts
      const h = Math.max(0, y0 - y1 - gap);
      const top = k === shown.length - 1;
      const r = top ? Math.min(4, h, bw / 2) : 0;     // rounded data end, square at the baseline
      segs += `<path class="seg" style="fill:var(--s-${t})" d="M${x},${y0 - gap} v${-(h - r)} ${r ? `q0,${-r} ${r},${-r} h${bw - 2 * r} q${r},0 ${r},${r}` : `h${bw}`} v${h - r} z"/>`;
      acc += n;
    });
    out += `<g class="col" data-i="${i}">${segs}<rect class="hit" x="${L + i * band}" y="${T}" width="${band}" height="${ph}"/></g>`;
    if (i % every === 0) out += `<text x="${L + i * band + band / 2}" y="${H - 8}" text-anchor="middle">${esc(label(p.t))}</text>`;
  });
  svg.innerHTML = out;
  svg.onmousemove = e => {
    const g = e.target.closest("g.col");
    if (!g) return hideTip();
    const p = pts[g.dataset.i], total = totals[g.dataset.i];
    showTip(e, `<div class="t">${esc(p.t)}</div>` + types.filter(t => p.counts[t]).map(t =>
      `<div class="r"><span>${swatch(t)}${cap(t)}</span><span>${num(p.counts[t])}</span></div>`).join("")
      + `<div class="r"><span>Total</span><strong>${num(total)}</strong></div>`);
  };
  svg.onmouseleave = hideTip;
  // Table view of the same numbers
  $("a-table").innerHTML = `<thead><tr><th>Time</th>${types.map(t => `<th class="num">${cap(t)}</th>`).join("")}<th class="num">Total</th></tr></thead>
    <tbody>${pts.filter((p, i) => totals[i]).map(p => `<tr><td>${esc(p.t)}</td>${types.map(t => `<td class="num">${num(p.counts[t])}</td>`).join("")}
      <td class="num">${num(types.reduce((a, t) => a + (p.counts[t] || 0), 0))}</td></tr>`).join("")}</tbody>`;
}

// hover for the bar lists and status bar
document.addEventListener("mousemove", e => {
  const el = e.target.closest("[data-tip]");
  if (el) showTip(e, esc(el.dataset.tip));
  else if (!e.target.closest("#a-trend")) hideTip();
});

// =====================================================================
// Live view and cameras
// =====================================================================
let liveJob = null, liveSrc = "";

pages.live = {
  enter() { this.tick(); this.cameras(); },
  async tick() {
    if (!state) return;
    const run = state.running;
    const pick = $("l-pick");
    if (!run.some(j => j.id === liveJob)) liveJob = run.length ? run[0].id : null;
    pick.innerHTML = run.length ? run.map(j => `<option value="${j.id}">${esc(j.name)}${j.kind === "camera" ? " (camera)" : ""}</option>`).join("")
                                : `<option value="">Nothing running</option>`;
    pick.value = liveJob ?? "";
    const src = liveJob ? `/stream/${liveJob}.mjpg` : "";
    if (src !== liveSrc) {
      liveSrc = src;
      $("l-video").innerHTML = src ? `<img src="${src}" alt="Live annotated video">`
        : `<span class="placeholder">Nothing is being processed. Start a camera or upload a video.</span>`;
    }
    const j = run.find(r => r.id === liveJob);
    const live = j?.live || {};
    $("l-fps").textContent = j ? `${(live.fps || 0).toFixed(1)} fps` : "";
    $("l-info").innerHTML = j ? `<dt>Source</dt><dd>${esc(live.source || j.name)}</dd>
      <dt>Vehicles in view</dt><dd>${num(live.vehicles)}</dd>
      <dt>Plates in view</dt><dd>${num((live.active || []).length)}</dd>
      ${j.kind === "camera" || j.kind === "test" ? `<dt>Frames</dt><dd>${num(j.frames)}</dd>` : `<dt>Progress</dt><dd>${Math.round((j.progress || 0) * 100)}%</dd>`}
      <dt>Records</dt><dd>${num(j.records)}</dd>${j.error ? `<dt>${j.kind === "test" ? "Playing" : "Note"}</dt><dd>${esc(j.error)}</dd>` : ""}`
      : `<dd class="muted small">Nothing running</dd>`;
    $("l-chips").innerHTML = (live.active || []).map(t => `<span class="chip">${esc(t.plate || "…")}</span>`).join("");
    this.camTick = (this.camTick || 0) + 1;
    if (this.camTick % 3 === 0) this.cameras();
  },
  async cameras() {
    const cams = await api("/api/cameras");
    $("cams").innerHTML = cams.length ? `<table><tbody>${cams.map(c => `<tr>
        <td><div><strong>${esc(c.name)}</strong></div><div class="muted small">${esc(c.url)}</div></td>
        <td>${c.state === "running" ? '<span class="badge running">Running</span>'
             : c.state ? `<span class="badge queued" title="${esc(c.error || "")}">${cap(c.state)}</span>`
             : '<span class="badge">Stopped</span>'}</td>
        <td class="nowrap" style="text-align:right">
          <button data-cam="${c.id}" data-act="${c.job ? "stop" : "start"}">${c.job ? "Stop" : "Start"}</button>
          <button class="icon danger" data-cam="${c.id}" data-act="delete" aria-label="Remove camera">&times;</button></td></tr>`).join("")}</tbody></table>`
      : `<span class="muted small">No cameras yet. Add an RTSP/HTTP stream or a webcam number below.</span>`;
  },
};
$("l-pick").onchange = e => { liveJob = +e.target.value || null; pages.live.tick(); };
$("cams").onclick = async e => {
  const b = e.target.closest("button[data-cam]");
  if (!b) return;
  try {
    if (b.dataset.act === "delete") {
      if (!confirm("Remove this camera?")) return;
      await api(`/api/cameras/${b.dataset.cam}`, {method: "DELETE"});
    } else {
      const r = await api(`/api/cameras/${b.dataset.cam}/${b.dataset.act}`, {method: "POST"});
      if (r.job) liveJob = r.job;
    }
    setTimeout(() => { pages.live.cameras(); poll(); }, 300);
  } catch (err) { toast(esc(err.message), "error"); }
};
$("cam-form").onsubmit = async e => {
  e.preventDefault();
  $("cam-error").textContent = "";
  try {
    await api("/api/cameras", {json: {name: e.target.name.value, url: e.target.url.value}});
    e.target.reset();
    pages.live.cameras();
  } catch (err) { $("cam-error").textContent = err.message; }
};

// =====================================================================
// Watchlist
// =====================================================================
pages.watchlist = {
  enter() { this.load(); },
  tick(fresh) { if (fresh) this.load(); },
  async load() {
    const list = await api("/api/watchlist");
    $("w-rows").innerHTML = list.length ? list.map(w => `<tr>
        <td><span class="plate">${esc(w.plate)}</span></td><td>${watchBadge(w.kind)}</td>
        <td class="muted">${esc(w.note || "")}</td><td class="num">${num(w.seen)}</td>
        <td class="nowrap muted small">${esc(w.last_seen || "never")}</td>
        <td style="text-align:right">${w.seen ? `<button data-plate="${esc(w.plate)}">Records</button>` : ""}
          <button class="icon danger" data-del="${w.id}" aria-label="Remove">&times;</button></td></tr>`).join("")
      : `<tr><td colspan="6" class="empty">The watchlist is empty</td></tr>`;
  },
};
$("w-form").onsubmit = async e => {
  e.preventDefault();
  $("w-error").textContent = "";
  const f = e.target;
  try {
    await api("/api/watchlist", {json: {plate: f.plate.value, kind: f.kind.value, note: f.note.value}});
    f.plate.value = f.note.value = "";
    pages.watchlist.load();
  } catch (err) { $("w-error").textContent = err.message; }
};
$("w-rows").onclick = async e => {
  const del = e.target.closest("[data-del]"), plate = e.target.closest("[data-plate]");
  if (del) { await api(`/api/watchlist/${del.dataset.del}`, {method: "DELETE"}); pages.watchlist.load(); }
  if (plate) { setRecordFilter({q: plate.dataset.plate}); pages.records.load(); go("records"); }
};

// ---------- start ----------
measureTop();
for (const p of Object.values(pages)) p.enter?.();
poll();
setInterval(poll, 1000);
if (location.hash && SECTIONS.includes(location.hash.slice(1))) setTimeout(() => go(location.hash.slice(1)), 300);
spy();
