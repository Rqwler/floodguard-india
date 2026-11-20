/* FloodGuard India — dashboard controller
 * Charts: Chart.js 4 (vendored). Data: FastAPI /api/*.
 */
(() => {
  "use strict";

  const IST = "Asia/Kolkata";
  const REFRESH_POLL_MS = 60_000;
  const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;

  const LEVELS = [
    { label: "Normal", color: "#2dd4bf", bg: "rgba(45,212,191,.12)" },
    { label: "Watch", color: "#fbbf24", bg: "rgba(251,191,36,.12)" },
    { label: "Warning", color: "#fb923c", bg: "rgba(251,146,60,.13)" },
    { label: "Severe", color: "#fb7185", bg: "rgba(251,113,133,.14)" },
  ];

  const state = {
    dashboard: null,
    activeRegion: null,
    alerts: [],
    charts: {},
    loading: false,
    lastLoaded: 0,
    pollTimer: null,
    retryTimer: null,
  };

  const $ = (id) => document.getElementById(id);
  const nf = new Intl.NumberFormat("en-IN");

  /* ------------------------------------------------------------------ */
  /* Utilities                                                          */
  /* ------------------------------------------------------------------ */

  function esc(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[char]));
  }

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, Number(value) || 0));
  }

  function fmtNum(value, digits = 0) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
    return new Intl.NumberFormat("en-IN", {
      minimumFractionDigits: 0,
      maximumFractionDigits: digits,
    }).format(Number(value));
  }

  function fmtCompact(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
    return new Intl.NumberFormat("en-IN", {
      notation: "compact",
      maximumFractionDigits: 1,
    }).format(Number(value));
  }

  function parseStamp(value) {
    if (!value) return null;
    const raw = String(value);
    const stamp = raw.length <= 10 ? `${raw}T12:00:00+05:30` : raw;
    const parsed = new Date(stamp);
    return Number.isNaN(parsed.getTime()) ? null : parsed;
  }

  function fmtDate(value, extra = {}) {
    const parsed = parseStamp(value);
    if (!parsed) return "—";
    return new Intl.DateTimeFormat("en-IN", {
      timeZone: IST,
      day: "numeric",
      month: "short",
      ...extra,
    }).format(parsed);
  }

  function fmtDateTime(value) {
    const parsed = parseStamp(value);
    if (!parsed) return "—";
    return `${new Intl.DateTimeFormat("en-IN", {
      timeZone: IST,
      day: "numeric",
      month: "short",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    }).format(parsed)} IST`;
  }

  function timeAgo(value) {
    const parsed = parseStamp(value);
    if (!parsed) return "";
    const seconds = Math.round((Date.now() - parsed.getTime()) / 1000);
    if (seconds < 45) return "just now";
    if (seconds < 90) return "1 min ago";
    if (seconds < 3600) return `${Math.round(seconds / 60)} min ago`;
    if (seconds < 86400) return `${Math.round(seconds / 3600)} h ago`;
    if (seconds < 604800) return `${Math.round(seconds / 86400)} d ago`;
    return fmtDate(value);
  }

  function levelInfo(level) {
    return LEVELS[clamp(Number(level), 0, 3)];
  }

  function levelForScore(score) {
    const value = Number(score) || 0;
    if (value >= 85) return 3;
    if (value >= 65) return 2;
    if (value >= 40) return 1;
    return 0;
  }

  function riskStyle(level) {
    const info = levelInfo(level);
    return `--risk-color:${info.color};--risk-bg:${info.bg}`;
  }

  function riskPill(level, label) {
    const info = levelInfo(level);
    return `<span class="risk-pill" style="${riskStyle(level)}">${esc(label || info.label)}</span>`;
  }

  function hexToRgba(hex, alpha) {
    const clean = String(hex || "#22d3ee").replace("#", "");
    const value = clean.length === 3
      ? clean.split("").map((part) => part + part).join("")
      : clean.padEnd(6, "0").slice(0, 6);
    const number = parseInt(value, 16);
    const r = (number >> 16) & 255;
    const g = (number >> 8) & 255;
    const b = number & 255;
    return `rgba(${r},${g},${b},${alpha})`;
  }

  function setFormStatus(node, message, tone = "") {
    if (!node) return;
    node.textContent = message || "";
    node.className = `form-status${tone ? ` ${tone}` : ""}`.trim();
  }

  function toast(title, message, tone = "") {
    const container = $("toastContainer");
    if (!container) return;
    const node = document.createElement("div");
    node.className = `toast${tone ? ` ${tone}` : ""}`;
    node.setAttribute("role", "status");
    const heading = document.createElement("strong");
    heading.textContent = title;
    node.appendChild(heading);
    if (message) node.appendChild(document.createTextNode(message));
    node.addEventListener("click", () => node.remove());
    container.appendChild(node);
    window.setTimeout(() => {
      node.style.opacity = "0";
      node.style.transition = "opacity .25s ease";
      window.setTimeout(() => node.remove(), 260);
    }, 6200);
  }

  async function api(path, options = {}) {
    const init = { cache: "no-store", ...options };
    init.headers = {
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...(options.headers || {}),
    };
    const response = await fetch(path, init);
    let payload = null;
    try {
      payload = await response.json();
    } catch (_) {
      payload = null;
    }
    if (!response.ok) {
      const detail = payload && (payload.detail || payload.message);
      throw new Error(detail || `Request failed (${response.status})`);
    }
    return payload;
  }

  function savedEmail() {
    try {
      return localStorage.getItem("floodguard.email") || "";
    } catch (_) {
      return "";
    }
  }

  function rememberEmail(value) {
    try {
      localStorage.setItem("floodguard.email", value);
    } catch (_) {
      /* localStorage may be unavailable; not critical. */
    }
  }

  function setCardTone(node, level) {
    const card = node ? node.closest(".stat-card") : null;
    if (!card) return;
    card.classList.remove("warning", "severe");
    if (Number(level) === 2) card.classList.add("warning");
    if (Number(level) === 3) card.classList.add("severe");
  }

  function dischargePosition(discharge, thresholds) {
    const value = Number(discharge) || 0;
    const watch = Number(thresholds?.watch) || 0;
    const warning = Number(thresholds?.warning) || 0;
    const severe = Number(thresholds?.severe) || 0;
    if (value <= 0 || severe <= 0) return 2;
    if (value <= watch) return clamp((value / Math.max(watch, 1)) * 42, 2, 42);
    if (value <= warning) {
      return 42 + ((value - watch) / Math.max(warning - watch, 1)) * 25;
    }
    if (value <= severe) {
      return 67 + ((value - warning) / Math.max(severe - warning, 1)) * 19;
    }
    return 86 + Math.min(1, (value - severe) / Math.max(severe, 1)) * 12;
  }

  /* ------------------------------------------------------------------ */
  /* Chart.js setup                                                     */
  /* ------------------------------------------------------------------ */

  const riskBandsPlugin = {
    id: "riskBands",
    beforeDatasetsDraw(chart, _args, options) {
      if (!options || options.enabled !== true) return;
      const { ctx, chartArea } = chart;
      const y = chart.scales ? chart.scales.y : null;
      if (!ctx || !chartArea || !y) return;
      const bands = [
        [0, 40, "rgba(45,212,191,.04)"],
        [40, 65, "rgba(251,191,36,.05)"],
        [65, 85, "rgba(251,146,60,.055)"],
        [85, 100, "rgba(251,113,133,.06)"],
      ];
      ctx.save();
      ctx.beginPath();
      ctx.rect(chartArea.left, chartArea.top, chartArea.width, chartArea.height);
      ctx.clip();
      for (const [low, high, fill] of bands) {
        const top = y.getPixelForValue(high);
        const bottom = y.getPixelForValue(low);
        ctx.fillStyle = fill;
        ctx.fillRect(chartArea.left, top, chartArea.width, bottom - top);
      }
      ctx.strokeStyle = "rgba(148,190,220,.16)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 5]);
      for (const value of [40, 65, 85]) {
        const position = y.getPixelForValue(value);
        ctx.beginPath();
        ctx.moveTo(chartArea.left, position);
        ctx.lineTo(chartArea.right, position);
        ctx.stroke();
      }
      ctx.restore();
    },
  };

  const forecastShadePlugin = {
    id: "forecastShade",
    beforeDatasetsDraw(chart, _args, options) {
      if (!options || options.index === undefined || options.index === null) return;
      const { ctx, chartArea } = chart;
      const x = chart.scales ? chart.scales.x : null;
      if (!ctx || !chartArea || !x) return;
      const index = Number(options.index);
      if (!Number.isFinite(index) || index >= chart.data.labels.length) return;
      const boundary = x.getPixelForValue(index - 0.5);
      if (!Number.isFinite(boundary)) return;
      ctx.save();
      const left = Math.max(boundary, chartArea.left);
      ctx.fillStyle = "rgba(34,211,238,.035)";
      ctx.fillRect(left, chartArea.top, chartArea.right - left, chartArea.height);
      ctx.strokeStyle = "rgba(34,211,238,.28)";
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(boundary, chartArea.top);
      ctx.lineTo(boundary, chartArea.bottom);
      ctx.stroke();
      ctx.restore();
    },
  };

  if (window.Chart) {
    Chart.register(riskBandsPlugin);
    Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
    Chart.defaults.font.size = 10;
    Chart.defaults.color = "#8ea8bd";
    Chart.defaults.borderColor = "rgba(148,190,220,.08)";
    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    Chart.defaults.plugins.legend.labels.boxWidth = 7;
    Chart.defaults.plugins.legend.labels.boxHeight = 7;
    Chart.defaults.plugins.legend.labels.padding = 13;
    Chart.defaults.plugins.legend.labels.font = { size: 9 };
    Chart.defaults.plugins.tooltip.backgroundColor = "rgba(6,15,27,.96)";
    Chart.defaults.plugins.tooltip.borderColor = "rgba(148,190,220,.22)";
    Chart.defaults.plugins.tooltip.borderWidth = 1;
    Chart.defaults.plugins.tooltip.titleColor = "#e9f5ff";
    Chart.defaults.plugins.tooltip.bodyColor = "#c3d7e8";
    Chart.defaults.plugins.tooltip.footerColor = "#8ea8bd";
    Chart.defaults.plugins.tooltip.padding = 10;
    Chart.defaults.plugins.tooltip.cornerRadius = 7;
    Chart.defaults.plugins.tooltip.displayColors = true;
    Chart.defaults.plugins.tooltip.boxPadding = 3;
  }

  function destroyChart(name) {
    if (state.charts[name]) {
      state.charts[name].destroy();
      delete state.charts[name];
    }
  }

  function makeChart(name, canvasId, config) {
    if (!window.Chart) return null;
    const canvas = $(canvasId);
    if (!canvas) return null;
    destroyChart(name);
    state.charts[name] = new Chart(canvas, config);
    return state.charts[name];
  }

  const baseX = {
    grid: { color: "rgba(148,190,220,.055)", tickLength: 0 },
    border: { display: false },
    ticks: { color: "#617b91", font: { size: 9 }, maxRotation: 0, autoSkip: true, maxTicksLimit: 8 },
  };

  function dateTicks(labels, maxTicksLimit = 8) {
    const limit = window.innerWidth < 620 ? Math.min(4, maxTicksLimit) : maxTicksLimit;
    return {
      ...baseX.ticks,
      maxTicksLimit: limit,
      callback: (value, index) => fmtDate(labels[index] ?? value),
    };
  }

  /* ------------------------------------------------------------------ */
  /* Boot                                                               */
  /* ------------------------------------------------------------------ */

  document.addEventListener("DOMContentLoaded", init);

  function init() {
    startClock();
    bindEvents();

    const email = savedEmail();
    if (email) {
      if ($("alertEmail")) $("alertEmail").value = email;
      if ($("testEmail")) $("testEmail").value = email;
    }

    loadDashboard();
    state.pollTimer = window.setInterval(() => {
      if (document.visibilityState !== "hidden") loadDashboard({ quiet: true });
    }, REFRESH_POLL_MS);

    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "visible" && Date.now() - state.lastLoaded > REFRESH_POLL_MS) {
        loadDashboard({ quiet: true });
      }
    });
  }

  function startClock() {
    const clock = $("clock");
    if (!clock) return;
    const formatter = new Intl.DateTimeFormat("en-IN", {
      timeZone: IST,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
    const tick = () => {
      clock.textContent = `${formatter.format(new Date())} IST`;
    };
    tick();
    window.setInterval(tick, 1000);
  }

  function bindEvents() {
    $("refreshBtn")?.addEventListener("click", () => loadDashboard({ manual: true }));
    $("alertForm")?.addEventListener("submit", handleSubscribe);
    $("testEmailBtn")?.addEventListener("click", () => sendPipeline("test"));
    $("demoAlertBtn")?.addEventListener("click", () => sendPipeline("demo"));

    $("regionCards")?.addEventListener("click", (event) => {
      const card = event.target.closest(".region-card");
      if (card) selectRegion(card.dataset.region, true);
    });
    $("regionCards")?.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      const card = event.target.closest(".region-card");
      if (!card) return;
      event.preventDefault();
      selectRegion(card.dataset.region, true);
    });
    $("regionTabs")?.addEventListener("click", (event) => {
      const tab = event.target.closest(".region-tab");
      if (tab) selectRegion(tab.dataset.region, false);
    });
    $("regionTabs")?.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      const tab = event.target.closest(".region-tab");
      if (!tab) return;
      event.preventDefault();
      selectRegion(tab.dataset.region, false);
    });
  }

  /* ------------------------------------------------------------------ */
  /* Data loading                                                       */
  /* ------------------------------------------------------------------ */

  async function loadDashboard({ manual = false, quiet = false } = {}) {
    if (state.loading) return;
    const firstLoad = !state.dashboard;
    state.loading = true;
    const refreshButton = $("refreshBtn");
    if (refreshButton) {
      refreshButton.classList.add("spinning");
      refreshButton.disabled = true;
    }
    if (manual) $("dataStatusText") && ($("dataStatusText").textContent = "Syncing…");

    try {
      const payload = manual
        ? await api("/api/refresh", { method: "POST" })
        : await api("/api/dashboard");
      state.dashboard = payload;
      state.lastLoaded = Date.now();
      window.clearTimeout(state.retryTimer);
      renderAll(payload);
      loadAlerts();
      if (manual) {
        toast(
          "Live data refreshed",
          `${payload.summary.regions_monitored} basins analysed in ${payload.generated_in_seconds}s.`,
          "success"
        );
      } else if (firstLoad) {
        toast("Dashboard connected", "Live river intelligence is now streaming.", "success");
      }
    } catch (error) {
      console.error("Dashboard load failed", error);
      showLoadError(error);
    } finally {
      state.loading = false;
      if (refreshButton) {
        refreshButton.classList.remove("spinning");
        refreshButton.disabled = false;
      }
    }
  }

  function showLoadError(error) {
    const banner = $("offlineBanner");
    const status = $("dataStatus");
    const statusText = $("dataStatusText");
    if (status) {
      status.className = "status-chip demo";
      status.title = "The dashboard could not reach the local API";
    }
    if (statusText) statusText.textContent = "Offline";
    if (banner) {
      banner.classList.remove("hidden");
      banner.classList.add("demo");
      banner.textContent = `Could not reach the FloodGuard API: ${error.message}. Retrying automatically…`;
    }
    if (!state.dashboard) {
      const cards = $("regionCards");
      if (cards) {
        cards.innerHTML = `
          <div class="empty-state" style="grid-column:1/-1;">
            <div class="empty-icon">◎</div>
            <strong>Live basins are still loading</strong>
            <p>The first sync can take a minute while GloFAS data is fetched. The dashboard will retry automatically.</p>
          </div>`;
      }
    }
    window.clearTimeout(state.retryTimer);
    state.retryTimer = window.setTimeout(() => loadDashboard({ quiet: true }), 12_000);
  }

  async function loadAlerts() {
    try {
      const payload = await api("/api/alerts");
      state.alerts = Array.isArray(payload?.alerts) ? payload.alerts : [];
      renderAlertLog(state.alerts);
    } catch (error) {
      console.warn("Could not load alert log", error);
      if (!state.alerts.length && state.dashboard?.alerts) {
        state.alerts = state.dashboard.alerts;
        renderAlertLog(state.alerts);
      }
    }
  }

  /* ------------------------------------------------------------------ */
  /* Main render                                                        */
  /* ------------------------------------------------------------------ */

  function renderAll(data) {
    renderStatus(data);
    renderSummary(data);
    renderNational(data);
    renderRegionChecks(data.regions);
    renderRegionBoard(data);
    renderForecastChart(data);
    renderAlertSettings(data);
    renderAlertLog(state.alerts.length ? state.alerts : data.alerts || []);
    renderModel(data);

    const side = data.data_source || {};
    const last = $("lastUpdated");
    if (last) {
      last.textContent = fmtDateTime(data.generated_at);
      last.title = timeAgo(data.generated_at);
    }
    const cadence = $("refreshCadence");
    if (cadence) cadence.textContent = `Every ${side.refresh_minutes ?? 30} min`;

    const regions = data.regions || [];
    const preferred = state.activeRegion
      || data.summary?.peak_region_id
      || regions[0]?.id
      || null;
    if (preferred) selectRegion(preferred, false, true);
  }

  function renderStatus(data) {
    const source = data.data_source || {};
    const kind = ["live", "cached", "demo"].includes(source.status) ? source.status : "cached";
    const chip = $("dataStatus");
    const text = $("dataStatusText");
    if (chip) {
      chip.className = `status-chip ${kind}`;
      chip.title = `${source.provider || "Open-Meteo"} · ${source.live_regions ?? "?"}/${source.attempted_regions ?? "?"} basins live`;
    }
    if (text) {
      text.textContent = kind === "live"
        ? "Live · GloFAS"
        : kind === "cached"
          ? "Cached data"
          : "Demo fallback";
    }

    const banner = $("offlineBanner");
    if (!banner) return;
    const errors = Array.isArray(source.errors) ? source.errors : [];
    if (errors.length) {
      banner.classList.remove("hidden");
      banner.classList.toggle("demo", kind === "demo");
      const names = errors.map((item) => `${item.region} (${item.status})`).join(", ");
      banner.textContent = `Some basins are running on cached or fallback data: ${names}. Live values will resume at the next scheduled sync.`;
    } else {
      banner.classList.add("hidden");
    }
  }

  function renderSummary(data) {
    const summary = data.summary || {};
    const regions = data.regions || [];
    const score = Math.round(Number(summary.max_risk_score) || 0);
    const level = clamp(Number(summary.max_level) || 0, 0, 3);
    const info = levelInfo(level);

    $("statBasins").textContent = summary.regions_monitored ?? regions.length;
    $("statBasinsSub").textContent = regions.map((region) => region.name).join(" · ")
      || "Assam · Bihar · Kerala · Mumbai · West Bengal";

    $("statWarnings").textContent = summary.active_warnings ?? 0;
    $("statWarningsSub").textContent = summary.active_warnings
      ? `${summary.max_label || info.label} peak · ${summary.alert_threshold || "Warning"} threshold`
      : `No basin above ${summary.alert_threshold || "Warning"} in the alert horizon`;
    setCardTone($("statWarnings"), level);

    $("statPeak").textContent = score;
    $("statPeakSub").textContent = summary.peak_region
      ? `${summary.peak_region} · ${summary.max_label || info.label}`
      : "Across the active horizon";
    setCardTone($("statPeak"), level);

    $("statSubscribers").textContent = fmtNum(summary.subscribers ?? 0);
    const simulated = Number(summary.alerts_simulated) || 0;
    const failed = Number(summary.alerts_failed) || 0;
    $("statSubscribersSub").textContent = simulated || failed
      ? `${simulated} simulated · ${failed} failed`
      : "Email early-warning list";

    const test = data.model?.test;
    if (test && typeof test.macro_f1 === "number") {
      $("statF1").textContent = `${(test.macro_f1 * 100).toFixed(1)}%`;
      $("statF1Sub").textContent = `Accuracy ${(Number(test.accuracy || 0) * 100).toFixed(1)}% · ${fmtNum(test.samples || 0)} hold-out windows`;
    } else {
      $("statF1").textContent = "—";
      $("statF1Sub").textContent = "Model artifact not available";
    }
  }

  function renderNational(data) {
    const summary = data.summary || {};
    const score = clamp(Math.round(Number(summary.max_risk_score) || 0), 0, 100);
    const level = clamp(Number(summary.max_level) || 0, 0, 3);
    const info = levelInfo(level);
    const ring = $("nationalRing");

    if (ring) {
      ring.style.setProperty("--pct", score);
      ring.style.setProperty("--ring-color", info.color);
    }
    $("nationalScore").textContent = Number.isFinite(score) ? score : "—";

    const pill = $("nationalPill");
    if (pill) {
      pill.textContent = summary.max_label || info.label;
      pill.style.setProperty("--risk-color", info.color);
      pill.style.setProperty("--risk-bg", info.bg);
    }

    const active = Number(summary.active_warnings) || 0;
    const count = Number(summary.regions_monitored) || (data.regions || []).length;
    const peak = summary.peak_region || "—";
    let narrative;
    if (active > 0) {
      narrative = `${active} of ${count} basins are in the ${(summary.max_label || info.label).toLowerCase()} band. ${peak} carries the strongest signal at ${score}/100.`;
    } else if (score >= 40) {
      narrative = `${peak} is on watch at ${score}/100. The other monitored corridors remain inside their normal operating range.`;
    } else {
      narrative = `All ${count} basins are inside their normal operating range. ${peak} has the highest relative signal at ${score}/100.`;
    }
    $("nationalSummary").textContent = narrative;
    $("nationalPeak").textContent = peak === "—" ? "—" : `${peak} · ${summary.max_label || info.label}`;
  }

  function renderRegionChecks(regions) {
    const container = $("regionChecks");
    if (!container || !Array.isArray(regions) || !regions.length) return;
    if (container.dataset.ready === "1") return;
    container.innerHTML = regions.map((region) => `
      <label class="check-label">
        <input type="checkbox" value="${esc(region.id)}" checked>
        <span>${esc(region.name)}</span>
      </label>`).join("");
    container.dataset.ready = "1";
  }

  function renderRegionBoard(data) {
    const container = $("regionCards");
    const tabs = $("regionTabs");
    if (!container) return;
    const regions = data.regions || [];
    if (!regions.length) {
      container.innerHTML = `<div class="empty-state" style="grid-column:1/-1;"><strong>No basin data available</strong><p>The API returned no basin payloads.</p></div>`;
      if (tabs) tabs.innerHTML = "";
      return;
    }

    container.innerHTML = regions.map((region) => {
      const current = region.current || {};
      const thresholds = region.thresholds || {};
      const observedLevel = clamp(Number(current.observed_level) || 0, 0, 3);
      const observedScore = Math.round(Number(current.observed_score) || 0);
      const next = current.next_24h || region.forecast?.[0] || {};
      const peak = region.peak || {};
      const peakLevel = clamp(Number(peak.level) || 0, 0, 3);
      const series = (region.series || [])
        .filter((item) => !item.is_forecast && Number(item.discharge) > 0)
        .slice(-24);
      const maxDischarge = Math.max(...series.map((item) => Number(item.discharge) || 0), 1);
      const bars = series.map((item) => {
        const height = clamp(((Number(item.discharge) || 0) / maxDischarge) * 100, 5, 100);
        return `<i style="height:${height.toFixed(0)}%"></i>`;
      }).join("");
      const position = dischargePosition(current.discharge, thresholds);
      const info = levelInfo(observedLevel);

      return `
        <article class="region-card" style="--region-color:${esc(region.color)}" data-region="${esc(region.id)}"
                 role="button" tabindex="0" aria-label="Inspect ${esc(region.name)} basin details">
          <div class="card-top">
            <span class="region-label"><i></i>${esc(region.name)}</span>
            ${riskPill(observedLevel, current.observed_label || info.label)}
          </div>
          <h3>${esc(region.river)}</h3>
          <div class="region-sub">${esc(region.station)} · ${esc(region.area)}</div>
          <div class="card-main">
            <div class="mini-ring" style="--pct:${clamp(observedScore, 0, 100)};--ring-color:${info.color}" title="Observed risk score">
              <span>${Number.isFinite(observedScore) ? observedScore : "—"}</span>
            </div>
            <div class="card-stats">
              <div class="card-stat"><span>Discharge</span><strong>${fmtNum(current.discharge)} m³/s</strong></div>
              <div class="card-stat"><span>Rain · 24h</span><strong>${fmtNum(current.rain_24h, 1)} mm</strong></div>
              <div class="card-stat"><span>Next 24h</span><strong>${esc(next.label || "—")} · ${Math.round(Number(next.risk_score) || 0)}</strong></div>
            </div>
          </div>
          <div class="mini-bars" title="Discharge, last 24 days" aria-hidden="true">${bars}</div>
          <div class="threshold-line">
            <div class="track" title="Observed discharge relative to warning thresholds">
              <i style="left:${position.toFixed(1)}%"></i>
              <span>Warning ${fmtCompact(thresholds.warning)}</span>
            </div>
          </div>
          <div class="card-foot">
            <div class="card-outlook">
              <span>Peak · ${fmtDate(peak.date)}</span>
              <strong style="color:${levelInfo(peakLevel).color}">${esc(peak.label || "—")} · ${Math.round(Number(peak.risk_score) || 0)}/100</strong>
            </div>
            <span class="view-link">Inspect →</span>
          </div>
        </article>`;
    }).join("");

    if (tabs) {
      tabs.innerHTML = regions.map((region) => `
        <button class="region-tab" type="button" role="tab" aria-selected="false"
                data-region="${esc(region.id)}" style="--tab-color:${esc(region.color)}">
          <i></i>${esc(region.name)}
        </button>`).join("");
    }
  }

  /* ------------------------------------------------------------------ */
  /* Detail view                                                        */
  /* ------------------------------------------------------------------ */

  function selectRegion(regionId, scroll, force = false) {
    const data = state.dashboard;
    if (!data) return;
    const region = (data.regions || []).find((item) => item.id === regionId) || (data.regions || [])[0];
    if (!region) return;
    const changed = state.activeRegion !== region.id;
    state.activeRegion = region.id;

    document.querySelectorAll(".region-card").forEach((card) => {
      card.classList.toggle("selected", card.dataset.region === region.id);
    });
    document.querySelectorAll(".region-tab").forEach((tab) => {
      const active = tab.dataset.region === region.id;
      tab.classList.toggle("active", active);
      tab.setAttribute("aria-selected", String(active));
    });

    if (changed || force) renderDetail(region);
    if (scroll) {
      const section = $("detail");
      if (section) section.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  function renderDetail(region) {
    const current = region.current || {};
    const thresholds = region.thresholds || {};
    const active = region.active || {};
    const peak = region.peak || {};
    const forecast = region.forecast || [];
    const series = region.series || [];
    const ensemble = region.ensemble || state.dashboard?.model?.ensemble || {};

    $("detailKicker").textContent = `${region.state} · ${region.area}`;
    $("detailTitle").textContent = `${region.river} at ${region.station}`;
    $("detailDescription").textContent = region.description || "";

    const activeLevel = clamp(Number(active.level) || 0, 0, 3);
    const activeSource = active.source === "observed"
      ? `observed ${fmtDate(active.date)}`
      : `forecast for ${fmtDate(active.date)}`;
    $("detailBadge").innerHTML = `
      ${riskPill(activeLevel, active.label || levelInfo(activeLevel).label)}
      <span class="detail-badge-sub">${esc(activeSource)} · ${Math.round(Number(active.score) || 0)}/100</span>`;

    $("detailStats").innerHTML = `
      <div class="detail-stat">
        <span>River discharge</span>
        <strong>${fmtNum(current.discharge)} m³/s</strong>
        <small>${fmtNum(current.discharge_ratio_30, 2)}× the 30-day median</small>
      </div>
      <div class="detail-stat">
        <span>Rainfall · 7 days</span>
        <strong>${fmtNum(current.rain_7d, 1)} mm</strong>
        <small>${fmtNum(current.rain_24h, 1)} mm in the last 24 hours</small>
      </div>
      <div class="detail-stat">
        <span>Soil moisture</span>
        <strong>${current.soil_moisture == null ? "—" : Number(current.soil_moisture).toFixed(2)} m³/m³</strong>
        <small>Root-zone saturation</small>
      </div>
      <div class="detail-stat">
        <span>Peak outlook</span>
        <strong>${Math.round(Number(peak.risk_score) || 0)}/100</strong>
        <small>${esc(peak.label || "—")} · ${fmtDate(peak.date)}</small>
      </div>`;

    const next = current.next_24h || forecast[0] || {};
    const probabilities = next.probabilities || {};
    const maxProbability = Math.max(...LEVELS.map((level) => Number(probabilities[level.label]) || 0), 1);
    $("probabilityBars").innerHTML = LEVELS.map((level) => {
      const value = Number(probabilities[level.label]) || 0;
      const width = clamp((value / maxProbability) * 100, value > 0 ? 3 : 0, 100);
      return `
        <div class="prob-row">
          <span>${level.label}</span>
          <div class="prob-track"><i style="width:${width.toFixed(1)}%;--prob-color:${level.color}"></i></div>
          <strong>${value.toFixed(1)}%</strong>
        </div>`;
    }).join("");

    const modelWeight = Math.round((Number(ensemble.model_weight) || 0.4) * 100);
    const hydroWeight = Math.round((Number(ensemble.hydrology_weight) || 0.6) * 100);
    $("ensembleNote").textContent = `Next 24 h: ${modelWeight}% BiLSTM + ${hydroWeight}% GloFAS prior`
      + (next.model_risk_score != null ? ` · model ${Math.round(next.model_risk_score)}` : "")
      + (next.hydrology_risk_score != null ? `, hydrology ${Math.round(next.hydrology_risk_score)}` : "")
      + `. Ensemble ${Math.round(Number(next.risk_score) || 0)}/100.`;

    const driverIcons = ["R", "Q", "Δ", "S"];
    $("drivers").innerHTML = (region.drivers || []).map((driver, index) => `
      <div class="driver-item" data-impact="${esc(driver.impact)}">
        <span class="driver-icon">${esc(driverIcons[index] || "•")}</span>
        <div class="driver-copy">
          <strong>${esc(driver.label)} <b>${esc(driver.value)}</b></strong>
          <p>${esc(driver.detail)}</p>
        </div>
      </div>`).join("");

    $("forecastTableBody").innerHTML = forecast.length ? forecast.map((item) => {
      const level = clamp(Number(item.level) || 0, 0, 3);
      const info = levelInfo(level);
      return `
        <tr class="future-row">
          <td class="date-cell">${fmtDate(item.date, { weekday: "short" })}</td>
          <td>${fmtNum(item.rain, 1)} mm</td>
          <td>${fmtNum(item.discharge)} m³/s</td>
          <td>${Math.round(Number(item.alert_probability) || 0)}%</td>
          <td>
            <span class="risk-cell">
              <i class="risk-dot" style="--risk-color:${info.color}"></i>
              ${esc(item.label || info.label)} <span class="mini-score">${Math.round(Number(item.risk_score) || 0)}</span>
            </span>
          </td>
        </tr>`;
    }).join("") : `<tr><td colspan="5" class="empty-cell">No forecast available for this basin.</td></tr>`;

    const ensembleChip = $("ensembleChip");
    if (ensembleChip) {
      ensembleChip.textContent = `BiLSTM ${modelWeight}% + GloFAS ${hydroWeight}%`;
    }

    renderBasinChart(region, series);
    renderBasinRiskChart(region, series);
  }

  function renderBasinChart(region, series) {
    if (!window.Chart || !series.length) return;
    const labels = series.map((item) => item.date);
    const observedCount = series.filter((item) => !item.is_forecast).length;
    const rain = series.map((item) => (item.rain == null ? null : Number(item.rain)));
    const discharge = series.map((item) => (item.discharge == null ? null : Number(item.discharge)));
    const warning = Number(region.thresholds?.warning) || null;

    makeChart("basin", "basinChart", {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: "Rainfall (mm)",
            data: rain,
            yAxisID: "rain",
            backgroundColor: "rgba(56,189,248,.34)",
            hoverBackgroundColor: "rgba(56,189,248,.55)",
            borderRadius: 2,
            barPercentage: 0.72,
            categoryPercentage: 0.9,
            order: 3,
          },
          {
            label: "Discharge (m³/s)",
            data: discharge,
            yAxisID: "y",
            type: "line",
            borderColor: region.color,
            backgroundColor: "transparent",
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 3,
            tension: 0.28,
            spanGaps: true,
            order: 1,
          },
          ...(warning ? [{
            label: "Warning threshold",
            data: labels.map(() => warning),
            yAxisID: "y",
            type: "line",
            borderColor: "rgba(251,146,60,.65)",
            borderDash: [5, 4],
            borderWidth: 1,
            pointRadius: 0,
            fill: false,
            order: 2,
            threshold: true,
          }] : []),
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 450 },
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { position: "bottom", labels: { filter: (item) => item.text !== "Warning threshold" } },
          tooltip: {
            filter: (item) => item.dataset.label !== "Warning threshold",
            callbacks: {
              title: (items) => fmtDate(labels[items[0].dataIndex], { weekday: "short" }),
              label: (item) => {
                if (item.dataset.yAxisID === "rain") return `Rainfall: ${fmtNum(item.parsed.y, 1)} mm`;
                return `Discharge: ${fmtNum(item.parsed.y)} m³/s`;
              },
            },
          },
          forecastShade: { index: observedCount },
        },
        scales: {
          x: { ...baseX, ticks: dateTicks(labels, 7), stacked: false },
          y: {
            position: "left",
            grid: { color: "rgba(148,190,220,.055)" },
            border: { display: false },
            ticks: { color: "#617b91", font: { size: 9 }, callback: (value) => fmtCompact(value) },
          },
          rain: {
            position: "right",
            beginAtZero: true,
            grid: { display: false },
            border: { display: false },
            ticks: { color: "rgba(56,189,248,.65)", font: { size: 9 }, maxTicksLimit: 4 },
          },
        },
      },
      plugins: [forecastShadePlugin],
    });
  }

  function renderBasinRiskChart(region, series) {
    if (!window.Chart || !series.length) return;
    const labels = series.map((item) => item.date);
    const observedCount = series.filter((item) => !item.is_forecast).length;
    const risk = series.map((item) => Number(item.risk_score) || 0);
    const levels = series.map((item) => clamp(Number(item.level) || 0, 0, 3));

    makeChart("basinRisk", "basinRiskChart", {
      type: "line",
      data: {
        labels,
        datasets: [{
          label: "Flood risk",
          data: risk,
          borderColor: region.color,
          backgroundColor: hexToRgba(region.color, 0.12),
          fill: true,
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.28,
          spanGaps: true,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 450 },
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { display: false },
          riskBands: { enabled: true },
          tooltip: {
            callbacks: {
              title: (items) => fmtDate(labels[items[0].dataIndex], { weekday: "short" }),
              label: (item) => {
                const level = levelInfo(levels[item.dataIndex]);
                return `Risk: ${Math.round(item.parsed.y)}/100 · ${level.label}`;
              },
              afterBody: (items) => {
                const entry = series[items[0].dataIndex];
                return entry?.is_forecast ? "Model forecast" : "Historical model path";
              },
            },
          },
          forecastShade: { index: observedCount },
        },
        scales: {
          x: { ...baseX, ticks: dateTicks(labels, 7) },
          y: {
            min: 0,
            max: 100,
            grid: { color: "rgba(148,190,220,.055)" },
            border: { display: false },
            ticks: { color: "#617b91", font: { size: 9 }, stepSize: 25 },
          },
        },
      },
      plugins: [forecastShadePlugin],
    });
  }

  /* ------------------------------------------------------------------ */
  /* Forecast chart                                                     */
  /* ------------------------------------------------------------------ */

  function renderForecastChart(data) {
    if (!window.Chart) return;
    const regions = data.regions || [];
    const firstForecast = regions[0]?.forecast || [];
    if (!firstForecast.length) return;
    const labels = firstForecast.map((item) => item.date);

    const datasets = regions.map((region) => ({
      label: region.name,
      data: (region.forecast || []).map((item) => Number(item.risk_score) || 0),
      borderColor: region.color,
      backgroundColor: hexToRgba(region.color, 0.07),
      pointBackgroundColor: region.color,
      pointBorderColor: "rgba(6,15,27,.9)",
      pointBorderWidth: 1,
      pointRadius: 2,
      pointHoverRadius: 4,
      borderWidth: 2,
      tension: 0.34,
      spanGaps: true,
    }));

    datasets.push({
      label: "Warning · 65",
      data: labels.map(() => 65),
      borderColor: "rgba(251,146,60,.45)",
      borderDash: [5, 4],
      borderWidth: 1,
      pointRadius: 0,
      fill: false,
      threshold: true,
    });
    datasets.push({
      label: "Severe · 85",
      data: labels.map(() => 85),
      borderColor: "rgba(251,113,133,.40)",
      borderDash: [5, 4],
      borderWidth: 1,
      pointRadius: 0,
      fill: false,
      threshold: true,
    });

    $("forecastRange").textContent = `${fmtDate(labels[0])} → ${fmtDate(labels[labels.length - 1])}`;

    makeChart("forecast", "riskForecastChart", {
      type: "line",
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 500 },
        interaction: { mode: "index", intersect: false },
        plugins: {
          riskBands: { enabled: true },
          legend: {
            position: "bottom",
            labels: { filter: (item) => !item.text.startsWith("Warning") && !item.text.startsWith("Severe") },
          },
          tooltip: {
            filter: (item) => !item.dataset.threshold,
            callbacks: {
              title: (items) => fmtDate(labels[items[0].dataIndex], { weekday: "long" }),
              label: (item) => {
                const level = levelInfo(levelForScore(item.parsed.y));
                return `${item.dataset.label}: ${Math.round(item.parsed.y)}/100 · ${level.label}`;
              },
            },
          },
        },
        scales: {
          x: { ...baseX, ticks: dateTicks(labels, 8) },
          y: {
            min: 0,
            max: 100,
            grid: { color: "rgba(148,190,220,.055)" },
            border: { display: false },
            ticks: { color: "#617b91", font: { size: 9 }, stepSize: 25 },
          },
        },
      },
    });
  }

  /* ------------------------------------------------------------------ */
  /* Alerts                                                            */
  /* ------------------------------------------------------------------ */

  function renderAlertSettings(data) {
    const settings = data.alert_settings || {};
    const threshold = settings.threshold_label || "Warning";
    const horizon = settings.horizon_days ?? 3;
    $("alertThresholdLabel").textContent = threshold;
    $("alertHorizonLabel").textContent = `${horizon}-day`;

    const smtp = settings.email_mode === "smtp";
    const chip = $("emailModeChip");
    if (chip) chip.className = `status-chip ${smtp ? "live" : "cached"}`;
    $("emailModeText").textContent = smtp ? "SMTP delivery" : "Simulated delivery";
    $("pipelineMode").textContent = smtp ? "Live SMTP" : "Simulated · outbox";
    const note = $("pipelineNote");
    if (note) {
      note.innerHTML = smtp
        ? "Live email delivery is configured. Alerts are sent by SMTP and mirrored in the activity log below."
        : "SMTP is not configured. Every alert is rendered to an HTML file in <code>data/outbox</code> and logged below, so the full pipeline stays testable.";
    }
  }

  function renderAlertLog(alerts) {
    const list = Array.isArray(alerts) ? alerts : [];
    const log = $("alertLog");
    const count = $("alertLogCount");
    if (count) count.textContent = `${list.length} ${list.length === 1 ? "entry" : "entries"}`;
    if (!log) return;

    if (!list.length) {
      log.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">◎</div>
          <strong>No alerts dispatched yet</strong>
          <p>When a basin reaches Warning, the delivery log appears here. Use the demo button above to preview the full path.</p>
        </div>`;
      return;
    }

    log.innerHTML = list.map((alert) => {
      const level = clamp(Number(alert.level) || 0, 0, 3);
      const info = levelInfo(level);
      const status = String(alert.status || "unknown");
      const statusLabel = {
        sent: "Sent",
        simulated: "Simulated",
        partial: "Partial",
        failed: "Failed",
        no_subscribers: "No subscribers",
      }[status] || status.replace(/_/g, " ");
      const details = alert.details || {};
      const recipients = Number(alert.recipients) || 0;
      const meta = [];
      if (alert.target_date) meta.push(`target ${fmtDate(alert.target_date)}`);
      if (alert.risk_score != null) meta.push(`score ${Math.round(Number(alert.risk_score))}`);
      if (recipients) meta.push(`${recipients} recipient${recipients === 1 ? "" : "s"}`);
      if (details.demo) meta.push("demo");
      return `
        <div class="alert-row">
          <span class="alert-level-dot" style="--risk-color:${info.color}" title="${esc(alert.level_label)}"></span>
          <div class="alert-main">
            <strong title="${esc(alert.headline)}">${esc(alert.headline)}</strong>
            <span>${esc(alert.region_name)} · ${esc(meta.join(" · "))} · ${esc(alert.level_label)}</span>
          </div>
          <div class="alert-meta">
            <span class="alert-status ${esc(status)}">${esc(statusLabel)}</span>
            <span class="alert-time" title="${esc(fmtDateTime(alert.created_at))}">${esc(timeAgo(alert.created_at))}</span>
          </div>
        </div>`;
    }).join("");
  }

  async function handleSubscribe(event) {
    event.preventDefault();
    const emailInput = $("alertEmail");
    const email = (emailInput?.value || "").trim();
    const selected = Array.from(document.querySelectorAll("#regionChecks input:checked")).map((input) => input.value);
    const status = $("subscribeStatus");

    if (!EMAIL_RE.test(email)) {
      setFormStatus(status, "Enter a valid email address, for example you@example.com.", "error");
      emailInput?.focus();
      return;
    }
    if (!selected.length) {
      setFormStatus(status, "Select at least one basin to watch.", "error");
      return;
    }

    const button = $("subscribeBtn");
    if (button) {
      button.disabled = true;
      button.textContent = "Subscribing…";
    }
    setFormStatus(status, "Saving your alert preferences…");

    try {
      const result = await api("/api/subscribe", {
        method: "POST",
        body: JSON.stringify({ email, regions: selected }),
      });
      rememberEmail(result.email);
      if ($("testEmail") && !$("testEmail").value) $("testEmail").value = result.email;

      const createdAt = parseStamp(result.created_at);
      const isNew = createdAt && Date.now() - createdAt.getTime() < 30_000;
      if (isNew) {
        const counter = $("statSubscribers");
        if (counter) counter.textContent = fmtNum((parseInt(counter.textContent.replace(/[^\d]/g, ""), 10) || 0) + 1);
      }

      setFormStatus(
        status,
        `Subscribed ${result.email} for ${result.regions.length} basin${result.regions.length === 1 ? "" : "s"}. Alerts arrive when the ensemble reaches Warning.`,
        "success"
      );
      toast("Alert subscription saved", `${result.email} is watching ${result.regions.length} basin(s).`, "success");
    } catch (error) {
      setFormStatus(status, error.message, "error");
      toast("Subscription failed", error.message, "error");
    } finally {
      if (button) {
        button.disabled = false;
        button.textContent = "Subscribe";
      }
    }
  }

  async function sendPipeline(kind) {
    const testInput = $("testEmail");
    const email = (testInput?.value || $("alertEmail")?.value || savedEmail()).trim();
    const status = $("testStatus");
    const region = state.activeRegion || state.dashboard?.regions?.[0]?.id || "assam-brahmaputra";
    const button = kind === "test" ? $("testEmailBtn") : $("demoAlertBtn");

    if (!EMAIL_RE.test(email)) {
      setFormStatus(status, "Enter a valid email address for the pipeline test.", "error");
      testInput?.focus();
      return;
    }

    const original = button ? button.textContent : "";
    if (button) {
      button.disabled = true;
      button.textContent = kind === "test" ? "Sending…" : "Dispatching…";
    }
    setFormStatus(status, kind === "test" ? "Sending test email…" : "Dispatching demo Warning…");

    try {
      const result = await api(`/api/alerts/${kind}`, {
        method: "POST",
        body: JSON.stringify({ email, region_id: region }),
      });
      rememberEmail(email);
      if ($("alertEmail") && !$("alertEmail").value) $("alertEmail").value = email;
      const simulated = result.status === "simulated" || result.mode === "simulation";
      const message = result.message
        || (simulated
          ? `Simulated email saved to data/outbox for ${result.email}.`
          : `Email delivered to ${result.email}.`);
      setFormStatus(status, message, simulated ? "simulated" : "success");
      toast(
        kind === "test" ? "Test email processed" : "Demo Warning dispatched",
        simulated ? "Simulation mode — see data/outbox and the activity log." : message,
        simulated ? "warn" : "success"
      );
      await loadAlerts();
    } catch (error) {
      setFormStatus(status, error.message, "error");
      toast(kind === "test" ? "Test email failed" : "Demo alert failed", error.message, "error");
    } finally {
      if (button) {
        button.disabled = false;
        button.textContent = original;
      }
    }
  }

  /* ------------------------------------------------------------------ */
  /* Model card                                                         */
  /* ------------------------------------------------------------------ */

  function renderModel(data) {
    const model = data.model || {};
    const chip = $("modelStatusChip");
    const trained = model.status === "trained";

    if (chip) chip.textContent = trained ? "BiLSTM · trained" : "Hydrology fallback";
    $("architectureTitle").textContent = trained
      ? "BiLSTM · temporal attention"
      : "Hydrological fallback";
    $("architectureText").textContent = trained
      ? `Two-layer bidirectional LSTM with a learned basin embedding and additive attention pooling, trained ${fmtDate(model.trained_at, { year: "numeric" })}.`
      : model.note || "The deep-learning artifact is unavailable; risk is driven by GloFAS discharge thresholds.";

    if (!trained) {
      $("modelMeta").innerHTML = `<span class="meta-chip">Mode <b>heuristic</b></span>`;
      $("modelMetrics").innerHTML = "";
      $("confusionMatrix").innerHTML = `<p class="model-note" style="margin:0;">Train the model with <code>python -m app.bootstrap --train</code> to populate hold-out metrics.</p>`;
      $("featureChips").innerHTML = "";
      $("modelNote").textContent = "Predictions currently use the GloFAS discharge prior only.";
      $("modelEpochNote").textContent = "No training history";
      destroyChart("training");
      return;
    }

    const test = model.test || {};
    const validation = model.validation || {};
    const split = model.split || {};
    const history = model.history || {};
    const featureNames = model.feature_names || [];
    const featureLabels = model.feature_labels || {};

    $("modelMeta").innerHTML = [
      ["Trained", fmtDate(model.trained_at, { year: "numeric" })],
      ["Window", `${model.sequence_length || 14}d × ${featureNames.length || 19}`],
      ["Best epoch", `${model.best_epoch || "—"} / ${model.epochs_run || "—"}`],
      ["Train windows", fmtNum(split.train_samples || 0)],
      ["Test windows", fmtNum(test.samples || 0)],
      ["Data mode", `${model.data_mode || "cached"} history`],
    ].map(([label, value]) => `<span class="meta-chip">${esc(label)} <b>${esc(value)}</b></span>`).join("");

    $("modelMetrics").innerHTML = `
      <div class="model-metric"><span>Test accuracy</span><strong>${(Number(test.accuracy || 0) * 100).toFixed(1)}%</strong></div>
      <div class="model-metric"><span>Macro-F1</span><strong>${(Number(test.macro_f1 || 0) * 100).toFixed(1)}%</strong></div>
      <div class="model-metric"><span>Test loss</span><strong>${Number(test.loss || 0).toFixed(3)}</strong></div>
      <div class="model-metric"><span>Val macro-F1</span><strong>${(Number(validation.macro_f1 || 0) * 100).toFixed(1)}%</strong></div>`;

    const matrix = Array.isArray(test.confusion_matrix) ? test.confusion_matrix : [];
    if (matrix.length) {
      const totals = matrix.map((row) => row.reduce((sum, value) => sum + Number(value || 0), 0));
      $("confusionMatrix").innerHTML = `
        <table class="confusion-table">
          <thead><tr><th>Actual ↓ / Predicted →</th>${LEVELS.map((level) => `<th>${level.label}</th>`).join("")}</tr></thead>
          <tbody>
            ${matrix.map((row, rowIndex) => `
              <tr>
                <td>${LEVELS[rowIndex]?.label || rowIndex}</td>
                ${row.map((value, columnIndex) => {
                  const total = totals[rowIndex] || 1;
                  const share = Number(value || 0) / total;
                  const diagonal = rowIndex === columnIndex;
                  const alpha = value > 0 ? 0.06 + share * 0.5 : 0;
                  const color = diagonal ? "45,212,191" : "251,113,133";
                  return `<td class="confusion-cell" style="background:rgba(${color},${alpha.toFixed(2)});${diagonal && share > 0.35 ? "font-weight:700;" : ""}"
                              title="${LEVELS[rowIndex]?.label} → ${LEVELS[columnIndex]?.label}: ${value} windows (${(share * 100).toFixed(1)}%)">${value}</td>`;
                }).join("")}
              </tr>`).join("")}
          </tbody>
        </table>`;
    } else {
      $("confusionMatrix").innerHTML = `<p class="model-note" style="margin:0;">Confusion matrix is not available in the metadata.</p>`;
    }

    $("featureChips").innerHTML = featureNames.map((name) => {
      const label = featureLabels[name] || name.replace(/_/g, " ");
      return `<span class="feature-chip" title="${esc(name)}">${esc(label)}</span>`;
    }).join("");

    const testStart = split.val_end ? ` Test windows begin after ${fmtDate(split.val_end, { year: "numeric" })}.` : "";
    $("modelNote").textContent = `${model.notes || "Labels use region-specific discharge quantiles; validation and test periods are strictly later than training."}${testStart}`;
    $("modelEpochNote").textContent = `Best epoch ${model.best_epoch ?? "—"} of ${model.epochs_run ?? "—"}`;

    renderTrainingChart(history, model.best_epoch);
  }

  function renderTrainingChart(history, bestEpoch) {
    if (!window.Chart) return;
    const epochs = Array.isArray(history?.epoch) ? history.epoch : [];
    if (!epochs.length) {
      destroyChart("training");
      return;
    }
    const train = history.train_loss || [];
    const validation = history.val_loss || [];
    const f1 = (history.val_macro_f1 || []).map((value) => Number(value) * 100);

    makeChart("training", "trainingChart", {
      type: "line",
      data: {
        labels: epochs,
        datasets: [
          {
            label: "Train loss",
            data: train,
            borderColor: "#22d3ee",
            backgroundColor: "rgba(34,211,238,.08)",
            borderWidth: 1.8,
            pointRadius: 0,
            pointHoverRadius: 3,
            tension: 0.3,
            fill: true,
            yAxisID: "y",
          },
          {
            label: "Validation loss",
            data: validation,
            borderColor: "#fbbf24",
            backgroundColor: "transparent",
            borderWidth: 1.7,
            borderDash: [4, 3],
            pointRadius: 0,
            pointHoverRadius: 3,
            tension: 0.3,
            yAxisID: "y",
          },
          {
            label: "Val macro-F1",
            data: f1,
            borderColor: "rgba(45,212,191,.55)",
            backgroundColor: "transparent",
            borderWidth: 1.2,
            pointRadius: 0,
            pointHoverRadius: 3,
            tension: 0.3,
            yAxisID: "f1",
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 400 },
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { position: "bottom", labels: { font: { size: 8.5 } } },
          tooltip: {
            callbacks: {
              title: (items) => {
                const epoch = epochs[items[0].dataIndex];
                return `Epoch ${epoch}${bestEpoch && Number(epoch) === Number(bestEpoch) ? " · best" : ""}`;
              },
              label: (item) => item.dataset.yAxisID === "f1"
                ? `Val macro-F1: ${item.parsed.y.toFixed(1)}%`
                : `${item.dataset.label}: ${item.parsed.y.toFixed(3)}`,
            },
          },
        },
        scales: {
          x: {
            ...baseX,
            grid: { display: false },
            ticks: { ...baseX.ticks, maxTicksLimit: 8, callback: (value, index) => `E${epochs[index] ?? value}` },
          },
          y: {
            position: "left",
            grid: { color: "rgba(148,190,220,.055)" },
            border: { display: false },
            ticks: { color: "#617b91", font: { size: 9 } },
          },
          f1: {
            position: "right",
            min: 0,
            max: 100,
            grid: { display: false },
            border: { display: false },
            ticks: { color: "rgba(45,212,191,.55)", font: { size: 8 }, callback: (value) => `${value}%` },
          },
        },
      },
    });
  }
})();
