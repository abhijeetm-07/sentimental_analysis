/**
 * EmotionAI — Frontend Logic
 */

const API_BASE_URL =
  window.location.origin && window.location.origin.startsWith("http")
    ? window.location.origin
    : "http://127.0.0.1:8000";

const EMOJIS = { sadness:"😢", joy:"😄", love:"❤️", anger:"😠", fear:"😨", surprise:"😲" };
const BAR_COLORS = { sadness:"var(--c-sadness)", joy:"var(--c-joy)", love:"var(--c-love)", anger:"var(--c-anger)", fear:"var(--c-fear)", surprise:"var(--c-surprise)" };

const SAMPLES = {
  joy:      "After months of hard work, I finally got the job I had been dreaming about.",
  sadness:  "I tried to stay positive, but losing someone I cared about has left me feeling empty.",
  anger:    "I spent hours preparing everything, and they cancelled the meeting without even telling me.",
  fear:     "My heart started racing when I heard footsteps following me in the empty street.",
  surprise: "I opened the box expecting something ordinary, but I couldn't believe what was inside.",
  love:     "Even on my worst days, being around her makes me feel understood and cared for."
};

let busy = false;

document.addEventListener("DOMContentLoaded", () => {
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => document.querySelectorAll(s);

  const textInput       = $("#textInput");
  const charCount       = $("#charCount");
  const validation      = $("#validationMsg");
  const analyzeBtn      = $("#analyzeBtn");
  const clearBtn        = $("#clearBtn");
  const errorCard       = $("#errorCard");
  const errorTitle      = $("#errorTitle");
  const errorDesc       = $("#errorDesc");
  const dismissError    = $("#dismissError");
  const resultsSection  = $("#resultsSection");
  const resultEmoji     = $("#resultEmoji");
  const resultName      = $("#resultEmotionName");
  const resultConf      = $("#resultConfidence");
  const probList        = $("#probabilitiesList");
  const interpText      = $("#interpretationText");
  const recapText       = $("#recapText");
  const statusDot       = $("#statusDot");
  const statusText      = $("#statusText");

  // ---- Health ----
  async function checkHealth() {
    try {
      const r = await fetch(`${API_BASE_URL}/health`);
      if (!r.ok) throw new Error(r.status);
      const d = await r.json();
      setStatus(d.model_loaded ? "online" : "offline", d.model_loaded ? "Online" : "Model offline");
    } catch {
      setStatus("warn", "Unreachable");
    }
  }

  function setStatus(state, label) {
    statusDot.className = "status-dot " + state;
    statusText.textContent = label;
  }

  checkHealth();
  setInterval(checkHealth, 30000);

  // ---- Input ----
  textInput.addEventListener("input", () => {
    charCount.textContent = `${textInput.value.length} / 2000`;
    textInput.style.height = "auto";
    textInput.style.height = Math.min(textInput.scrollHeight, 320) + "px";
    if (textInput.value.length) hideValidation();
  });

  textInput.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); analyze(); }
  });

  clearBtn.addEventListener("click", () => {
    textInput.value = "";
    charCount.textContent = "0 / 2000";
    textInput.style.height = "auto";
    hideValidation(); hideError();
    textInput.focus();
  });

  $$(".pill").forEach((p) =>
    p.addEventListener("click", () => {
      const emo = p.dataset.emotion;
      if (SAMPLES[emo]) {
        textInput.value = SAMPLES[emo];
        textInput.dispatchEvent(new Event("input"));
        hideValidation(); hideError();
        textInput.focus();
      }
    })
  );

  analyzeBtn.addEventListener("click", analyze);
  dismissError.addEventListener("click", hideError);

  function showValidation(msg) { validation.textContent = msg; validation.classList.add("show"); textInput.focus(); }
  function hideValidation()    { validation.textContent = ""; validation.classList.remove("show"); }
  function hideError()         { errorCard.style.display = "none"; }

  function showError(title, desc) {
    errorTitle.textContent = title || "Unable to analyze.";
    errorDesc.textContent  = desc  || "Make sure the FastAPI server is running.";
    errorCard.style.display = "flex";
  }

  // ---- Analyze ----
  async function analyze() {
    if (busy) return;
    const text = textInput.value.trim();
    if (!text) { showValidation("Enter some text first."); return; }

    busy = true;
    analyzeBtn.disabled = true;
    analyzeBtn.classList.add("loading");
    hideError();

    try {
      const r = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      });

      if (!r.ok) {
        let msg = `Server error (${r.status})`;
        try { const e = await r.json(); if (e.detail) msg = e.detail; } catch {}
        throw new Error(msg);
      }

      const data = await r.json();
      showResult(data);
    } catch (err) {
      console.error("[EmotionAI]", err);
      showError(err.message, "The model may still be loading. Try again shortly.");
    } finally {
      busy = false;
      analyzeBtn.disabled = false;
      analyzeBtn.classList.remove("loading");
    }
  }

  // ---- Results ----
  function showResult(data) {
    const emo   = (data.predicted_emotion || "").toLowerCase();
    const conf  = data.confidence || 0;
    const probs = data.all_probabilites || {};

    resultEmoji.textContent = EMOJIS[emo] || "🎭";
    resultName.textContent  = emo.toUpperCase();
    animateCounter(resultConf, conf * 100, "%");

    // Probabilities sorted descending
    probList.innerHTML = "";
    const sorted = Object.keys(probs).sort((a, b) => probs[b] - probs[a]);
    sorted.forEach((e) => {
      const pct = (probs[e] * 100).toFixed(1);
      const row = document.createElement("div");
      row.className = "prob-row";
      row.innerHTML = `
        <span class="prob-label">${e}</span>
        <div class="prob-track"><div class="prob-bar" data-pct="${pct}" style="background:${BAR_COLORS[e] || "var(--text-3)"}"></div></div>
        <span class="prob-val">${pct}%</span>`;
      probList.appendChild(row);
    });

    requestAnimationFrame(() => {
      probList.querySelectorAll(".prob-bar").forEach((b) => { b.style.width = b.dataset.pct + "%"; });
    });

    interpText.textContent = `The model predicts ${emo.toUpperCase()} as the dominant emotion with ${(conf * 100).toFixed(1)} % confidence.`;
    recapText.textContent  = `"${data.text || textInput.value}"`;

    resultsSection.style.display = "block";
    resultsSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function animateCounter(el, target, suffix) {
    const start = performance.now();
    const dur = 600;
    (function tick(now) {
      const p = Math.min((now - start) / dur, 1);
      const ease = 1 - Math.pow(1 - p, 3);
      el.textContent = (ease * target).toFixed(1) + " " + suffix;
      if (p < 1) requestAnimationFrame(tick);
    })(start);
  }
});
