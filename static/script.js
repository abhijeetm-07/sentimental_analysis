/**
 * EmotionAI — Frontend Logic
 * Connects to FastAPI backend (/health and /predict)
 * Matches the complete HTML structure and neutral zinc dark palette.
 */

const API_BASE_URL =
  window.location.origin && window.location.origin.startsWith("http")
    ? window.location.origin
    : "http://127.0.0.1:8000";

const EMOJIS = {
  sadness: "😢",
  joy: "😄",
  love: "❤️",
  anger: "😠",
  fear: "😨",
  surprise: "😲"
};

const EMOTION_COLORS = {
  sadness: "var(--color-sadness)",
  joy: "var(--color-joy)",
  love: "var(--color-love)",
  anger: "var(--color-anger)",
  fear: "var(--color-fear)",
  surprise: "var(--color-surprise)"
};

const SAMPLE_TEXTS = {
  joy: "After months of hard work, I finally achieved something I had been dreaming about for years!",
  sadness: "I tried my best to keep everything together, but losing what mattered most has left me feeling empty.",
  love: "Even on the hardest days, having people who understand and care for you makes everything worthwhile.",
  anger: "I spent weeks preparing the report and they discarded it entirely without even reviewing the findings.",
  fear: "Walking down that unlit alley late at night, every sudden sound sent a chill straight down my spine.",
  surprise: "I opened the plain cardboard parcel expecting spare parts, but inside was an unexpected gift."
};

let isAnalyzing = false;

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const textInput          = document.getElementById("textInput");
  const charCounter        = document.getElementById("charCounter");
  const inlineValidation   = document.getElementById("inlineValidation");
  const clearBtn           = document.getElementById("clearBtn");
  const analyzeBtn         = document.getElementById("analyzeBtn");
  const examplePills       = document.querySelectorAll(".example-pill");

  const healthBadge        = document.getElementById("healthBadge");
  const statusDot          = document.getElementById("statusDot");
  const statusText         = document.getElementById("statusText");

  const errorCard          = document.getElementById("errorCard");
  const errorMessage       = document.getElementById("errorMessage");
  const dismissErrorBtn    = document.getElementById("dismissErrorBtn");

  const resultsSection     = document.getElementById("resultsSection");
  const predictionHighlight= document.getElementById("predictionHighlight");
  const resultEmoji        = document.getElementById("resultEmoji");
  const resultEmotionName  = document.getElementById("resultEmotionName");
  const resultConfidence   = document.getElementById("resultConfidence");
  const probabilitiesList  = document.getElementById("probabilitiesList");
  const interpretationCard = document.getElementById("interpretationCard");
  const interpretationText = document.getElementById("interpretationText");
  const recapText          = document.getElementById("recapText");

  // ==========================================
  // 1. Health Status Polling
  // ==========================================
  async function checkServerHealth() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`, {
        method: "GET",
        headers: { "Accept": "application/json" }
      });

      if (!response.ok) {
        throw new Error(`Status ${response.status}`);
      }

      const data = await response.json();
      if (data.model_loaded) {
        setHealthStatus("status-online", "Online");
      } else {
        setHealthStatus("status-warning", "Model Loading");
      }
    } catch (err) {
      setHealthStatus("status-offline", "Offline");
    }
  }

  function setHealthStatus(statusClass, label) {
    if (!statusDot || !statusText) return;
    statusDot.className = `status-dot ${statusClass}`;
    statusText.textContent = label;
  }

  // Initial check and periodic polling
  checkServerHealth();
  setInterval(checkServerHealth, 30000);

  // ==========================================
  // 2. Input Handling & Character Counter
  // ==========================================
  function updateCharCounter() {
    const length = textInput.value.length;
    charCounter.textContent = `${length} / 2000`;

    // Auto-adjust textarea height up to 320px
    textInput.style.height = "auto";
    textInput.style.height = `${Math.min(textInput.scrollHeight, 320)}px`;

    if (length > 0) {
      clearValidation();
    }
  }

  textInput.addEventListener("input", updateCharCounter);

  // Keyboard shortcut: Ctrl+Enter or Cmd+Enter to analyze
  textInput.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      handleAnalyze();
    }
  });

  // Clear button
  clearBtn.addEventListener("click", () => {
    textInput.value = "";
    charCounter.textContent = "0 / 2000";
    textInput.style.height = "auto";
    clearValidation();
    hideError();
    textInput.focus();
  });

  // Example pills
  examplePills.forEach((pill) => {
    pill.addEventListener("click", () => {
      const emotion = pill.getAttribute("data-emotion");
      if (SAMPLE_TEXTS[emotion]) {
        textInput.value = SAMPLE_TEXTS[emotion];
        updateCharCounter();
        clearValidation();
        hideError();
        textInput.focus();
      }
    });
  });

  // ==========================================
  // 3. Error and Validation Helpers
  // ==========================================
  function showValidation(msg) {
    inlineValidation.textContent = msg;
    inlineValidation.classList.add("active");
    textInput.focus();
  }

  function clearValidation() {
    inlineValidation.textContent = "";
    inlineValidation.classList.remove("active");
  }

  function showError(message) {
    if (errorMessage) {
      errorMessage.textContent = message || "Make sure the FastAPI server is running on http://127.0.0.1:8000.";
    }
    if (errorCard) {
      errorCard.style.display = "flex";
      errorCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }

  function hideError() {
    if (errorCard) {
      errorCard.style.display = "none";
    }
  }

  if (dismissErrorBtn) {
    dismissErrorBtn.addEventListener("click", hideError);
  }

  // ==========================================
  // 4. Prediction Execution
  // ==========================================
  analyzeBtn.addEventListener("click", handleAnalyze);

  async function handleAnalyze() {
    if (isAnalyzing) return;

    const text = textInput.value.trim();
    if (!text) {
      showValidation("Please enter a sentence to analyze.");
      return;
    }

    clearValidation();
    hideError();
    setLoadingState(true);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json"
        },
        body: JSON.stringify({ text })
      });

      if (!response.ok) {
        let errorMsg = `Server error (${response.status})`;
        try {
          const errData = await response.json();
          if (errData.detail) errorMsg = errData.detail;
        } catch (_) {}
        throw new Error(errorMsg);
      }

      const result = await response.json();
      displayResults(result, text);
    } catch (err) {
      console.error("[EmotionAI Error]:", err);
      showError(err.message || "Failed to communicate with FastAPI backend.");
    } finally {
      setLoadingState(false);
    }
  }

  function setLoadingState(loading) {
    isAnalyzing = loading;
    analyzeBtn.disabled = loading;
    if (loading) {
      analyzeBtn.classList.add("loading");
    } else {
      analyzeBtn.classList.remove("loading");
    }
  }

  // ==========================================
  // 5. Render Results
  // ==========================================
  function displayResults(data, inputText) {
    const rawEmotion = (data.predicted_emotion || "").toLowerCase();
    const confidenceVal = typeof data.confidence === "number" ? data.confidence : 0;
    const confidencePct = (confidenceVal * 100).toFixed(1);
    const probabilities = data.all_probabilites || data.all_probabilities || {};

    const emotionColor = EMOTION_COLORS[rawEmotion] || "var(--accent-dim)";
    const emoji = EMOJIS[rawEmotion] || "🎭";

    // Update Highlight card
    if (resultEmoji) resultEmoji.textContent = emoji;
    if (resultEmotionName) {
      resultEmotionName.textContent = rawEmotion.toUpperCase();
    }
    if (predictionHighlight) {
      predictionHighlight.style.setProperty("--highlight-accent", emotionColor);
    }

    // Animate confidence number
    if (resultConfidence) {
      animateCounter(resultConfidence, confidenceVal * 100, "%");
    }

    // Render Probability Distribution
    if (probabilitiesList) {
      probabilitiesList.innerHTML = "";
      const sortedKeys = Object.keys(probabilities).sort(
        (a, b) => (probabilities[b] || 0) - (probabilities[a] || 0)
      );

      sortedKeys.forEach((key) => {
        const val = probabilities[key] || 0;
        const pct = (val * 100).toFixed(1);
        const rowColor = EMOTION_COLORS[key.toLowerCase()] || "var(--accent-dim)";
        const rowEmoji = EMOJIS[key.toLowerCase()] || "•";

        const row = document.createElement("div");
        row.className = "prob-row";
        row.innerHTML = `
          <span class="prob-label"><span>${rowEmoji}</span> ${key}</span>
          <div class="prob-track">
            <div class="prob-bar" data-pct="${pct}" style="--bar-color: ${rowColor}; width: 0%;"></div>
          </div>
          <span class="prob-value">${pct}%</span>
        `;
        probabilitiesList.appendChild(row);
      });

      // Smoothly expand bars
      requestAnimationFrame(() => {
        setTimeout(() => {
          probabilitiesList.querySelectorAll(".prob-bar").forEach((bar) => {
            bar.style.width = `${bar.dataset.pct}%`;
          });
        }, 50);
      });
    }

    // Interpretation text
    if (interpretationText) {
      const dominantPct = confidencePct;
      let note = "";
      if (confidenceVal >= 0.8) {
        note = `Strong positive identification. The Bidirectional GRU network detected unambiguous semantic and contextual indicators characteristic of ${rawEmotion}.`;
      } else if (confidenceVal >= 0.5) {
        note = `Moderate confidence. The BiGRU synthesized context across both directions, identifying ${rawEmotion} as the most probable emotional tone with subtle secondary nuances.`;
      } else {
        note = `Mixed emotional sentiment. The model identifies ${rawEmotion} as primary, though the sequence contains overlapping emotional cues.`;
      }
      interpretationText.textContent = `The model classified this sentence as "${rawEmotion.toUpperCase()}" with ${dominantPct}% confidence. ${note}`;
    }

    // Input text recap
    if (recapText) {
      recapText.textContent = `"${data.text || inputText}"`;
    }

    // Make results visible and smoothly scroll to them
    if (resultsSection) {
      resultsSection.style.display = "block";
      resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  // Helper for numeric counter ease-out animation
  function animateCounter(element, target, suffix = "%") {
    const start = performance.now();
    const duration = 650;
    (function frame(currentTime) {
      const elapsed = currentTime - start;
      const progress = Math.min(elapsed / duration, 1);
      const easeOut = 1 - Math.pow(1 - progress, 3);
      const currentVal = (easeOut * target).toFixed(1);
      element.textContent = `${currentVal}${suffix}`;
      if (progress < 1) {
        requestAnimationFrame(frame);
      }
    })(start);
  }
});
