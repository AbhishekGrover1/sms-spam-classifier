// SMS Spam Classifier front end.
// Author: Abhishek Grover

const SAMPLES = {
  spam: "WINNER!! You have been selected to receive a £900 prize. Call 09061701234 now or text CLAIM to 81010 to collect. T&Cs apply.",
  ham: "Hey, are we still on for dinner tonight? I can book a table for 8.",
};

const input = document.getElementById("message");
const classifyButton = document.getElementById("classify");
const notice = document.getElementById("notice");
const result = document.getElementById("result");
const verdict = document.getElementById("verdict");
const meterFill = document.getElementById("meter-fill");
const score = document.getElementById("score");
const signals = document.getElementById("signals");
const chips = document.getElementById("chips");

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// 100.0% or 0.0% would suggest the model is certain, which it never is
function formatPercent(percent) {
  if (percent > 99.9) return ">99.9%";
  if (percent < 0.1) return "<0.1%";
  return `${percent.toFixed(1)}%`;
}

function showNotice(text) {
  notice.textContent = text;
}

function setBusy(busy) {
  classifyButton.disabled = busy;
  classifyButton.textContent = busy ? "Classifying" : "Classify";
  classifyButton.setAttribute("aria-busy", String(busy));
}

function showResult(data) {
  const isSpam = data.label === "spam";
  const percent = data.spam_probability * 100;

  result.dataset.tone = isSpam ? "spam" : "ham";
  verdict.textContent = isSpam ? "Spam" : "Not spam";

  const figure = document.createElement("strong");
  figure.textContent = formatPercent(percent);
  score.replaceChildren(figure, " spam probability");

  // words come from what the user typed, so they go in as text, never as html
  chips.replaceChildren();
  const strongest = Math.max(...data.signals.map((signal) => signal.weight), 0);
  for (const signal of data.signals) {
    const chip = document.createElement("li");
    chip.className = "chip";
    chip.textContent = signal.word;
    chip.style.opacity = (0.78 + 0.22 * (signal.weight / strongest)).toFixed(2);
    chips.append(chip);
  }
  signals.hidden = data.signals.length === 0;

  // start from an empty meter and unlit card so the animation plays on every result
  result.hidden = false;
  result.classList.remove("is-lit");
  meterFill.style.width = "0%";
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      result.classList.add("is-lit");
      meterFill.style.width = `${percent.toFixed(1)}%`;
    });
  });

  result.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "nearest" });
}

async function classify() {
  const message = input.value.trim();
  showNotice("");

  if (!message) {
    showNotice("Enter a message to classify.");
    input.focus();
    return;
  }

  setBusy(true);
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
      signal: AbortSignal.timeout(20000),
    });
    if (response.status === 422) {
      showNotice("Messages must be between 1 and 1000 characters.");
      return;
    }
    if (!response.ok) {
      throw new Error(`Server answered ${response.status}`);
    }
    showResult(await response.json());
  } catch (error) {
    showNotice("Could not reach the classifier. Check your connection and try again.");
  } finally {
    setBusy(false);
  }
}

classifyButton.addEventListener("click", classify);

input.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    classify();
  }
});

// the example buttons fill the box and classify right away
document.querySelectorAll("[data-sample]").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = SAMPLES[button.dataset.sample];
    classify();
  });
});

// the glow on the surfaces follows the cursor
document.querySelectorAll(".panel, .result").forEach((surface) => {
  surface.addEventListener("pointermove", (event) => {
    const box = surface.getBoundingClientRect();
    surface.style.setProperty("--mx", `${event.clientX - box.left}px`);
    surface.style.setProperty("--my", `${event.clientY - box.top}px`);
  });
});

// the keys lean toward the cursor
if (!reduceMotion) {
  document.querySelectorAll(".key").forEach((key) => {
    key.addEventListener("pointermove", (event) => {
      const box = key.getBoundingClientRect();
      const x = (event.clientX - box.left) / box.width - 0.5;
      const y = (event.clientY - box.top) / box.height - 0.5;
      key.style.setProperty("--ry", `${x * 16}deg`);
      key.style.setProperty("--rx", `${-y * 16}deg`);
    });
    key.addEventListener("pointerleave", () => {
      key.style.setProperty("--rx", "0deg");
      key.style.setProperty("--ry", "0deg");
    });
  });
}
