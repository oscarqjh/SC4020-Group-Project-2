(() => {
  const instructionsEl = document.getElementById("instructions-text");
  const chatHistoryEl = document.getElementById("chat-history");
  const chatForm = document.getElementById("chat-form");
  const chatInput = document.getElementById("chat-input");
  const tabButtons = document.querySelectorAll(".tab-button");
  const tabContents = document.querySelectorAll(".tab-content");

  const symptomGrid = document.getElementById("symptom-grid");
  const resetSymptomsBtn = document.getElementById("reset-symptoms");
  const runSymptomAnalysisBtn = document.getElementById("run-symptom-analysis");
  const formsOutput = document.getElementById("forms-output");
  const breastForm = document.getElementById("breast-form");

  let sessionId = "";
  const selectedSymptoms = new Set();

  // Tab switching
  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabButtons.forEach((b) => b.classList.remove("active"));
      tabContents.forEach((content) => content.classList.remove("active"));
      btn.classList.add("active");
      const target = document.getElementById(btn.dataset.target);
      target.classList.add("active");
    });
  });

  async function createSession() {
    const res = await fetch("/api/session", { method: "POST" });
    if (!res.ok) throw new Error("Failed to create session");
    const data = await res.json();
    sessionId = data.session_id;
    instructionsEl.textContent = data.instructions;
    appendSystemMessage("Session initialised. Ask your question to begin.");
  }

  function appendUserMessage(text) {
    const wrapper = document.createElement("div");
    wrapper.className = "chat-message user";
    wrapper.textContent = `You: ${text}`;
    chatHistoryEl.appendChild(wrapper);
    chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;
  }

  function formatAgentResponse(text) {
    const escaped = text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
    const withStrong = escaped.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    return withStrong.replace(/\n/g, "<br>");
  }

  function appendSystemMessage(text) {
    const wrapper = document.createElement("div");
    wrapper.className = "chat-message agent";
    wrapper.innerHTML = formatAgentResponse(text);
    chatHistoryEl.appendChild(wrapper);
    chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;
  }

  async function sendChatMessage(message) {
    appendUserMessage(message);
    appendSystemMessage("Analyzing...");

    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message }),
    });

    if (!res.ok) {
      appendSystemMessage("Error: Unable to process the request.");
      return;
    }

    const data = await res.json();
    appendSystemMessage(data.response);
  }

  chatForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const message = chatInput.value.trim();
    if (!message) return;
    chatInput.value = "";
    sendChatMessage(message);
  });

  // Symptom checklist rendering
  function renderSymptomGrid() {
    SYMPTOM_LIST.forEach((symptom) => {
      const chip = document.createElement("div");
      chip.className = "symptom-chip";
      chip.textContent = symptom.replace(/_/g, " ");
      chip.dataset.value = symptom;
      chip.addEventListener("click", () => toggleSymptom(symptom, chip));
      symptomGrid.appendChild(chip);
    });
  }

  function toggleSymptom(symptom, element) {
    if (selectedSymptoms.has(symptom)) {
      selectedSymptoms.delete(symptom);
      element.classList.remove("selected");
    } else {
      selectedSymptoms.add(symptom);
      element.classList.add("selected");
    }
  }

  function resetSymptoms() {
    selectedSymptoms.clear();
    document
      .querySelectorAll(".symptom-chip.selected")
      .forEach((chip) => chip.classList.remove("selected"));
  }

  resetSymptomsBtn.addEventListener("click", resetSymptoms);

  runSymptomAnalysisBtn.addEventListener("click", async () => {
    if (!sessionId) {
      appendSystemMessage("Session not ready yet.");
      return;
    }

    const symptoms = Array.from(selectedSymptoms);
    appendFormsOutput("Analyzing selected symptoms...");

    const res = await fetch("/api/disease/checklist", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, symptoms }),
    });

    if (!res.ok) {
      appendFormsOutput("Error: Unable to run disease analysis.");
      return;
    }

    const data = await res.json();
    appendFormsOutput(data.response);
  });

  function appendFormsOutput(text) {
    formsOutput.textContent = text;
  }

  breastForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!sessionId) {
      appendFormsOutput("Session not ready yet.");
      return;
    }

    const formData = new FormData(breastForm);
    const measurements = {};
    formData.forEach((value, key) => {
      if (value !== "") {
        measurements[key] = Number(value);
      }
    });

    if (Object.keys(measurements).length === 0) {
      appendFormsOutput("Please enter at least one measurement.");
      return;
    }

    appendFormsOutput("Analyzing measurements...");

    const res = await fetch("/api/breast/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, measurements }),
    });

    if (!res.ok) {
      appendFormsOutput("Error: Unable to run breast cancer analysis.");
      return;
    }

    const data = await res.json();
    appendFormsOutput(data.response);
  });

  // Initialise
  renderSymptomGrid();
  createSession().catch(() => {
    appendSystemMessage("Failed to initialise session. Please reload the page.");
  });
})();
