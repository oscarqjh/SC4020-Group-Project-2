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
  
  // Additional measurement fields (excluding the 4 required ones)
  const ADDITIONAL_MEASUREMENTS = [
    { name: "radius_mean", label: "Radius (Mean)" },
    { name: "radius_se", label: "Radius (SE)" },
    { name: "radius_worst", label: "Radius (Worst)" },
    { name: "texture_mean", label: "Texture (Mean)" },
    { name: "texture_se", label: "Texture (SE)" },
    { name: "texture_worst", label: "Texture (Worst)" },
    { name: "perimeter_mean", label: "Perimeter (Mean)" },
    { name: "perimeter_se", label: "Perimeter (SE)" },
    { name: "perimeter_worst", label: "Perimeter (Worst)" },
    { name: "area_worst", label: "Area (Worst)" },
    { name: "smoothness_mean", label: "Smoothness (Mean)" },
    { name: "smoothness_se", label: "Smoothness (SE)" },
    { name: "smoothness_worst", label: "Smoothness (Worst)" },
    { name: "compactness_mean", label: "Compactness (Mean)" },
    { name: "compactness_se", label: "Compactness (SE)" },
    { name: "compactness_worst", label: "Compactness (Worst)" },
    { name: "concavity_mean", label: "Concavity (Mean)" },
    { name: "concavity_se", label: "Concavity (SE)" },
    { name: "concavity_worst", label: "Concavity (Worst)" },
    { name: "concave_points_worst", label: "Concave Points (Worst)" },
    { name: "symmetry_mean", label: "Symmetry (Mean)" },
    { name: "symmetry_se", label: "Symmetry (SE)" },
    { name: "symmetry_worst", label: "Symmetry (Worst)" },
    { name: "fractal_dimension_mean", label: "Fractal Dimension (Mean)" },
    { name: "fractal_dimension_se", label: "Fractal Dimension (SE)" },
    { name: "fractal_dimension_worst", label: "Fractal Dimension (Worst)" },
  ];
  
  const selectedMeasurements = new Set();

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

  // Breast cancer measurement dropdown functionality
  const measurementDropdown = document.getElementById("measurement-dropdown");
  const measurementSearch = document.getElementById("measurement-search");
  const additionalFieldsGroup = document.getElementById("additional-fields-group");
  const additionalFields = document.getElementById("additional-fields");

  function renderMeasurementDropdown(measurementsList = ADDITIONAL_MEASUREMENTS) {
    measurementDropdown.innerHTML = "";
    
    measurementsList.forEach((measurement) => {
      const item = document.createElement("div");
      item.className = "measurement-checkbox-item";
      
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.id = `checkbox-${measurement.name}`;
      checkbox.checked = selectedMeasurements.has(measurement.name);
      checkbox.addEventListener("change", () => toggleMeasurement(measurement));
      
      const label = document.createElement("label");
      label.htmlFor = `checkbox-${measurement.name}`;
      label.textContent = measurement.label;
      
      item.appendChild(checkbox);
      item.appendChild(label);
      measurementDropdown.appendChild(item);
    });
  }

  function toggleMeasurement(measurement) {
    const checkbox = document.getElementById(`checkbox-${measurement.name}`);
    if (checkbox && checkbox.checked) {
      // Checkbox was checked - add measurement
      selectedMeasurements.add(measurement.name);
      addMeasurementField(measurement);
    } else {
      // Checkbox was unchecked - remove measurement
      selectedMeasurements.delete(measurement.name);
      removeMeasurementField(measurement.name);
    }
    updateAdditionalFieldsVisibility();
  }

  function addMeasurementField(measurement) {
    // Check if field already exists
    if (document.querySelector(`input[name="${measurement.name}"]`)) {
      return;
    }
    
    const label = document.createElement("label");
    label.innerHTML = `
      ${measurement.label}
      <input type="number" step="0.0000001" name="${measurement.name}" />
    `;
    additionalFields.appendChild(label);
  }

  function removeMeasurementField(measurementName) {
    const input = document.querySelector(`input[name="${measurementName}"]`);
    if (input && input.closest("label")) {
      input.closest("label").remove();
    }
  }

  function updateAdditionalFieldsVisibility() {
    if (selectedMeasurements.size > 0) {
      additionalFieldsGroup.style.display = "block";
    } else {
      additionalFieldsGroup.style.display = "none";
    }
  }

  // Search functionality for measurements
  measurementSearch.addEventListener("input", (e) => {
    const searchTerm = e.target.value.toLowerCase().trim();
    if (searchTerm === "") {
      renderMeasurementDropdown(ADDITIONAL_MEASUREMENTS);
    } else {
      const filtered = ADDITIONAL_MEASUREMENTS.filter((m) =>
        m.label.toLowerCase().includes(searchTerm) ||
        m.name.toLowerCase().includes(searchTerm)
      );
      renderMeasurementDropdown(filtered);
    }
  });

  // Handle form reset
  breastForm.addEventListener("reset", () => {
    selectedMeasurements.clear();
    additionalFields.innerHTML = "";
    updateAdditionalFieldsVisibility();
    measurementSearch.value = "";
    renderMeasurementDropdown(ADDITIONAL_MEASUREMENTS);
  });

  // Initialise
  renderSymptomGrid();
  renderMeasurementDropdown();
  createSession().catch(() => {
    appendSystemMessage("Failed to initialise session. Please reload the page.");
  });
})();
