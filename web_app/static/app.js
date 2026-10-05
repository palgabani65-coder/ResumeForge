document.addEventListener("DOMContentLoaded", () => {
  const resumeInput = document.getElementById("resume-input");
  const charCounter = document.getElementById("char-counter");
  const classifyBtn = document.getElementById("classify-btn");
  const clearBtn = document.getElementById("clear-btn");
  const emptyState = document.getElementById("empty-state");
  const resultContainer = document.getElementById("result-container");
  const topCategoryName = document.getElementById("top-category-name");
  const topConfidencePill = document.getElementById("top-confidence-pill");
  const probabilityList = document.getElementById("probability-list");
  const keywordsCloud = document.getElementById("keywords-cloud");
  const cleanTextSnippet = document.getElementById("clean-text-snippet");
  const confidenceIndicator = document.getElementById("confidence-indicator");
  const benchmarkTableBody = document.getElementById("benchmark-table-body");
  const fileInput = document.getElementById("resume-file");
  const uploadZone = document.getElementById("upload-zone");
  const uploadStatus = document.getElementById("upload-status");
  // Enforce pure Light Theme (FlipSentiment Nordic Style)
  document.documentElement.classList.remove("dark");
  try { localStorage.removeItem("rf_theme"); } catch (e) {}

  let sampleResumes = {
    software_engineer: { text: "Senior Software Engineer with seven years of experience building Python, FastAPI, React, PostgreSQL, Docker, AWS, and cloud microservices. Designed CI/CD pipelines and scalable production services." },
    accountant: { text: "CPA and Senior Accountant with eight years of experience in audit, general ledger reconciliation, GAAP compliance, tax filings, balance sheets, QuickBooks, SAP, and financial reporting." },
    hr_manager: { text: "Human Resources Manager experienced in talent acquisition, employee relations, onboarding, Workday HRIS, payroll, performance management, and employment compliance." },
    chef: { text: "Executive Chef with ten years of culinary leadership. Experienced in menu development, kitchen operations, food safety, inventory control, pastry, and team training." }
  };

  // Update counters
  function updateCounters() {
    const text = resumeInput.value;
    const chars = text.length;
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    charCounter.textContent = `${chars} characters | ${words} words`;
  }

  resumeInput.addEventListener("input", updateCounters);

  function setUploadStatus(message, type = "") {
    uploadStatus.textContent = message;
    uploadStatus.className = `upload-status ${type}`;
  }

  async function extractResumeText(file) {
    const extension = file.name.split(".").pop().toLowerCase();
    if (["txt", "md"].includes(extension)) return file.text();

    const buffer = await file.arrayBuffer();
    if (extension === "pdf") {
      if (!window.pdfjsLib) throw new Error("PDF reader is still loading. Please try again.");
      const pdf = await window.pdfjsLib.getDocument({ data: buffer }).promise;
      const pages = [];
      for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += 1) {
        const page = await pdf.getPage(pageNumber);
        const content = await page.getTextContent();
        pages.push(content.items.map(item => item.str).join(" "));
      }
      return pages.join("\n");
    }

    if (extension === "docx") {
      if (!window.mammoth) throw new Error("DOCX reader is still loading. Please try again.");
      const result = await window.mammoth.extractRawText({ arrayBuffer: buffer });
      return result.value;
    }
    throw new Error("Please choose a PDF, DOCX, TXT, or MD resume.");
  }

  async function handleUploadedFile(file) {
    if (!file) return;
    if (file.size > 5 * 1024 * 1024) {
      setUploadStatus("This file is over the 5 MB limit.", "is-error");
      return;
    }
    setUploadStatus(`Reading ${file.name}...`);
    try {
      const extractedText = (await extractResumeText(file)).trim();
      if (extractedText.length < 20) throw new Error("No readable resume text was found in this file.");
      resumeInput.value = extractedText;
      updateCounters();
      setUploadStatus(`${file.name} loaded. Review the text, then select Analyze & Classify.`, "is-success");
    } catch (error) {
      setUploadStatus(error.message, "is-error");
    } finally {
      fileInput.value = "";
    }
  }

  fileInput.addEventListener("change", event => handleUploadedFile(event.target.files[0]));
  ["dragenter", "dragover"].forEach(eventName => uploadZone.addEventListener(eventName, event => {
    event.preventDefault();
    uploadZone.classList.add("is-dragging");
  }));
  ["dragleave", "drop"].forEach(eventName => uploadZone.addEventListener(eventName, event => {
    event.preventDefault();
    uploadZone.classList.remove("is-dragging");
  }));
  uploadZone.addEventListener("drop", event => handleUploadedFile(event.dataTransfer.files[0]));

  // Clear button
  clearBtn.addEventListener("click", () => {
    resumeInput.value = "";
    document.querySelectorAll(".sample-btn").forEach(b => b.classList.remove("is-active"));
    updateCounters();
    emptyState.style.display = "block";
    resultContainer.style.display = "none";
    confidenceIndicator.textContent = "Awaiting input...";
  });

  // Fetch sample resumes
  fetch("/api/sample_resumes")
    .then(res => res.json())
    .then(data => {
      sampleResumes = data;
      const sampleButtons = document.querySelectorAll(".sample-btn");
      sampleButtons.forEach(btn => {
        btn.addEventListener("click", () => {
          sampleButtons.forEach(b => b.classList.remove("is-active"));
          btn.classList.add("is-active");
          const key = btn.getAttribute("data-key");
          if (sampleResumes[key]) {
            resumeInput.value = sampleResumes[key].text;
            updateCounters();
            classifyResume();
          }
        });
      });
    })
    .catch(() => {
      // The embedded examples above keep the static frontend usable without an API server.
    });

  // Fetch benchmark metrics
  fetch("/api/metrics")
    .then(res => res.json())
    .then(data => {
      if (data.model_comparison) {
        benchmarkTableBody.innerHTML = "";
        const models = Object.entries(data.model_comparison);
        // Sort descending by validation Macro-F1
        models.sort((a, b) => b[1].val_macro_f1 - a[1].val_macro_f1);
        for (const [name, metrics] of models) {
          const isBest = name === data.best_model;
          const tr = document.createElement("tr");
          tr.innerHTML = `
            <td><strong>${name.replace(/_/g, " ")}</strong> ${isBest ? '<span class="badge-status-best">Best</span>' : ''}</td>
            <td>${name.includes("MLP") ? "Deep Neural" : "Classical ML"}</td>
            <td>${(metrics.val_accuracy * 100).toFixed(2)}%</td>
            <td><strong>${metrics.val_macro_f1.toFixed(4)}</strong></td>
            <td>${metrics.val_weighted_f1.toFixed(4)}</td>
          `;
          benchmarkTableBody.appendChild(tr);
        }
      }
    })
    .catch(err => console.error("Error loading benchmarks:", err));

  // Classify Action
  async function classifyResume() {
    const text = resumeInput.value.trim();
    if (!text) {
      alert("Please enter resume text to classify.");
      return;
    }

    classifyBtn.disabled = true;
    classifyBtn.querySelector(".btn-text").textContent = "Classifying...";
    classifyBtn.querySelector(".btn-loader").style.display = "inline";
    confidenceIndicator.textContent = "Analyzing patterns...";

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text, top_k: 5 })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed");
      }

      // Display results
      emptyState.style.display = "none";
      resultContainer.style.display = "block";

      topCategoryName.textContent = data.predicted_category;
      const confPercent = (data.confidence * 100).toFixed(1);
      topConfidencePill.textContent = `${confPercent}% Confidence`;
      confidenceIndicator.textContent = `Predicted: ${data.predicted_category} (${confPercent}%)`;

      // Probabilities breakdown
      probabilityList.innerHTML = "";
      data.top_predictions.forEach(item => {
        const div = document.createElement("div");
        div.className = "prob-item";
        div.innerHTML = `
          <div class="prob-meta">
            <span class="prob-cat">${item.category}</span>
            <span class="prob-val">${item.percentage}%</span>
          </div>
          <div class="prob-bar-track">
            <div class="prob-bar-fill" style="width: ${item.percentage}%;"></div>
          </div>
        `;
        probabilityList.appendChild(div);
      });

      // Extracted keywords
      keywordsCloud.innerHTML = "";
      if (data.top_matching_keywords && data.top_matching_keywords.length > 0) {
        data.top_matching_keywords.forEach(kw => {
          const span = document.createElement("span");
          span.className = "kw-badge";
          span.textContent = kw;
          keywordsCloud.appendChild(span);
        });
      } else {
        keywordsCloud.innerHTML = '<span class="text-muted" style="font-size:0.8rem;">No explicit high-weight keywords found.</span>';
      }

      // Cleaned text snippet
      cleanTextSnippet.textContent = data.cleaned_text_preview || "N/A";

    } catch (error) {
      alert("The frontend loaded the resume successfully, but prediction is available when the Phase 6 notebook inference service is running.");
      confidenceIndicator.textContent = "Inference service unavailable";
    } finally {
      classifyBtn.disabled = false;
      classifyBtn.querySelector(".btn-text").textContent = "Analyze & Classify";
      classifyBtn.querySelector(".btn-loader").style.display = "none";
    }
  }

  classifyBtn.addEventListener("click", classifyResume);
});
