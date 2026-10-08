import React, { useEffect, useRef, useState } from "react";

const API_URL = "http://127.0.0.1:8000";
const REVIEW_THRESHOLD = 70;

const riskInfo = {
  Low: {
    className: "risk-low",
    icon: "✓",
    description: "Low risk document",
  },
  Medium: {
    className: "risk-medium",
    icon: "!",
    description: "Moderate risk document",
  },
  High: {
    className: "risk-high",
    icon: "!",
    description: "High risk document",
  },
  Critical: {
    className: "risk-critical",
    icon: "⚠",
    description: "Critical risk document",
  },
};

function App() {
  const fileInputRef = useRef(null);

  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [statistics, setStatistics] = useState(null);

  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [error, setError] = useState("");

  // Manual review interaction
  const [reviewExpanded, setReviewExpanded] = useState(false);
  const [reviewAcknowledged, setReviewAcknowledged] = useState(false);

  useEffect(() => {
    loadDashboard();
  }, []);

  async function loadDashboard() {
    await Promise.all([loadPredictions(), loadStatistics()]);
  }

  async function loadPredictions() {
    try {
      setHistoryLoading(true);

      const response = await fetch(`${API_URL}/predictions`);

      if (!response.ok) {
        throw new Error("Could not load prediction history.");
      }

      const data = await response.json();

      const items = Array.isArray(data)
        ? data
        : data.predictions || data.data || [];

      setPredictions(items);
    } catch (err) {
      console.error(err);
    } finally {
      setHistoryLoading(false);
    }
  }

  async function loadStatistics() {
    try {
      const response = await fetch(`${API_URL}/statistics`);

      if (!response.ok) {
        throw new Error("Could not load statistics.");
      }

      const data = await response.json();
      setStatistics(data);
    } catch (err) {
      console.error(err);
    }
  }

  function handleFileChange(event) {
    const selectedFile = event.target.files?.[0];

    if (!selectedFile) return;

    const extension = selectedFile.name.split(".").pop().toLowerCase();

    if (!["pdf", "docx"].includes(extension)) {
      setError("Please select a PDF or DOCX file.");
      setFile(null);
      return;
    }

    setError("");
    setFile(selectedFile);
    setPrediction(null);

    // Reset review interaction
    setReviewExpanded(false);
    setReviewAcknowledged(false);
  }

  function openFilePicker() {
    fileInputRef.current?.click();
  }

  async function analyzeDocument() {
    if (!file) {
      setError("Please select a PDF or DOCX document first.");
      return;
    }

    try {
      setLoading(true);
      setError("");
      setPrediction(null);

      // Reset review interaction
      setReviewExpanded(false);
      setReviewAcknowledged(false);

      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Document analysis failed.");
      }

      if (!data.success) {
        throw new Error(data.message || "Document analysis failed.");
      }

      /*
        Backend returns:

        {
          success: true,
          message: "...",
          prediction: {
            id,
            filename,
            risk_level,
            policy_category,
            confidence,
            review_required,
            review_status,
            important_terms,
            class_probabilities,
            created_at
          }
        }

        So we use data.prediction.
      */
      const result = data.prediction || data;

      setPrediction(result);

      await loadDashboard();
    } catch (err) {
      console.error(err);
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  async function clearHistory() {
    const confirmed = window.confirm(
      "Are you sure you want to clear all prediction history?"
    );

    if (!confirmed) return;

    try {
      const response = await fetch(`${API_URL}/predictions`, {
        method: "DELETE",
      });

      if (!response.ok) {
        throw new Error("Could not clear history.");
      }

      setPredictions([]);
      setPrediction(null);
      setReviewExpanded(false);
      setReviewAcknowledged(false);

      await loadStatistics();
    } catch (err) {
      setError(err.message);
    }
  }

  function getRiskClass(risk) {
    return riskInfo[risk]?.className || "risk-medium";
  }

  function formatDate(dateString) {
    if (!dateString) return "—";

    const date = new Date(dateString);

    if (Number.isNaN(date.getTime())) {
      return dateString;
    }

    return date.toLocaleString();
  }

  const totalDocuments =
    statistics?.total_documents ??
    statistics?.total_predictions ??
    predictions.length ??
    0;

  const manualReviews =
    statistics?.manual_reviews ??
    statistics?.review_required ??
    predictions.filter((item) => item.review_required).length;

  const automaticClassifications =
    statistics?.automatic_classifications ??
    totalDocuments - manualReviews;

  const riskCounts = {
    Low: 0,
    Medium: 0,
    High: 0,
    Critical: 0,
  };

  predictions.forEach((item) => {
    if (riskCounts[item.risk_level] !== undefined) {
      riskCounts[item.risk_level] += 1;
    }
  });

  const confidence = Number(prediction?.confidence) || 0;

  const needsManualReview =
    prediction?.review_required === true ||
    confidence < REVIEW_THRESHOLD;

  return (
    <div className="app">
      {/* ================= HEADER ================= */}

      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">AI</div>

          <div>
            <h1>Document Risk Classifier</h1>
            <p>AI-powered risk & compliance analysis</p>
          </div>
        </div>

        <div className="status">
          <span className="status-dot"></span>
          API Connected
        </div>
      </header>

      {/* ================= MAIN ================= */}

      <main className="container">

        {/* ================= HERO ================= */}

        <section className="hero">
          <div>
            <span className="eyebrow">DOCUMENT INTELLIGENCE</span>

            <h2>
              Analyze document
              <br />
              <span>risk automatically.</span>
            </h2>

            <p>
              Upload a business document and let the machine-learning model
              classify its compliance risk level and identify important terms.
            </p>
          </div>

          <div className="hero-badge">
            <span>ML</span>
            <small>TF-IDF + Logistic Regression</small>
          </div>
        </section>

        {/* ================= ERROR ================= */}

        {error && (
          <div className="error-banner">
            <strong>Error:</strong> {error}
          </div>
        )}

        {/* ================= UPLOAD ================= */}

        <section className="upload-card">
          <button
            className="upload-icon"
            onClick={openFilePicker}
            type="button"
            aria-label="Choose document"
          >
            ↑
          </button>

          <h3>
            {file ? file.name : "Upload a document"}
          </h3>

          <p>
            {file
              ? `${(file.size / 1024).toFixed(1)} KB selected`
              : "Supported formats: PDF and DOCX"}
          </p>

          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.docx"
            onChange={handleFileChange}
            hidden
          />

          <div className="upload-actions">
            <button
              className="secondary-btn"
              onClick={openFilePicker}
              type="button"
            >
              {file ? "Choose another file" : "Choose document"}
            </button>

            <button
              className="primary-btn"
              onClick={analyzeDocument}
              disabled={!file || loading}
              type="button"
            >
              {loading ? "Analyzing..." : "Analyze document →"}
            </button>
          </div>
        </section>

        {/* ================= STATS ================= */}

        <section className="stats-grid">
          <div className="stat-card">
            <span className="stat-label">DOCUMENTS ANALYZED</span>
            <strong>{totalDocuments}</strong>
            <span className="stat-sub">Total predictions</span>
          </div>

          <div className="stat-card">
            <span className="stat-label">AUTO CLASSIFIED</span>
            <strong>{automaticClassifications}</strong>
            <span className="stat-sub">
              Above confidence threshold
            </span>
          </div>

          <div className="stat-card">
            <span className="stat-label">MANUAL REVIEW</span>
            <strong>{manualReviews}</strong>
            <span className="stat-sub">
              Requires human review
            </span>
          </div>

          <div className="stat-card">
            <span className="stat-label">MODEL</span>
            <strong>93.75%</strong>
            <span className="stat-sub">
              Evaluation accuracy
            </span>
          </div>
        </section>

        {/* ================= RESULT ================= */}

        {prediction && (
          <section className="result-section">

            <div className="section-heading">
              <div>
                <span className="eyebrow">
                  LATEST ANALYSIS
                </span>

                <h2>Classification result</h2>
              </div>
            </div>

            <div className="result-grid">

              {/* ================= RESULT MAIN ================= */}

              <div className="result-main">

                <div className="result-top">

                  <div>
                    <span className="file-label">
                      DOCUMENT
                    </span>

                    <h3>
                      {prediction.filename || file?.name}
                    </h3>
                  </div>

                  <span
                    className={`risk-badge ${getRiskClass(
                      prediction.risk_level
                    )}`}
                  >
                    {riskInfo[prediction.risk_level]?.icon}{" "}
                    {prediction.risk_level}
                  </span>
                </div>

                {/* CONFIDENCE */}

                <div className="confidence">

                  <div className="confidence-header">
                    <span>Confidence</span>

                    <strong>
                      {confidence.toFixed(2)}%
                    </strong>
                  </div>

                  <div className="progress-track">

                    <div
                      className={`progress-fill ${getRiskClass(
                        prediction.risk_level
                      )}`}
                      style={{
                        width: `${Math.min(
                          confidence,
                          100
                        )}%`,
                      }}
                    />

                  </div>
                </div>

                {/* DETAILS */}

                <div className="result-details">

                  <div>
                    <span>Policy category</span>

                    <strong>
                      {prediction.policy_category ||
                        "Not specified"}
                    </strong>
                  </div>

                  <div>
                    <span>Review status</span>

                    <strong>
                      {prediction.review_status || "—"}
                    </strong>
                  </div>

                </div>
              </div>

              {/* ================= INTERACTIVE REVIEW CARD ================= */}

              <div
                className={`review-card ${
                  reviewExpanded ? "review-expanded" : ""
                }`}
                onClick={() =>
                  setReviewExpanded((current) => !current)
                }
                role="button"
                tabIndex={0}
                onKeyDown={(event) => {
                  if (
                    event.key === "Enter" ||
                    event.key === " "
                  ) {
                    event.preventDefault();
                    setReviewExpanded((current) => !current);
                  }
                }}
              >

                <span className="review-icon">
                  {needsManualReview ? "!" : "✓"}
                </span>

                <h3>
                  {needsManualReview
                    ? "Manual review required"
                    : "Automatically classified"}
                </h3>

                <p>
                  {needsManualReview
                    ? "The model confidence is below the automatic classification threshold."
                    : "The model confidence is high enough for automatic classification."}
                </p>

                {/* CLICK HINT */}

                <div className="review-toggle">
                  {reviewExpanded
                    ? "▲ Hide review details"
                    : "▼ View review details"}
                </div>

                {/* EXPANDED REVIEW CONTENT */}

                {reviewExpanded && (
                  <div
                    className="review-details"
                    onClick={(event) => event.stopPropagation()}
                  >

                    <div className="review-detail-row">
                      <span>Predicted risk</span>

                      <strong>
                        {prediction.risk_level}
                      </strong>
                    </div>

                    <div className="review-detail-row">
                      <span>Model confidence</span>

                      <strong>
                        {confidence.toFixed(2)}%
                      </strong>
                    </div>

                    <div className="review-detail-row">
                      <span>Automatic threshold</span>

                      <strong>
                        {REVIEW_THRESHOLD}%
                      </strong>
                    </div>

                    <div className="review-detail-row">
                      <span>Decision</span>

                      <strong>
                        {confidence >= REVIEW_THRESHOLD
                          ? "Automatic classification"
                          : "Human review required"}
                      </strong>
                    </div>

                    {/* CONFIDENCE EXPLANATION */}

                    <div className="review-explanation">

                      {confidence < REVIEW_THRESHOLD ? (
                        <>
                          <strong>
                            Why is review required?
                          </strong>

                          <p>
                            The model considers{" "}
                            <strong>
                              {prediction.risk_level}
                            </strong>{" "}
                            the most probable risk class,
                            but its confidence is only{" "}
                            <strong>
                              {confidence.toFixed(2)}%
                            </strong>
                            . Since this is below the{" "}
                            <strong>
                              {REVIEW_THRESHOLD}%
                            </strong>{" "}
                            threshold, the document should
                            be checked by a human reviewer.
                          </p>
                        </>
                      ) : (
                        <>
                          <strong>
                            Automatic classification available
                          </strong>

                          <p>
                            The model confidence is above
                            the configured automatic
                            classification threshold.
                          </p>
                        </>
                      )}

                    </div>

                    {/* ACKNOWLEDGE BUTTON */}

                    {needsManualReview && (
                      <button
                        type="button"
                        className={
                          reviewAcknowledged
                            ? "review-action reviewed"
                            : "review-action"
                        }
                        onClick={() =>
                          setReviewAcknowledged(
                            (current) => !current
                          )
                        }
                      >
                        {reviewAcknowledged
                          ? "✓ Review acknowledged"
                          : "Mark review as acknowledged"}
                      </button>
                    )}

                  </div>
                )}
              </div>
            </div>

            {/* ================= LOWER ANALYSIS ================= */}

            <div className="analysis-lower">

              {/* IMPORTANT TERMS */}

              <div className="panel">

                <div className="panel-heading">
                  <h3>Important terms</h3>

                  <span>
                    Risk indicators
                  </span>
                </div>

                <div className="terms">

                  {prediction.important_terms?.length ? (
                    prediction.important_terms.map(
                      (term, index) => (
                        <span
                          key={`${term}-${index}`}
                          className="term"
                        >
                          {term}
                        </span>
                      )
                    )
                  ) : (
                    <span className="empty">
                      No important terms detected.
                    </span>
                  )}

                </div>
              </div>

              {/* RISK PROBABILITIES */}

              <div className="panel">

                <div className="panel-heading">

                  <h3>
                    Risk probabilities
                  </h3>

                  <span>
                    Model output
                  </span>

                </div>

                <div className="probabilities">

                  {Object.entries(
                    prediction.class_probabilities || {}
                  ).map(([risk, value]) => (

                    <div
                      className="probability-row"
                      key={risk}
                    >

                      <span>{risk}</span>

                      <div className="mini-track">

                        <div
                          className={`mini-fill ${getRiskClass(
                            risk
                          )}`}
                          style={{
                            width: `${Math.min(
                              Number(value) || 0,
                              100
                            )}%`,
                          }}
                        />

                      </div>

                      <strong>
                        {value}%
                      </strong>

                    </div>

                  ))}

                </div>
              </div>

            </div>
          </section>
        )}

        {/* ================= RISK OVERVIEW ================= */}

        <section className="overview-section">

          <div className="section-heading">

            <div>
              <span className="eyebrow">
                OVERVIEW
              </span>

              <h2>
                Risk distribution
              </h2>
            </div>

          </div>

          <div className="risk-overview">

            {Object.entries(riskCounts).map(
              ([risk, count]) => (

                <div
                  className="risk-count-card"
                  key={risk}
                >

                  <div
                    className={`risk-circle ${getRiskClass(
                      risk
                    )}`}
                  >
                    {count}
                  </div>

                  <div>
                    <strong>{risk}</strong>
                    <span>documents</span>
                  </div>

                </div>

              )
            )}

          </div>
        </section>

        {/* ================= HISTORY ================= */}

        <section className="history-section">

          <div className="section-heading">

            <div>
              <span className="eyebrow">
                HISTORY
              </span>

              <h2>
                Recent analyses
              </h2>
            </div>

            {predictions.length > 0 && (
              <button
                className="danger-btn"
                onClick={clearHistory}
                type="button"
              >
                Clear history
              </button>
            )}

          </div>

          <div className="history-card">

            {historyLoading ? (

              <div className="empty-history">
                Loading history...
              </div>

            ) : predictions.length === 0 ? (

              <div className="empty-history">

                <div className="empty-icon">
                  📄
                </div>

                <h3>
                  No documents analyzed yet
                </h3>

                <p>
                  Upload your first PDF or DOCX
                  document to see results here.
                </p>

              </div>

            ) : (

              <div className="history-table">

                <div className="history-row history-header">
                  <span>DOCUMENT</span>
                  <span>RISK</span>
                  <span>CONFIDENCE</span>
                  <span>REVIEW</span>
                  <span>DATE</span>
                </div>

                {predictions
                  .slice(0, 10)
                  .map((item, index) => (

                    <div
                      className="history-row"
                      key={item.id || index}
                    >

                      <span className="document-name">
                        {item.filename ||
                          "Unknown document"}
                      </span>

                      <span
                        className={`small-risk ${getRiskClass(
                          item.risk_level
                        )}`}
                      >
                        {item.risk_level || "—"}
                      </span>

                      <span>
                        {item.confidence !==
                        undefined
                          ? `${item.confidence}%`
                          : "—"}
                      </span>

                      <span>

                        {item.review_required ? (

                          <span className="review-pill warning">
                            Review
                          </span>

                        ) : (

                          <span className="review-pill success">
                            Automatic
                          </span>

                        )}

                      </span>

                      <span className="date">
                        {formatDate(
                          item.created_at
                        )}
                      </span>

                    </div>

                  ))}

              </div>

            )}

          </div>
        </section>

      </main>

      {/* ================= FOOTER ================= */}

      <footer>
        <span>
          AI Document Risk & Compliance Classifier
        </span>

        <span>
          TF-IDF • Logistic Regression • FastAPI • React
        </span>
      </footer>

    </div>
  );
}

export default App;