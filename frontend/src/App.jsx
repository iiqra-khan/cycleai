// import './App.css'
import { useState } from "react";

// const API_BASE = "http://127.0.0.1:8000";
// To this — works both locally and in Docker
const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"

// ── Colour palette ────────────────────────────────────────────────────────────
const COLORS = {
  bg:        "#0d0f14",
  surface:   "#13161e",
  border:    "#1e2330",
  accent:    "#c084fc",   // soft violet
  accent2:   "#f472b6",   // rose
  accent3:   "#38bdf8",   // sky blue
  text:      "#e2e8f0",
  muted:     "#64748b",
  low:       "#4ade80",
  moderate:  "#fbbf24",
  high:      "#f87171",
};

// ── Tiny components ───────────────────────────────────────────────────────────

function RiskBadge({ level, probability }) {
  const color = level === "HIGH" ? COLORS.high
              : level === "MODERATE" ? COLORS.moderate
              : COLORS.low;
  return (
    <div style={{
      display: "inline-flex", alignItems: "center", gap: 8,
      padding: "6px 14px", borderRadius: 999,
      border: `1px solid ${color}22`,
      background: `${color}11`,
    }}>
      <div style={{
        width: 8, height: 8, borderRadius: "50%",
        background: color,
        boxShadow: `0 0 8px ${color}`,
      }} />
      <span style={{ color, fontSize: 13, fontWeight: 600, fontFamily: "monospace" }}>
        {level} · {(probability * 100).toFixed(1)}%
      </span>
    </div>
  );
}

function ProbBar({ value, color }) {
  return (
    <div style={{
      height: 6, borderRadius: 3,
      background: COLORS.border, overflow: "hidden",
      marginTop: 6,
    }}>
      <div style={{
        height: "100%", width: `${value * 100}%`,
        background: `linear-gradient(90deg, ${color}88, ${color})`,
        borderRadius: 3,
        transition: "width 0.8s cubic-bezier(0.16,1,0.3,1)",
      }} />
    </div>
  );
}

function Card({ children, style }) {
  return (
    <div style={{
      background: COLORS.surface,
      border: `1px solid ${COLORS.border}`,
      borderRadius: 16, padding: 24,
      ...style,
    }}>
      {children}
    </div>
  );
}

function Label({ children }) {
  return (
    <div style={{
      fontSize: 11, fontWeight: 700, letterSpacing: "0.1em",
      color: COLORS.muted, textTransform: "uppercase", marginBottom: 6,
    }}>
      {children}
    </div>
  );
}

function Input({ label, value, onChange, type = "number", step = "0.1", min = "0" }) {
  return (
    <div style={{ display: "flex", flexDirection: "column" }}>
      <Label>{label}</Label>
      <input
        type={type} step={step} min={min} value={value}
        onChange={e => onChange(e.target.value)}
        style={{
          background: COLORS.bg,
          border: `1px solid ${COLORS.border}`,
          borderRadius: 8, padding: "8px 12px",
          color: COLORS.text, fontSize: 14,
          outline: "none", transition: "border 0.2s",
        }}
        onFocus={e => e.target.style.border = `1px solid ${COLORS.accent}66`}
        onBlur={e => e.target.style.border = `1px solid ${COLORS.border}`}
      />
    </div>
  );
}

function SectionTitle({ children, accent }) {
  return (
    <div style={{
      fontSize: 12, fontWeight: 700, letterSpacing: "0.12em",
      textTransform: "uppercase", color: accent || COLORS.accent,
      marginBottom: 16, display: "flex", alignItems: "center", gap: 8,
    }}>
      <div style={{ width: 20, height: 1, background: accent || COLORS.accent }} />
      {children}
    </div>
  );
}

// ── Default form values ───────────────────────────────────────────────────────
const DEFAULT_DAY = {
  lh_imputed: 4.6,
  estrogen_imputed: 138.4,
  pdg_imputed: 3.6,
  cramps_imputed: 1,
  sorebreasts_imputed: 1,
  bloating_imputed: 1,
  moodswing_imputed: 1,
  fatigue_imputed: 1,
  headaches_imputed: 1,
  foodcravings_imputed: 1,
  indigestion_imputed: 1,
  exerciselevel_imputed: 3,
  stress_imputed: 3,
  sleepissue_imputed: 1,
  appetite_imputed: 3,
  high_estrogen_flag: false,
  estrogen_capped_flag: false,
  is_weekend: false,
  id: 1,
  day_in_study: 1,
};

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [formData, setFormData]     = useState({ ...DEFAULT_DAY });
  const [result, setResult]         = useState(null);
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState(null);
  const [activeTab, setActiveTab]   = useState("input");
  const [includeRag, setIncludeRag] = useState(false);

  const updateField = (field, val) =>
    setFormData(prev => ({ ...prev, [field]: parseFloat(val) || 0 }));

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          days: [{ ...formData }],
          include_rag:  includeRag,
          include_shap: true,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Prediction failed");
      }

      const data = await res.json();
      setResult(data);
      setActiveTab("results");
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      minHeight: "100vh",
      background: COLORS.bg,
      color: COLORS.text,
      fontFamily: "'DM Sans', 'Segoe UI', sans-serif",
      padding: "32px 24px",
    }}>

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div style={{ maxWidth: 900, margin: "0 auto 40px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 8 }}>
          <div style={{
            width: 40, height: 40, borderRadius: 12,
            background: `linear-gradient(135deg, ${COLORS.accent}, ${COLORS.accent2})`,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 20,
          }}>
            🌙
          </div>
          <div>
            <h1 style={{
              margin: 0, fontSize: 24, fontWeight: 700,
              background: `linear-gradient(90deg, ${COLORS.accent}, ${COLORS.accent2})`,
              WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
            }}>
              CycleAI
            </h1>
            <div style={{ fontSize: 12, color: COLORS.muted, marginTop: 2 }}>
              Menstrual Cycle Prediction · Explainable AI · Clinical DSS
            </div>
          </div>
        </div>
      </div>

      {/* ── Tabs ───────────────────────────────────────────────────────────── */}
      <div style={{ maxWidth: 900, margin: "0 auto 24px" }}>
        <div style={{
          display: "flex", gap: 4,
          background: COLORS.surface,
          border: `1px solid ${COLORS.border}`,
          borderRadius: 12, padding: 4, width: "fit-content",
        }}>
          {["input", "results"].map(tab => (
            <button key={tab}
              onClick={() => setActiveTab(tab)}
              style={{
                padding: "8px 20px", borderRadius: 8, border: "none",
                cursor: "pointer", fontSize: 13, fontWeight: 600,
                transition: "all 0.2s",
                background: activeTab === tab
                  ? `linear-gradient(135deg, ${COLORS.accent}22, ${COLORS.accent2}22)`
                  : "transparent",
                color: activeTab === tab ? COLORS.accent : COLORS.muted,
                borderBottom: activeTab === tab
                  ? `2px solid ${COLORS.accent}` : "2px solid transparent",
              }}
            >
              {tab === "input" ? "📋 Input Data" : "📊 Results"}
            </button>
          ))}
        </div>
      </div>

      <div style={{ maxWidth: 900, margin: "0 auto" }}>

        {/* ── INPUT TAB ──────────────────────────────────────────────────────── */}
        {activeTab === "input" && (
          <div style={{ display: "grid", gap: 20 }}>

            {/* Hormones */}
            <Card>
              <SectionTitle accent={COLORS.accent3}>Hormonal Markers</SectionTitle>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
                <Input label="LH (IU/L)"          value={formData.lh_imputed}       onChange={v => updateField("lh_imputed", v)} />
                <Input label="Estrogen (pg/mL)"   value={formData.estrogen_imputed} onChange={v => updateField("estrogen_imputed", v)} />
                <Input label="Progesterone (PDG)"  value={formData.pdg_imputed}      onChange={v => updateField("pdg_imputed", v)} />
              </div>
            </Card>

            {/* Symptoms */}
            <Card>
              <SectionTitle accent={COLORS.accent2}>Symptom Scores (0–5)</SectionTitle>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 16 }}>
                {[
                  ["Cramps",       "cramps_imputed"],
                  ["Sore Breasts", "sorebreasts_imputed"],
                  ["Bloating",     "bloating_imputed"],
                  ["Mood Swings",  "moodswing_imputed"],
                  ["Fatigue",      "fatigue_imputed"],
                  ["Headaches",    "headaches_imputed"],
                  ["Food Cravings","foodcravings_imputed"],
                  ["Indigestion",  "indigestion_imputed"],
                  ["Exercise",     "exerciselevel_imputed"],
                  ["Stress",       "stress_imputed"],
                  ["Sleep Issues", "sleepissue_imputed"],
                  ["Appetite",     "appetite_imputed"],
                ].map(([label, field]) => (
                  <Input key={field} label={label}
                    value={formData[field]}
                    onChange={v => updateField(field, v)}
                    step="1" min="0"
                  />
                ))}
              </div>
            </Card>

            {/* Cycle info */}
            <Card>
              <SectionTitle>Cycle Info</SectionTitle>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
                <Input label="Participant ID"  value={formData.id}           onChange={v => updateField("id", v)} step="1" />
                <Input label="Day in Study"    value={formData.day_in_study} onChange={v => updateField("day_in_study", v)} step="1" />
                <div style={{ display: "flex", flexDirection: "column", justifyContent: "flex-end" }}>
                  <Label>Options</Label>
                  <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer", fontSize: 13 }}>
                    <input type="checkbox" checked={includeRag}
                      onChange={e => setIncludeRag(e.target.checked)}
                      style={{ accentColor: COLORS.accent }}
                    />
                    Include RAG explanation (slower)
                  </label>
                </div>
              </div>
            </Card>

            {/* Error */}
            {error && (
              <div style={{
                padding: 16, borderRadius: 12,
                background: `${COLORS.high}11`,
                border: `1px solid ${COLORS.high}33`,
                color: COLORS.high, fontSize: 13,
              }}>
                ⚠ {error}
              </div>
            )}

            {/* Submit */}
            <button
              onClick={handlePredict}
              disabled={loading}
              style={{
                padding: "14px 32px", borderRadius: 12, border: "none",
                background: loading
                  ? COLORS.border
                  : `linear-gradient(135deg, ${COLORS.accent}, ${COLORS.accent2})`,
                color: "#fff", fontSize: 15, fontWeight: 700,
                cursor: loading ? "not-allowed" : "pointer",
                transition: "all 0.2s",
                boxShadow: loading ? "none" : `0 4px 24px ${COLORS.accent}44`,
              }}
            >
              {loading ? "⏳ Predicting..." : "🔮 Run Prediction"}
            </button>
          </div>
        )}

        {/* ── RESULTS TAB ────────────────────────────────────────────────────── */}
        {activeTab === "results" && (
          <div style={{ display: "grid", gap: 20 }}>

            {!result && (
              <Card>
                <div style={{ textAlign: "center", padding: 40, color: COLORS.muted }}>
                  No results yet. Go to Input Data and run a prediction.
                </div>
              </Card>
            )}

            {result && (
              <>
                {/* Period + Ovulation side by side */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>

                  {/* Period */}
                  <Card style={{ borderColor: `${COLORS.accent2}33` }}>
                    <SectionTitle accent={COLORS.accent2}>Period Onset</SectionTitle>
                    <RiskBadge
                      level={result.period_risk.level}
                      probability={result.period_probability}
                    />
                    <ProbBar value={result.period_probability} color={COLORS.accent2} />
                    <div style={{ fontSize: 12, color: COLORS.muted, marginTop: 10 }}>
                      {result.period_risk.description}
                    </div>
                    <div style={{
                      marginTop: 16, fontSize: 13, color: COLORS.muted,
                      padding: "8px 12px", background: COLORS.bg,
                      borderRadius: 8,
                    }}>
                      Prediction: <span style={{
                        color: result.period_prediction ? COLORS.high : COLORS.low,
                        fontWeight: 700
                      }}>
                        {result.period_prediction ? "POSITIVE" : "NEGATIVE"}
                      </span>
                    </div>
                  </Card>

                  {/* Ovulation */}
                  <Card style={{ borderColor: `${COLORS.accent3}33` }}>
                    <SectionTitle accent={COLORS.accent3}>LH Surge / Ovulation</SectionTitle>
                    <RiskBadge
                      level={result.ovulation_risk.level}
                      probability={result.ovulation_probability}
                    />
                    <ProbBar value={result.ovulation_probability} color={COLORS.accent3} />
                    <div style={{ fontSize: 12, color: COLORS.muted, marginTop: 10 }}>
                      {result.ovulation_risk.description}
                    </div>
                    <div style={{
                      marginTop: 16, fontSize: 13, color: COLORS.muted,
                      padding: "8px 12px", background: COLORS.bg,
                      borderRadius: 8,
                    }}>
                      Prediction: <span style={{
                        color: result.ovulation_prediction ? COLORS.high : COLORS.low,
                        fontWeight: 700
                      }}>
                        {result.ovulation_prediction ? "POSITIVE" : "NEGATIVE"}
                      </span>
                    </div>
                  </Card>
                </div>

                {/* Top SHAP features */}
                {result.top_features?.length > 0 && (
                  <Card>
                    <SectionTitle accent={COLORS.accent}>Top SHAP Features</SectionTitle>
                    <div style={{ display: "grid", gap: 10 }}>
                      {result.top_features.slice(0, 8).map((f, i) => {
                        const maxVal = result.top_features[0]?.shap_value || 1;
                        const pct    = Math.abs(f.shap_value) / Math.abs(maxVal);
                        return (
                          <div key={i} style={{ display: "grid", gridTemplateColumns: "200px 1fr 80px", alignItems: "center", gap: 12 }}>
                            <div style={{ fontSize: 13, color: COLORS.text }}>
                              {f.feature.replace(/_/g, " ")}
                            </div>
                            <div style={{ height: 6, borderRadius: 3, background: COLORS.border }}>
                              <div style={{
                                height: "100%", width: `${pct * 100}%`,
                                background: `linear-gradient(90deg, ${COLORS.accent}88, ${COLORS.accent})`,
                                borderRadius: 3,
                              }} />
                            </div>
                            <div style={{ fontSize: 12, color: COLORS.accent, textAlign: "right", fontFamily: "monospace" }}>
                              +{f.shap_value?.toFixed(3)}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </Card>
                )}

                {/* Clinical explanation */}
                <Card>
                  <SectionTitle>Clinical Explanation</SectionTitle>
                  <pre style={{
                    fontSize: 13, color: COLORS.text,
                    whiteSpace: "pre-wrap", lineHeight: 1.7,
                    fontFamily: "inherit", margin: 0,
                  }}>
                    {result.clinical_explanation}
                  </pre>
                </Card>

                {/* RAG explanation */}
                {result.rag_explanation && (
                  <Card style={{ borderColor: `${COLORS.accent}33` }}>
                    <SectionTitle accent={COLORS.accent}>
                      Agentic RAG — Clinical Evidence
                    </SectionTitle>
                    <div style={{
                      fontSize: 13, color: COLORS.text,
                      lineHeight: 1.8, whiteSpace: "pre-wrap",
                    }}>
                      {result.rag_explanation}
                    </div>
                  </Card>
                )}

                {/* Model metadata */}
                <div style={{
                  display: "flex", gap: 12, flexWrap: "wrap",
                  padding: "12px 0", borderTop: `1px solid ${COLORS.border}`,
                }}>
                  {[
                    ["Model",    result.model_used],
                    ["Features", result.features_used],
                  ].map(([k, v]) => (
                    <div key={k} style={{ fontSize: 12, color: COLORS.muted }}>
                      <span style={{ color: COLORS.accent }}>{k}:</span> {v}
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}