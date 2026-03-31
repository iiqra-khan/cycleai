// import './App.css'
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// const API_BASE = "http://127.0.0.1:8000";
// To this — works both locally and in Docker
const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"
const DAYS_STORAGE_KEY = "cycleai_days_v1";
const AUTH_STORAGE_KEY = "cycleai_auth_v1";

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

const IMPORT_DAY_DEFAULTS = {
  lh_imputed: 0,
  estrogen_imputed: 0,
  pdg_imputed: 0,
  cramps_imputed: 0,
  sorebreasts_imputed: 0,
  bloating_imputed: 0,
  moodswing_imputed: 0,
  fatigue_imputed: 0,
  headaches_imputed: 0,
  foodcravings_imputed: 0,
  indigestion_imputed: 0,
  exerciselevel_imputed: 0,
  stress_imputed: 0,
  sleepissue_imputed: 0,
  appetite_imputed: 0,
  high_estrogen_flag: false,
  estrogen_capped_flag: false,
  is_weekend: false,
  id: 1,
  day_in_study: 1,
};

function normalizeStoredDays(rawDays) {
  if (!Array.isArray(rawDays) || rawDays.length === 0) {
    return [{ ...DEFAULT_DAY }];
  }
  return rawDays.map((d, idx) => ({
    ...DEFAULT_DAY,
    ...(d || {}),
    day_in_study: Number.isFinite(Number(d?.day_in_study))
      ? Number(d.day_in_study)
      : idx + 1,
  }));
}

function normalizeImportedDays(rawDays) {
  if (!Array.isArray(rawDays) || rawDays.length === 0) {
    return [{ ...DEFAULT_DAY }];
  }
  return rawDays.map((d, idx) => ({
    ...IMPORT_DAY_DEFAULTS,
    ...(d || {}),
    id: Number.isFinite(Number(d?.id)) ? Number(d.id) : 1,
    day_in_study: Number.isFinite(Number(d?.day_in_study)) ? Number(d.day_in_study) : idx + 1,
  }));
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [days, setDays]             = useState(() => {
    try {
      const stored = localStorage.getItem(DAYS_STORAGE_KEY);
      if (!stored) return [{ ...DEFAULT_DAY }];
      return normalizeStoredDays(JSON.parse(stored));
    } catch {
      return [{ ...DEFAULT_DAY }];
    }
  });
  const [activeDayIdx, setActiveDayIdx] = useState(0);
  const [result, setResult]         = useState(null);
  const [ragTrace, setRagTrace]     = useState([]);
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState(null);
  const [activeTab, setActiveTab]   = useState("input");
  const [includeRag, setIncludeRag] = useState(false);
  const [jobId, setJobId]           = useState(null);
  const [showImportBox, setShowImportBox] = useState(false);
  const [importJsonText, setImportJsonText] = useState("");
  const [authToken, setAuthToken] = useState(() => localStorage.getItem(AUTH_STORAGE_KEY) || "");
  const [authMode, setAuthMode] = useState("login");
  const [authUsername, setAuthUsername] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authLoading, setAuthLoading] = useState(false);
  const [me, setMe] = useState(null);
  const sseRef                      = useRef(null);
  const currentDay                  = days[activeDayIdx] || days[0] || { ...DEFAULT_DAY };

  useEffect(() => {
    localStorage.setItem(DAYS_STORAGE_KEY, JSON.stringify(days));
  }, [days]);

  useEffect(() => {
    if (authToken) {
      localStorage.setItem(AUTH_STORAGE_KEY, authToken);
    } else {
      localStorage.removeItem(AUTH_STORAGE_KEY);
      setMe(null);
    }
  }, [authToken]);

  useEffect(() => {
    if (!authToken) return;

    const loadMe = async () => {
      try {
        const res = await fetch(`${API_BASE}/auth/me`, {
          headers: { Authorization: `Bearer ${authToken}` },
        });

        if (!res.ok) {
          setAuthToken("");
          return;
        }

        const data = await res.json();
        setMe(data);
      } catch {
        setAuthToken("");
      }
    };

    loadMe();
  }, [authToken]);

  const authFetch = async (url, options = {}) => {
    const headers = {
      ...(options.headers || {}),
      Authorization: `Bearer ${authToken}`,
    };

    const response = await fetch(url, { ...options, headers });
    if (response.status === 401) {
      setAuthToken("");
      throw new Error("Session expired. Please log in again.");
    }
    return response;
  };

  const handleAuthSubmit = async (evt) => {
    evt.preventDefault();
    setError(null);

    const username = authUsername.trim().toLowerCase();
    if (username.length < 3) {
      setError("Username must be at least 3 characters.");
      return;
    }

    if (authPassword.length < 8) {
      setError("Password must be at least 8 characters.");
      return;
    }

    setAuthLoading(true);

    try {
      const endpoint = authMode === "register" ? "/auth/register" : "/auth/login";
      const res = await fetch(`${API_BASE}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password: authPassword }),
      });

      let data = null;
      try {
        data = await res.json();
      } catch {
        data = null;
      }
      if (!res.ok) {
        throw new Error(data?.detail || `Authentication failed (${res.status})`);
      }

      setAuthToken(data.access_token);
      setAuthPassword("");
      setError(null);
    } catch (e) {
      setError(e.message || "Authentication failed");
    } finally {
      setAuthLoading(false);
    }
  };

  const handleLogout = () => {
    if (sseRef.current) {
      sseRef.current.close();
      sseRef.current = null;
    }
    setAuthToken("");
    setResult(null);
    setRagTrace([]);
    setJobId(null);
    setLoading(false);
  };

  const updateField = (field, val) => {
    const parsed = parseFloat(val);
    const numeric = Number.isNaN(parsed) ? 0 : parsed;

    setDays(prev => prev.map((day, idx) => (
      idx === activeDayIdx ? { ...day, [field]: numeric } : day
    )));
  };

  const addDay = () => {
    let newIndex = 0;
    setDays(prev => {
      const last = prev[prev.length - 1] || DEFAULT_DAY;
      const next = [
        ...prev,
        {
          ...last,
          day_in_study: (Number(last.day_in_study) || prev.length) + 1,
        },
      ];
      newIndex = next.length - 1;
      return next;
    });
    setActiveDayIdx(newIndex);
  };

  const duplicateActiveDay = () => {
    let newIndex = 0;
    setDays(prev => {
      const base = prev[activeDayIdx] || prev[prev.length - 1] || DEFAULT_DAY;
      const next = [
        ...prev,
        {
          ...base,
          day_in_study: (Number(base.day_in_study) || prev.length) + 1,
        },
      ];
      newIndex = next.length - 1;
      return next;
    });
    setActiveDayIdx(newIndex);
  };

  const removeActiveDay = () => {
    if (days.length <= 1) return;

    let nextIndex = 0;
    setDays(prev => {
      const idx = Math.min(activeDayIdx, prev.length - 1);
      const next = prev.filter((_, i) => i !== idx);
      nextIndex = Math.max(0, idx - 1);
      return next;
    });
    setActiveDayIdx(nextIndex);
  };

  const clearAllDays = () => {
    setDays([{ ...DEFAULT_DAY }]);
    setActiveDayIdx(0);
  };

  const applyIdToAllDays = () => {
    const participantId = Number(currentDay.id) || 1;
    setDays(prev => prev.map(day => ({ ...day, id: participantId })));
  };

  const importDaysFromJson = () => {
    try {
      const parsed = JSON.parse(importJsonText);
      const candidate = Array.isArray(parsed) ? parsed : parsed?.days;
      const normalized = normalizeImportedDays(candidate);

      setDays(normalized);
      setActiveDayIdx(0);
      setShowImportBox(false);
      setImportJsonText("");
      setError(null);
    } catch {
      setError("Invalid JSON. Paste either an array of day objects or an object with a days array.");
    }
  };

  const handlePredict = async () => {
    if (sseRef.current) {
      sseRef.current.close();
      sseRef.current = null;
    }

    setLoading(true);
    setError(null);
    setResult(null);
    setRagTrace([]);
    setJobId(null);

    if (!authToken) {
      setError("Please log in before running predictions.");
      setLoading(false);
      return;
    }

    try {
      if (days.length < 7) {
        throw new Error("Please provide at least 7 days to match draft5 temporal feature behavior.");
      }

      const participantIds = new Set(days.map(d => Math.round(Number(d.id) || 1)));
      if (participantIds.size !== 1) {
        throw new Error("All days must belong to the same participant id.");
      }

      const normalizedDays = days.map(day => ({
        ...day,
        id: Math.round(Number(day.id) || 1),
        day_in_study: Math.round(Number(day.day_in_study) || 1),
      }));

      const hasMissingHormones = normalizedDays.some(day => (
        !Number.isFinite(Number(day.lh_imputed))
        || !Number.isFinite(Number(day.estrogen_imputed))
        || !Number.isFinite(Number(day.pdg_imputed))
      ));
      if (hasMissingHormones) {
        throw new Error("Each day must include numeric lh_imputed, estrogen_imputed, and pdg_imputed values.");
      }

      const sortedDays = [...normalizedDays].sort((a, b) => a.day_in_study - b.day_in_study);

      const requestBody = {
        days: sortedDays,
        include_rag:  includeRag,
        include_shap: true,
      };

      // Use SSE job flow when RAG is enabled to stream live trace events.
      if (includeRag) {
        const jobRes = await authFetch(`${API_BASE}/predict/jobs`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(requestBody),
        });

        if (!jobRes.ok) {
          const err = await jobRes.json();
          throw new Error(err.detail || "Failed to create prediction job");
        }

        const job = await jobRes.json();
        setJobId(job.job_id);
        setActiveTab("results");

        const es = new EventSource(`${API_BASE}${job.stream_url}?token=${encodeURIComponent(authToken)}`);
        sseRef.current = es;

        es.onmessage = (evt) => {
          try {
            const eventObj = JSON.parse(evt.data);
            setRagTrace(prev => [...prev, eventObj]);

            if (eventObj.type === "final_result") {
              setResult(eventObj.payload);
            }

            if (eventObj.type === "error") {
              const msg = eventObj.payload?.error || eventObj.message || "Prediction job failed";
              setError(msg);
            }

            if (eventObj.type === "completed" || eventObj.type === "error") {
              setLoading(false);
              es.close();
              sseRef.current = null;
            }
          } catch (_e) {
            setError("Failed to parse prediction stream event");
            setLoading(false);
            es.close();
            sseRef.current = null;
          }
        };

        es.onerror = () => {
          setError("Prediction stream disconnected");
          setLoading(false);
          es.close();
          sseRef.current = null;
        };

        return;
      }

      const res = await authFetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
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
          <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
            {authToken && me && (
              <div style={{ fontSize: 12, color: COLORS.muted }}>
                Logged in as <span style={{ color: COLORS.accent }}>{me.username}</span>
              </div>
            )}
            {authToken && (
              <button
                onClick={handleLogout}
                style={{
                  padding: "8px 12px",
                  borderRadius: 8,
                  border: `1px solid ${COLORS.border}`,
                  background: COLORS.bg,
                  color: COLORS.text,
                  cursor: "pointer",
                  fontSize: 12,
                }}
              >
                Logout
              </button>
            )}
          </div>
        </div>
      </div>

      {!authToken && (
        <div style={{ maxWidth: 900, margin: "0 auto 24px" }}>
          <Card style={{ borderColor: `${COLORS.accent}33` }}>
            <SectionTitle accent={COLORS.accent}>{authMode === "register" ? "Create Account" : "Login Required"}</SectionTitle>
            <form onSubmit={handleAuthSubmit} style={{ display: "grid", gap: 12 }}>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <Input label="Username" value={authUsername} onChange={setAuthUsername} type="text" step={undefined} min={undefined} />
                <Input label="Password" value={authPassword} onChange={setAuthPassword} type="password" step={undefined} min={undefined} />
              </div>
              {error && (
                <div style={{
                  padding: 10,
                  borderRadius: 10,
                  background: `${COLORS.high}11`,
                  border: `1px solid ${COLORS.high}33`,
                  color: COLORS.high,
                  fontSize: 13,
                }}>
                  {error}
                </div>
              )}
              <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                <button
                  type="submit"
                  disabled={authLoading}
                  style={{
                    padding: "10px 16px",
                    borderRadius: 10,
                    border: "none",
                    background: `linear-gradient(135deg, ${COLORS.accent}, ${COLORS.accent2})`,
                    color: "#fff",
                    cursor: authLoading ? "not-allowed" : "pointer",
                    fontWeight: 700,
                    fontSize: 13,
                  }}
                >
                  {authLoading ? "Please wait..." : authMode === "register" ? "Register" : "Login"}
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setAuthMode(prev => (prev === "login" ? "register" : "login"));
                    setError(null);
                  }}
                  style={{
                    padding: "10px 14px",
                    borderRadius: 10,
                    border: `1px solid ${COLORS.border}`,
                    background: COLORS.bg,
                    color: COLORS.text,
                    cursor: "pointer",
                    fontSize: 13,
                  }}
                >
                  Switch to {authMode === "login" ? "Register" : "Login"}
                </button>
              </div>
            </form>
          </Card>
        </div>
      )}

      {/* ── Tabs ───────────────────────────────────────────────────────────── */}
      {authToken && (
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
      )}

      <div style={{ maxWidth: 900, margin: "0 auto" }}>

        {/* ── INPUT TAB ──────────────────────────────────────────────────────── */}
        {authToken && activeTab === "input" && (
          <div style={{ display: "grid", gap: 20 }}>

            {/* Hormones */}
            <Card>
              <SectionTitle>Day History</SectionTitle>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 14 }}>
                {days.map((day, idx) => (
                  <button
                    key={`day-btn-${idx}`}
                    onClick={() => setActiveDayIdx(idx)}
                    style={{
                      padding: "6px 10px",
                      borderRadius: 999,
                      border: `1px solid ${idx === activeDayIdx ? COLORS.accent : COLORS.border}`,
                      background: idx === activeDayIdx ? `${COLORS.accent}22` : COLORS.bg,
                      color: idx === activeDayIdx ? COLORS.accent : COLORS.text,
                      fontSize: 12,
                      cursor: "pointer",
                    }}
                  >
                    Day {idx + 1} • study {Math.round(Number(day.day_in_study) || idx + 1)}
                  </button>
                ))}
              </div>

              <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 18 }}>
                <button onClick={addDay} style={{
                  padding: "8px 12px", borderRadius: 8, border: `1px solid ${COLORS.border}`,
                  background: COLORS.bg, color: COLORS.text, cursor: "pointer", fontSize: 12,
                }}>
                  + Add Day
                </button>
                <button onClick={duplicateActiveDay} style={{
                  padding: "8px 12px", borderRadius: 8, border: `1px solid ${COLORS.border}`,
                  background: COLORS.bg, color: COLORS.text, cursor: "pointer", fontSize: 12,
                }}>
                  Duplicate Active Day
                </button>
                <button onClick={removeActiveDay} style={{
                  padding: "8px 12px", borderRadius: 8, border: `1px solid ${COLORS.border}`,
                  background: COLORS.bg, color: COLORS.text, cursor: "pointer", fontSize: 12,
                }}>
                  Remove Active Day
                </button>
                <button onClick={clearAllDays} style={{
                  padding: "8px 12px", borderRadius: 8, border: `1px solid ${COLORS.high}33`,
                  background: `${COLORS.high}11`, color: COLORS.high, cursor: "pointer", fontSize: 12,
                }}>
                  Clear All
                </button>
                <button onClick={() => setShowImportBox(prev => !prev)} style={{
                  padding: "8px 12px", borderRadius: 8, border: `1px solid ${COLORS.accent}55`,
                  background: `${COLORS.accent}11`, color: COLORS.accent, cursor: "pointer", fontSize: 12,
                }}>
                  {showImportBox ? "Close JSON Import" : "Import JSON"}
                </button>
                <div style={{ marginLeft: "auto", fontSize: 12, color: COLORS.muted, alignSelf: "center" }}>
                  Days in payload: {days.length}
                </div>
              </div>

              {showImportBox && (
                <div style={{
                  border: `1px solid ${COLORS.border}`,
                  background: COLORS.bg,
                  borderRadius: 12,
                  padding: 12,
                  marginBottom: 18,
                }}>
                  <Label>Paste Days JSON</Label>
                  <textarea
                    value={importJsonText}
                    onChange={(e) => setImportJsonText(e.target.value)}
                    placeholder={'[{"id":1,"day_in_study":1,"lh_imputed":4.2,...},{"id":1,"day_in_study":2,...}]'}
                    style={{
                      width: "100%",
                      minHeight: 120,
                      background: COLORS.surface,
                      color: COLORS.text,
                      border: `1px solid ${COLORS.border}`,
                      borderRadius: 8,
                      padding: 10,
                      fontSize: 12,
                      fontFamily: "monospace",
                      resize: "vertical",
                    }}
                  />
                  <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
                    <button onClick={importDaysFromJson} style={{
                      padding: "8px 12px", borderRadius: 8, border: "none",
                      background: `linear-gradient(135deg, ${COLORS.accent}, ${COLORS.accent2})`,
                      color: "#fff", cursor: "pointer", fontSize: 12, fontWeight: 700,
                    }}>
                      Load Days
                    </button>
                    <button onClick={() => setImportJsonText("")} style={{
                      padding: "8px 12px", borderRadius: 8, border: `1px solid ${COLORS.border}`,
                      background: COLORS.surface, color: COLORS.text, cursor: "pointer", fontSize: 12,
                    }}>
                      Clear
                    </button>
                  </div>
                </div>
              )}

              <SectionTitle accent={COLORS.accent3}>Hormonal Markers</SectionTitle>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
                <Input label="LH (IU/L)"          value={currentDay.lh_imputed}       onChange={v => updateField("lh_imputed", v)} />
                <Input label="Estrogen (pg/mL)"   value={currentDay.estrogen_imputed} onChange={v => updateField("estrogen_imputed", v)} />
                <Input label="Progesterone (PDG)"  value={currentDay.pdg_imputed}      onChange={v => updateField("pdg_imputed", v)} />
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
                    value={currentDay[field]}
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
                <Input label="Participant ID"  value={currentDay.id}           onChange={v => updateField("id", v)} step="1" />
                <Input label="Day in Study"    value={currentDay.day_in_study} onChange={v => updateField("day_in_study", v)} step="1" />
                <div style={{ display: "flex", flexDirection: "column", justifyContent: "flex-end" }}>
                  <Label>Options</Label>
                  <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer", fontSize: 13 }}>
                    <input type="checkbox" checked={includeRag}
                      onChange={e => setIncludeRag(e.target.checked)}
                      style={{ accentColor: COLORS.accent }}
                    />
                    Include RAG explanation (slower)
                  </label>
                  <button onClick={applyIdToAllDays} style={{
                    marginTop: 10,
                    padding: "7px 10px",
                    borderRadius: 8,
                    border: `1px solid ${COLORS.border}`,
                    background: COLORS.bg,
                    color: COLORS.text,
                    cursor: "pointer",
                    fontSize: 12,
                    width: "fit-content",
                  }}>
                    Apply ID to all days
                  </button>
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

            {loading && includeRag && (
              <div style={{ fontSize: 12, color: COLORS.muted }}>
                Streaming live RAG trace events...
              </div>
            )}
          </div>
        )}

        {/* ── RESULTS TAB ────────────────────────────────────────────────────── */}
        {authToken && activeTab === "results" && (
          <div style={{ display: "grid", gap: 20 }}>

            {/* Live trace */}
            {(ragTrace.length > 0 || loading) && (
              <Card style={{ borderColor: `${COLORS.accent}33` }}>
                <SectionTitle accent={COLORS.accent}>Live RAG Trace</SectionTitle>
                {jobId && (
                  <div style={{ fontSize: 12, color: COLORS.muted, marginBottom: 10 }}>
                    Job ID: <span style={{ color: COLORS.accent, fontFamily: "monospace" }}>{jobId}</span>
                  </div>
                )}
                <div style={{
                  maxHeight: 240,
                  overflowY: "auto",
                  border: `1px solid ${COLORS.border}`,
                  borderRadius: 10,
                  padding: 10,
                  background: COLORS.bg,
                  display: "grid",
                  gap: 8,
                }}>
                  {ragTrace.length === 0 && (
                    <div style={{ fontSize: 12, color: COLORS.muted }}>
                      Waiting for events...
                    </div>
                  )}
                  {ragTrace.map((evt, idx) => (
                    <div key={`${evt.event_id || idx}-${evt.type || "event"}`} style={{
                      fontSize: 12,
                      border: `1px solid ${COLORS.border}`,
                      borderRadius: 8,
                      padding: "8px 10px",
                      background: COLORS.surface,
                    }}>
                      <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 4 }}>
                        <span style={{
                          color: COLORS.accent,
                          fontFamily: "monospace",
                          fontSize: 11,
                        }}>
                          {evt.type || "event"}
                        </span>
                        <span style={{ color: COLORS.muted, fontSize: 11, fontFamily: "monospace" }}>
                          {evt.timestamp ? new Date(evt.timestamp).toLocaleTimeString() : ""}
                        </span>
                      </div>
                      <div style={{ color: COLORS.text }}>{evt.message}</div>
                    </div>
                  ))}
                </div>
              </Card>
            )}

            {!result && ragTrace.length === 0 && !loading && (
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
                    <div style={{ fontSize: 13, color: COLORS.text, lineHeight: 1.8 }}>
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                          p: ({ children }) => <p style={{ margin: "0 0 10px", color: COLORS.text }}>{children}</p>,
                          h1: ({ children }) => <h3 style={{ margin: "0 0 10px", color: COLORS.accent }}>{children}</h3>,
                          h2: ({ children }) => <h4 style={{ margin: "12px 0 8px", color: COLORS.accent }}>{children}</h4>,
                          h3: ({ children }) => <h5 style={{ margin: "10px 0 6px", color: COLORS.accent2 }}>{children}</h5>,
                          ul: ({ children }) => <ul style={{ margin: "0 0 10px 18px" }}>{children}</ul>,
                          ol: ({ children }) => <ol style={{ margin: "0 0 10px 18px" }}>{children}</ol>,
                          li: ({ children }) => <li style={{ marginBottom: 4 }}>{children}</li>,
                          strong: ({ children }) => <strong style={{ color: "#ffffff" }}>{children}</strong>,
                          code: ({ children }) => (
                            <code style={{
                              background: COLORS.bg,
                              border: `1px solid ${COLORS.border}`,
                              borderRadius: 6,
                              padding: "2px 6px",
                              fontFamily: "monospace",
                            }}>
                              {children}
                            </code>
                          ),
                        }}
                      >
                        {result.rag_explanation}
                      </ReactMarkdown>
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