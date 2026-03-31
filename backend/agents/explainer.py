# =============================================================================
# explainer.py — AgenticRAG: full agentic RAG with tool calling
# =============================================================================

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Any, Callable
from openai import OpenAI
from dotenv import load_dotenv

from ..rag.knowledge_base import build_vector_store
from ..ml.shap_utils import get_shap_context

load_dotenv()


class AgenticRAG:
    """
    Agentic RAG system using tool calling.

    The LLM is given two tools:
      1. retrieve_evidence    — search the clinical knowledge base
      2. finalize_explanation — signal it has enough context to answer

    The LLM decides WHAT to search, HOW MANY TIMES, and WHEN to stop.
    Python only executes what the LLM requests.
    """

    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "retrieve_evidence",
                "description": (
                    "Search the clinical knowledge base for medical evidence. "
                    "Call this to retrieve passages about hormonal markers, "
                    "symptoms, or cycle physiology. Call multiple times if needed."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Clinical search query, e.g. "
                                "'progesterone drop luteal phase menstruation'"
                            )
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of passages to retrieve (1-5)",
                            "default": 3
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "finalize_explanation",
                "description": (
                    "Call this when you have retrieved enough evidence and are "
                    "ready to write the final clinical explanation."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "explanation": {
                            "type": "string",
                            "description": "The complete clinical explanation with citations"
                        }
                    },
                    "required": ["explanation"]
                }
            }
        }
    ]

    def __init__(self, trace_callback: Callable[[dict], None] | None = None):
        self.api_key  = os.environ.get("OPENAI_API_KEY")
        self.base_url = os.environ.get("OPENAI_BASE_URL",
                                        "https://openrouter.ai/api/v1")
        self.trace_callback = trace_callback

        self._trace("rag_init", "Initialising ChromaDB + sentence-transformers.")

        print("\n[RAG] Initialising ChromaDB + sentence-transformers...")
        self._collection, self._chroma = build_vector_store()
        print(f"  [RAG] Vector store ready.")
        self._trace("rag_ready", "Vector store ready.")

    def _trace(self, event_type: str, message: str, payload: dict | None = None):
        if not self.trace_callback:
            return

        self.trace_callback({
            "type": event_type,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": payload or {},
        })

    # ── Tool execution ────────────────────────────────────────────────────────
    def _execute_tool(self, tool_name: str, tool_args: dict) -> str:
        if tool_name == "retrieve_evidence":
            query  = tool_args.get("query", "")
            top_k  = tool_args.get("top_k", 3)
            print(f"  [Tool] retrieve_evidence | query='{query}' | top_k={top_k}")
            self._trace(
                "tool_call",
                "LLM called retrieve_evidence.",
                {"tool": "retrieve_evidence", "query": query, "top_k": top_k}
            )

            results  = self._collection.query(
                query_texts=[query],
                n_results=min(top_k, 7)
            )
            documents = (results.get("documents") or [[]])[0]
            metadatas = (results.get("metadatas") or [[]])[0]
            passages = []
            retrieved = []
            for i, (doc, meta) in enumerate(
                zip(documents, metadatas)
            ):
                passages.append(
                    f"[Evidence {i+1} — {meta.get('source','?')}]\n{doc}"
                )
                print(f"  [Tool] Retrieved: "
                      f"{meta.get('source')} — {meta.get('topic')}")
                retrieved.append({
                    "source": meta.get("source", "?"),
                    "topic": meta.get("topic", "?"),
                })

            self._trace(
                "tool_result",
                "Retrieved evidence passages.",
                {"count": len(retrieved), "items": retrieved}
            )
            return "\n\n".join(passages)

        elif tool_name == "finalize_explanation":
            self._trace("tool_call", "LLM called finalize_explanation.")
            return tool_args.get("explanation", "")

        return f"Unknown tool: {tool_name}"

    # ── LLM call with retry ───────────────────────────────────────────────────
    def _call_llm(self, messages: list, use_tools: bool = True):
        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        kwargs = {
            "model"   : "openrouter/auto",
            "messages": messages,
        }
        if use_tools:
            kwargs["tools"]       = self.TOOLS
            kwargs["tool_choice"] = "auto"

        for attempt in range(3):
            try:
                self._trace(
                    "llm_call",
                    "Calling LLM.",
                    {"attempt": attempt + 1, "use_tools": use_tools}
                )
                return client.chat.completions.create(**kwargs)
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    print(f"  [LLM] Rate limited "
                          f"(attempt {attempt+1}/3) — waiting 15s...")
                    self._trace(
                        "llm_rate_limited",
                        "LLM rate-limited; retrying.",
                        {"attempt": attempt + 1}
                    )
                    time.sleep(15)
                else:
                    self._trace(
                        "llm_error",
                        "LLM call failed.",
                        {"error": str(e)}
                    )
                    raise e
        return None

    # ── Agentic tool loop ─────────────────────────────────────────────────────
    def _agentic_tool_loop(self, system: str, user: str,
                            max_iterations: int = 5) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user}
        ]
        print(f"\n  [Agent] Starting agentic tool loop "
              f"(max {max_iterations} steps)...")
        self._trace(
            "agent_start",
            "Starting agentic tool loop.",
            {"max_iterations": max_iterations}
        )

        for iteration in range(max_iterations):
            print(f"  [Agent] Step {iteration + 1}...")
            self._trace(
                "agent_step",
                f"Agent step {iteration + 1}.",
                {"step": iteration + 1}
            )
            response = self._call_llm(messages, use_tools=True)

            if response is None:
                self._trace("agent_warning", "LLM unavailable after retries.")
                return (
                    "[RAG] LLM unavailable — all models rate limited.\n"
                    f"User query:\n{user}"
                )

            msg = response.choices[0].message

            if msg.tool_calls:
                messages.append({
                    "role"      : "assistant",
                    "content"   : msg.content or "",
                    "tool_calls": [
                        {
                            "id"      : tc.id,
                            "type"    : "function",
                            "function": {
                                "name"     : tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in msg.tool_calls
                    ]
                })

                for tool_call in msg.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    print(f"  [Agent] LLM called tool: '{tool_name}'")
                    self._trace(
                        "agent_tool_call",
                        "LLM requested tool call.",
                        {"tool": tool_name}
                    )

                    if tool_name == "finalize_explanation":
                        explanation = tool_args.get("explanation", "")
                        print(f"  [Agent] LLM finalized at step {iteration+1}")
                        self._trace(
                            "agent_finalized",
                            "LLM finalized explanation.",
                            {"step": iteration + 1}
                        )
                        return explanation

                    tool_result = self._execute_tool(tool_name, tool_args)
                    messages.append({
                        "role"        : "tool",
                        "tool_call_id": tool_call.id,
                        "content"     : tool_result
                    })
            else:
                print(f"  [Agent] LLM responded directly at step {iteration+1}")
                self._trace(
                    "agent_direct_response",
                    "LLM responded directly without tool call.",
                    {"step": iteration + 1}
                )
                return msg.content or "[No explanation generated]"

        self._trace("agent_max_iterations", "Reached max agent iterations.")
        return "[Agent] Max iterations reached without finalization."

    # ── Public API ────────────────────────────────────────────────────────────
    def retrieve_and_explain(self, prediction_context: dict,
                              X_test: pd.DataFrame | None = None) -> str:
        """
        Main entry point. Drop-in replacement for AgenticRAGStub.

        Args:
            prediction_context: dict with keys:
                target, prediction, top_features, model, feature_names
            X_test: pass X_test to enable SHAP-based context
        """
        target = prediction_context.get("target", "period_start")
        proba  = prediction_context.get("prediction", 0.5)
        event  = ("period onset (menstruation)"
                  if target == "period_start"
                  else "LH surge (ovulation)")
        risk   = ("HIGH"   if proba > 0.7
                  else "MODERATE" if proba > 0.4
                  else "LOW")

        print(f"\n{'='*60}")
        print(f"  AGENTIC RAG (Tool Calling) — Target: {target.upper()}")
        print(f"{'='*60}")
        self._trace(
            "rag_context",
            "Prepared RAG context for explanation.",
            {"target": target, "probability": proba, "risk": risk}
        )

        model         = prediction_context.get("model")
        feature_names = prediction_context.get("feature_names", [])

        if model is not None and X_test is not None and len(feature_names) > 0:
            shap_ctx = get_shap_context(model, X_test, feature_names)
        else:
            top_feats = prediction_context.get("top_features", [])
            shap_ctx  = {
                "base_value"             : 0.05,
                "total_shap_contribution": proba,
                "top_positive_features"  : [
                    {"feature": f, "shap_value": 0.1} for f in top_feats
                ],
                "top_negative_features": [],
            }

        pos_drivers = "\n".join([
            f"  • {f['feature'].replace('_',' ')}: SHAP = +{f['shap_value']:.4f}"
            for f in shap_ctx["top_positive_features"]
        ])
        neg_drivers = "\n".join([
            f"  • {f['feature'].replace('_',' ')}: SHAP = {f['shap_value']:.4f}"
            for f in shap_ctx["top_negative_features"]
        ]) or "  • None significant"

        system = """You are a clinical AI agent explaining menstrual cycle predictions.
You have two tools:
  1. retrieve_evidence — search the clinical knowledge base
  2. finalize_explanation — submit your final explanation when ready

Workflow:
  - Call retrieve_evidence with a relevant clinical query
  - Review evidence; call again if needed for a different angle
  - Call finalize_explanation with your complete structured response

Final explanation must include:
  1. PREDICTION SUMMARY
  2. KEY HORMONAL SIGNALS
  3. SYMPTOM INDICATORS
  4. EVIDENCE BASIS (cite sources)
  5. CLINICAL INTERPRETATION
  6. CAVEATS
Keep under 400 words."""

        user = f"""Explain this menstrual cycle prediction:

Event:       {event}
Probability: {proba:.1%}
Risk Level:  {risk}
Base Rate:   {shap_ctx['base_value']:.1%}

SHAP — Features driving prediction UP:
{pos_drivers}

SHAP — Features driving prediction DOWN:
{neg_drivers}

Use your tools to retrieve relevant clinical evidence, then finalize."""

        explanation = self._agentic_tool_loop(system, user, max_iterations=5)
        self._trace("rag_explanation_done", "RAG explanation generated.")
        return explanation


def generate_clinical_explanation(prediction_proba: float,
                                   top_features: pd.DataFrame | None,
                                   target: str = "period_start",
                                   negative_features: pd.DataFrame | None = None) -> str:
    """
    Plain-language clinical explanation (fast, no LLM).
    Used as a quick summary before AgenticRAG runs.
    """
    risk_level = ("High"     if prediction_proba > 0.7
                  else "Moderate" if prediction_proba > 0.4
                  else "Low")
    top_3 = (top_features.head(3)["feature"].tolist()
             if top_features is not None and not top_features.empty else [])
    neg_3 = (negative_features.head(3)["feature"].tolist()
             if negative_features is not None and not negative_features.empty else [])

    explanation = (
        f"PREDICTION SUMMARY\n"
        f"{'='*40}\n"
        f"Target Event:    "
        f"{'Period Onset' if target == 'period_start' else 'LH Surge / Ovulation'}\n"
        f"Probability:     {prediction_proba:.1%}\n"
        f"Risk Level:      {risk_level}\n"
        f"\nKEY CONTRIBUTING FACTORS\n"
        f"{'-'*40}\n"
    )
    if top_3:
        for i, feat in enumerate(top_3, 1):
            explanation += f"  {i}. {feat.replace('_', ' ').title()} (upward driver)\n"
    elif neg_3:
        for i, feat in enumerate(neg_3, 1):
            explanation += f"  {i}. {feat.replace('_', ' ').title()} (downward/protective driver)\n"
    else:
        explanation += "  No dominant SHAP drivers were detected for this sample.\n"

    explanation += (
        f"\nINTERPRETATION\n"
        f"{'-'*40}\n"
        f"The model identifies {risk_level.lower()} probability of "
        f"{'menstruation onset' if target == 'period_start' else 'ovulation'} "
        f"based on hormonal and symptomatic patterns observed.\n"
    )
    return explanation