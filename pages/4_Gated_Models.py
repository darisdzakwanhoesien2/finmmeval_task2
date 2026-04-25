from __future__ import annotations

import os
import streamlit as st
from huggingface_hub import HfApi

st.set_page_config(page_title="Gated Models Browser", page_icon="🔒", layout="wide")
st.title("🔒 Gated Models Browser")
st.caption("Browse, search, and check your access status for gated Hugging Face models.")

# ── Auth ──────────────────────────────────────────────────────────────────────
def _safe_secrets_get(key: str, default=None):
    try:
        return st.secrets.get(key, default)
    except FileNotFoundError:
        return default

default_token = (
    _safe_secrets_get("HF_TOKEN")
    or st.session_state.get("hf_token")
    or os.getenv("HF_TOKEN")
    or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    or ""
)

with st.sidebar:
    st.subheader("Authentication")
    hf_token = st.text_input(
        "Hugging Face token",
        value=default_token,
        type="password",
        help="Required to check gated model access status.",
    )
    if hf_token:
        st.session_state["hf_token"] = hf_token

    st.divider()
    st.subheader("Search Filters")
    search_query  = st.text_input("Search model name", placeholder="e.g. llama, gemma, mistral")
    filter_task   = st.selectbox(
        "Pipeline / task",
        ["All", "text-generation", "conversational", "text2text-generation",
         "question-answering", "summarization", "feature-extraction"],
    )
    filter_access = st.selectbox(
        "Access status",
        ["All", "✅ Accessible", "🔒 Gated (no access)", "❓ Unknown"],
    )
    max_results   = st.slider("Max models to fetch", min_value=10, max_value=200, value=50, step=10)
    fetch_btn     = st.button("🔍 Fetch Gated Models", type="primary", use_container_width=True)

# ── Known curated gated models (always shown) ─────────────────────────────────
CURATED_GATED: list[dict] = [
    {"model_id": "meta-llama/Llama-3.1-8B-Instruct",        "pipeline": "text-generation",  "org": "Meta"},
    {"model_id": "meta-llama/Llama-3.2-3B-Instruct",        "pipeline": "text-generation",  "org": "Meta"},
    {"model_id": "meta-llama/Llama-3.3-70B-Instruct",       "pipeline": "text-generation",  "org": "Meta"},
    {"model_id": "meta-llama/Llama-2-7b-chat-hf",           "pipeline": "text-generation",  "org": "Meta"},
    {"model_id": "meta-llama/Llama-2-13b-chat-hf",          "pipeline": "text-generation",  "org": "Meta"},
    {"model_id": "meta-llama/Llama-2-70b-chat-hf",          "pipeline": "text-generation",  "org": "Meta"},
    {"model_id": "meta-llama/Meta-Llama-3-8B-Instruct",     "pipeline": "text-generation",  "org": "Meta"},
    {"model_id": "meta-llama/Meta-Llama-3-70B-Instruct",    "pipeline": "text-generation",  "org": "Meta"},
    {"model_id": "google/gemma-7b-it",                       "pipeline": "text-generation",  "org": "Google"},
    {"model_id": "google/gemma-2b-it",                       "pipeline": "text-generation",  "org": "Google"},
    {"model_id": "google/gemma-2-9b-it",                     "pipeline": "text-generation",  "org": "Google"},
    {"model_id": "google/gemma-2-27b-it",                    "pipeline": "text-generation",  "org": "Google"},
    {"model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",    "pipeline": "text-generation",  "org": "Mistral"},
    {"model_id": "mistralai/Mistral-7B-Instruct-v0.2",      "pipeline": "text-generation",  "org": "Mistral"},
    {"model_id": "mistralai/Mistral-Large-Instruct-2407",   "pipeline": "text-generation",  "org": "Mistral"},
    {"model_id": "microsoft/phi-4",                          "pipeline": "text-generation",  "org": "Microsoft"},
    {"model_id": "microsoft/Phi-3-medium-128k-instruct",    "pipeline": "text-generation",  "org": "Microsoft"},
    {"model_id": "apple/OpenELM-3B-Instruct",               "pipeline": "text-generation",  "org": "Apple"},
    {"model_id": "tiiuae/falcon-40b-instruct",              "pipeline": "text-generation",  "org": "TII"},
    {"model_id": "01-ai/Yi-34B-Chat",                       "pipeline": "text-generation",  "org": "01.AI"},
]


def _check_access(api: HfApi, model_id: str) -> tuple[str, str]:
    """
    Returns (status_emoji, detail_message).
    status: '✅ Accessible' | '🔒 Gated (no access)' | '❓ Unknown'
    """
    try:
        info = api.model_info(model_id, token=hf_token)
        gated = getattr(info, "gated", False)
        if gated:
            return "🔒 Gated (no access)", "License not accepted"
        return "✅ Accessible", "Token has access"
    except Exception as exc:
        err = str(exc)
        if "403" in err or "gated" in err.lower():
            return "🔒 Gated (no access)", "License not accepted"
        if "404" in err:
            return "🔒 Gated (no access)", "404 – model may be private or gated"
        return "❓ Unknown", str(exc)[:120]


def _fetch_gated_from_hub(api: HfApi, query: str, task: str, limit: int) -> list[dict]:
    """Search HF Hub for gated models matching the query."""
    results = []
    try:
        kwargs: dict = dict(limit=limit, sort="downloads", direction=-1, fetch_config=False)
        if query:
            kwargs["search"] = query
        if task and task != "All":
            kwargs["pipeline_tag"] = task
        for m in api.list_models(**kwargs):
            if getattr(m, "gated", False):
                results.append({
                    "model_id": m.modelId,
                    "pipeline":  getattr(m, "pipeline_tag", "—") or "—",
                    "org":       m.modelId.split("/")[0] if "/" in m.modelId else "—",
                })
    except Exception as exc:
        st.warning(f"Hub search failed: {exc}")
    return results


# ── Main content ──────────────────────────────────────────────────────────────
tab_curated, tab_hub = st.tabs(["📋 Curated List", "🌐 Live Hub Search"])

with tab_curated:
    st.markdown("### Well-known gated models")
    st.caption("Curated list of commonly used gated models. Click **Check My Access** to verify your token.")

    # apply sidebar filters to curated list
    filtered = CURATED_GATED
    if search_query:
        filtered = [m for m in filtered if search_query.lower() in m["model_id"].lower()]
    if filter_task != "All":
        filtered = [m for m in filtered if m["pipeline"] == filter_task]

    if not hf_token:
        st.info("Add your Hugging Face token in the sidebar to check access status.", icon="ℹ️")

    check_all = st.button("Check My Access for All", use_container_width=True, disabled=not hf_token)

    rows = []
    for entry in filtered:
        status, detail = ("—", "Token required")
        if hf_token and check_all:
            api = HfApi(token=hf_token)
            status, detail = _check_access(api, entry["model_id"])
        rows.append({
            "Model": entry["model_id"],
            "Org": entry["org"],
            "Pipeline": entry["pipeline"],
            "Access": status,
            "Detail": detail,
            "HF Page": f"https://huggingface.co/{entry['model_id']}",
        })

    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)

        # apply access filter
        if filter_access != "All":
            df = df[df["Access"] == filter_access]

        # colour-code Access column
        def _colour(val: str):
            if val.startswith("✅"):
                return "color: green"
            if val.startswith("🔒"):
                return "color: red"
            return "color: grey"

        st.dataframe(
            df.style.applymap(_colour, subset=["Access"]),
            use_container_width=True,
            hide_index=True,
            column_config={
                "HF Page": st.column_config.LinkColumn("HF Page", display_text="Open ↗"),
            },
        )

        # ── Per-model detail expander ─────────────────────────────────────────
        st.markdown("#### Per-model detail")
        for entry in filtered:
            with st.expander(f"🔒 {entry['model_id']}  ({entry['org']})"):
                c1, c2 = st.columns([2, 1])
                c1.markdown(
                    f"**Pipeline:** `{entry['pipeline']}`  \n"
                    f"**Org:** `{entry['org']}`  \n"
                    f"**HF page:** [huggingface.co/{entry['model_id']}](https://huggingface.co/{entry['model_id']})"
                )
                with c2:
                    if hf_token:
                        if st.button("Check access", key=f"chk_{entry['model_id']}"):
                            with st.spinner("Checking…"):
                                api = HfApi(token=hf_token)
                                status, detail = _check_access(api, entry["model_id"])
                            if status.startswith("✅"):
                                st.success(f"{status} — {detail}")
                            elif status.startswith("🔒"):
                                st.error(
                                    f"{status}  \n"
                                    f"Accept the license at:  \n"
                                    f"https://huggingface.co/{entry['model_id']}"
                                )
                            else:
                                st.warning(f"{status} — {detail}")
                    else:
                        st.caption("Add token to check")
    else:
        st.info("No models match the current filters.")


with tab_hub:
    st.markdown("### Live Hub search for gated models")
    st.caption("Searches HuggingFace Hub for gated models matching your filters. Requires a token.")

    if not hf_token:
        st.info("Add your Hugging Face token in the sidebar to search the Hub.", icon="ℹ️")
    elif fetch_btn:
        with st.spinner(f"Searching Hub for gated models (up to {max_results})…"):
            api = HfApi(token=hf_token)
            hub_models = _fetch_gated_from_hub(
                api,
                query=search_query,
                task=filter_task if filter_task != "All" else "",
                limit=max_results,
            )

        if not hub_models:
            st.warning("No gated models found matching your search. Try a broader query.")
        else:
            st.success(f"Found **{len(hub_models)}** gated model(s). Checking access…")
            progress = st.progress(0)
            results = []
            for i, m in enumerate(hub_models):
                status, detail = _check_access(api, m["model_id"])
                results.append({
                    "Model":    m["model_id"],
                    "Org":      m["org"],
                    "Pipeline": m["pipeline"],
                    "Access":   status,
                    "Detail":   detail,
                    "HF Page":  f"https://huggingface.co/{m['model_id']}",
                })
                progress.progress((i + 1) / len(hub_models))
            progress.empty()

            import pandas as pd
            df_hub = pd.DataFrame(results)
            if filter_access != "All":
                df_hub = df_hub[df_hub["Access"] == filter_access]

            st.dataframe(
                df_hub,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "HF Page": st.column_config.LinkColumn("HF Page", display_text="Open ↗"),
                },
            )

            accessible = df_hub[df_hub["Access"] == "✅ Accessible"]
            gated_no   = df_hub[df_hub["Access"] == "🔒 Gated (no access)"]
            st.markdown(
                f"**Summary:** {len(accessible)} accessible &nbsp;|&nbsp; "
                f"{len(gated_no)} need license acceptance &nbsp;|&nbsp; "
                f"{len(df_hub) - len(accessible) - len(gated_no)} unknown"
            )
    else:
        st.info("Set your filters in the sidebar and click **Fetch Gated Models**.", icon="👈")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "To gain access to a gated model: log in at [huggingface.co](https://huggingface.co), "
    "open the model page, and click **Agree and access repository**. "
    "Then make sure your token has **read** scope at "
    "[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)."
)