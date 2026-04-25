from __future__ import annotations

import os
import streamlit as st
from src.data_utils import build_prompt_text, load_dataset, parse_prompt, row_label
from src.hf_streaming import (
    stream_chat_completion,
    validate_hf_token,
    check_model_access,
    RECOMMENDED_FREE_MODELS,
    GATED_MODELS,
    DEFAULT_MODEL,
)

st.set_page_config(page_title="Model Playground", page_icon=":robot_face:", layout="wide")
st.title("Model Playground")
st.caption("Stream answers from a hosted Hugging Face model using the dataset prompt. Model weights are never stored locally.")

split = st.sidebar.selectbox("Dataset split", ["polyfiqa-easy", "polyfiqa-expert"])
frame = load_dataset(split).assign(company=lambda df: df["task_id"].str.split("_").str[0])

selected_index = st.selectbox(
    "Prompt row",
    options=frame.index.tolist(),
    format_func=lambda idx: row_label(frame.loc[idx]),
)
row = frame.loc[selected_index]
parsed = parse_prompt(row["query"], row["question"])

default_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or ""

settings_col, preview_col = st.columns([1, 1.2])

with settings_col:
    st.subheader("Model Settings")
    hf_token = st.text_input("Hugging Face API token", value=default_token, type="password")

    # ── Token validation ──────────────────────────────────────────────────────
    val_col1, val_col2 = st.columns([2, 1])
    with val_col1:
        validate_btn = st.button("Validate Token", use_container_width=True)
    with val_col2:
        token_ok = st.session_state.get("token_valid", False)
        st.markdown(f"{'🟢' if token_ok else '🔴'} {'Valid' if token_ok else 'Not verified'}")

    if validate_btn:
        if not hf_token.strip():
            st.error("Enter a token first.")
            st.session_state["token_valid"] = False
        else:
            with st.spinner("Validating token…"):
                is_valid, msg = validate_hf_token(hf_token.strip())
            st.session_state["token_valid"] = is_valid
            if is_valid:
                st.success(msg)
            else:
                st.error(msg)

    st.divider()

    # ── Model picker ──────────────────────────────────────────────────────────
    st.markdown("**Model**")
    use_custom = st.toggle("Enter custom model ID", value=False)

    if use_custom:
        model_id = st.text_input("Custom model ID", value=DEFAULT_MODEL)
    else:
        model_id = st.selectbox(
            "Select a free model",
            options=RECOMMENDED_FREE_MODELS,
            index=0,
            help="These models are publicly accessible — no license approval needed.",
        )

    # Warn immediately if a known gated model is typed
    if model_id.strip() in GATED_MODELS:
        st.warning(
            f"⚠️ **`{model_id}`** requires license acceptance before use.\n\n"
            f"[Click here to accept the license](https://huggingface.co/{model_id})",
            icon="🔒",
        )

    # ── Model access check ────────────────────────────────────────────────────
    acc_col1, acc_col2 = st.columns([2, 1])
    with acc_col1:
        check_access_btn = st.button("Check Model Access", use_container_width=True)
    with acc_col2:
        access_ok = st.session_state.get("model_accessible", False)
        st.markdown(f"{'🟢' if access_ok else '🔴'} {'OK' if access_ok else 'Unknown'}")

    if check_access_btn:
        if not hf_token.strip():
            st.error("Validate your token first.")
        else:
            with st.spinner(f"Checking access to `{model_id.strip()}`…"):
                accessible, access_msg = check_model_access(hf_token.strip(), model_id.strip())
            st.session_state["model_accessible"] = accessible
            if accessible:
                st.success(access_msg)
            else:
                st.error(access_msg)

    # Reset access status when model changes
    if model_id != st.session_state.get("_last_model_id"):
        st.session_state["model_accessible"] = False
        st.session_state["_last_model_id"] = model_id

    st.divider()

    system_prompt = st.text_area(
        "System prompt",
        value=(
            "You are a financial QA assistant. Follow the user's instructions exactly, stay grounded in the provided "
            "financial statements and multilingual news, and keep the response concise."
        ),
        height=100,
    )

    provider = st.selectbox(
        "Provider",
        options=["hf-inference", "auto", "nebius", "together"],
        index=0,
        help="'hf-inference' is the safe default. 'auto' may route to Cerebras which only supports 'conversational'.",
    )
    resolved_display = provider if provider != "auto" else "hf-inference (auto → forced)"
    st.caption(f"Resolved provider: `{resolved_display}`")

    max_tokens  = st.slider("Max new tokens", min_value=64, max_value=1024, value=256, step=32)
    temperature = st.slider("Temperature",    min_value=0.0, max_value=1.2, value=0.2, step=0.1)

    use_curated_prompt = st.checkbox(
        "Use curated prompt wrapper",
        value=True,
        help="Reconstructs a cleaner prompt from parsed sections instead of sending the raw query verbatim.",
    )

    if use_curated_prompt:
        prompt_text = build_prompt_text(parsed, "curated")
    else:
        prompt_text = row["query"]

    run = st.button("Stream Response", type="primary", use_container_width=True)

with preview_col:
    st.subheader("Prompt Preview")
    st.markdown(f"**Task ID:** `{row['task_id']}`")
    st.markdown(f"**Reference question:** {parsed.question}")
    st.text_area("Prompt sent to the model", prompt_text, height=420)

st.subheader("Reference Answer")
st.write(row["answer"])

if run:
    # ── Guard 1: token ────────────────────────────────────────────────────────
    if not hf_token.strip():
        st.error("Enter a Hugging Face API token or set `HF_TOKEN` before running inference.")
        st.stop()

    if not st.session_state.get("token_valid", False):
        with st.spinner("Validating token…"):
            is_valid, msg = validate_hf_token(hf_token.strip())
        st.session_state["token_valid"] = is_valid
        if not is_valid:
            st.error(f"Token validation failed — {msg}")
            st.stop()

    # ── Guard 2: model access pre-check ──────────────────────────────────────
    with st.spinner(f"Checking access to `{model_id.strip()}`…"):
        accessible, access_msg = check_model_access(hf_token.strip(), model_id.strip())
    st.session_state["model_accessible"] = accessible

    if not accessible:
        st.error(access_msg)
        st.info(
            "💡 **Try one of these confirmed-free models** (no license needed):\n\n"
            + "\n".join(f"- `{m}`" for m in RECOMMENDED_FREE_MODELS),
        )
        st.stop()

    # ── Stream ────────────────────────────────────────────────────────────────
    st.subheader("Streaming Output")
    output_box = st.empty()

    def response_stream():
        accumulated = ""
        try:
            for token in stream_chat_completion(
                token=hf_token.strip(),
                model=model_id.strip(),
                prompt=prompt_text,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                provider=provider,
            ):
                accumulated += token
                output_box.markdown(accumulated)
                yield token
        except Exception as exc:
            st.error(str(exc))
            return

    st.write_stream(response_stream)
