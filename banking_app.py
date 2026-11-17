
import streamlit as st
import importlib.util
import pandas as pd
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
BACKEND_FILE = PROJECT / "banking_app_backend.py"

st.set_page_config(
    page_title="Banking & Credit Risk Assistant",
    page_icon="🏦",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    max-width: 1250px;
}
.main-heading {
    font-size: 2rem;
    font-weight: 750;
}
.muted {
    color: #8b95a5;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_backend():

    spec = importlib.util.spec_from_file_location(
        "banking_app_backend_loaded",
        BACKEND_FILE
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


backend = load_backend()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "page" not in st.session_state:
    st.session_state.page = "Chat Assistant"


# --------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------

with st.sidebar:

    st.markdown("## 🏦 Banking Assistant")
    st.caption("Local AI-powered banking analysis")

    st.divider()

    st.markdown("### Navigation")

    st.session_state.page = st.radio(
        "Choose view",
        ["Chat Assistant", "Grade Analysis", "About Project"],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### System components")

    st.success("Ollama — Qwen3 4B")
    st.success("DuckDB — LendingClub data")
    st.success("Python — Financial calculations")

    st.divider()

    st.metric(
        "Labelled accepted loans",
        "1,190,174"
    )

    st.caption("Local processing through Ollama and DuckDB")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------------------
# CHAT VIEW
# --------------------------------------------------------------

if st.session_state.page == "Chat Assistant":

    st.markdown(
        '<div class="main-heading">'
        'Banking & Credit Risk Assistant</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="muted">'
        'Ask about banking concepts, debt ratios, loan repayment '
        'and historical credit-risk statistics.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    examples = [
        "What is credit risk?",
        "What is the overall Charged Off rate?",
        "Explain default rate for grade D loans.",
        "Compare default rates across all grades.",
        "Calculate DTI for Rs. 10,000 debt and Rs. 40,000 income."
    ]

    with st.expander("Try an example question"):

        for example in examples:

            if st.button(
                example,
                key="example_" + str(examples.index(example)),
                use_container_width=True
            ):
                st.session_state.pending_question = example
                st.rerun()

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message["role"] == "assistant":

                st.caption(
                    f"Source: {message.get('source', 'Unknown')} "
                    f"| {message.get('latency', 0):.3f} sec"
                )

                if message.get("evidence"):
                    with st.expander("View evidence"):

                        evidence = message["evidence"]

                        if "grades" in evidence:

                            table = pd.DataFrame(
                                evidence["grades"]
                            )

                            st.dataframe(
                                table,
                                use_container_width=True,
                                hide_index=True
                            )

                            chart_data = table.set_index(
                                "grade"
                            )[["charged_off_rate"]]

                            st.bar_chart(chart_data)

                        else:
                            st.json(evidence)

    question = st.chat_input(
        "Ask your banking question..."
    )

    if st.session_state.get("pending_question"):

        question = st.session_state.pop("pending_question")

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Processing..."):

                try:

                    result = backend.ask(question)

                    answer = result.get(
                        "answer",
                        "No answer returned."
                    )

                    st.markdown(answer)

                    latency = result.get(
                        "latency_seconds", 0
                    )

                    st.caption(
                        f"Source: {result.get('source')} "
                        f"| {latency:.3f} sec"
                    )

                    evidence = result.get("evidence", {})

                    if evidence:

                        with st.expander("View supporting evidence"):

                            if "grades" in evidence:

                                table = pd.DataFrame(
                                    evidence["grades"]
                                )

                                st.dataframe(
                                    table,
                                    use_container_width=True,
                                    hide_index=True
                                )

                                st.bar_chart(
                                    table.set_index("grade")[
                                        ["charged_off_rate"]
                                    ]
                                )

                            else:
                                st.json(evidence)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "source": result.get("source"),
                        "latency": latency,
                        "grounded": result.get("grounded"),
                        "evidence": evidence
                    })

                except Exception as error:

                    st.error(
                        "The request could not be processed. "
                        "Check the Ollama service and data engine."
                    )

                    st.caption(str(error))


# --------------------------------------------------------------
# GRADE ANALYSIS VIEW
# --------------------------------------------------------------

elif st.session_state.page == "Grade Analysis":

    st.title("LendingClub Grade Analysis")
    st.caption("Historical Charged Off rates by loan grade.")

    import importlib.util

    # This data is computed and verified during Batch 51.
    grade_data = pd.read_json(
        PROJECT / "reports" / "grade_statistics.json"
    )

    st.dataframe(
        grade_data,
        use_container_width=True,
        hide_index=True
    )

    chart = grade_data.set_index("grade")[
        ["charged_off_rate"]
    ]

    st.subheader("Historical Charged Off rate by grade")
    st.bar_chart(chart)

    highest = grade_data.loc[
        grade_data["charged_off_rate"].idxmax()
    ]

    lowest = grade_data.loc[
        grade_data["charged_off_rate"].idxmin()
    ]

    c1, c2 = st.columns(2)

    c1.metric(
        "Highest rate",
        f"{highest['charged_off_rate']:.2f}%",
        f"Grade {highest['grade']}"
    )

    c2.metric(
        "Lowest rate",
        f"{lowest['charged_off_rate']:.2f}%",
        f"Grade {lowest['grade']}"
    )

    st.caption(
        "Historical associations only; these rates are not future predictions."
    )


# --------------------------------------------------------------
# ABOUT VIEW
# --------------------------------------------------------------

else:

    st.title("About the Project")

    st.markdown("""
    **Banking and Credit Risk Analysis using a Domain-Specific
    Small Language Model and Explainable Transformer Architecture**

    This research prototype combines:

    - A custom 80.8M-parameter Transformer architecture experiment.
    - Ollama-hosted Qwen3 4B for natural-language explanations.
    - DuckDB for exact analysis of LendingClub accepted loans.
    - Python arithmetic for deterministic financial calculations.

    The language model and custom research model are distinct.
    The deployed language component is Qwen3 4B.
    Historical loan outcomes should not be interpreted as
    guaranteed future default predictions.
    """)

st.divider()

st.caption(
    "Banking & Credit Risk Analysis | Local research prototype"
)
