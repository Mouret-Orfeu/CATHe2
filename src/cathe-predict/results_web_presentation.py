from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Predictions Results", layout="wide")
st.title("Predictions – Results viewer")

default_path = "./src/cathe-predict/Results.csv"
results_path = default_path


@st.cache_data(show_spinner=False)
def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


p = Path(results_path)
if not p.exists() or p.stat().st_size == 0:
    st.error("Results.csv not found or empty.")
    st.write("Expected after running cathe_predictions.py. Confirm the path and try again.")
else:
    try:
        df = load_results(str(p))
    except Exception as e:
        st.error(f"Failed to load results: {e}")
    else:
        required = [
            "Record",
            "CATHe_Predicted_SFAM",
            "CATHe_Prediction_Probability",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error("Missing required columns in Results.csv: " + ", ".join(missing))
        else:
            # Column selection: include Sequence only if requested
            st.subheader("Results")
            include_seq = st.checkbox("Include sequence column", value=False)

            columns_to_show = required.copy()
            if include_seq:
                if "Sequence" in df.columns:
                    columns_to_show.append("Sequence")
                else:
                    st.info("Column 'Sequence' not found in Results.csv.")

            display_df = df[columns_to_show].rename(
                columns={
                    "CATHe_Predicted_SFAM": "predicted SF",
                    "CATHe_Prediction_Probability": "model confidence score",
                    "Sequence": "sequence",
                }
            )

            # If sequence is shown, render a wrapped, multi-line table
            if include_seq and "sequence" in display_df.columns:
                # Ensure it's string and wrap-able
                display_df["sequence"] = display_df["sequence"].astype(str)

                # Add CSS to allow long tokens to wrap across lines
                st.markdown(
                    """
                    <style>
                    [data-testid="stTable"] td, [data-testid="stTable"] th {
                        white-space: pre-wrap !important;
                        word-break: break-word !important;
                        overflow-wrap: anywhere !important;
                    }
                    /* Make columns share width to favor wrapping over horizontal scroll */
                    [data-testid="stTable"] table { table-layout: fixed; width: 100%; }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                # Use st.table so CSS above applies (st.dataframe truncates cells)
                styled = display_df.style.set_properties(
                    subset=["sequence"],
                    **{"font-family": "monospace"},
                )
                st.table(styled)
            else:
                st.dataframe(display_df, use_container_width=True)
