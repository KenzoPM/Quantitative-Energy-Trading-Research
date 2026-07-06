import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Alpha Radar", layout="wide")

@st.cache_data
def load_data():
    try:
        df_bess = pd.read_parquet('MaStR_V2_Stromspeicher_Cleaned.parquet')
        df_solar = pd.read_parquet('MaStR_V2_Solar_Cleaned.parquet')
        df_bess['Type'] = 'BESS'
        df_solar['Type'] = 'Solar PV'
        return pd.concat([df_bess, df_solar], ignore_index=True)
    except FileNotFoundError:
        st.error("Data files not found. Please run the parser engine first.")
        return pd.DataFrame()

df = load_data()

if not df.empty:
    st.sidebar.header("Alpha Generation Filters")
    # Filter logic and UI implementation omitted for brevity in public repo
    st.markdown("### Congestion & Volatility Heatmap System Active")
    st.markdown("Connect the Parquet outputs to visualize physical BESS/Solar coverage ratios per postal code.")
