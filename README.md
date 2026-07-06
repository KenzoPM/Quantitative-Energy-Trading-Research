# Quantitative Research & Algorithmic Trading Notes

A centralized repository documenting algorithmic execution strategies, stochastic price modeling, and spatial arbitrage infrastructure for the European Power Markets (EPEX Spot). 

The transition of the German power grid towards intermittent renewables has induced severe physical imbalances, replacing stable baseload pricing with extreme intraday polarization. This repository contains my quantitative research and codebase aimed at exploiting these market inefficiencies through Battery Energy Storage Systems (BESS).

### 📁 01. Stochastic Pricing & Additive OU Jump Diffusion
* **Objective:** Model residual grid noise and optimize BESS dispatch strategies.
* **Context:** Institutional Hourly Price Forward Curves (HPFC) structurally smooth out within-day extremes. The internal CKE baseline projected an unrealistic stabilization at 294 negative hours for 2027, completely contradicting the empirical trajectory pointing toward 621 hours.
* **Mathematical Framework:** Applying a standard Mean-Reverting Jump Diffusion directly to an HPFC creates volatility double-counting. This engine models residual noise using an Additive Ornstein-Uhlenbeck process with jumps, strictly bounded by EPEX clearing limits (-600 EUR to 1000 EUR).
* **Stack:** Python, NumPy, Maximum Likelihood Estimation (MLE), Euler-Maruyama discretization.

### 📁 02. Spatial Arbitrage Data Engine (MaStR Radar)
* **Objective:** Identify localized grid bottlenecks and commercial arbitrage opportunities based on solar PV density.
* **Architecture:** A robust Python pipeline that bypasses static APIs by parsing the massive raw XML dump from the German national registry (Marktstammdatenregister). 
* **Execution:** Extracts and cleans BESS and Solar asset data, compresses it into Parquet files, and feeds a real-time Streamlit dashboard to calculate physical coverage ratios via interactive Plotly heatmaps.
* **Stack:** Python, Streamlit, Plotly, XML Parsing, Pandas.

### 📁 03. Market Microstructure & Empirical Analysis
* **Objective:** Document the structural collapse of the traditional low-margin baseline and the explosion of extreme pricing events.
* **Findings:** Hours priced in the dead zone (0 to 70 EUR/MWh) collapsed by 50.1% YoY. April 2026 recorded a 64.0% surge in negative pricing compared to the previous year. Extreme events, such as the -499 EUR/MWh all-time low on May 1st, 2026, define current profitability.
* **Conclusion:** Deterministic models mathematically eliminate the true arbitrage spread, fundamentally underpricing algorithmic trading and flexible physical assets.

***
*Disclaimer: All proprietary data, forward curves, and internal forecasts have been strictly anonymized or removed to comply with confidentiality agreements.*
