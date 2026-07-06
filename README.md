# EPEX Spot : Quantitative Trading & Flexible Asset Analytics

A centralized repository documenting algorithmic execution strategies, stochastic price modeling, and spatial arbitrage infrastructure for the European Power Markets (EPEX Spot). 

The transition of the German power grid has replaced stable baseload pricing with extreme intraday polarization. My research and codebase focus on bridging quantitative modeling with market execution to capture this volatility through Battery Energy Storage Systems (BESS).

### 📁 01. Intraday Execution : RL Arbitrage Engine
* **Objective:** Maximize PnL through aggressive algorithmic bidding on the continuous Intraday market.
* **Strategy:** Development of a Reinforcement Learning model to transition from passive asset management to a highly competitive market-taker environment. 
* **Execution Logic:** Operates on a Dual-Cycle Engine. Cycle 1 targets Solar Rescue (Abregelung) to absorb negative price pools. Cycle 2 executes Pure Grid Arbitrage during evening scarcity spikes.

### 📁 02. Stochastic Pricing : Additive OU Jump Diffusion
* **Objective:** Correct deterministic baselines to secure the BESS Flex Case valuation and accurately price extreme market jumps.
* **The Problem:** Institutional Hourly Price Forward Curves (HPFC) structurally smooth out within-day extremes. The internal CKE baseline projected an unrealistic stabilization at 294 negative hours for 2027, completely contradicting the empirical trajectory pointing toward 621 hours.
* **The Solution:** An Additive Ornstein-Uhlenbeck jump diffusion model, strictly bounded by EPEX clearing limits (-600 EUR to 1000 EUR). This avoids the volatility double-counting inherent to standard MRJD models applied to HPFCs. 
* **Validation:** Calibrated via Maximum Likelihood Estimation on 2024 data, the out-of-sample backtest achieved an error margin of under 2.5% against the 2025 outturn.

### 📁 03. Spatial Arbitrage : MaStR Data Engine
* **Objective:** Source localized grid bottlenecks to deploy flexible assets where physical congestion is highest.
* **Architecture:** A robust Python data pipeline bypassing static APIs to parse massive XML dumps directly from the German national registry (Marktstammdatenregister).
* **Deployment:** Extracts BESS and Solar asset parameters, compresses them into Parquet files, and feeds a real-time Streamlit dashboard. This generates interactive Plotly heatmaps to calculate physical coverage ratios and identify actionable DSO network nodes.

***
*Disclaimer: All proprietary data, forward curves, and internal commercial forecasts have been strictly anonymized or removed to comply with confidentiality agreements.*
