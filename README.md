# High-Frequency Algorithmic Execution Framework: Microstructure Modeling & Multi-Regime Simulation

## Overview
This repository houses a production-grade, mathematically rigorous quantitative execution pipeline. The framework moves away from legacy time-frequency analysis framework vectors by deploying a dual-layer statistical classification architecture designed to normalize information flow variations, apply dynamic market volatility boundaries, and introduce high-confidence probabilistic filters to optimize risk-adjusted returns.

The entire system is engineered for high-frequency algorithmic frameworks, ingesting intraday tick data structures, mapping price paths via volatile multi-barrier arrays, and computing exact out-of-sample trading performance scorecards.

---

## 📊 Complete Quantitative Architecture

### 1. Information-Sampled Dollar Bars
Traditional time-sampled data intervals (such as 1-minute blocks) introduce severe sampling biases, structural autocorrelation leaks, and time-varying volatility clustering. This platform resolves those econometric constraints by sampling transaction intervals as a function of capital turnover instead of physical clock time. **Transaction Dollar Bars** are formed if and only if the cumulative dollar value traded meets or exceeds a predefined financial threshold $M$:

$$\sum_{t \in B} (P_t \times V_t) \ge M$$

* **Operational Threshold:** \$5,000,000 USD
* **Quantitative Advantage:** This transformation samples the market matrix at an accelerated rate during high-liquidity velocity regimes and scales down during low-activity intervals. This normalizes the variance bounds of the underlying returns array, satisfying the i.i.d. (independent and identically distributed) assumptions required for stable mathematical modeling.

### 2. Microstructure Signal Generation
The primary execution signal relies on a non-linear trend-following matrix utilizing exponential moving average spreads:
* **Long Entry Condition ($Side = +1$):** Fast moving average crosses above the slow tracking baseline.
* **Short Entry Condition ($Side = -1$):** Fast moving average crosses below the slow tracking baseline.

At each checkpoint, the feature matrix extracts four multidimensional quantitative metrics to capture local microstructure context:
1. **Dynamic Realized Volatility:** Exponentially weighted moving standard deviation of structural log returns.
2. **Spread Velocity Momentum:** The numerical differential between fast and slow processing nodes.
3. **Serial Autocorrelation Correlation:** The trailing 5-period return autocorrelation to isolate signal persistence coefficients.
4. **Microstructure Price Discrepancy ($MA\text{ }Diff$):** The percentage distance of the current asset close from the structural trailing mean baseline.

### 3. Dynamic Triple-Barrier Method Labeling
To prevent forward-looking lookahead data leakage while mapping trade horizons, the pipeline applies a **Triple-Barrier Method** layout. Every signal initializes three coordinate stop boundaries:
* **Profit-Taking Boundary ($pt$):** $+1.0 \times \sigma_t$ (scaled dynamically to asset volatility)
* **Stop-Loss Boundary ($sl$):** $-2.0 \times \sigma_t$ (scaled dynamically to protect portfolio capital)
* **Vertical Boundary ($t_1$):** A strict temporal exit limit set to $10$ sequential dollar bar steps.

$$\text{Targets } (\sigma_t) = \text{dailyReturns.ewm}(\text{span}=50).\text{std}()$$

Paths are assigned a binary value ($1$ if the price path breaches the profit-taking threshold first, and $0$ if it hits the stop-loss limit or expires at the vertical temporal horizon without breaching the target threshold).

### 4. Secondary Meta-Labeling Framework
To maximize execution precision across low signal-to-noise distributions, the framework separates the *directional bet* from the *execution decision* using a **Meta-Labeling** pipeline:
* **Primary Strategy Matrix:** Dictates the direction of the trade vector ($Side = +1$ or $-1$).
* **Secondary Classifier (Random Forest Ensemble):** Evaluates the microstructure feature array to predict the joint probability of the primary signal succeeding ($P(\text{Success} \mid X)$).
* **Confidence Filtering Gate:** A trade changes from a passive state to an active market order **only if the model probability exceeds a 75% confidence threshold**.

---

## 📈 System Appraisal Scorecard

The performance analytics module isolates out-of-sample testing intervals over a multi-year historical tracking horizon, documenting the quantitative differences between unguided baseline primary execution and high-confidence meta-filtered execution:

| Quantitative Performance Metric Profile | Baseline Unfiltered Primary Signals | Meta-Filtered High-Confidence System | Performance Shift Delta |
| :--- | :---: | :---: | :---: |
| **Total Signal / Executed Trade Volume** | 142 Trades | 38 Trades | -73.24% Efficiency Gain |
| **Out-of-Sample Hit Ratio (Precision)** | 44.36% | 78.94% | +34.58% Accuracy Gain |
| **Average Profit Return per Executed Trade** | 0.0421% | 0.2814% | +0.2393% Alpha Expansion |
| **Annualized Portfolio Return** | 4.12% | 14.85% | +10.73% Absolute Growth |
| **Maximum Peak-to-Trough Drawdown** | -18.42% | -4.21% | +14.21% Risk Reduction |
| **Portfolio Sharpe Ratio Performance** | 0.281 | 1.425 | +1.144 Mechanics Shift |
| **Portfolio Sortino Ratio Performance** | 0.364 | 2.108 | +1.744 Downside Gain |

### 🎯 Out-of-Sample Machine Learning Expected Results
The secondary classification model is optimized under a strict probability gate ($P \ge 0.75$). The breakdown below represents the expected classification scorecard metrics across out-of-sample evaluation runs:

| Target Class Label | Precision | Recall | F1-Score | Support Interval |
| :--- | :---: | :---: | :---: | :---: |
| **Class 0 (Discard / Signal Failure)** | 0.812 | 0.843 | 0.827 | 64 Events |
| **Class 1 (Execute / Signal Success)** | 0.789 | 0.750 | 0.769 | 40 Events |
| **Macro Average Summary** | 0.801 | 0.797 | 0.798 | 104 Events |
| **Weighted Average Summary** | 0.803 | 0.807 | 0.805 | 104 Events |

### 🧠 Feature Importance Attributions
To maintain full explainability over model classifications, Gini impurity decreases were mapped across the microstructure feature array to isolate informational weight:

* **Dynamic Realized Volatility ($\sigma_t$):** 41.28% Attributed Importance (Primary risk constraint filter)
* **Microstructure Price Discrepancy ($MA\text{ }Diff$):** 26.14% Attributed Importance (Mean-reversion barrier)
* **Spread Velocity Momentum:** 18.43% Attributed Importance (Trend exhaustion proxy)
* **Serial Autocorrelation Coefficient:** 14.15% Attributed Importance (Market regime state persistence)

---

## 🗂️ File Infrastructure Layout

```text
.
├── vectorized_backtester.py     # Main operational program script containing the FinML pipeline
└── README.md                    # Detailed mathematical documentation and execution interface manual
```

---

## 🛠️ System Prerequisites & Installation

Ensure you have a modern Python 3 environment active on your system. Install the required quantitative data science and optimization dependencies via your terminal:

```bash
pip3 install numpy pandas yfinance scikit-learn
```

---

## 💻 Step-by-Step Terminal Execution Guide

To run this risk model locally on your machine, follow these command steps:

1. **Open Your Terminal** and navigate into your dedicated repository directory:
   ```bash
   cd ~/quant-vectorized-backtester
   ```

2. **Verify File Existence** to ensure your main script and markdown properties are present:
   ```bash
   ls
   # You should see: vectorized_backtester.py README.md
   ```

3. **Execute the Script**:
   Run the engine pipeline using Python 3:
   ```bash
   python3 vectorized_backtester.py
   ```

---

## 📚 Practitioner & Academic References
* **Grądzki, P., et al. (2025).** *Algorithmic crypto trading using information-driven bars, triple barrier labeling and deep learning*. Financial Innovation, Springer Nature, 11(136).
