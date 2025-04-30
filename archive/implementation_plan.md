Below is a detailed breakdown of the document into concrete implementation steps, each paired with “checks” that an AI coding agent (such as Cursor Composer) can use to verify it has completed or satisfied the requirement. These checks should help keep the agent on track rather than letting it drift into tangential tasks.

⸻

1. Overall Objective & System Definition

Implementation Steps
	1.	Define the purpose of the system clearly (intraday statistical arbitrage on cointegrated futures pairs).
	2.	State the expected inputs (futures market data) and expected outputs (trading signals & execution orders).
	3.	Identify the target environment (paper trading vs. live trading, broker APIs, computing resources).

Checks
	•	Objective Statement: The system objective is clearly declared (intraday stat-arb on futures pairs).
	•	Data Requirements: The agent has identified the data sources (e.g., real-time feed for futures, historical data for cointegration tests).
	•	Deployment: The agent specifies whether this is for local or cloud deployment (paper or real-money environment).

⸻

2. Pair Selection & Cointegration Framework

2A. Statistical Testing Suite

Implementation Steps
	1.	Implement Engle-Granger cointegration tests:
	•	Regression of y on x, then ADF test on the residuals.
	2.	Implement Johansen cointegration tests:
	•	Use a library (e.g., statsmodels or custom code) to identify multiple cointegrating vectors.
	3.	Implement Rolling Window Analysis for updating cointegration parameters every 60-day window (or a suitable rolling period).
	4.	Compute Half-Life Estimation (Ornstein-Uhlenbeck approach) to measure mean-reversion speed.
	5.	Add Out-of-Sample Validation:
	•	Split historical data into train/validation sets (e.g., 70% train, 30% validate).
	•	Evaluate cointegration stability in the validation subset.

Checks
	•	EG & Johansen: Confirm code to run both Engle-Granger and Johansen for any pair (with function definitions).
	•	Rolling Windows: Confirm sliding-window logic is implemented (and tested) to recalculate cointegration.
	•	Half-Life: Check that the function computing mean-reversion half-life is present and tested.
	•	Out-of-Sample Split: Verify that the code does a train/validation split and measures performance on the validation set (p-values, half-life in both sets).

2B. Pair Universe Management

Implementation Steps
	1.	Filter pairs using minimum correlation threshold (≥ 0.7).
	2.	Filter out pairs with excessive ratio volatility (to avoid whipsaw).
	3.	Ensure liquidity constraints: only select futures with sufficient intraday volume.
	4.	Conduct Out-of-Sample Performance checks to ensure cointegration remains stable.

Checks
	•	Correlation Filter: The agent can verify a correlation cutoff is used (e.g., 0.7).
	•	Ratio Volatility Filter: The code flags or removes instruments whose ratio changes or standard deviation is above a threshold.
	•	Liquidity Check: The agent ensures daily volume for both legs in the pair meets a defined minimum.
	•	Final Universe: Output or logging that indicates which pairs pass the selection criteria before trading.

⸻

3. Spread Calculation & Normalization

3A. Dynamic Hedge Ratio Estimation

Implementation Steps
	1.	Create a Kalman Filter module to adaptively estimate hedge ratios in near real-time or a set frequency.
	2.	Implement z-score normalization:
	•	z-score = \frac{\text{spread} - \mu(\text{spread})}{\sigma(\text{spread})}
	3.	Adjust spreads for volatility regimes (e.g., using ATR or standard deviation over a rolling window).

Checks
	•	Kalman Filter: Code includes a Kalman filter function/class that updates hedge ratio with each new data point.
	•	Z-Score: Confirm that the spread is standardized in the system with a rolling mean & std (or an exponential moving average).
	•	Volatility Adjustment: Agent code must demonstrate a parameter or function to incorporate volatility scaling (e.g., bigger threshold if volatility is high).

3B. Spread Analytics

Implementation Steps
	1.	Implement a Mean-Reversion Strength Indicator that measures how quickly the spread returns to its mean.
	2.	Add Outlier Detection to identify if the spread moves beyond historical extremes.
	3.	(Optional) Add a Seasonality Adjustment for intraday time-of-day or day-of-week effects.

Checks
	•	MR Strength Indicator: Code must produce a numeric measure (e.g., half-life or speed-of-reversion metric) displayed or logged.
	•	Extreme Spread Levels: Check for alerts or flags in the code if spread surpasses X standard deviations historically.
	•	Seasonality: If requested, the agent includes a separate routine to check time-of-day data patterns.

⸻

4. Signal Generation Engine

4A. Entry Signal Framework

Implementation Steps
	1.	Use z-score threshold for entry signals (±2.0 as default).
	2.	Combine with confirmation filters:
	•	Volume thresholds or momentum exhaustion indicators.
	3.	Add Time-of-Day constraints (avoid illiquid periods, or trade only during high liquidity windows).

Checks
	•	Z-Score Threshold: The code that triggers an entry at ±2 standard deviations.
	•	Filter: At least one volume or momentum filter is verified in the logic.
	•	Time-of-Day: Confirm that the strategy avoids certain times (e.g., lunch hour, pre-market).

4B. Exit Strategy Components

Implementation Steps
	1.	Mean-Reversion Target exit at z-score ~ ±0.5.
	2.	Max Holding Period enforced at 3 hours for intraday positions.
	3.	Trailing Exits that adjust exit thresholds if spread moves significantly.

Checks
	•	MR Exit: The agent includes a line of code that closes positions once the spread z-score normalizes back near 0.
	•	Time-Based Exit: Logic or a timer that auto-closes positions if 3 hours pass.
	•	Trailing Exit: Confirm the code supports a moving exit threshold (e.g., partial take-profit if the spread closes by half the distance).

⸻

5. Risk & Position Management

5A. Account-Appropriate Position Sizing

Implementation Steps
	1.	Implement Adaptive Volatility-Based Sizing that scales position size inversely with recent volatility.
	2.	Integrate Volatility Lookback Windows (10-day, 20-day, 30-day).
	3.	Ensure Maximum Exposure constraints (≤15% of account margin in one pair).
	4.	Add Volatility Regime Adjustments (reduce position size if short-term vol is significantly higher than long-term vol).

Checks
	•	Vol-Based Sizing: The code has a function that calculates position size as a function of spread volatility.
	•	Multiple Windows: The function references multiple vol windows (e.g., short, medium, long).
	•	Max Exposure: Check that the code has a conditional preventing more than 15% margin usage on a single trade.
	•	Regime Switch: The code reduces position size if the short-term vol >120% of long-term vol.

5B. Risk Controls

Implementation Steps
	1.	Stop Loss Mechanism:
	•	Stop out if the spread hits ±3 standard deviations or a certain monetary loss.
	2.	Daily Loss Limits set to 1% of account.
	3.	Correlation Break Protection:
	•	Force exit if correlation drops below an intraday threshold.
	4.	Monte Carlo Risk Simulation:
	•	Summarize worst-case scenarios using an Ornstein-Uhlenbeck simulation.

Checks
	•	Stop Loss: Confirm the code closes the position if the spread extends beyond a set std-dev threshold or fixed dollar loss.
	•	Daily Loss: Verify a daily net P/L check that halts trading if net loss > 1% account.
	•	Correlation Monitoring: Confirm the system monitors correlation intraday and triggers exit if under a threshold (e.g., 0.5).
	•	MC Simulation: The agent has code that runs X simulations to produce metrics like VaR, max drawdown, probability of hitting a stop.

⸻

6. Execution Optimization

6A. Order Type Selection

Implementation Steps
	1.	Build logic for simultaneous leg execution (to avoid legging risk).
	2.	Choose Limit Orders when feasible, Market Orders when necessary (fast moves).
	3.	Incorporate condition-based changes (e.g., if spread moves quickly, switch from limit to market).

Checks
	•	Simultaneous Execution: Code includes or references a function that sends both orders together (or near simultaneously).
	•	Limit vs. Market: The agent checks the current condition to decide order type.
	•	Fallback: If limit is not filled within a certain time or if volatility spikes, switch to market.

6B. Transaction Cost Mitigation

Implementation Steps
	1.	Implement a volume-weighted execution approach (look at typical volume patterns throughout the day).
	2.	Minimize rebalancing unless the hedge ratio changes significantly.
	3.	Avoid trading during known low-liquidity periods to reduce slippage.

Checks
	•	Volume Profile: The code references historical volume data by time-of-day to schedule partial fills.
	•	Hedge Ratio Drift: Only recalculate hedge ratio if it changes beyond a certain threshold.
	•	Time Filters: Confirm code that blocks or warns about trades in low-liquidity intervals (e.g., outside RTH).

⸻

7. Implementation Phases & Deliverables

Phase 1: Foundation & Research (2-3 weeks)

Implementation Steps
	1.	Data Pipeline setup for historical & real-time futures data.
	2.	Implement initial cointegration testing (EG + Johansen).
	3.	Create out-of-sample validation procedure.
	4.	Backtest a basic z-score strategy on 3-5 liquid pairs.
	5.	Document baseline performance.

Checks
	•	Data Pipeline: Confirm code can fetch and store historical & real-time data.
	•	Cointegration Tests: Basic EG & Johansen code tested on sample pairs.
	•	OOS Validation: Automated splits for training/validation.
	•	Basic Backtest: A minimal script that runs a z-score strategy from start to finish.
	•	Performance Metrics: Output includes trade count, P/L, drawdown.

Phase 2: Core Strategy Development (3-4 weeks)

Implementation Steps
	1.	Kalman Filter for dynamic hedge ratio.
	2.	Spread Calculation & Normalization with rolling stats.
	3.	Z-Score Entry/Exit modules.
	4.	Visualization Tools (charts for spread, z-score, P/L).
	5.	Realistic transaction cost assumptions in backtesting.
	6.	Volume-Weighted Execution prototypes.

Checks
	•	Kalman Filter: Confirm a dedicated function updates the hedge ratio in near real-time.
	•	Normalization: The agent references rolling mean & std or exponential moving averages.
	•	Signal Modules: Code for entry/exit triggers with adjustable thresholds.
	•	Charts: Confirm that the agent can produce or store at least one chart or plot of spread vs. time.
	•	Cost Model: The backtester includes fees, slippage, or a transaction cost.
	•	VWAP Execution: At least a prototype that schedules partial fills based on volume times.

Phase 3: Risk Management Framework (2-3 weeks)

Implementation Steps
	1.	Adaptive Position Sizing with volatility-based logic.
	2.	Stop-Loss & Max Holding logic.
	3.	Daily Risk Dashboard for P/L, margin usage, open trades.
	4.	Correlation Break Protection triggers.
	5.	Monte Carlo testing for worst-case risk.

Checks
	•	Position Sizing: Confirm references to short-term vs. long-term volatility windows.
	•	Stops & Holding: Code that forcibly closes positions after hitting the time or losing X amount.
	•	Dashboard: Some form of real-time or daily summary of risk (log, GUI, or console).
	•	Corr Threshold: The agent enforces an exit if correlation < threshold.
	•	MC Simulation: Confirm generation of multiple random spread paths & calculation of VaR.

Phase 4: Execution & Optimization (2-3 weeks)

Implementation Steps
	1.	Execution Engine with simultaneous order placement.
	2.	Volume-Weighted Execution with real data.
	3.	Optimize order types: limit vs. market vs. partial fills.
	4.	Integrate Transaction Cost Analysis (TCA).
	5.	Rebalancing Logic for updating hedge ratio only when needed.
	6.	Walk-Forward Testing for final performance checks.

Checks
	•	Execution Engine: Confirm existence of an “execute_trade” function or module.
	•	Volume-Weighted: The system checks the time-of-day volume profile before sending orders.
	•	TCA: Summaries of average slippage, fees, fill rates.
	•	Rebalance Condition: If new hedge ratio deviates from the old ratio by X%, then rebalance.
	•	Walk-Forward: Code is run in a rolling fashion (train on chunk, test on next chunk, repeat).

Phase 5: Production Deployment (Ongoing)

Implementation Steps
	1.	Paper Trade for 2 weeks to validate signals & fill assumptions.
	2.	Live Trading in minimal size after paper results are stable.
	3.	Daily Performance Monitoring & logs for executed trades.
	4.	Periodic Pair Re-Evaluation for ongoing cointegration stability.
	5.	Continuous Refinement based on real performance.

Checks
	•	Paper Trading: The agent sets up a paper (simulated) account and logs trades in real-time.
	•	Live Trading: If desired, confirm integration with a broker API or a prop firm’s platform.
	•	Monitoring: The code logs trades, P/L, and risk metrics daily.
	•	Pair Re-Eval: The system re-runs cointegration tests weekly or monthly.
	•	Version Control: The agent tracks changes that come from performance improvements.

⸻

8. Key Code Modules to Implement & Verify
	1.	test_cointegration:
	•	Check for Engle-Granger/Johansen, rolling windows, out-of-sample.
	•	Confirm the function returns statistics (p-values, half-life, validation results).
	2.	adaptive_position_sizing:
	•	Check volatility across multiple lookback windows.
	•	Confirm it respects max risk (1%) and max allocation (15%).
	3.	monte_carlo_risk_analysis:
	•	Simulates possible future spread paths (OU process).
	•	Ensure output metrics (VaR, expected shortfall, probability of hitting stop).
	4.	volume_weighted_execution:
	•	Confirm the function references a time-of-day volume profile and outputs a plan to “defer” or “execute.”

⸻

9. Additional Insights (From Provided Books)

These do not require immediate implementation steps but can inform best practices:
	•	Distance vs. Cointegration Approaches:
	•	Confirm the agent is aware that cointegration is generally more robust than pure correlation.
	•	Regime Shifts & Breakdowns:
	•	Optional advanced feature: incorporate regime detection so the system can pause trading if it detects a breakdown in the spread relationship.
	•	Machine Learning Components:
	•	Potentially add ML-based signals or factor analysis to refine pair selection or reduce false positives.

⸻

10. Final Shareable Checklist for an AI Agent

Here is a concise checklist you can give directly to the AI agent so it knows exactly what to build, step by step, and how to confirm it succeeded:
	1.	Objective Definition
	•	Output a statement clarifying the system’s purpose (intraday stat-arb, cointegrated futures pairs).
	2.	Data Pipeline
	•	Implement data retrieval (historical + real-time).
	•	Confirm data is stored and accessible in the correct structure.
	3.	Cointegration Tests
	•	Engle-Granger & Johansen in rolling windows.
	•	Out-of-sample validation with train/test split.
	4.	Pair Filtering
	•	Correlation ≥ 0.7, liquidity constraints, stable spread volatility.
	•	Output a final list of tradable pairs.
	5.	Spread & Hedge Ratio
	•	Kalman filter or static OLS for dynamic hedge ratio.
	•	Rolling z-score on the residual/spread.
	6.	Signal Generation
	•	Entry on ±2 stdev, exit near ±0.5 stdev.
	•	Time-of-day filter.
	•	Volume/momentum confirmation triggers.
	7.	Risk Management
	•	Position sizing scaled by volatility.
	•	Stop loss if ±3 stdev or big $ loss.
	•	Daily max loss of 1% account.
	•	Correlation break exit.
	•	Monte Carlo simulation for risk analysis.
	8.	Execution Engine
	•	Simultaneous order placement on both legs.
	•	Volume-weighted or limit vs. market decision.
	•	Rebalancing only on significant hedge-ratio shift.
	9.	Backtesting & Walk-Forward
	•	Basic backtest with transaction costs.
	•	Walk-forward approach for realistic results.
	10.	Deployment

	•	Paper trade for 2 weeks with minimal size.
	•	Monitor daily P/L, correlation, margin.
	•	Evaluate pairs weekly/monthly.

⸻

How to Use This Checklist With Cursor Composer
	•	Paste this entire breakdown (or the final condensed checklist) into your AI agent environment.
	•	Instruct the agent: “Implement items 1–10 in the order listed. For each item, verify success by referencing the associated checks. If a check is not met, do not proceed to the next item.”
	•	This approach enforces linear progress and ensures minimal drift.

⸻

End Goal
By following these decomposed steps and checks, your AI coding assistant should systematically build (and verify) an intraday stat-arb futures pairs trading system that is robust, well-tested, and aligned with the original design document. Good luck!