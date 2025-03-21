# Reinforcement Learning for Algorithmic Trading

## Project Overview

This project explores the application of Reinforcement Learning (RL) for automated trading within financial markets. Built primarily for educational purposes only.

## Methodology

### Feature Engineering

Due to my background not being in quantitative finance, I leverage publicly available technical indicators as input features. These include:

* **Average True Range (ATR)**: Measures market volatility.
* **Bollinger Bands (BB)**: Indicate price volatility and potential overbought/oversold conditions.

For additional feature inspiration, I referenced the research paper ["101 Formulaic Alphas"](https://arxiv.org/pdf/1601.00991).

### Reinforcement Learning Approach

The core challenge lies in designing an effective reward function that aligns with desired trading behaviors. The initial reward function aimed to incentivize:

* Outperforming a benchmark.
* Achieving positive Profit and Loss (PNL).

However, the agent hack the reward function , exploiting its limitations rather than learning genuine trading strategies.

### Reward Function Limitations

The primary weaknesses of the initial reward function were:

* **Oversimplification of Market Dynamics:** It assumed that market outperformance could be solely achieved through a predefined set of rewards, neglecting the nuanced and unpredictable nature of financial markets.
* **Exclusive Reliance on PNL:** PNL alone is insufficient to evaluate trading performance. Risk management and other critical factors were overlooked.

These limitations led to unintended agent behaviors, as illustrated in the following observations:

* **Premature Termination:** The agent frequently executed a series of detrimental trades, depleting the available capital and prematurely ending the trading episode.
    * ![Premature Termination](https://github.com/Alireza93336333393/A2C_for_trading/blob/main/Screenshot%20from%202025-03-20%2010-34-50%20(1).png)
    * the length of the data that was traind on (3685)
* **Bias Towards Buy-and-Hold:** The agent learned to favor a buy-and-hold strategy, as this behavior consistently yielded higher rewards within the flawed reward structure.
    * ![Buy and Hold 1](https://github.com/Alireza93336333393/A2C_for_trading/blob/main/Screenshot%20from%202025-03-21%2008-30-10.png)
    * ![Buy and Hold 2](https://github.com/Alireza93336333393/A2C_for_trading/blob/main/Screenshot%20from%202025-03-21%2008-32-28.png)
* **Reward Maximization at the Expense of losing money:** The agent prioritized maximizing rewards, even if it resulted in financial losses.
    * ![Reward vs PNL](https://github.com/Alireza93336333393/A2C_for_trading/blob/main/Screenshot%20from%202025-03-20%2010-34-50.png)
    * This demonstrates that solely encouraging high PNL and outperforming the market without considering risk can lead to undesirable outcomes.

## Future Directions (TODO)

* [ ] **Refine the Reward Function:** Develop a more robust reward function.
* [ ] **Explore Alternative RL Algorithms:** Investigate the performance of other RL algorithms.
* [ ] **Incorporate Advanced Features:** Explore the use of more sophisticated features beyond basic technical indicators.
* [ ] **Add Risk Managment**: add stop losses and take profits to the agent.

## Trading Environment

The project utilizes the `gym-trading-env` library, which provides a flexible and customizable trading environment. Understanding the `History Object` within this environment is crucial for implementing custom reward functions.
* [gym-trading-env Documentation](https://gym-trading-env.readthedocs.io/en/latest/)
* [History Object Documentation](https://gym-trading-env.readthedocs.io/en/latest/history.html)
