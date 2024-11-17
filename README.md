# Heston Model Asset Pricing and Implied Volatility

## Introduction
This project simulates asset prices and variance under the **Heston model**, a stochastic volatility model. It generates asset prices and implied volatilities for options pricing, allowing for the calculation of both **European call and put option prices**. The model also enables the estimation of **implied volatilities** based on option prices.

The **Heston model** is widely used in financial mathematics to model the stochastic volatility of asset prices, providing more realistic modeling compared to simpler models like Geometric Brownian Motion (GBM). In this project, we simulate asset prices and variance, perform Monte Carlo simulations, and derive implied volatilities for European options.

## Methodology

### Heston Stochastic Volatility Model
The Heston model assumes that asset prices and volatility are driven by a set of correlated stochastic processes. The asset price is modeled as:  
<br>![](https://quicklatex.com/cache3/82/ql_42fc16c9e81cb0ed4b73a2b29d399782_l3.png)

The variance follows a mean-reverting process:  
<br>![](https://quicklatex.com/cache3/12/ql_eb664787e3888fd6dea6766a57c66812_l3.png)  
<br>The asset price ![](https://quicklatex.com/cache3/74/ql_0870d8c7a29dbaf572ae5493ec851274_l3.png) and the variance ![](https://quicklatex.com/cache3/f8/ql_762042dcf9425b2c639a588b72d1d5f8_l3.png) are simulated using correlated Brownian motions with parameters defined by the user.

### Implied Volatility Calculation
Implied volatility is the volatility value that, when input into an option pricing model, returns the market price of the option. In this project, the implied volatility for both **call** and **put options** is calculated using the `py_vollib_vectorized` library, which provides an efficient method to compute implied volatilities from option prices.

The formula for implied volatility is derived from the Black-Scholes model, adjusted for market conditions.

## Approach

1. **Simulation of Asset Prices and Volatility:**
   The Heston model is implemented using a discretized version of the SDEs for asset prices and variance, with correlated random variables to simulate the price dynamics.

2. **Monte Carlo Simulations:**
   Monte Carlo simulations are used to generate multiple scenarios of asset prices and variance, which are used to calculate option prices and implied volatilities.

3. **Implied Volatility Smile:**
   The implied volatility smile is computed for a range of strikes and plotted against strike prices to visualize the volatility skew in the market.

## Results

### Simulated Asset Prices and Variance
The plots illustrate the Heston model's ability to capture the positive correlation between asset prices and their volatility. The asset price paths and variance process exhibit synchronous fluctuations, confirming this relationship.  
![](https://github.com/1Aditya7/VaR-and-CVaR-Modelling/blob/main/hestonMedia/assetsAndVariance.png)


### Implied Volatility Smile
The plot shows the implied volatility smile from the Heston model. The implied volatility for call options is higher for out-of-the-money and at-the-money strikes, while the implied volatility for put options is higher for deep out-of-the-money strikes. This pattern is consistent with the volatility smile observed in real market data.  
![Implied Volatility Smile](https://github.com/1Aditya7/VaR-and-CVaR-Modelling/blob/main/hestonMedia/impliedVolatility.png)

## Code

### Heston Model Simulation

```python
import numpy as np

def heston_model_sim(S0, v0, rho, kappa, theta, sigma, T, N, M):
    # Initialise other parameters
    dt = T / N
    mu = np.array([0, 0])
    cov = np.array([[1, rho], [rho, 1]])

    # Arrays for storing prices and variances
    S = np.full(shape=(N+1, M), fill_value=S0)
    v = np.full(shape=(N+1, M), fill_value=v0)

    # Sampling correlated Brownian motions under risk-neutral measure
    Z = np.random.multivariate_normal(mu, cov, (N, M))

    for i in range(1, N+1):
        S[i] = S[i-1] * np.exp((r - 0.5 * v[i-1]) * dt + np.sqrt(v[i-1] * dt) * Z[i-1, :, 0])
        v[i] = np.maximum(v[i-1] + kappa * (theta - v[i-1]) * dt + sigma * np.sqrt(v[i-1] * dt) * Z[i-1, :, 1], 0)

    return S, v
```

### Option Pricing and Implied Volatility Calculation

```python
from py_vollib_vectorized import vectorized_implied_volatility as implied_vol

puts = np.array([np.exp(-r * T) * np.mean(np.maximum(k - S, 0)) for k in K])
calls = np.array([np.exp(-r * T) * np.mean(np.maximum(S - k, 0)) for k in K])

put_ivs = implied_vol(puts, S0, K, T, r, flag='p', q=0, return_as='numpy', on_error='ignore')
call_ivs = implied_vol(calls, S0, K, T, r, flag='c', q=0, return_as='numpy')
```

## Limitations and Future Scope
- **Parameter Sensitivity:** The accuracy of the model depends heavily on the choice of model parameters. Future improvements could include parameter calibration based on market data.
- **Numerical Stability:** In some cases, numerical instability may arise due to the discretization of stochastic processes, which could be addressed with more refined numerical schemes.
- **Real Market Data:** The model assumes risk-neutral pricing and may not fully capture real-world dynamics. Future work could focus on calibrating the model to real market data to improve accuracy.

## Conclusion
This project demonstrates the application of the Heston model to simulate asset prices and estimate implied volatilities for European options. By simulating both positive and negative correlations between asset returns and variance, we were able to observe how these correlations influence the asset price dynamics and the implied volatility smile.
