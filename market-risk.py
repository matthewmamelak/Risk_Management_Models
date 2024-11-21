import numpy as np
from scipy.stats import norm

# Black-Scholes formula for a European call option
def black_scholes_call(S, K, T, r, q, sigma):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# Initial parameters (yesterday's values)
K = 105                # Strike price
S_yesterday = 100      # Yesterday's stock price
T_yesterday = 1        # Time to maturity in years
r_yesterday = 0.02     # Yesterday's risk-free rate
q_yesterday = 0.04     # Yesterday's dividend yield
sigma_yesterday = 0.30 # Yesterday's volatility

# Today's values after parameter changes
S_today = 101          # Today's stock price
T_today = T_yesterday - 1/252  # One day less to maturity (assuming 252 trading days in a year)
r_today = 0.021        # Today's risk-free rate
q_today = 0.04         # Today's dividend yield (unchanged)
sigma_today = 0.32     # Today's volatility

# Step-by-step P&L attribution
pnl_attribution = []

# 1. Initial call price
initial_price = black_scholes_call(S_yesterday, K, T_yesterday, r_yesterday, q_yesterday, sigma_yesterday)
pnl_attribution.append(("Initial Price", initial_price))

# 2. Update for stock price
stock_price_change = black_scholes_call(S_today, K, T_yesterday, r_yesterday, q_yesterday, sigma_yesterday) - initial_price
pnl_attribution.append(("Stock Price Change", stock_price_change))

# 3. Update for time decay
time_decay_change = black_scholes_call(S_today, K, T_today, r_yesterday, q_yesterday, sigma_yesterday) - \
                    black_scholes_call(S_today, K, T_yesterday, r_yesterday, q_yesterday, sigma_yesterday)
pnl_attribution.append(("Time Decay Change", time_decay_change))

# 4. Update for dividend yield
dividend_yield_change = black_scholes_call(S_today, K, T_today, r_yesterday, q_today, sigma_yesterday) - \
                        black_scholes_call(S_today, K, T_today, r_yesterday, q_yesterday, sigma_yesterday)
pnl_attribution.append(("Dividend Yield Change", dividend_yield_change))

# 5. Update for risk-free rate
risk_free_rate_change = black_scholes_call(S_today, K, T_today, r_today, q_today, sigma_yesterday) - \
                        black_scholes_call(S_today, K, T_today, r_yesterday, q_today, sigma_yesterday)
pnl_attribution.append(("Risk-Free Rate Change", risk_free_rate_change))

# 6. Update for volatility
volatility_change = black_scholes_call(S_today, K, T_today, r_today, q_today, sigma_today) - \
                    black_scholes_call(S_today, K, T_today, r_today, q_today, sigma_yesterday)
pnl_attribution.append(("Volatility Change", volatility_change))

# 7. Final call price
final_price = black_scholes_call(S_today, K, T_today, r_today, q_today, sigma_today)
pnl_attribution.append(("Final Price", final_price))

# Display the P&L attribution
for change, value in pnl_attribution:
    print(f"{change}: {value:.4f}")

# Reverse order of P&L attribution for comparison
# (Repeat above steps in reverse order and compare results)
