import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Constants
PD = 0.01  # Probability of Default (1%)
LGD = 0.40  # Loss Given Default (40%)
EAD = 1000000  # Exposure at Default ($1 million)
M = 1  # Maturity
R = 0.15  # Correlation factor
c_range = np.linspace(0, 2, 100)  # Climate add-on range from 0 to 2

# Base RWA Calculation
K_base = LGD * norm.cdf(
    (norm.ppf(PD) + np.sqrt(R) * norm.ppf(0.999)) / np.sqrt(1 - R)
) - LGD * PD
RWA_base = 12.5 * EAD * K_base

# Climate-adjusted RWA as a function of c
def climate_adjusted_PD(PD, c):
    """Adjusts PD based on climate add-on parameter c."""
    return 1 / (1 + np.exp(-np.log(PD / (1 - PD)) - c))


def climate_adjusted_RWA(c):
    adj_PD = climate_adjusted_PD(PD, c)
    K_climate = LGD * norm.cdf(
        (norm.ppf(adj_PD) + np.sqrt(R) * norm.ppf(0.999)) / np.sqrt(1 - R)
    ) - LGD * adj_PD
    return 12.5 * EAD * K_climate


climate_RWA = np.array([climate_adjusted_RWA(c) for c in c_range])

# Ratio of climateRWA to baseRWA
RWA_ratio = climate_RWA / RWA_base

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(c_range, RWA_ratio, label="Climate RWA / Base RWA")
plt.axhline(y=1, color="r", linestyle="--", label="Base RWA Reference (Ratio=1)")
plt.title("Ratio of Climate RWA to Base RWA vs Climate Add-on (c)")
plt.xlabel("Climate Add-on (c)")
plt.ylabel("RWA Ratio (Climate RWA / Base RWA)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()