import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Constants
PD = 0.01  # Probability of Default (1%)
LGD = 0.40  # Loss Given Default (40%)
EAD = 1_000_000  # Exposure at Default ($1 million)
M = 1  # Maturity
R = 0.15  # Correlation factor
c_range = np.linspace(0, 2, 100)  # Climate add-on range from 0 to 2

# Function to calculate the Basel maturity adjustment factor b
def calculate_b(PD):
    return (0.11852 - 0.05478 * np.log(PD)) ** 2

# Function to calculate K with intermediate steps
def calculate_K(PD, LGD, R, M):
    phi_inv_PD = norm.ppf(PD)  
    phi_inv_999 = norm.ppf(0.999)  
    sqrt_R = np.sqrt(R)  
    sqrt_1_minus_R = np.sqrt(1 - R)  

    # Term inside the normal CDF
    adjusted_term = (phi_inv_PD + sqrt_R * phi_inv_999) / sqrt_1_minus_R
    N_adjusted = norm.cdf(adjusted_term) 

    # First term of K
    first_term = LGD * N_adjusted - LGD * PD

    # Maturity adjustment
    b = calculate_b(PD)
    maturity_adjustment = (1 + (M - 2.5) * b) / (1 - 1.5 * b)

    # Final K
    K = first_term * maturity_adjustment
    return K

# Base RWA Calculation
K_base = calculate_K(PD, LGD, R, M)
RWA_base = 12.5 * EAD * K_base

# Climate-adjusted PD
def climate_adjusted_PD(PD, c):
    """Adjusts PD based on climate add-on parameter c."""
    return 1 / (1 + np.exp(-np.log(PD / (1 - PD)) - c))

# Climate-adjusted RWA as a function of c
def climate_adjusted_RWA(c):
    adj_PD = climate_adjusted_PD(PD, c)
    K_climate = calculate_K(adj_PD, LGD, R, M)
    return 12.5 * EAD * K_climate

# Compute climate-adjusted RWA and ratio
climate_RWA = np.array([climate_adjusted_RWA(c) for c in c_range])
RWA_ratio = climate_RWA / RWA_base

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(c_range, RWA_ratio, label="Climate RWA / Base RWA", color="blue")
plt.title("Climate RWA to Base RWA Ratio vs Climate Add-on (c)")
plt.xlabel("Climate Add-on (c)")
plt.ylabel("RWA Ratio (Climate RWA / Base RWA)")
plt.legend()
plt.grid(True)
plt.show()
