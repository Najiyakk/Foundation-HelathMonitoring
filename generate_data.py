
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# GEOFLUX: Synthetic Foundation Reference Data
# These are simulated values, NOT real sensor measurements.

np.random.seed(42)

# Applied loads in kilonewtons
loads = np.arange(100, 1100, 100)

# Assumed reference stiffness (kN/mm)
reference_stiffness = 100.0

# Underlying settlement relationship
ideal_settlement = loads / reference_stiffness

# Add simulated sensor measurement noise
noise = np.random.normal(0, 0.03, len(loads))
synthetic_settlement = ideal_settlement + noise

# Create the dataset
data = pd.DataFrame({
    "load_kN": loads,
    "synthetic_settlement_mm": np.round(
        synthetic_settlement, 4
    )
})

# Save the CSV file
data.to_csv("synthetic_reference_data.csv", index=False)

print("\nGEOFLUX Synthetic Reference Data")
print(data.to_string(index=False))
print("\nSaved: synthetic_reference_data.csv")

# Plot the reference data
plt.figure(figsize=(9, 5))

plt.scatter(
    loads,
    synthetic_settlement,
    label="Synthetic reference data"
)

plt.plot(
    loads,
    ideal_settlement,
    "--",
    label="Underlying assumed relationship"
)

plt.xlabel("Applied Load (kN)")
plt.ylabel("Settlement (mm)")
plt.title("GEOFLUX Foundation Settlement")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("synthetic_reference_plot.png", dpi=300)
plt.show()
