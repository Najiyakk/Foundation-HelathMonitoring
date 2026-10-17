import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# GEOFLUX preliminary foundation model
# Simplified spring idealization, not a full FEM mesh.

INPUT_FILE = "synthetic_reference_data.csv"

# Load the synthetic reference dataset
data = pd.read_csv(INPUT_FILE)

loads = data["load_kN"].to_numpy()
reference = data["synthetic_settlement_mm"].to_numpy()

# Initial assumed model stiffness (kN/mm)
initial_stiffness = 80.0

# Model prediction
initial_prediction = loads / initial_stiffness

# Compare initial predictions with the reference
initial_rmse = np.sqrt(
    np.mean((reference - initial_prediction) ** 2)
)

print("GEOFLUX: Initial Model Comparison")
print("---------------------------------")
print(f"Initial stiffness: {initial_stiffness:.2f} kN/mm")
print(f"Initial RMSE: {initial_rmse:.4f} mm")

# Plot the comparison
plt.figure(figsize=(9, 5))

plt.scatter(
    loads,
    reference,
    label="Synthetic reference data"
)

plt.plot(
    loads,
    initial_prediction,
    "r--",
    label="Initial model prediction"
)

plt.xlabel("Applied Load (kN)")
plt.ylabel("Settlement (mm)")
plt.title("GEOFLUX: Initial Model Comparison")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("initial_model_comparison.png", dpi=300)
plt.show()
