import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 10000

# Generate synthetic features
feature1 = np.random.normal(loc=0, scale=1, size=n_samples)
feature2 = np.random.normal(loc=1, scale=2, size=n_samples)
feature3 = np.random.normal(loc=2, scale=1.5, size=n_samples)
feature4 = np.random.normal(loc=3, scale=0.5, size=n_samples)

# Generate target column (binary classification)
# The target is a function of the features with some noise
target = (
    0.5 * feature1 + 1.5 * feature2 - 2.0 * feature3 + 0.8 * feature4 + np.random.normal(loc=0, scale=1, size=n_samples)
)
target = (target > target.mean()).astype(int)  # Convert to binary (0 or 1)

# Create a DataFrame
data = pd.DataFrame(
    {
        "feature1": feature1,
        "feature2": feature2,
        "feature3": feature3,
        "feature4": feature4,
        "target": target,
    }
)

# Save to CSV
data.to_csv("data/sample_data.csv", index=False)
print("sample_data.csv generated with 10,000 samples!")