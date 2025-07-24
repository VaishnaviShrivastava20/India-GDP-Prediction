# -----------------------------
# Linear Regression for GDP Prediction
# -----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1) Load and clean data -------------------------------------------------------
df = pd.read_csv("India_GDP_1960-2022.csv")

# Rename columns for simplicity
df = df.rename(columns={
    "GDP in (Billion) $": "GDP",
    "Growth %": "Growth_pct"
})

# Clean GDP column: remove commas, convert to numeric, and multiply to actual USD
df["GDP"] = df["GDP"].astype(str).str.replace(",", "")
df["GDP"] = pd.to_numeric(df["GDP"], errors="coerce") * 1e9  # Convert Billion USD to actual USD

# Convert Year to numeric
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

# Drop any invalid rows
df = df.dropna(subset=["Year", "GDP"])

# 2) Prepare data for regression ----------------------------------------------
X_reg = df[["Year"]].values
y_reg = df["GDP"].values

# Split into training and testing sets
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# 3) Build a pipeline with polynomial regression ------------------------------
linreg = Pipeline([
    ("poly", PolynomialFeatures(degree=3, include_bias=False)),  # Polynomial features
    ("scaler", StandardScaler()),                               # Scale features
    ("lin", LinearRegression())                                 # Linear regression
])
linreg.fit(Xr_train, yr_train)

# 4) Evaluate the model -------------------------------------------------------
yr_pred = linreg.predict(Xr_test)

mse = mean_squared_error(yr_test, yr_pred)
rmse = np.sqrt(mse)

print("\n=== Linear Regression: Predicting GDP values ===")
print("RMSE:", rmse)
print("R^2 :", r2_score(yr_test, yr_pred))

# 5) Predict future GDP (2023–2035) -------------------------------------------
future_years = np.arange(df["Year"].max() + 1, 2036).reshape(-1, 1)
future_gdp_pred = linreg.predict(future_years)

# Print future GDP predictions
print("\n=== Predicted GDP (2023–2035) ===")
for year, gdp in zip(future_years.flatten(), future_gdp_pred):
    print(f"{year}: {gdp/1e9:.2f} Billion USD")  # Divide by 1e9 to show in billions

# 6) Plot historical + predicted GDP -----------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(df["Year"], df["GDP"]/1e9, marker="o", linestyle="-", label="Historical GDP (Billion USD)")
plt.plot(future_years.flatten(), future_gdp_pred/1e9, marker="x", linestyle="--", label="Predicted GDP (Billion USD)")
plt.title("India GDP: Historical & Predicted (Linear Regression)")
plt.xlabel("Year")
plt.ylabel("GDP (Billion USD)")
plt.legend()
plt.tight_layout()
plt.show()
