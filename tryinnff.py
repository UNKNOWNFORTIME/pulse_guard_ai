import joblib
import pandas as pd

# Load the trained model
model = joblib.load("transformer_failure_model.pkl")

# Load the new dataset
new_data = pd.read_csv(r"C:\Users\SEAN\Downloads\Dataset_Year_2020_Iworkdone2.csv")

# Check columns
print("--- Columns found in new file: ---")
print(new_data.columns)

# Define the features the model was trained on
feature_columns = [
    "LOCATION", 
    "POWER", 
    "SELF_PROTECTION", 
    "Average_earth_discharge_density_DDT_Rays_km_2_a_o", 
    "Maximum_ground_discharge_density_DDT_Rays_km_2_a_o", 
    "Burning_rate_Failures_year", 
    "Criticality_according_to_previous_study_for_ceramics_level", 
    "Removable_connectors", 
    "Type_of_clients", 
    "Number_of_users", 
    "Electric_power_not_supplied_EENS_kWh", 
    "Type_of_installation", 
    "Air_network", 
    "Circuit_Queue", 
    "km_of_network_LT"]


# --- ✅ NEW CLEANING STEP ---
# This forces all your feature columns to be numeric.
# Text like 'STRATUM 2' will become NaN (Not a Number).
print("\n--- Cleaning and converting data to numeric... ---")
for col in feature_columns:
    if col in new_data.columns:
        # 'errors='coerce'' turns all non-numeric text into NaN
        new_data[col] = pd.to_numeric(new_data[col], errors='coerce')

# 'fillna(0)' replaces all NaNs (like 'STRATUM 2') with 0
new_data = new_data.fillna(0)
print("Cleaning complete.")
# --- END OF NEW STEP ---

# Extract features from new dataset
print("\n--- Features being sent to model: ---")
X_new = new_data[feature_columns]

# Predict class labels (0 = healthy, 1 = likely to fail)
print("\n--- Running prediction... ---")
predictions = model.predict(X_new)  # This will now work

# Predict probabilities (optional)
probabilities = model.predict_proba(X_new)[:, 1]  # probability of failure

# Add predictions to your dataframe
new_data["predicted_failure"] = predictions
new_data["failure_probability"] = probabilities

# See the results
print("\n--- Prediction Results (Top 5 rows): ---")
print(new_data[["LOCATION", "POWER", "predicted_failure", "failure_probability"]].head())

# Save all results to a new file
new_data.to_csv("predictions_2020.csv", index=False)
print("\n🎉 Success! Full results saved to 'predictions_2020.csv'")