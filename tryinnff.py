import joblib
import pandas as pd

# Load the trained model
model = joblib.load("transformer_failure_model.pkl")

# ✅ 1. Load the new dataset (with 'r' for the path)
try:
    new_data = pd.read_csv(r"C:\Users\SEAN\Downloads\Dataset_Year_2020_Iworkdone2.csv")
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

print("Columns found in new CSV:")
print(new_data.columns)

# ✅ 2. List the features *used in training*
# ⚠️ WARNING: This list MUST match your ACTUAL_AI_TEAM.py script.
# I have REMOVED "Burned transformers 2020" because it is a target, not a feature.
feature_columns = [
    "LOCATION", 
    "POWER", 
    "SELF-PROTECTION", 
    "Average earth discharge density DDT [Rays/km^2-a駉]", 
    "Maximum ground discharge density DDT [Rays/km^2-a駉]", 
    "Burning rate  [Failures/year]", 
    "Criticality according to previous study for ceramics level", 
    "Removable connectors", 
    "Type of clients", 
    "Number of users", 
    "Electric power not supplied EENS [kWh]", 
    "Type of installation", 
    "Air network", 
    "Circuit Queue", 
    "km of network LT:"
]

# ✅ 3. Extract features from new dataset
try:
    X_new = new_data[feature_columns]
except KeyError as e:
    print(f"KeyError: {e}")
    print("A column in your 'feature_columns' list is NOT in your new CSV file.")
    print("Please check the column names above and fix your list.")
    exit()

# ✅ 4. Make predictions
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)[:, 1]  # Get probability of '1' (failure)

# Add predictions to your dataframe
new_data["predicted_failure"] = predictions
new_data["failure_probability"] = probabilities

# See the results
print("\n--- Predictions ---")
print(new_data[["LOCATION", "POWER", "predicted_failure", "failure_probability"]].head())

# Save the results to a new CSV (optional)
new_data.to_csv("transformer_predictions_2020.csv", index=False)
print("Saved predictions to 'transformer_predictions_2020.csv'")
