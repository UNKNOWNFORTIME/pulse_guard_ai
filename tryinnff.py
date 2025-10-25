import joblib
import pandas as pd

# Load the trained model
model = joblib.load("transformer_failure_model.pkl")



# Load the new dataset
new_data = pd.read_csv("C:\Users\SEAN\Downloads\Dataset_Year_2020_Iworkdone2.csv")

# Check columns
print(new_data.columns)
print(new_data.head())

# List the features used in training
feature_columns = ["LOCATION", "POWER", "SELF-PROTECTION", "Average earth discharge density DDT [Rays/km^2-a駉]", "Maximum ground discharge density DDT [Rays/km^2-a駉]", "Burning rate  [Failures/year]", "Criticality according to previous study for ceramics level", "Removable connectors", "Type of clients", "Number of users", "Electric power not supplied EENS [kWh]", "Type of installation", "Air network", "Circuit Queue", "km of network LT:", "Burned transformers 2020"]


# Extract features from new dataset
X_new = new_data[feature_columns]
# Predict class labels (0 = healthy, 1 = likely to fail)
predictions = model.predict(X_new)

# Predict probabilities (optional)
probabilities = model.predict_proba(X_new)[:, 1]  # probability of failure

# Add predictions to your dataframe
new_data["predicted_failure"] = predictions
new_data["failure_probability"] = probabilities

# See the results
print(new_data.head())

