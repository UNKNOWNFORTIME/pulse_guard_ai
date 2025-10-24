import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

# ✅ 1. Load dataset
data = pd.read_csv(r"C:\Users\SEAN\Downloads\cleaned_transformer_dataset212.csv")

# ✅ 2. Force-remove commas and clean spaces
for col in data.columns:
    data[col] = data[col].astype(str).str.replace(',', '').str.strip()

# ✅ 3. Identify your target column
target_col = "Burned_transformers_2019"

# ✅ 4. Convert all columns EXCEPT target to numeric where possible
for col in data.columns:
    if col != target_col:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# ✅ 5. Encode target if it’s categorical
if data[target_col].dtype == 'object':
    data[target_col] = LabelEncoder().fit_transform(data[target_col])

# ✅ 6. Replace NaN values with 0
data = data.fillna(0)

# ✅ 7. Separate features and target
X = data.drop(target_col, axis=1)
y = data[target_col]

# ✅ 8. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ 9. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ 10. Evaluate accuracy
accuracy = model.score(X_test, y_test)
print(f"✅ Model accuracy: {accuracy * 100:.2f}%")

# ✅ 11. Save model
joblib.dump(model, "transformer_failure_model.pkl")
print("🎉 Model saved as transformer_failure_model.pkl")

