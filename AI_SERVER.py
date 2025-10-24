from flask import Flask, request, jsonify
import joblib
import os
import hmac
from functools import wraps

app = Flask(__name__)

# Load model from environment variable or default
MODEL_PATH = os.getenv("MODEL_PATH", "transformer_failure_model.pkl")
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# Load API key from environment variable
API_KEY = os.getenv("API_KEY")

# Decorator to protect endpoints with API key
def require_api_key(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth.split(" ", 1)[1].strip()
        else:
            return jsonify({"error": "Missing API key"}), 401

        if not API_KEY or not hmac.compare_digest(token, API_KEY):
            return jsonify({"error": "Invalid API key"}), 403
        return func(*args, **kwargs)
    return wrapper

# Root route
@app.route("/", methods=["GET"])
def index():
    return "Transformer Failure Prediction API is running!", 200

# Health check route
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

# Prediction route
@app.route("/predict", methods=["POST"])
@require_api_key
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    if not data or "input" not in data:
        return jsonify({"error": "JSON must contain 'input' field"}), 400

    features = data["input"]
    # Ensure 2D array
    if isinstance(features[0], (int, float)):
        features = [features]

    try:
        prediction = model.predict(features)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Development use only
    app.run(host="0.0.0.0", port=5000, debug=False)
