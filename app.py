import pickle
import numpy as np
from flask_cors import CORS
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)
# Load the ML model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the scaler (Ensure you saved it while training)
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract input data from JSON request
        print("dsaasdfasdf")
        data = request.get_json()
        if not data or "input" not in data:
            return jsonify({"error": "Invalid input. Please send JSON with 'input' key."}), 400

        data = request.json["input"]
        print(data)
        print("--------------------------------     ")

        # Convert to numpy array and reshape for prediction
        data_array = np.array(data).reshape(1, -1)

        # Scale input using the saved scaler
        data_scaled = scaler.transform(data_array)

        # Make prediction
        prediction = model.predict(data_scaled)

        # Return prediction result
        result = "FAULTY" if prediction[0] == 1 else "NORMAL"
        print(result)
        return jsonify({"prediction": result})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Use default 5000
    app.run(host="0.0.0.0", port=port, debug=False)