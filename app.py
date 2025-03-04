from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("churn_model.pkl")

# print(X.shape)

@app.route("/")
def home():
    return "Welcome to the Churn Prediction API!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Convert data to NumPy array (assuming input is a list of feature values)
        features = np.array(data["features"]).reshape(1, -1)

        print("Received features shape:", features.shape)
        print("Expected feature count:", len(model.feature_names_in_))
        print("Feature names:", model.feature_names_in_)

        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].tolist()  

        # Return response
        return jsonify({
            "prediction": int(prediction),  
            "probability": probability
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
