from flask import Flask, render_template, request
import numpy as np
import joblib

# Load the trained model
model_path = "best_mlp_model_scaled6.pkl"
clf = joblib.load(model_path)

scaler_path = "scaler6.pkl"
scaler = joblib.load(scaler_path)

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
         # Capture the employee name from the form
        employee_name = request.form["employee_name"]
        
        # Extract input features from the form
        features = [float(request.form[key]) for key in [
            "satisfaction_level", "last_evaluation", "number_project",
            "average_montly_hours", "time_spend_company", "Work_accident",
            "promotion_last_5years", "Departments", "salary"
        ]]
        features = np.array(features).reshape(1, -1)

        # Scale the input features
        scaled_features = scaler.transform(features)

       
        # Predict using the loaded model
        prediction = clf.predict(scaled_features)
        result = "will likely leave" if prediction[0] == 1 else "will likely stay"

        # Combine result with the employee name
        final_result = f"Employee {employee_name} {result}"
    except Exception as e:
        final_result = f"Error occurred: {e}"

    # Pass the result and employee name back to the frontend
    return render_template("index.html", prediction_text=final_result)

if __name__ == "__main__":
    app.run(debug=True)
