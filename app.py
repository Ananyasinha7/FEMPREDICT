from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)


rf = pickle.load(open("pcos_rf_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = request.form
        features = [
            float(form_data["age"]),
            float(form_data["weight"]),
            float(form_data["height"]),
            int(form_data["blood_group"]),
            int(form_data["period_freq"]),
            int(form_data["weight_gain"]),
            int(form_data["hair_growth"]),
            int(form_data["skin_darkening"]),
            int(form_data["hair_loss"]),
            int(form_data["acne"]),
            int(form_data["fast_food"]),
            int(form_data["exercise"]),
            int(form_data["mood_swings"]),
            int(form_data["regular_periods"]),
            int(form_data["period_length"]),
        ]

        user_data = np.array([features])
        prediction = rf.predict(user_data)[0]

        if prediction == 1:
            message = "The model predicts that you may have PCOS. Please consult a healthcare professional for further evaluation."
        else:
            message = "The model predicts that you are unlikely to have PCOS. Still, consulting a healthcare provider is always a good idea."

        return render_template("result.html", prediction_message=message)

    except Exception as e:
        return render_template("result.html", prediction_message=f"Error in prediction: {e}")

if __name__ == "__main__":
    app.run(debug=True)
