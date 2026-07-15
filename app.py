
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    annual_rainfall = float(request.form["AnnualRainfall"])
    cloud_visibility = float(request.form["CloudVisibility"])
    temperature = float(request.form["Temperature"])
    humidity = float(request.form["Humidity"])
    seasonal_rainfall = float(request.form["SeasonalRainfall"])

    features = np.array([[annual_rainfall, cloud_visibility, temperature, humidity, seasonal_rainfall]])
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]

    if prediction == 1:
        prediction_text = "High Flood Risk - Flood Likely"
        css_class = "flood"
    else:
        prediction_text = "Low Flood Risk - No Flood Expected"
        css_class = "noflood"

    return render_template("result.html", prediction_text=prediction_text, css_class=css_class)

if __name__ == "__main__":
    app.run(debug=True)
