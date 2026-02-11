from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained pipeline
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = {
        "PID": [int(request.form["PID"])],
        "Neighborhood": [request.form["Neighborhood"]],
        "Year Built": [int(request.form["Year Built"])],
        "Overall Qual": [int(request.form["Overall Qual"])],
        "Kitchen Qual": [request.form["Kitchen Qual"]],
        "Exter Qual": [request.form["Exter Qual"]],
        "Lot Area": [int(request.form["Lot Area"])]
    }

    df = pd.DataFrame(data)

    prediction = model.predict(df)[0]

    return f"<h2>Predicted House Price: ${prediction:,.2f}</h2>"

if __name__ == "__main__":
    app.run()

