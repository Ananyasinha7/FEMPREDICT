from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        data = [float(request.form.get(k)) for k in request.form]
        prediction = model.predict([data])[0]
        return render_template("index.html", result=prediction)
    return render_template("index.html")


def handler(environ, start_response):
    return app(environ, start_response)
