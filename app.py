from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("linear_model.joblib")

@app.route("/", methods=["GET","POST"])
def index():
    pred = None
    if request.method == "POST":
        hours = float(request.form['hours'])
        pred = model.predict(np.array([[hours]]))[0]
        pred = round(pred, 2)
    return render_template("index.html", prediction=pred)

if __name__ == "__main__":
    app.run(debug=True)
