from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

app = Flask(__name__)

@app.route('/', methods=('GET', 'POST'))
def cal_pressure():
    # Calculate pressdure
    result = ""

    if request.method == 'POST':
        curAge = int(request.form['age'])
        curWeight = int(request.form['weight'])
        result = load(curAge, curWeight)

    return render_template('index.html', result = result)

def train():
    df = pd.read_csv("/home/jim48949/mysite/SBP.csv")

    x = df[["Age", "Weight"]]
    y = df["SBP"]

    regr = LinearRegression()
    regr.fit(x, y)

    joblib.dump(regr, "regr.pkl")


def load(age, weight):
    clf = joblib.load("/home/jim48949/mysite/regr.pkl")
    x = pd.DataFrame([[age, weight]], columns=["Age", "Weight"])
    prediction = clf.predict(x)[0]
    #print(prediction)
    return prediction

if __name__ == '__main__':
    #train()
    app.run()