from flask import Flask
from flask import render_template,request

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('hello.html')

@app.route("/send",methods=["POST"])
def send():
    question = request.form.get("message")

    return question

@app.route("/clear")
def clear():
    return "clear"
app.run(debug=True)