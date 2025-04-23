from flask import Flask, render_template, redirect, url_for
import threading
import subprocess
import signal
import os

app = Flask(__name__)
process = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start')
def start():
    global process
    if process is None:
        process = subprocess.Popen(["python", "focus_analyzer.py"])
    return redirect(url_for('home'))

@app.route('/stop')
def stop():
    global process
    if process:
        os.kill(process.pid, signal.SIGTERM)
        process = None
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
