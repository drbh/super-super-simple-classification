import threading
from flask import Flask
from flask import request
from flask import render_template

from datetime import datetime
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import json


import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

path = "./config/"
CONFIG = {}


class MyHandler(FileSystemEventHandler):
    def on_modified(self, event):
        resetConfig()


def watch():
    event_handler = MyHandler()
    observer = Observer()
    observer.schedule(event_handler, path=path, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def resetConfig():
    global CONFIG
    with open('config/groups.json') as json_file:
        data = json.load(json_file)
        CONFIG = data
    print("Reset Configuration File")
    # print(json.dumps(CONFIG, indent=4))


def getLabels():
    return CONFIG


thr = threading.Thread(target=watch, args=(), kwargs={})
thr.start()

app = Flask(__name__, static_folder="static")

module_url = "./universal-sentence-encoder-large"

# set inital config
resetConfig()

# Create graph and finalize (optional but recommended).
g = tf.Graph()
with g.as_default():
    text_input = tf.placeholder(dtype=tf.string, shape=[None])
    embed = hub.Module(module_url)
    my_result = embed(text_input)
    init_op = tf.group(
        [tf.global_variables_initializer(), tf.tables_initializer()])


g.finalize()

# Create session and initialize.
session = tf.Session(graph=g)
session.run(init_op)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@app.route("/", methods=['GET'])
def land():
    return render_template('index.html')


@app.route("/groups", methods=['POST'])
def groups():
    start = datetime.now()
    return json.dumps({"status": "reset"})


@app.route("/similar", methods=['GET', 'POST'])
def similar():
    start = datetime.now()
    data = json.loads(request.data)
    my_result_out = session.run(
        my_result, feed_dict={text_input: [data["a"], data["b"]]})
    corr = np.inner(my_result_out, my_result_out)
    diff = datetime.now() - start
    elapsed_ms = (diff.days * 86400000) + \
        (diff.seconds * 1000) + (diff.microseconds / 1000)
    return json.dumps({
        "value": float(corr[0][1]),
        "latency": elapsed_ms}, cls=NumpyEncoder)


@app.route("/classify", methods=['GET', 'POST'])
def classify():
    start = datetime.now()
    data = json.loads(request.data)

    results = {}

    # 400
    labels = getLabels()

    for key, item in labels.items():
        mylist = [data["a"]] + item

        my_result_out = session.run(
            my_result, feed_dict={text_input: mylist})
        corr = np.inner(my_result_out, my_result_out)
        val = [float(x) for x in corr[0]]
        val.pop(0)  # remove self
        results[key] = val

    diff1 = datetime.now() - start

    summaries = {}
    for key, data in results.items():

        quartiles = np.percentile(data, [25, 50, 75, 90])
        # calculate min/max
        data_min, data_max, data_std = np.min(data), np.max(data), np.std(data)

        summaries[key] = {
            'min:':  data_min,
            'q1:':  quartiles[0],
            'median:': quartiles[1],
            'q3:':  quartiles[2],
            'p90:':  quartiles[3],
            'max:':  data_max,
            "n": len(data),
            "data_std": data_std,
            "range": data_max - data_min,
        }

    diff = datetime.now() - start
    elapsed_ms = (diff.days * 86400000) + \
        (diff.seconds * 1000) + (diff.microseconds / 1000)
    elapsed_ms1 = (diff1.days * 86400000) + \
        (diff1.seconds * 1000) + (diff1.microseconds / 1000)
    return json.dumps({
        # "results": results,
        "summaries": summaries,
        "latency": elapsed_ms,
        "latency1": elapsed_ms1
    }, cls=NumpyEncoder)


if __name__ == "__main__":
    app.run(host='0.0.0.0')
