from flask import redirect, render_template, url_for
from flask import request
import requests


def index():
    return redirect(url_for('blueprint.form_view'))


def optimize():
    id = request.form['id']
    pipeline = request.files['pipeline']
    print(pipeline.filename)
    computing_infra = request.files['computing_infra']
    network_infra = request.files['network_infra']
    population_size = request.form['population_size']
    generations_check = request.form['generations_check']
    max_generations = request.form['max_generations']
    time_check = request.form['time_check']
    max_time = request.form['max_time']

    generations_check = True if generations_check == 'on' else False
    time_check = True if time_check == 'on' else False

    data = {
        'id': id,
        'population_size': int(population_size),
        'generations_check': bool(generations_check),
        'max_generations': int(max_generations),
        'time_check': bool(time_check),
        'max_time': int(max_time)
    }

    files = {
        'pipeline': (pipeline.filename, pipeline.read(), 'application/octet-stream'),
        'computing_infra': (computing_infra.filename, computing_infra.read(), 'application/octet-stream'),
        'network_infra': (network_infra.filename, network_infra.read(), 'application/octet-stream')
    }

    res = requests.post('http://192.168.0.17:8080/api/v1/optimize', data=data, files=files)
    print(res)
    return res


def form_view():
    return render_template('form.html')


def report_view():
    # return render_template('report.html')
    return "On progress"


def log_visualizer(task_id):
    # return redirect('localhost:6006:/#timeseries?runFilter={}'.format(task_id))
    return "On progress"
