from flask import redirect, render_template, url_for
import requests


def index():
    return redirect(url_for('blueprint.form_view'))


def optimize(pipeline, computing_infra, network_infra, population_size, max_generations, max_time):
    form_data = {
        'pipeline': pipeline,
        'computing_infra': computing_infra,
        'network_infra': network_infra,
        'population_size': population_size,
        'max_generations': max_generations,
        'max_time': max_time
    }
    # res = requests.post('localhost:8080/optimize', data=form_data)

    print("Hello from controller optimization Calibra Server")
    return "Hello from controller optimization Calibra Server"


def form_view():
    return render_template('form.html')


def report_view():
    # return render_template('report.html')
    return "On progress"


def log_visualizer(task_id):
    # return redirect('localhost:6006:/#timeseries?runFilter={}'.format(task_id))
    return "On progress"
