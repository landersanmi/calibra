import io
import base64
import time
import threading
import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import csv
import tempfile
import os

from flask import redirect, render_template, url_for, session
from flask import request
from io import BytesIO, StringIO
from datetime import datetime

IMAGE_DIRECTORY = './tmp'
IMAGE_LIFETIME = 20  #seconds


def index():
    return redirect(url_for('blueprint.form_view'))


def form_view():
    return render_template('form.html')


def report_view():
    deployment_report = session.get('deployment_report')
    image_paths = session.get('image_paths')

    print(deployment_report)
    images = {}
    for i, image_path in enumerate(image_paths):
        with open(image_path, 'rb') as f:
            image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            images[f'sol{i}_img'] = image_base64

    return render_template('report.html', deployment_report=deployment_report, images=images)


def optimize():
    run_id = request.form['id']
    pipeline = request.files['pipeline']
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
        'id': run_id,
        'population_size': int(population_size),
        'generations_check': bool(generations_check),
        'max_generations': int(max_generations),
        'time_check': bool(time_check),
        'max_time': int(max_time)*60 # Slider select minutes, but api receives seconds
    }

    files = {
        'pipeline': (pipeline.filename, pipeline.read(), 'application/octet-stream'),
        'computing_infra': (computing_infra.filename, computing_infra.read(), 'application/octet-stream'),
        'network_infra': (network_infra.filename, network_infra.read(), 'application/octet-stream')
    }

    res = requests.post('http://127.0.0.1:8080/api/v1/optimize', data=data, files=files)
    deployment_report = json.loads(res.content)

    final_deployment_report, figures = _generate_final_report_(deployment_report,
                                                               generations_check,
                                                               time_check,
                                                               request.files)

    image_filenames = _save_images_(figures)
    image_paths = [os.path.join(IMAGE_DIRECTORY, filename) for filename in image_filenames]

    session['image_paths'] = image_paths
    session['deployment_report'] = final_deployment_report

    delete_thread = threading.Thread(target=_delete_images_after_delay_, args=(image_paths,))
    delete_thread.start()
    return redirect(url_for('blueprint.report_view'))


def _generate_final_report_(deployment_report, generations_check, time_check, request_files):
    max_generations = deployment_report['max_generations'] if generations_check else "NA"
    max_time = deployment_report['max_time'] if time_check else "NA"

    objectives = ['best_solution', 'best_sol_performance_fitness', 'best_sol_cost_fitness',
                  'best_sol_net_cost_fitness', 'best_sol_net_fail_prob_fitness']
    variables = ['model_perf_sol', 'comp_cost_sol', 'net_cost_sol', 'net_fail_sol']
    solutions_ids = ['0', '1', '2', '3', '4']

    sol0_img = _get_deployment_plot_(deployment_report['best_solution'], request_files)
    sol1_img = _get_deployment_plot_(deployment_report['best_sol_performance_fitness'], request_files)
    sol2_img = _get_deployment_plot_(deployment_report['best_sol_cost_fitness'], request_files)
    sol3_img = _get_deployment_plot_(deployment_report['best_sol_net_cost_fitness'], request_files)
    sol4_img = _get_deployment_plot_(deployment_report['best_sol_net_fail_prob_fitness'], request_files)

    final_deployment_report = {
        'id': deployment_report['id'],
        'report_date': deployment_report['report_date'],
        'num_models': deployment_report['num_models'],
        'num_computing_devices': deployment_report['num_computing_devices'],
        'num_net_devices': deployment_report['num_net_devices'],
        'population_size': deployment_report['population_size'],
        'max_generations': max_generations,
        'max_time': max_time,
        'pareto_front_size': deployment_report['pareto_front_size'],
        'total_time': deployment_report['total_time'],
        'time_to_met_constraints': deployment_report['time_to_met_constraints'],
    }

    for var in variables:
        for num in solutions_ids:
            key = f'{var}{num}'
            num = int(num)
            value = json.loads(deployment_report[objectives[num]])['objectives'][variables.index(var)]
            final_deployment_report[key] = value

    figures = [sol0_img, sol1_img, sol2_img, sol3_img, sol4_img]
    return final_deployment_report, figures


def _get_deployment_plot_(solution, request_files):
    solution = json.loads(solution)
    device_solution = solution['variables_0']
    network_solution = solution['variables_1']

    computing_infra = request_files['computing_infra']
    computing_infra.seek(0)
    computing_infra_bytes = BytesIO(computing_infra.read())
    df = pd.read_csv(computing_infra_bytes)
    device_names = df.id.to_numpy()

    network_infra = request_files['network_infra']
    network_infra.seek(0)
    network_infra_bytes = BytesIO(network_infra.read())
    df2 = pd.read_csv(network_infra_bytes, sep=";")
    net_device_names = df2.id.to_numpy()

    device_solution_df = pd.DataFrame(device_solution, columns=device_names)
    net_solution_df = pd.DataFrame(network_solution, columns=device_names)

    device_matrix = np.array(device_solution_df[device_names].values, dtype=float).T
    net_matrix = np.array(net_solution_df[device_names].values, dtype=float).T
    models_indexes = device_solution_df.index

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9), dpi=90,
                           gridspec_kw={'width_ratios': [len(models_indexes), len(net_device_names)]})

    ax[0].imshow(device_matrix, cmap='GnBu', interpolation='nearest')
    plt.sca(ax[0])
    plt.yticks(range(device_matrix.shape[0]), device_names, fontsize=8)
    plt.xticks(range(device_matrix.shape[1]), models_indexes, fontsize=8)  # , rotation=30)
    plt.ylabel('Computing devices')
    plt.xlabel('Pipeline Models')
    plt.title("Device Solution")

    # place '0' or '1' values centered in the individual squares
    for x in range(device_matrix.shape[0]):
        for y in range(device_matrix.shape[1]):
            ax[0].annotate(str(device_matrix[x, y])[0], xy=(y, x),
                           horizontalalignment='center', verticalalignment='center')

    ax[1].imshow(net_matrix, cmap='GnBu', interpolation='nearest')
    plt.sca(ax[1])
    plt.yticks(range(net_matrix.shape[0]), device_names, fontsize=8)
    plt.xticks(range(net_matrix.shape[1]), net_device_names, fontsize=8)  # , rotation=30)
    plt.ylabel('Computing devices')
    plt.xlabel('Network Devices')
    plt.title("Network Solution")

    # place '0' or '1' values centered in the individual squares
    for x in range(net_matrix.shape[0]):
        for y in range(net_matrix.shape[1]):
            ax[1].annotate(str(net_matrix[x, y])[0], xy=(y, x),
                           horizontalalignment='center', verticalalignment='center')

    """
    # Save the plot to bytes
    image_bytes = io.BytesIO()
    plt.savefig(image_bytes, format='png')
    plt.close()
    encoded_image = base64.b64encode(image_bytes.getvalue()).decode('utf-8')
    """
    return fig


def _save_images_(figures):
    image_filenames = []
    for i, figure in enumerate(figures):
        str_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f'image[{i}]{str_datetime}.png'
        image_path = os.path.join(IMAGE_DIRECTORY, filename)
        figure.savefig(image_path)
        plt.close(figure)
        image_filenames.append(filename)
    return image_filenames


def _delete_images_after_delay_(image_paths):
    time.sleep(IMAGE_LIFETIME)
    for path in image_paths:
        if os.path.exists(path):
            os.remove(path)

'''
def log_visualizer(task_id):
    # return redirect('localhost:6006:/#timeseries?runFilter={}'.format(task_id))
    return "On progress"
'''