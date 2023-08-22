from datetime import datetime
from flask import render_template, redirect, url_for, session


from .forms import HomePageForm, ScenarioSelectionForm, Scenario1Form, Scenario1Form2, Scenario2Form
from . import main
import pandas as pd
from flask import request
import os
from werkzeug.utils import secure_filename
from flask import current_app
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#from celery import Celery
from app import celery
from sklearn.metrics import roc_auc_score
from app.main.tasks import run_training
import time
from flask import flash
from itertools import product
from flask_socketio import emit
from celery.result import AsyncResult
from flask import jsonify
from app.main.models import ModelResults, FairnessMetrics 
import plotly
import plotly.graph_objs as go
from plotly.express import scatter
import plotly.express as px
import pandas as pd
import json
from collections import defaultdict
import numpy as np
# from app import socketio





UPLOAD_FOLDER = 'main/uploads'
ALLOWED_EXTENSIONS = set(['csv'])

def recursive_to_list(item):
    if isinstance(item, np.ndarray):
        return item.tolist()
    elif isinstance(item, dict):
        return {k: recursive_to_list(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [recursive_to_list(i) for i in item]
    else:
        return item

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_hyperparameters():
    learning_rate_values = [0.01, 0.1, 0.002]
    lambda_fairness_values = [0.1, 1, 0.01]
    num_epochs = 100
    batch_size = 32

    for learning_rate, lambda_fairness in product(learning_rate_values, lambda_fairness_values):
        yield {
            'learning_rate': learning_rate,
            'lambda_fairness': lambda_fairness,
            'num_epochs': num_epochs,
            'batch_size': batch_size
        }


@main.route('/', methods=['GET', 'POST'])
def index():
    form = HomePageForm()
    if form.validate_on_submit():
        session['fairness_definition'] = form.fairness_definition.data
        # print(session['fairness_definition'])
        file = form.file_upload.data

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            uploads_dir = os.path.join(current_app.root_path, 'main', 'uploads')
            os.makedirs(uploads_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            new_filename = f'{filename.split(".")[0]}_{timestamp}.csv'
            save_location = os.path.join(uploads_dir, new_filename)
            file.save(save_location)

            # read the file and extract headers
            df = pd.read_csv(save_location)
            headers = df.columns.tolist()
            session['column_headers'] = headers
            session['file_path'] = save_location  # Save the file path to the session
            return redirect(url_for('main.scenario_selection'))
        
       
    return render_template('index.html', form=form)

@main.route('/scenario_selection', methods=['GET', 'POST'])
def scenario_selection():
    form = ScenarioSelectionForm()
    if form.validate_on_submit():
        session['scenario'] = form.scenario.data
        if form.scenario.data == '1':
            return redirect(url_for('main.scenario_1'))
        else:
            return redirect(url_for('main.scenario_2'))
    return render_template('scenario_selection.html', form=form)

@main.route('/scenario_1', methods=['GET', 'POST'])
def scenario_1():
    form = Scenario1Form()
    form.sensitive_attribute.choices = session.get('column_headers', [])
    form.target_variable.choices = session.get('column_headers', [])
    if form.validate_on_submit():
        if form.sensitive_attribute.data != form.target_variable.data:
            session['sensitive_attribute'] = form.sensitive_attribute.data
            session['target_variable'] = form.target_variable.data
            session['model_type'] = form.model_type.data
            hyperparameters = generate_hyperparameters()
            
            task_ids = []
            
            # Loop through the hyperparameters and call the run_training function for each combination
            for hyperparam in hyperparameters:
                task = run_training.delay(
                    session['file_path'],
                    session['target_variable'],
                    session['sensitive_attribute'],
                    session['model_type'],
                    session['fairness_definition'],
                    hyperparam['learning_rate'],
                    hyperparam['lambda_fairness'],
                    hyperparam['num_epochs'],
                    hyperparam['batch_size']
                )
            task_ids.append(str(task.id))
            session['task_ids'] = task_ids
            return render_template('waiting.html', task_ids=session['task_ids'], form=form)

        else:
            flash('The sensitive attribute and target variable cannot be the same.')

    return render_template('scenario_1.html', form=form)


        
    

@main.route('/scenario_1_2', methods=['GET', 'POST'])
def scenario_1_2():
    form = Scenario1Form2()
    if form.validate_on_submit():
        return redirect(url_for('main.index'))
    return render_template('scenario_1_2.html', form=form)

@main.route('/scenario_2', methods=['GET', 'POST'])
def scenario_2():
    form = Scenario2Form()
    form.target_variable.choices = session.get('column_headers', [])
    if form.validate_on_submit():
        return redirect(url_for('main.index'))
    return render_template('scenario_2.html', form=form)

@main.route('/results', methods=['GET'])
def results():
    task_id = session.get('task_id')
    task = celery.AsyncResult(task_id) if task_id else None
    if task and task.state == 'SUCCESS':
        # If task finished successfully, display the results
        predictions, model_accuracy, auc_score = task.result
        return render_template('results.html', predictions=predictions, model_accuracy=model_accuracy, auc_score=auc_score)
    else:
        traceback = task.traceback if task else None
        # If task is still processing or failed, display a waiting or error page
        return render_template('waiting_or_error.html', task=task, traceback=traceback)
    

# @main.route('/results/<model_type>', methods=['GET'])
# def show_model_results(model_type):
#     # Query the database for the model results
#     results = ModelResults.query.filter_by(model_class=model_type).all()
#     ## Noticed an issue here: Model class is Logistic Regression but model type is logistic_regression_demographic_parity
#     print("Results Object:", results)
#     print("Model Accuracy:", results[0].model_accuracy)
#     print("Fairness Score:", results[0].fairness_score)
#     # Prepare the data for the scatter plot
#     model_accuracies, fairness_scores = zip(*[(result.model_accuracy, result.fairness_score) for result in results])

#     # Create the scatter plot
#     scatter = go.Scatter(x = model_accuracies, y = fairness_scores, mode = 'markers')
#     layout = go.Layout(title = 'Model Accuracy vs Fairness Score', xaxis = dict(title = 'Model Accuracy'), yaxis = dict(title = 'Fairness Score'))
#     figure = go.Figure(data = [scatter], layout = layout)

#     # Convert the plot to HTML
#     plot_html = plotly.offline.plot(figure, include_plotlyjs = True, output_type = 'div')

#     return render_template('model_results.html', plot = plot_html)



@main.route('/results/<model_type>', methods=['GET'])
def show_model_results(model_type):
    fairness_metrics_mapping = {
    "equal_opportunity": ["true_positive_rate", "false_negative_error"],
    "equalized_odds": ["true_positive_rate", "false_positive_rate", "false_negative_error", "true_negative_rate"],
    "demographic_parity": ["selection_rate"],
    "false_positive_rate_parity": ["false_positive_rate", "true_negative_error"],
    "error_rate_parity": ["balanced_acc_error", "true_positive_error", "true_negative_error", "false_positive_error", "false_negative_error"]
}
    # Query the database for the model results
    results = ModelResults.query.filter_by(model_class=model_type).all()
    # fair_results = FairnessMetrics.query.filter_by(model_results_id=result.id).first().metrics

    groups = defaultdict(list)
    for result in results:
        key = (result.model_accuracy, result.fairness_score, result.learning_rate, result.lambda_fairness)
        groups[key].append(result)
    # Select representative models
    representative_models = []
    for key, group_models in groups.items():
        # Here, we're simply taking the first model as the representative.
        # You can modify this to select based on other criteria.
        representative_model = group_models[0]
        representative_models.append(representative_model)
    
    metrics = {}
    model_signatures = set()  # A set to keep track of unique metric signatures
    filtered_models = [] 

    for model in representative_models:
        fair_metric = FairnessMetrics.query.filter_by(model_results_id=model.id).first()
        if fair_metric:
            # Generate a unique signature for the model based on its metric values
            signature = str(sorted(fair_metric.metrics.items()))
            if signature not in model_signatures:
                metrics[model.id] = fair_metric.metrics
                model_signatures.add(signature)
                filtered_models.append(model)

    representative_models = filtered_models

    # Define the expected metrics and groups
    expected_metrics = set([metric_name for model in metrics.values() for metric_name in model.keys()])
    expected_groups = set([group for model in metrics.values() for metric in model.values() for group in metric.keys()])

    # Initialize the metric_data dictionary
    metric_data = {metric_name: {group: [] for group in expected_groups} for metric_name in expected_metrics}

    for model in representative_models:
        model_id = model.id
        model_metrics = metrics.get(model_id, {})
        
        for metric_name in expected_metrics:
            for group in expected_groups:
                value = model_metrics.get(metric_name, {}).get(group, None)  # Use None as a placeholder for missing values
                metric_data[metric_name][group].append(value)
    # Populate the metric_data dictionary
                # metric_data[metric_name][group].append(value)

    # Prepare the data for the scatter plot
    data = [{
        'model_id': model.id,
        'model_accuracy': model.model_accuracy, 
         'fairness_score': model.fairness_score, 
         'learning_rate': model.learning_rate, 
         'lambda_fairness': model.lambda_fairness} for model in representative_models]
    df = pd.DataFrame(data)
    print("Model ID:", df['model_id'].tolist())
    print("Length of Model IDs:", len(df['model_id'].tolist()))
    
    # Create the scatter plot
    fig = px.scatter(df, x='model_accuracy', y='fairness_score',
                    hover_data=['learning_rate', 'lambda_fairness'],
                     custom_data=['model_id'],
                     color_discrete_sequence=['black'],  # This sets the color sequence to just black
                    color=df['model_id']*0,
                     labels={'fairness_score': 'Fairness Metric (Lower is Better)',
                             'model_accuracy': 'Error Rate (Lower is Better)',
                             },
                     title='Error Rate vs. Equalized Odds with Hover Tooltip')

    print("Number of traces in the scatter plot:", len(fig.data))
    scatter_data = fig.to_dict()["data"][0]
    scatter_data = recursive_to_list(scatter_data)
    scatter_raw_data = [trace.to_plotly_json() for trace in fig.data]
    scatter_raw_data = recursive_to_list(scatter_raw_data)
    scatter_layout = fig.layout.to_plotly_json()
    scatter_layout = recursive_to_list(scatter_layout)
    


    # Convert the plot to HTML
    # plot_html = plotly.offline.plot(fig, include_plotlyjs = True, output_type = 'div')

    return render_template('model_results.html', scatter_raw_data=scatter_raw_data, scatter_layout=scatter_layout, metric_data=metric_data, df=df, scatter_data=scatter_data, fairness_metrics_mapping=fairness_metrics_mapping, current_fairness=session['fairness_definition'])


# @main.route('/results/<model_type>', methods=['GET'])
# def show_model_results(model_type):
#     # Query the database for the model results
#     results = ModelResults.query.filter_by(model_class=model_type).all()

#     # Prepare the data for the scatter plot
#     data = [{'model_id': result.id, 
#              'model_accuracy': result.model_accuracy, 
#              'fairness_score': result.fairness_score, 
#              'learning_rate': result.learning_rate, 
#              'lambda_fairness': result.lambda_fairness} for result in results]
#     df = pd.DataFrame(data)

#     # Create the scatter plot
#     fig = px.scatter(df, x='fairness_score', y='model_accuracy',
#                      size='model_accuracy', hover_data=['learning_rate', 'lambda_fairness'],
#                      labels={'fairness_score': 'Fairness Metric (Lower is Better)',
#                              'model_accuracy': 'Model Accuracy (Higher is Better)',
#                              },
#                      title='Accuracy vs. Fairness with Hover Tooltip')
    
    # Add click events to the markers
    # for trace in fig.data:
    #     trace['customdata'] = df['model_id']
    #     trace['hovertemplate'] = '<b>Model ID</b>: %{customdata}<br>' + trace['hovertemplate']
    #     trace['mode'] = 'markers+text'
    #     trace['text'] = df['model_id'].tolist()
    #     trace['textposition'] = 'top center'

    # # Convert the plot to HTML
    # plot_html = plotly.offline.plot(fig, include_plotlyjs = True, output_type = 'div')

    # return render_template('model_results.html', plot = plot_html)

@main.route('/model_metrics/<model_id>', methods=['GET'])
def show_model_metrics(model_id):
    # Fetch the latest (or any specific) model's fairness metrics from the database
    model_metric = FairnessMetrics.query.filter_by(model_results_id=model_id).first()

    if not model_metric:
        # Handle case where no metrics are found for the given model_id
        # This is just an example. Modify as per your application's error handling strategy.
        return "No metrics found for the given model ID", 404

    # Render a template that displays the metrics
    return render_template('model_metrics.html', metrics=model_metric.metrics, model_id=model_id)








# Route to check the status of the tasks


@main.route('/check_tasks', methods=['POST'])
def check_tasks():
    task_ids = request.json['task_ids']
    tasks_finished = all(celery.AsyncResult(task_id).ready() for task_id in task_ids)
    if tasks_finished:
        model_type = session['model_type']
        print("Model Type:", model_type)
        redirect_url = url_for('main.show_model_results', model_type=model_type)
    else:
        redirect_url = None
    return jsonify(tasks_finished=tasks_finished, redirect_url=redirect_url)




