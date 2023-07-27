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
from app.main.models import ModelResults
import plotly
import plotly.graph_objs as go
# from app import socketio





UPLOAD_FOLDER = 'main/uploads'
ALLOWED_EXTENSIONS = set(['csv'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_hyperparameters():
    learning_rate_values = [0.01, 0.1, 1]
    lambda_fairness_values = [0.1, 1, 10]
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
            task_ids = []
            task = run_training.delay(session['file_path'], 
                                          session['target_variable'], 
                                          session['sensitive_attribute'], 
                                          session['model_type'], 
                                          session['fairness_definition'], 
                                          0.01,
                                            0.1,
                                            100,
                                            32)
            task_ids.append(str(task.id))
            session['task_ids'] = task_ids
            return render_template('waiting.html', task_ids=session['task_ids'], form=form)

        else:
            flash('The sensitive attribute and target variable cannot be the same.')

    return render_template('scenario_1.html', form=form)

# def scenario_1():
#     form = Scenario1Form()
#     form.sensitive_attribute.choices = session.get('column_headers', [])
#     form.target_variable.choices = session.get('column_headers', [])
#     if form.validate_on_submit():
#         if form.sensitive_attribute.data != form.target_variable.data:
#             session['sensitive_attribute'] = form.sensitive_attribute.data
#             session['target_variable'] = form.target_variable.data
#             session['model_type'] = form.model_type.data
#             task_ids = []
#             # for params in generate_hyperparameters():
#             task = run_training.delay(session['file_path'], 
#                                           session['target_variable'], 
#                                           session['sensitive_attribute'], 
#                                           session['model_type'], 
#                                           session['fairness_definition'], 
#                                           0.01,
#                                             0.1,
#                                             100,
#                                             32)
#                                         #   params['learning_rate'], 
#                                         #   params['lambda_fairness'],
#                                         #   params['num_epochs'],
#                                         #   params['batch_size'])
#             task_ids.append(str(task.id))
#             session['task_ids'] = task_ids
#             socketio.emit('wait_for_tasks', session['task_ids'], namespace='/')
#             return render_template('waiting.html', form=form)

#         else:
#             flash('The sensitive attribute and target variable cannot be the same.')

#     return render_template('scenario_1.html', form=form)
        

#         # print('file_path:', session['file_path'])
#         # print('target_variable:', session['target_variable'])
#         # print('sensitive_attribute:', session['sensitive_attribute'])
#         # print('fairness_definition:', session['fairness_definition'])
#         # print('model_type:', session['model_type'])

#         # Start the Celery task and save the task ID to the session
#         #task = celery.send_task('app.main.views.train_model', args=[session['file_path'], session['target_variable'], session['sensitive_attribute']])
#         #task = add.delay(4, 6)
        
    

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
    

@main.route('/results/<model_type>', methods=['GET'])
def show_model_results(model_type):
    # Query the database for the model results
    results = ModelResults.query.filter_by(model_class=model_type).all()
    ## Noticed an issue here: Model class is Logistic Regression but model type is logistic_regression_demographic_parity
    print("Results Object:", results)
    print("Model Accuracy:", results[0].model_accuracy)
    print("Fairness Score:", results[0].fairness_score)
    # Prepare the data for the scatter plot
    model_accuracies, fairness_scores = zip(*[(result.model_accuracy, result.fairness_score) for result in results])

    # Create the scatter plot
    scatter = go.Scatter(x = model_accuracies, y = fairness_scores, mode = 'markers')
    layout = go.Layout(title = 'Model Accuracy vs Fairness Score', xaxis = dict(title = 'Model Accuracy'), yaxis = dict(title = 'Fairness Score'))
    figure = go.Figure(data = [scatter], layout = layout)

    # Convert the plot to HTML
    plot_html = plotly.offline.plot(figure, include_plotlyjs = False, output_type = 'div')

    return render_template('results.html', plot = plot_html)



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




