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



UPLOAD_FOLDER = 'main/uploads'
ALLOWED_EXTENSIONS = set(['csv'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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
    form.target_variable.choices = [header for header in session.get('column_headers', []) if header != session.get('sensitive_attribute')]
    if form.validate_on_submit():
        session['sensitive_attribute'] = form.sensitive_attribute.data
        session['target_variable'] = form.target_variable.data
        session['model_type'] = form.model_type.data

        print('file_path:', session['file_path'])
        print('target_variable:', session['target_variable'])
        print('sensitive_attribute:', session['sensitive_attribute'])
        print('fairness_definition:', session['fairness_definition'])
        print('model_type:', session['model_type'])

        # Start the Celery task and save the task ID to the session
        #task = celery.send_task('app.main.views.train_model', args=[session['file_path'], session['target_variable'], session['sensitive_attribute']])
        task = run_training.delay(session['file_path'], session['target_variable'], session['sensitive_attribute'], session['model_type'], session['fairness_definition'])
        #task = add.delay(4, 6)
        session['task_id'] = str(task.id)
        time.sleep(80)
        return redirect(url_for('main.results'))
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

