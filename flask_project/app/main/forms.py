from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, FileField, SelectField
from wtforms.validators import DataRequired
from wtforms import RadioField

class HomePageForm(FlaskForm):
    fairness_definitions = {
        'demographic_parity': 'Strives to assign positive outcomes proportionally  \
            across different demographic groups, regardless of their true outcomes \
                [University Scholarship Allocation System (Promoting diversity by awarding scholarships proportionally to applicants from different socio-economic backgrounds)]',
        'equalized_odds': 'Seeks to balance both True Positive Rate and False Positive Rate parity, \
            mitigating biases in both correct and incorrect positive classifications \
                [Job Recruitment System (Balancing correct hiring and avoiding mismatches across urban and rural applicants)]',
        'equal_opportunity': 'Aims to mitigate disparities in correctly identifying deserving individuals as positive across different demographic groups \
            [Medical Diagnostics (Equally effective disease detection across genders)]',
        'false_positive_rate_parity': 'Aims to mitigate disparities in wrongly identifying undeserving individuals as \
            positive across different demographic groups \
                [Criminal Predictive Policing Systems (Preventing over-policing of one group over another)]',
        'error_rate_parity': 'Aims to achieve constant error rates in predictions (both false positives and false negatives) across different demographic groups \
            [Self Driving Car Collision Avoidance System (Promoting industry-wide safety standards across different car brands)]',

    }
    fairness_definition = RadioField('Fairness Definition', choices=[(key, key.replace('_', ' ').title()) for key in fairness_definitions.keys()])
    
    file_upload = FileField('Upload File', validators=[DataRequired()])
    submit = SubmitField('Submit')

class ScenarioSelectionForm(FlaskForm):
    scenario = StringField('Scenario', validators=[DataRequired()])
    submit = SubmitField('Submit')

class Scenario1Form(FlaskForm):
    sensitive_attribute = SelectField('Sensitive Attribute', validators=[DataRequired()], choices = [])
    target_variable = SelectField('Target Variable', validators=[DataRequired()], choices = [])
    model_type = SelectField('Model Type', validators=[DataRequired()], choices=[('logistic_regression', 'Logistic Regression'), ('perceptron', 'Perceptron'), ('neural_network', 'Neural Network')])

    submit = SubmitField('Submit')

class Scenario1Form2(FlaskForm):
    submit = SubmitField('Submit')

class Scenario2Form(FlaskForm):
    submit = SubmitField('Submit')