from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, FileField, SelectField
from wtforms.validators import DataRequired
from wtforms import RadioField

class HomePageForm(FlaskForm):
    fairness_definitions = {
        'group_unawareness': "Model doesn't use the sensitive attribute for prediction",
        'demographic_parity': "Proportion of each segment of a protected class should receive a positive outcome at equal rates",
        'disparate_impact': "Instead of aiming for equal approval rate, aims to achieve a higher than specified ratio",
        'equal_opportunity': "True Positive Rate is looked at for predictions (doesn't account for existing gaps between groups)",
        'equal_odds': "Enforces equal TPR and FPR cross groups (example use case: People going to jail)",
        'positive_predictive_value_parity': "Equalizes the chances of success given a positive prediction",
        'false_positive_ratio_parity': "Ratio of people that the model approved but defaulted on their loan out of all the groups that defaulted is the same for all groups",
        'negative_predictive_value_parity': "Ratio of a correct negative decision out of all the negative decisions from the model is the same for all groups",
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