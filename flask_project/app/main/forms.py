from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, FileField, SelectField
from wtforms.validators import DataRequired

class HomePageForm(FlaskForm):
    fairness_definition = StringField('Fairness Definition', validators=[DataRequired()])
    file_upload = FileField('Upload File', validators=[DataRequired()])
    submit = SubmitField('Submit')

class ScenarioSelectionForm(FlaskForm):
    scenario = StringField('Scenario', validators=[DataRequired()])
    submit = SubmitField('Submit')

class Scenario1Form(FlaskForm):
    sensitive_attribute = SelectField('Sensitive Attribute', validators=[DataRequired()], choices = [])
    target_variable = SelectField('Target Variable', validators=[DataRequired()], choices = [])
    submit = SubmitField('Submit')

class Scenario1Form2(FlaskForm):
    submit = SubmitField('Submit')

class Scenario2Form(FlaskForm):
    submit = SubmitField('Submit')