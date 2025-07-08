print('app/main/forms.py loaded')
from flask_wtf import FlaskForm
from wtforms import StringField, FloatField, BooleanField, SubmitField, TextAreaField, IntegerField, SelectField
from wtforms.validators import DataRequired, NumberRange

class PredictionForm(FlaskForm):
    workout_duration = FloatField('Workout Duration (minutes)', 
        validators=[
            DataRequired(),
            NumberRange(min=1, max=300, message="Duration must be between 1 and 300 minutes")
        ])
    heart_rate = FloatField('Average Heart Rate (bpm)', 
        validators=[
            DataRequired(),
            NumberRange(min=40, max=220, message="Heart rate must be between 40 and 220 bpm")
        ])
    age = IntegerField('Age', 
        validators=[
            DataRequired(),
            NumberRange(min=18, max=100, message="Age must be between 18 and 100")
        ])
    weight = FloatField('Weight (kg)', 
        validators=[
            DataRequired(),
            NumberRange(min=40, max=200, message="Weight must be between 40 and 200 kg")
        ])
    height = FloatField('Height (cm)', 
        validators=[
            DataRequired(),
            NumberRange(min=120, max=250, message="Height must be between 120 and 250 cm")
        ])
    body_temp = FloatField('Body Temperature (°C)', 
        validators=[
            DataRequired(),
            NumberRange(min=35.0, max=42.0, message="Body temperature must be between 35.0 and 42.0°C")
        ])
    gender = SelectField('Gender', choices=[('male', 'Male'), ('female', 'Female')], validators=[DataRequired()])
    is_public = BooleanField('Make this prediction public')
    submit = SubmitField('Calculate Calories')

class SurveyForm(FlaskForm):
    question1 = TextAreaField('What is your experience with AI?', validators=[DataRequired()])
    question2 = TextAreaField('What features would you like to see in the future?', validators=[DataRequired()])
    question3 = TextAreaField('How can we improve our service?', validators=[DataRequired()])
    submit = SubmitField('Submit Survey') 