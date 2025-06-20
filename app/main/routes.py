from flask import render_template, flash, redirect, url_for, request, jsonify
from flask_login import current_user, login_required
from app import db
from app.main import bp
from app.models.user import User, UserProfile
from app.models.ai_model import Prediction, Survey
from app.main.forms import PredictionForm, SurveyForm
from app.auth.forms import ChangePasswordForm
import numpy as np
import matplotlib.pyplot as plt
import os

@bp.route('/')
@bp.route('/index')
def index():
    if current_user.is_authenticated:
        return render_template('main/index.html', title='Home')
    return render_template('main/index.html', title='Welcome')

@bp.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    form = ChangePasswordForm()
    password_changed = False
    if form.validate_on_submit():
        if current_user.check_password(form.old_password.data):
            current_user.set_password(form.new_password.data)
            db.session.commit()
            flash('Your password has been updated.', 'success')
            password_changed = True
        else:
            flash('Current password is incorrect.', 'danger')
    return render_template('main/profile.html', title='Profile', form=form, password_changed=password_changed)

@bp.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    print('Entered predict route')
    form = PredictionForm()
    print('PredictionForm instantiated')
    diagram_url = None
    prediction_result = None
    if form.validate_on_submit():
        print('Form validated')
        # Gather all input data
        workout_duration = form.workout_duration.data or 0
        heart_rate = form.heart_rate.data or 0
        age = form.age.data or 0
        weight = form.weight.data or 0
        gender = form.gender.data
        print(f'Inputs: duration={workout_duration}, hr={heart_rate}, age={age}, weight={weight}, gender={gender}')
        input_data = {
            'workout_duration': workout_duration,
            'heart_rate': heart_rate,
            'age': age,
            'weight': weight,
            'gender': gender
        }
        # Use the correct formula for men and women
        if gender == 'male':
            calories_burned = (
                ((age * 0.2017) - (weight * 0.09036) + (heart_rate * 0.6309) - 55.0969)
                * workout_duration / 4.184
            )
        else:  # female
            calories_burned = (
                ((age * 0.074) - (weight * 0.05741) + (heart_rate * 0.4472) - 20.4022)
                * workout_duration / 4.184
            )
        # Prevent negative values
        prediction_result = max(calories_burned, 0)
        print(f'Calories burned: {prediction_result}')
        prediction = Prediction(
            user_id=current_user.id,
            input_data=input_data,
            prediction_result=prediction_result,
            is_public=form.is_public.data
        )
        db.session.add(prediction)
        db.session.commit()
        print('Prediction saved')
        # Generate diagram
        plt.figure(figsize=(6,4))
        features = ['Workout Duration', 'Heart Rate', 'Age', 'Weight']
        values = [workout_duration, heart_rate, age, weight]
        plt.bar(features, values, color=['blue', 'red', 'green', 'orange'])
        plt.title('Input Features')
        plt.ylabel('Value')
        plt.tight_layout()
        diagram_filename = f'diagram_{prediction.id}.png'
        diagram_path = os.path.join('app', 'static', 'diagrams')
        os.makedirs(diagram_path, exist_ok=True)
        plt.savefig(os.path.join(diagram_path, diagram_filename))
        plt.close()
        diagram_url = url_for('static', filename=f'diagrams/{diagram_filename}')
        flash('Prediction completed successfully!')
        return render_template('main/predict.html', title='Make Prediction', form=form, prediction_result=prediction_result, diagram_url=diagram_url)
    print('Rendering predict.html')
    return render_template('main/predict.html', title='Make Prediction', form=form, prediction_result=prediction_result, diagram_url=diagram_url)

@bp.route('/predictions')
@login_required
def predictions():
    user_predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.desc()).all()
    public_predictions = Prediction.query.filter_by(is_public=True).order_by(Prediction.created_at.desc()).all()
    return render_template('main/predictions.html', 
                         title='Predictions',
                         user_predictions=user_predictions,
                         public_predictions=public_predictions)

@bp.route('/survey', methods=['GET', 'POST'])
@login_required
def survey():
    form = SurveyForm()
    if form.validate_on_submit():
        survey = Survey(
            user_id=current_user.id,
            answers={
                'question1': form.question1.data,
                'question2': form.question2.data,
                'question3': form.question3.data
            }
        )
        db.session.add(survey)
        db.session.commit()
        flash('Thank you for completing the survey!')
        return redirect(url_for('main.index'))
    return render_template('main/survey.html', title='Survey', form=form) 