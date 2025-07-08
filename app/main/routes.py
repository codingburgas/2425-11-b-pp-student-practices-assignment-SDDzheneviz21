import matplotlib
matplotlib.use('Agg')
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
from app.models.ml_utils import train_logistic_regression
from app.models.calorie_model import predict_calories_burned, get_model_feature_importance, initialize_calorie_model
from datetime import timedelta

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
    feature_importance = None
    
    if form.validate_on_submit():
        print('Form validated')
        # Gather all input data
        workout_duration = form.workout_duration.data or 0
        heart_rate = form.heart_rate.data or 0
        age = form.age.data or 0
        weight = form.weight.data or 0
        height = form.height.data or 0
        body_temp = form.body_temp.data or 0
        gender = form.gender.data
        
        print(f'Inputs: duration={workout_duration}, hr={heart_rate}, age={age}, weight={weight}, height={height}, temp={body_temp}, gender={gender}')
        
        input_data = {
            'workout_duration': workout_duration,
            'heart_rate': heart_rate,
            'age': age,
            'weight': weight,
            'height': height,
            'body_temp': body_temp,
            'gender': gender
        }
        
        try:
            # Use ML model for prediction
            prediction_result = predict_calories_burned(
                gender=gender,
                age=age,
                height=height,
                weight=weight,
                duration=workout_duration,
                heart_rate=heart_rate,
                body_temp=body_temp
            )
            
            # Get feature importance
            feature_importance = get_model_feature_importance()
            
            print(f'ML Model Calories burned: {prediction_result}')
            
        except Exception as e:
            print(f'ML model error: {e}')
            # Fallback to simple formula if ML model fails
            met = estimate_met(heart_rate)
            prediction_result = workout_duration * (met * 3.5 * weight) / 200
            prediction_result = max(prediction_result, 0)
            print(f'Fallback Calories burned: {prediction_result}')
        
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
        plt.figure(figsize=(8,6))
        features = ['Duration', 'Heart Rate', 'Age', 'Weight', 'Height', 'Body Temp']
        values = [workout_duration, heart_rate, age, weight, height, body_temp]
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        plt.bar(features, values, color=colors)
        plt.title('Input Features for Calorie Prediction')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        diagram_filename = f'diagram_{prediction.id}.png'
        diagram_path = os.path.join('app', 'static', 'diagrams')
        os.makedirs(diagram_path, exist_ok=True)
        plt.savefig(os.path.join(diagram_path, diagram_filename))
        plt.close()
        diagram_url = url_for('static', filename=f'diagrams/{diagram_filename}')
        flash('Prediction completed successfully using ML model!')
        
        # --- ML вероятностен изход ---
        # Вземи всички предсказания за обучение
        predictions = Prediction.query.all()
        probability = None
        if predictions and len(predictions) >= 5:
            feature_names = ['workout_duration', 'heart_rate', 'age', 'weight', 'gender']
            X = []
            y = []
            threshold = 200
            for p in predictions:
                d = p.input_data
                gender_num = 1 if d.get('gender') == 'male' else 0
                X.append([
                    d.get('workout_duration', 0),
                    d.get('heart_rate', 0),
                    d.get('age', 0),
                    d.get('weight', 0),
                    gender_num
                ])
                y.append(1 if p.prediction_result >= threshold else 0)
            X = np.array(X)
            y = np.array(y)
            model = train_logistic_regression(X, y)
            # Подготви входа за текущото предсказание
            gender_num = 1 if gender == 'male' else 0
            input_vec = np.array([[workout_duration, heart_rate, age, weight, gender_num]])
            proba = model.predict_proba(input_vec)[0][1]  # вероятност за >200 kcal
            probability = proba
        
        return render_template('main/predict.html', 
                             title='Make Prediction', 
                             form=form, 
                             prediction_result=prediction_result, 
                             diagram_url=diagram_url, 
                             probability=probability,
                             feature_importance=feature_importance)
    
    print('Rendering predict.html')
    return render_template('main/predict.html', 
                         title='Make Prediction', 
                         form=form, 
                         prediction_result=prediction_result, 
                         diagram_url=diagram_url)

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

def estimate_met(heart_rate):
    if heart_rate < 100:
        return 3.5  # light
    elif heart_rate < 130:
        return 6    # moderate
    else:
        return 8    # vigorous 