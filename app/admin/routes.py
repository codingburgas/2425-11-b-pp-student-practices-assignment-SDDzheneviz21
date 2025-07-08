import matplotlib
matplotlib.use('Agg')
from flask import render_template, flash, redirect, url_for, request
from flask_login import login_required, current_user
from app import db
from app.admin import bp
from app.models.user import User, UserProfile
from app.models.ai_model import Prediction, Survey
from app.admin.forms import EditUserForm
from app.models.ml_utils import train_logistic_regression, calculate_metrics
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

def admin_required(f):
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash('You do not have permission to access this page.')
            return redirect(url_for('main.index'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

@bp.route('/dashboard')
@login_required
@admin_required
def dashboard():
    users = User.query.all()
    predictions = Prediction.query.all()
    surveys = Survey.query.all()
    return render_template('admin/dashboard.html',
                         title='Admin Dashboard',
                         users=users,
                         predictions=predictions,
                         surveys=surveys)

@bp.route('/user/<int:id>', methods=['GET', 'POST'])
@login_required
@admin_required

def edit_user(id):
    user = User.query.get_or_404(id)
    form = EditUserForm()
    if form.validate_on_submit():
        user.username = form.username.data
        user.email = form.email.data
        user.is_admin = form.is_admin.data
        db.session.commit()
        flash('User has been updated.')
        return redirect(url_for('admin.dashboard'))
    elif request.method == 'GET':
        form.username.data = user.username
        form.email.data = user.email
        form.is_admin.data = user.is_admin
    return render_template('admin/edit_user.html', title='Edit User', form=form)

@bp.route('/user/<int:id>/delete', methods=['POST'])
@login_required
@admin_required
def delete_user(id):
    user = User.query.get_or_404(id)
    if user == current_user:
        flash('You cannot delete your own account!')
        return redirect(url_for('admin.dashboard'))
    db.session.delete(user)
    db.session.commit()
    flash('User has been deleted.')
    return redirect(url_for('admin.dashboard'))

@bp.route('/prediction/<int:id>/delete', methods=['POST'])
@login_required
@admin_required
def delete_prediction(id):
    prediction = Prediction.query.get_or_404(id)
    db.session.delete(prediction)
    db.session.commit()
    flash('Prediction has been deleted.')
    return redirect(url_for('admin.dashboard'))

@bp.route('/survey/<int:id>/delete', methods=['POST'])
@login_required
@admin_required
def delete_survey(id):
    survey = Survey.query.get_or_404(id)
    db.session.delete(survey)
    db.session.commit()
    flash('Survey has been deleted.')
    return redirect(url_for('admin.dashboard'))

@bp.route('/metrics')
@login_required
@admin_required
def metrics():
    # Вземи всички предсказания
    predictions = Prediction.query.all()
    if not predictions or len(predictions) < 5:
        flash('Not enough data for metrics calculation.')
        return redirect(url_for('admin.dashboard'))
    # Подготви X и y
    feature_names = ['workout_duration', 'heart_rate', 'age', 'weight', 'gender']
    X = []
    y = []
    threshold = 200  # примерен праг за бинарна класификация
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
    # Тренирай модел и изчисли метриките
    model = train_logistic_regression(X, y)
    metrics = calculate_metrics(model, X, y)

    # --- Generate static images ---
    diagram_path = os.path.join('app', 'static', 'diagrams')
    os.makedirs(diagram_path, exist_ok=True)
    # Confusion matrix image
    cm = np.array(metrics['confusion_matrix'])
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Predicted ≤ 200 calories', 'Predicted > 200 calories'],
                yticklabels=['Burned ≤ 200 calories', 'Burned > 200 calories'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    cm_filename = 'confusion_matrix.png'
    plt.tight_layout()
    plt.savefig(os.path.join(diagram_path, cm_filename))
    plt.close()
    # URLs for template
    cm_url = url_for('static', filename=f'diagrams/{cm_filename}')

    return render_template('admin/metrics.html', title='Model Metrics', metrics=metrics, cm_url=cm_url)

@bp.route('/prediction/<int:id>')
@login_required
@admin_required
def view_prediction(id):
    prediction = Prediction.query.get_or_404(id)
    # Подготви данните за диаграмата
    input_data = prediction.input_data
    diagram_url = None
    diagram_filename = f'diagram_{prediction.id}.png'
    diagram_path = os.path.join('app', 'static', 'diagrams', diagram_filename)
    if os.path.exists(diagram_path):
        diagram_url = url_for('static', filename=f'diagrams/{diagram_filename}')
    # --- ML вероятностен изход ---
    from app.models.ai_model import Prediction as PredictionModel
    predictions = PredictionModel.query.all()
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
        gender_num = 1 if input_data.get('gender') == 'male' else 0
        input_vec = np.array([[input_data.get('workout_duration', 0), input_data.get('heart_rate', 0), input_data.get('age', 0), input_data.get('weight', 0), gender_num]])
        proba = model.predict_proba(input_vec)[0][1]
        probability = proba
    return render_template('admin/view_prediction.html', prediction=prediction, diagram_url=diagram_url, probability=probability) 