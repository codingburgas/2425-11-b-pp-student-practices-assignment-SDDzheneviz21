from flask import render_template, flash, redirect, url_for, request
from flask_login import login_required, current_user
from app import db
from app.admin import bp
from app.models.user import User, UserProfile
from app.models.ai_model import Prediction, Survey
from app.admin.forms import EditUserForm

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