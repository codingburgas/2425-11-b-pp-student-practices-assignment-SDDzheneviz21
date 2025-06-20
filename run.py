from app import create_app, db
from app.models.user import User, UserProfile
from app.models.ai_model import Prediction, Survey

app = create_app()

@app.shell_context_processor
def make_shell_context():
    return {
        'db': db,
        'User': User,
        'UserProfile': UserProfile,
        'Prediction': Prediction,
        'Survey': Survey
    }

# Create database tables
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True) 