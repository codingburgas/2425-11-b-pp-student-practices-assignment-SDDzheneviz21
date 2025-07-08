from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_bootstrap import Bootstrap
from flask_migrate import Migrate
from config import Config
# from app.main.routes import to_bulgarian_time  # Премахнато, за да няма цикличен импорт
from app.utils.time_utils import to_bulgarian_time

db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to access this page.'
bootstrap = Bootstrap()
migrate = Migrate()

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)
    bootstrap.init_app(app)
    migrate.init_app(app, db)

    # Register blueprints
    from app.auth import bp as auth_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')

    from app.main import bp as main_bp
    app.register_blueprint(main_bp)

    from app.admin import bp as admin_bp
    app.register_blueprint(admin_bp, url_prefix='/admin')

    # Create database tables
    with app.app_context():
        db.create_all()

    # Register Jinja2 filter
    app.jinja_env.filters['to_bulgarian_time'] = to_bulgarian_time

    # Initialize ML model
    with app.app_context():
        try:
            from app.models.calorie_model import initialize_calorie_model
            initialize_calorie_model()
            print("ML model initialized successfully!")
        except Exception as e:
            print(f"Warning: Could not initialize ML model: {e}")

    return app

from app import models 