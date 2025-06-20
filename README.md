# AI Prediction System

A Flask-based web application that allows users to make predictions using an AI model and participate in surveys to improve the system.

## Features

- User authentication (login/register)
- User roles (admin/regular user)
- AI prediction system
- User surveys
- Admin dashboard
- Public/private predictions
- User profiles

## Requirements

- Python 3.8+
- Flask and its extensions
- SQLite database

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Initialize the database:
```bash
flask db init
flask db migrate
flask db upgrade
```

5. Create an admin user:
```bash
flask shell
>>> from app import db
>>> from app.models.user import User, UserProfile
>>> admin = User(username='admin', email='admin@example.com', is_admin=True)
>>> admin.set_password('your-password')
>>> profile = UserProfile(user=admin)
>>> db.session.add(admin)
>>> db.session.add(profile)
>>> db.session.commit()
>>> exit()
```

## Running the Application

1. Start the Flask development server:
```bash
flask run
```

2. Open your web browser and navigate to `http://localhost:5000`

## Project Structure

```
project/
├── app/
│   ├── __init__.py
│   ├── models/
│   ├── static/
│   ├── templates/
│   ├── auth/
│   ├── main/
│   └── admin/
├── config.py
├── requirements.txt
└── run.py
```

## Usage

1. Register a new account or log in with existing credentials
2. Make predictions using the AI model
3. Take surveys to help improve the system
4. View your profile and prediction history
5. Admins can access the dashboard to manage users and view system statistics

## Contributing

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
