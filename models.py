# models.py
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """User model for authentication and authorization."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationship to predictions
    predictions = db.relationship('Prediction', backref='user', lazy=True)

    def set_password(self, password):
        """Hash and store the password."""
        from werkzeug.security import generate_password_hash
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Verify the password against the stored hash."""
        from werkzeug.security import check_password_hash
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'


class Prediction(db.Model):
    """Store each prediction made by the user."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    features_json = db.Column(db.Text, nullable=False)   # JSON string of input features
    prediction = db.Column(db.Integer, nullable=False)   # 0 = not default, 1 = default
    probability = db.Column(db.Float, nullable=False)    # probability of default (class 1)
    child_age = db.Column(db.Integer, nullable=False)
    woman_age = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f'<Prediction {self.id} by user {self.user_id}>'