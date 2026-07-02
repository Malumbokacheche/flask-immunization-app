import os
import json
from datetime import datetime, timedelta
from flask import Flask, render_template, redirect, url_for, flash, request
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
import pandas as pd
import joblib

# Database and models
from models import db, User, Prediction
from forms import LoginForm, RegistrationForm, ChangePasswordForm, PredictForm
from utils import load_model, prepare_input_for_model, get_recommendations

# -------------------------------------------------------------------
# App initialization
# -------------------------------------------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this'

# ----------------------------
# Fix: Absolute database path
# ----------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, 'instance', 'app.db')
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# -------------------------------------------------------------------
# Load the model (using joblib)
# -------------------------------------------------------------------
MODEL_PATH = os.path.join(BASE_DIR, 'immunization_model.pkl')
try:
    model_pipeline, feature_names = load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
    print(f"Expected features: {list(feature_names)}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model_pipeline = None
    feature_names = []

# -------------------------------------------------------------------
# User loader for Flask-Login
# -------------------------------------------------------------------
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# -------------------------------------------------------------------
# Create admin user if none exists
# -------------------------------------------------------------------
def create_admin():
    admin = User.query.filter_by(username='admin').first()
    if not admin:
        admin = User(username='admin', is_admin=True)
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()
        print("✅ Admin user created: admin / admin123")

# -------------------------------------------------------------------
# Create tables and admin on startup
# -------------------------------------------------------------------
with app.app_context():
    db.create_all()
    create_admin()

# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.check_password(form.password.data):
            login_user(user)
            flash('Logged in successfully.', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid username or password.', 'danger')
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
@login_required
def register():
    if not current_user.is_admin:
        flash('Only admin can create users.', 'danger')
        return redirect(url_for('dashboard'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, is_admin=form.is_admin.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash(f'User {form.username.data} created.', 'success')
        return redirect(url_for('admin_users'))
    return render_template('register.html', form=form)

@app.route('/change_password', methods=['GET', 'POST'])
@login_required
def change_password():
    form = ChangePasswordForm()
    if form.validate_on_submit():
        if current_user.check_password(form.old_password.data):
            current_user.set_password(form.new_password.data)
            db.session.commit()
            flash('Password changed successfully.', 'success')
            return redirect(url_for('dashboard'))
        flash('Old password is incorrect.', 'danger')
    return render_template('change_password.html', form=form)

@app.route('/admin/users')
@login_required
def admin_users():
    if not current_user.is_admin:
        flash('Access denied.', 'danger')
        return redirect(url_for('dashboard'))
    users = User.query.all()
    return render_template('admin_users.html', users=users)

@app.route('/admin/delete_user/<int:user_id>')
@login_required
def delete_user(user_id):
    if not current_user.is_admin:
        flash('Access denied.', 'danger')
        return redirect(url_for('dashboard'))
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        flash('Cannot delete yourself.', 'warning')
    else:
        db.session.delete(user)
        db.session.commit()
        flash(f'User {user.username} deleted.', 'success')
    return redirect(url_for('admin_users'))

@app.route('/dashboard')
@login_required
def dashboard():
    total_predictions = Prediction.query.filter_by(user_id=current_user.id).count()
    default_predictions = Prediction.query.filter_by(user_id=current_user.id, prediction=1).count()
    default_rate = (default_predictions / total_predictions * 100) if total_predictions > 0 else 0

    months = []
    counts = []
    defaults = []
    now = datetime.utcnow()
    for i in range(5, -1, -1):
        month_start = datetime(now.year, now.month, 1) - timedelta(days=30*i)
        month_end = month_start + timedelta(days=31)
        month_label = month_start.strftime('%b %Y')
        months.append(month_label)
        q = Prediction.query.filter(
            Prediction.user_id == current_user.id,
            Prediction.timestamp >= month_start,
            Prediction.timestamp < month_end
        )
        total = q.count()
        counts.append(total)
        default = q.filter_by(prediction=1).count()
        defaults.append(default)

    return render_template('dashboard.html',
                           total=total_predictions,
                           default_count=default_predictions,
                           default_rate=default_rate,
                           months=months,
                           counts=counts,
                           defaults=defaults)

@app.route('/monthly_report')
@login_required
def monthly_report():
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp).all()
    report = {}
    for p in predictions:
        key = p.timestamp.strftime('%Y-%m')
        if key not in report:
            report[key] = {'total': 0, 'default': 0}
        report[key]['total'] += 1
        if p.prediction == 1:
            report[key]['default'] += 1

    sorted_months = sorted(report.keys())
    report_list = []
    for m in sorted_months:
        total = report[m]['total']
        default = report[m]['default']
        rate = (default / total * 100) if total > 0 else 0
        report_list.append((m, total, default, rate))
    return render_template('monthly_report.html', report=report_list)
@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    # Check if model is loaded
    if model_pipeline is None:
        flash('Model not loaded. Please contact administrator.', 'danger')
        return redirect(url_for('dashboard'))

    form = PredictForm()

    # Handle POST (form submission)
    if form.validate_on_submit():
        try:
            # Collect raw input
            raw_data = {
                'Age_of_child': int(form.Age_of_child.data),
                'Age_of_woman': form.Age_of_woman.data,
                'Wealth_index_quintile': form.Wealth_index_quintile.data,
                'Highest_level_of_school_attended': form.Highest_level_of_school_attended.data,
                'Frequency_of_listening_to_radio': form.Frequency_of_listening_to_radio.data,
                'Mobile_phone_usage': form.Mobile_phone_usage.data,
                'Received_prenatal_care': form.Received_prenatal_care.data,
                'Marital_Status': form.Marital_Status.data
            }

            # Build input DataFrame
            df_input = prepare_input_for_model(raw_data, feature_names)

            # Get raw prediction (1 = immunised, 0 = default)
            raw_pred = int(model_pipeline.predict(df_input)[0])
            raw_prob = float(model_pipeline.predict_proba(df_input)[0][1])

            # Invert to get probability of default
            prob_default = 1 - raw_prob

            # ---------- Age-aware threshold with indicator-based adjustment ----------
            age = raw_data['Age_of_child']

            if age == 0:
                # Define "good" indicators
                good_indicators = (
                    raw_data['Wealth_index_quintile'] in ['MIDDLE', 'FOURTH', 'RICHEST'] and
                    raw_data['Highest_level_of_school_attended'] in ['PRIMARY', 'LOWER SECONDARY', 'UPPER SECONDARY', 'HIGHER', 'VOCATIONAL TRAINING'] and
                    raw_data['Received_prenatal_care'] == 'YES' and
                    raw_data['Marital_Status'] == 'MARRIED'
                )

                if good_indicators:
                    threshold = 0.95   # conservative for good indicators
                else:
                    threshold = 0.70   # more sensitive for poor indicators

            elif age == 1:
                threshold = 0.55
            else:  # age == 2
                threshold = 0.5

            pred = 1 if prob_default >= threshold else 0
            # -----------------------------------------------

            # Save prediction
            pred_record = Prediction(
                user_id=current_user.id,
                features_json=json.dumps(raw_data),
                prediction=pred,
                probability=prob_default,
                child_age=raw_data['Age_of_child'],
                woman_age=raw_data['Age_of_woman']
            )
            db.session.add(pred_record)
            db.session.commit()

            # Get recommendations
            recs = get_recommendations(pred, prob_default, raw_data)

            return render_template('result.html',
                                   prediction=pred,
                                   probability=prob_default,
                                   recommendations=recs,
                                   data=raw_data)

        except Exception as e:
            import traceback
            traceback.print_exc()
            flash(f'Prediction error: {str(e)}', 'danger')
            return render_template('predict.html', form=form)

    # If form validation fails, display errors
    if form.errors:
        for field, errors in form.errors.items():
            for error in errors:
                field_label = getattr(form, field).label.text if hasattr(form, field) else field
                flash(f'Error in {field_label}: {error}', 'danger')
        return render_template('predict.html', form=form)

    # GET request – show empty form
    return render_template('predict.html', form=form)
# -------------------------------------------------------------------
# Run the app
# -------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)