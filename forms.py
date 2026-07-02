from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, IntegerField, SelectField, SubmitField
from wtforms.validators import DataRequired, Length, NumberRange, EqualTo, ValidationError
from models import User

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=80)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    is_admin = BooleanField('Admin')
    submit = SubmitField('Create User')

    def validate_username(self, field):
        if User.query.filter_by(username=field.data).first():
            raise ValidationError('Username already taken.')

class ChangePasswordForm(FlaskForm):
    old_password = PasswordField('Old Password', validators=[DataRequired()])
    new_password = PasswordField('New Password', validators=[DataRequired(), Length(min=6)])
    confirm = PasswordField('Confirm New Password', validators=[DataRequired(), EqualTo('new_password')])
    submit = SubmitField('Change Password')

class PredictForm(FlaskForm):
    Age_of_child = SelectField('Child Age (0–2 years)', 
                               choices=[('0', '0 years'), ('1', '1 year'), ('2', '2 years')],
                               default='0',
                               validators=[DataRequired()],
                               render_kw={'required': False})
    Age_of_woman = IntegerField('Woman Age (15–49 years)', 
                                validators=[DataRequired(), NumberRange(min=15, max=49)])
    # ... other fields remain the same
    Wealth_index_quintile = SelectField('Wealth Index Quintile', choices=[
        ('POOREST', 'Poorest'), ('SECOND', 'Second'), ('MIDDLE', 'Middle'),
        ('FOURTH', 'Fourth'), ('RICHEST', 'Richest')
    ], validators=[DataRequired()])
    Highest_level_of_school_attended = SelectField('Highest School Level', choices=[
        ('ECE', 'Early Childhood Education'), ('PRIMARY', 'Primary'),
        ('LOWER SECONDARY', 'Lower Secondary'), ('UPPER SECONDARY', 'Upper Secondary'),
        ('HIGHER', 'Higher'), ('VOCATIONAL TRAINING', 'Vocational Training')
    ], validators=[DataRequired()])
    Frequency_of_listening_to_radio = SelectField('Radio Listening Frequency', choices=[
        ('NOT AT ALL', 'Not at All'), ('LESS THAN ONCE A WEEK', 'Less Than Once a Week'),
        ('AT LEAST ONCE A WEEK', 'At Least Once a Week'), ('ALMOST EVERY DAY', 'Almost Every Day'),
        ('NO RESPONSE', 'No Response')
    ], validators=[DataRequired()])
    Mobile_phone_usage = SelectField('Mobile Phone Usage', choices=[
        ('NOT AT ALL', 'Not at All'), ('LESS THAN ONCE A WEEK', 'Less Than Once a Week'),
        ('AT LEAST ONCE A WEEK', 'At Least Once a Week'), ('ALMOST EVERY DAY', 'Almost Every Day'),
        ('NO RESPONSE', 'No Response')
    ], validators=[DataRequired()])
    Received_prenatal_care = SelectField('Received Prenatal Care', choices=[('NO', 'No'), ('YES', 'Yes')], validators=[DataRequired()])
    Marital_Status = SelectField('Marital Status', choices=[
        ('NOT_IN_UNION', 'Not in Union'), ('LIVING_WITH_PARTNER', 'Living with Partner'),
        ('MARRIED', 'Married')
    ], validators=[DataRequired()])
    submit = SubmitField('Predict')