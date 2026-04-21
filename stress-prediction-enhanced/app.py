from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import os
import json
import secrets
from functools import wraps
import sqlalchemy as sa

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///stress_detection.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'




CSI_WEIGHTS = {
    'sleep_deprivation':     0.28,
    'night_noisy_compound':  0.18,
    'overwork_compound':     0.20,
    'thermal_discomfort':    0.12,
    'demanding_overwork':    0.14,
    'body_signal':           0.08,
}

def compute_compound_stress_index(record):
    """
    PATENTABLE FEATURE 1: Compound Stress Index (CSI)
    Computes a novel composite score using weighted non-linear
    interactions between biophysical and environmental factors.
    Returns CSI score (0-100) and a breakdown dict.
    """
    breakdown = {}

    
    sleep = float(record.sleep_hours)
    if sleep < 6:
        sleep_score = ((6 - sleep) / 6) ** 1.5 * 100
    elif sleep > 9:
        sleep_score = ((sleep - 9) / 3) ** 1.2 * 40
    else:
        sleep_score = 0
    breakdown['sleep_deprivation'] = round(sleep_score, 2)

   
    is_night = 1 if record.working_shift == 'Night shift' else 0
    is_noisy = 1 if record.noise_levels == 'Noisy' else 0
    night_noisy_score = (is_night * 60) + (is_noisy * 40) + (is_night * is_noisy * 50)
    breakdown['night_noisy_compound'] = round(night_noisy_score, 2)

   
    heavy_workload = 1 if record.workload == 'Heavy' else 0
    long_hours = 1 if record.working_hours in ['Long Day', 'Extreme Overtime'] else 0
    extreme_hours = 1 if record.working_hours == 'Extreme Overtime' else 0
    overwork_score = (heavy_workload * 50) + (long_hours * 40) + (heavy_workload * extreme_hours * 60)
    breakdown['overwork_compound'] = round(overwork_score, 2)

   
    body_abnormal = 1 if record.body_temperature == 'Not Normal' else 0
    workspace_uncomfortable = 1 if record.working_area_temperature == 'Uncomfortable' else 0
    thermal_score = (body_abnormal * 70) + (workspace_uncomfortable * 50) + (body_abnormal * workspace_uncomfortable * 40)
    breakdown['thermal_discomfort'] = round(thermal_score, 2)

   
    demanding = 1 if record.type_of_work == 'Demanding' else 0
    demanding_score = (demanding * 50) + (demanding * heavy_workload * 60)
    breakdown['demanding_overwork'] = round(demanding_score, 2)

    
    body_score = 80 if body_abnormal else 0
    breakdown['body_signal'] = round(body_score, 2)

    max_values = {
        'sleep_deprivation':    100,
        'night_noisy_compound': 150,
        'overwork_compound':    150,
        'thermal_discomfort':   160,
        'demanding_overwork':   110,
        'body_signal':          80,
    }

    csi = 0.0
    for key, weight in CSI_WEIGHTS.items():
        normalised = min(breakdown[key] / max_values[key], 1.0)
        csi += normalised * weight * 100

    breakdown['csi_total'] = round(csi, 2)
    return round(csi, 2), breakdown

class SuggestionFeedback(db.Model):
    id             = db.Column(db.Integer, primary_key=True)
    user_id        = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    suggestion_key = db.Column(db.String(100), nullable=False)
    shown_at       = db.Column(db.DateTime, default=datetime.utcnow)
    stress_before  = db.Column(db.Float, nullable=False)
    stress_after   = db.Column(db.Float, nullable=True)
    improved       = db.Column(db.Boolean, nullable=True)


def get_adaptive_suggestion_weights(user_id):
    feedbacks = SuggestionFeedback.query.filter_by(
        user_id=user_id
    ).filter(SuggestionFeedback.improved.isnot(None)).all()

    counts  = {}
    success = {}
    for fb in feedbacks:
        k = fb.suggestion_key
        counts[k]  = counts.get(k, 0) + 1
        success[k] = success.get(k, 0) + (1 if fb.improved else 0)

    weights = {}
    for k, total in counts.items():
        rate = success[k] / total
        weights[k] = round(0.5 + (rate * 1.5), 2)
    return weights


def close_feedback_loop(user_id, new_stress_pct):
    pending = SuggestionFeedback.query.filter_by(
        user_id=user_id, improved=None, stress_after=None
    ).all()
    for fb in pending:
        fb.stress_after = new_stress_pct
        fb.improved = bool(new_stress_pct < fb.stress_before)
    db.session.commit()


def record_suggestions_shown(user_id, suggestion_keys, stress_pct):
    for key in suggestion_keys:
        fb = SuggestionFeedback(
            user_id=user_id,
            suggestion_key=key,
            stress_before=stress_pct,
        )
        db.session.add(fb)
    db.session.commit()


def compute_esl_score(record):
    components = {}

    noise_map = {'Quiet': 0, 'Moderate': 35, 'Noisy': 80}
    noise_val = noise_map.get(record.noise_levels, 40)
    components['noise'] = noise_val

    temp_map = {'Comfortable': 0, 'Moderate': 30, 'Uncomfortable': 75}
    temp_val = temp_map.get(record.working_area_temperature, 30)
    components['workspace_temp'] = temp_val

    body_map = {'Normal': 0, 'Slightly Elevated': 40, 'Not Normal': 80}
    body_val = body_map.get(record.body_temperature, 0)
    components['body_temp'] = body_val

    shift_map = {'Day shift': 0, 'Evening shift': 30, 'Night shift': 70}
    shift_val = shift_map.get(record.working_shift, 0)
    components['circadian_penalty'] = shift_val

    active_stressors = sum([
        1 if noise_val > 50 else 0,
        1 if temp_val > 50 else 0,
        1 if body_val > 50 else 0,
        1 if shift_val > 50 else 0,
    ])
    amplifier = 1.0 + (active_stressors * 0.15)
    raw_esl = (noise_val * 0.30 + temp_val * 0.25 + body_val * 0.25 + shift_val * 0.20) * amplifier
    esl_score = min(round(raw_esl, 2), 100)
    components['amplifier'] = round(amplifier, 2)
    components['esl_total'] = esl_score
    return esl_score, components


class User(UserMixin, db.Model):
    id             = db.Column(db.Integer, primary_key=True)
    username       = db.Column(db.String(100), unique=True, nullable=False)
    email          = db.Column(db.String(100), unique=True, nullable=False)
    password       = db.Column(db.String(200), nullable=False)
    is_admin       = db.Column(db.Boolean, default=False)
    date_joined    = db.Column(db.DateTime, default=datetime.utcnow)
    stress_records = db.relationship('StressRecord', backref='user', lazy=True)


class StressRecord(db.Model):
    id                       = db.Column(db.Integer, primary_key=True)
    user_id                  = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date                     = db.Column(db.DateTime, default=datetime.utcnow)
    sleep_hours              = db.Column(db.Float, nullable=False)
    body_temperature         = db.Column(db.String(50), nullable=False)
    noise_levels             = db.Column(db.String(50), nullable=False)
    working_hours            = db.Column(db.String(50), nullable=False)
    working_area_temperature = db.Column(db.String(50), nullable=False)
    workload                 = db.Column(db.String(50), nullable=False)
    type_of_work             = db.Column(db.String(50), nullable=False)
    working_shift            = db.Column(db.String(50), nullable=False)
    stress_level             = db.Column(db.String(50), nullable=False)
    stress_percentage        = db.Column(db.Float, nullable=False)
    model_version            = db.Column(db.String(50), nullable=True, default="improved")
    csi_score                = db.Column(db.Float, nullable=True)
    esl_score                = db.Column(db.Float, nullable=True)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash('You need admin privileges to access this page.', 'danger')
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated_function




def load_model_resources():
    try:
        with open('workplace_stress_encoders_improved.pkl', 'rb') as f:
            encoders_info = pickle.load(f)
        with open('workplace_stress_ensemble_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return encoders_info, model
    except FileNotFoundError:
        try:
            with open('workplace_stress_encoders.pkl', 'rb') as f:
                encoders_info = pickle.load(f)
            with open('workplace_stress_xgboost_model.pkl', 'rb') as f:
                model = pickle.load(f)
            return encoders_info, model
        except FileNotFoundError:
            raise Exception("No model files found.")


def predict_stress_level(input_data, encoders_info, model):
    input_df = pd.DataFrame([input_data])
    input_df['sleep_squared']              = input_df['sleep_hours'] ** 2
    input_df['sleep_low']                  = (input_df['sleep_hours'] < 6).astype(int)
    input_df['sleep_optimal']              = ((input_df['sleep_hours'] >= 7) & (input_df['sleep_hours'] <= 8)).astype(int)
    input_df['night_noisy']                = ((input_df['working_shift'] == 'Night shift') & (input_df['noise_levels'] == 'Noisy')).astype(int)
    input_df['bad_conditions']             = ((input_df['working_area_temperature'] == 'Uncomfortable') & (input_df['type_of_work'] == 'Demanding')).astype(int)
    input_df['overworked']                 = ((input_df['workload'] == 'Heavy') & (input_df['working_hours'].isin(['Long Day', 'Extreme Overtime']))).astype(int)
    input_df['sleep_workload_interaction'] = input_df.apply(
        lambda r: 1 if (r['sleep_hours'] < 6 and r['workload'] == 'Heavy') else 0, axis=1)
    stress_conditions = [
        input_df['sleep_low'] == 1,
        input_df['working_shift'] == 'Night shift',
        input_df['workload'] == 'Heavy',
        input_df['noise_levels'] == 'Noisy',
        input_df['working_area_temperature'] == 'Uncomfortable',
        input_df['type_of_work'] == 'Demanding',
    ]
    input_df['stress_factors_count'] = sum(stress_conditions)
    for col in encoders_info['cat_mappings'].keys():
        le = encoders_info['label_encoders'][col]
        input_df[col + '_encoded'] = le.transform(input_df[col])
    result_df = pd.DataFrame(index=input_df.index)
    for feature in encoders_info['feature_columns']:
        if feature in input_df.columns:
            if feature in ['sleep_hours', 'sleep_squared', 'stress_factors_count']:
                feature_values = input_df[[feature]].values
                scaled_values  = encoders_info['numeric_scaler'].transform(feature_values)
                result_df[feature] = scaled_values.flatten()
            else:
                result_df[feature] = input_df[feature]
        else:
            raise ValueError(f"Required feature '{feature}' is missing")
    pred_encoded = model.predict(result_df)[0]
    if hasattr(model, 'predict_proba'):
        pred_proba = model.predict_proba(result_df)[0]
    else:
        pred_proba = np.zeros(len(encoders_info['label_encoders']['calculated_stress_level'].classes_))
        pred_proba[pred_encoded] = 1.0
    pred_label = encoders_info['label_encoders']['calculated_stress_level'].inverse_transform([pred_encoded])[0]
    return pred_label, pred_encoded, pred_proba




SUGGESTION_LIBRARY = {
    'sleep_low':            {'title': 'Improve Sleep Quality',         'content': "You're sleeping under 6 hours. Aim for 7-8 hrs by setting a fixed bedtime. Even 30 extra minutes can lower cortisol levels significantly."},
    'night_shift':          {'title': 'Night Shift Adaptation',        'content': 'Night shifts disrupt your circadian rhythm. Use blackout curtains, avoid screens 1hr before sleep, and maintain a consistent sleep window.'},
    'heavy_workload':       {'title': 'Workload Management',           'content': 'Your workload is heavy. Use time-blocking and the 2-minute rule: tasks under 2 minutes are done immediately, the rest get scheduled.'},
    'uncomfortable_temp':   {'title': 'Adjust Your Environment',       'content': 'Your workspace temperature is uncomfortable. A desk fan or heater can reduce thermal stress and improve focus.'},
    'noisy_env':            {'title': 'Manage Noise Levels',           'content': 'Noisy environments increase cortisol. Noise-cancelling headphones or white noise apps can restore focus and lower stress.'},
    'long_hours':           {'title': 'Work-Life Boundary',            'content': 'Long working hours erode recovery time. Schedule hard stops and protect at least 1 hour of non-work wind-down time each evening.'},
    'demanding_work':       {'title': 'Manage Demanding Work',         'content': 'Demanding tasks drain cognitive reserves. Use the Pomodoro technique: 25 min focused work, 5 min break.'},
    'body_temp_abnormal':   {'title': 'Monitor Your Health',           'content': 'An abnormal body temperature amplifies perceived stress. Rest, hydration, and a health check are recommended.'},
    'medium_stress':        {'title': 'Regular Breaks',                'content': 'Take 5-minute micro-breaks every hour. Stand, stretch, and hydrate. This alone reduces afternoon stress spikes noticeably.'},
    'high_stress':          {'title': 'Stress Management Techniques',  'content': 'Practice box breathing: inhale 4 sec, hold 4 sec, exhale 4 sec, hold 4 sec. Repeat 3x to activate your parasympathetic nervous system.'},
    'extreme_stress':       {'title': 'Professional Support',          'content': 'Your stress indicators are at a critical level. Please consider speaking with a counsellor or occupational health professional this week.'},
    'maintain_routine':     {'title': 'Maintain Your Routine',         'content': 'Your stress is low — great! Consistency is key. Keep your current sleep and work habits to sustain this balance long-term.'},
}


def get_adaptive_suggestions(stress_record, user_id=None):
    weights = get_adaptive_suggestion_weights(user_id) if user_id else {}
    keys = []
    if stress_record.stress_level == 'Low':           keys.append('maintain_routine')
    elif stress_record.stress_level == 'Medium':      keys.append('medium_stress')
    elif stress_record.stress_level == 'High':        keys.append('high_stress')
    else:                                              keys.append('extreme_stress')
    if float(stress_record.sleep_hours) < 6:                         keys.append('sleep_low')
    if stress_record.working_shift == 'Night shift':                 keys.append('night_shift')
    if stress_record.workload == 'Heavy':                            keys.append('heavy_workload')
    if stress_record.working_area_temperature == 'Uncomfortable':    keys.append('uncomfortable_temp')
    if stress_record.noise_levels == 'Noisy':                        keys.append('noisy_env')
    if stress_record.working_hours in ['Long Day', 'Extreme Overtime']: keys.append('long_hours')
    if stress_record.type_of_work == 'Demanding':                    keys.append('demanding_work')
    if stress_record.body_temperature == 'Not Normal':               keys.append('body_temp_abnormal')

    suggestions = []
    for key in keys:
        if key in SUGGESTION_LIBRARY:
            weight = weights.get(key, 1.0)
            suggestions.append({
                'key':     key,
                'title':   SUGGESTION_LIBRARY[key]['title'],
                'content': SUGGESTION_LIBRARY[key]['content'],
                'weight':  weight,
                'badge':   'Proven Effective' if weight >= 1.5 else ('Mixed Results' if weight < 0.8 else ''),
            })
    suggestions.sort(key=lambda x: x['weight'], reverse=True)
    return suggestions, keys


@app.context_processor
def inject_year():
    return {'current_year': datetime.now().year}


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username         = request.form.get('username')
        email            = request.form.get('email')
        password         = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        if not username or not email or not password:
            flash('All fields are required', 'danger')
            return redirect(url_for('signup'))
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('signup'))
        user_exists = User.query.filter((User.username == username) | (User.email == email)).first()
        if user_exists:
            flash('Username or email already exists', 'danger')
            return redirect(url_for('signup'))
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, email=email, password=hashed_password)
        if User.query.count() == 0:
            new_user.is_admin = True
        db.session.add(new_user)
        db.session.commit()
        flash('Account created successfully! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('admin_dashboard') if user.is_admin else url_for('user_dashboard'))
        flash('Invalid username or password', 'danger')
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))


@app.route('/stress-predictor', methods=['GET', 'POST'])
def stress_predictor():
    try:
        encoders_info, model = load_model_resources()
        body_temp_options  = list(encoders_info['cat_mappings']['body_temperature'].keys())
        noise_options      = list(encoders_info['cat_mappings']['noise_levels'].keys())
        work_hours_options = list(encoders_info['cat_mappings']['working_hours'].keys())
        temp_options       = list(encoders_info['cat_mappings']['working_area_temperature'].keys())
        workload_options   = list(encoders_info['cat_mappings']['workload'].keys())
        work_type_options  = list(encoders_info['cat_mappings']['type_of_work'].keys())
        shift_options      = list(encoders_info['cat_mappings']['working_shift'].keys())

        if request.method == 'POST':
            input_data = {
                'sleep_hours':              float(request.form.get('sleep_hours')),
                'body_temperature':         request.form.get('body_temperature'),
                'noise_levels':             request.form.get('noise_levels'),
                'working_hours':            request.form.get('working_hours'),
                'working_area_temperature': request.form.get('working_area_temperature'),
                'workload':                 request.form.get('workload'),
                'type_of_work':             request.form.get('type_of_work'),
                'working_shift':            request.form.get('working_shift'),
            }
            pred_label, pred_encoded, pred_proba = predict_stress_level(input_data, encoders_info, model)
            stress_pct = round(float(np.max(pred_proba)) * 100, 1)
            messages = {
                'Low':           'Great! Your workplace stress levels are low. Keep maintaining your healthy habits!',
                'Medium':        'Your stress levels are moderate. Consider implementing some stress management strategies.',
                'High':          'Your stress levels are high. It\'s important to take action to reduce your stress.',
                'Extremely High':'Your stress levels are extremely high! Please seek support immediately.',
            }

            if current_user.is_authenticated and not current_user.is_admin:
                close_feedback_loop(current_user.id, stress_pct)
                new_record = StressRecord(
                    user_id=current_user.id,
                    sleep_hours=input_data['sleep_hours'],
                    body_temperature=input_data['body_temperature'],
                    noise_levels=input_data['noise_levels'],
                    working_hours=input_data['working_hours'],
                    working_area_temperature=input_data['working_area_temperature'],
                    workload=input_data['workload'],
                    type_of_work=input_data['type_of_work'],
                    working_shift=input_data['working_shift'],
                    stress_level=pred_label,
                    stress_percentage=stress_pct,
                )
                csi_score, csi_breakdown = compute_compound_stress_index(new_record)
                new_record.csi_score = csi_score
                esl_score, esl_components = compute_esl_score(new_record)
                new_record.esl_score = esl_score
                db.session.add(new_record)
                db.session.commit()
                _, suggestion_keys = get_adaptive_suggestions(new_record, current_user.id)
                record_suggestions_shown(current_user.id, suggestion_keys, stress_pct)
                return render_template('stress_predictor.html',
                    show_result=True, prediction=pred_label,
                    percentage=f"{stress_pct}%", message=messages.get(pred_label, ''),
                    input_data=new_record, csi_score=csi_score, csi_breakdown=csi_breakdown,
                    esl_score=esl_score, esl_components=esl_components,
                    body_temp_options=body_temp_options, noise_options=noise_options,
                    work_hours_options=work_hours_options, temp_options=temp_options,
                    workload_options=workload_options, work_type_options=work_type_options,
                    shift_options=shift_options)

            # Anonymous: compute scores on temp object
            class TempRecord:
                pass
            tmp = TempRecord()
            for k, v in input_data.items():
                setattr(tmp, k, v)
            tmp.stress_level = pred_label
            tmp.stress_percentage = stress_pct
            csi_score, csi_breakdown = compute_compound_stress_index(tmp)
            esl_score, esl_components = compute_esl_score(tmp)
            return render_template('stress_predictor.html',
                show_result=True, prediction=pred_label,
                percentage=f"{stress_pct}%", message=messages.get(pred_label, ''),
                input_data=tmp, csi_score=csi_score, csi_breakdown=csi_breakdown,
                esl_score=esl_score, esl_components=esl_components,
                body_temp_options=body_temp_options, noise_options=noise_options,
                work_hours_options=work_hours_options, temp_options=temp_options,
                workload_options=workload_options, work_type_options=work_type_options,
                shift_options=shift_options)

        return render_template('stress_predictor.html', show_result=False,
            body_temp_options=body_temp_options, noise_options=noise_options,
            work_hours_options=work_hours_options, temp_options=temp_options,
            workload_options=workload_options, work_type_options=work_type_options,
            shift_options=shift_options)

    except Exception as e:
        flash(f'Error: {str(e)}', 'danger')
        return redirect(url_for('home'))


@app.route('/user-dashboard')
@login_required
def user_dashboard():
    stress_records = StressRecord.query.filter_by(
        user_id=current_user.id).order_by(StressRecord.date.desc()).all()
    return render_template('user_dashboard.html', stress_records=stress_records)


@app.route('/admin-dashboard')
@login_required
@admin_required
def admin_dashboard():
    users = User.query.all()
    stress_records = StressRecord.query.order_by(StressRecord.date.desc()).limit(20).all()
    return render_template('admin_dashboard.html', users=users, stress_records=stress_records,
        total_users=User.query.count(), total_records=StressRecord.query.count())


@app.route('/record/<int:record_id>')
@login_required
def view_record(record_id):
    record = StressRecord.query.get_or_404(record_id)
    if record.user_id != current_user.id and not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
    return jsonify({
        'sleep_hours': record.sleep_hours, 'body_temperature': record.body_temperature,
        'noise_levels': record.noise_levels, 'working_area_temperature': record.working_area_temperature,
        'working_hours': record.working_hours, 'workload': record.workload,
        'type_of_work': record.type_of_work, 'working_shift': record.working_shift,
        'stress_level': record.stress_level, 'stress_percentage': record.stress_percentage,
        'csi_score': record.csi_score, 'esl_score': record.esl_score,
        'date': record.date.strftime('%Y-%m-%d %H:%M'),
    })


@app.route('/suggestions')
@login_required
def suggestions():
    latest_record = StressRecord.query.filter_by(
        user_id=current_user.id).order_by(StressRecord.date.desc()).first()
    if not latest_record:
        flash("No stress records found. Please complete a stress prediction first.", "info")
        return redirect(url_for('stress_predictor'))
    suggestions_list, _ = get_adaptive_suggestions(latest_record, current_user.id)
    csi_score, csi_breakdown = compute_compound_stress_index(latest_record)
    esl_score, esl_components = compute_esl_score(latest_record)
    return render_template('suggestions.html', record=latest_record,
        suggestions=suggestions_list, csi_score=csi_score, csi_breakdown=csi_breakdown,
        esl_score=esl_score, esl_components=esl_components)


@app.route('/analysis')
@login_required
def analysis():
    stress_records = StressRecord.query.filter_by(
        user_id=current_user.id).order_by(StressRecord.date.asc()).all()
    if not stress_records:
        flash("No stress records found. Please complete a stress prediction first.", "info")
        return redirect(url_for('stress_predictor'))

    dates = [r.date.strftime('%Y-%m-%d') for r in stress_records]
    num_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Extremely High': 4}
    stress_levels_list = [num_map.get(r.stress_level, 0) for r in stress_records]
    csi_scores = [r.csi_score or 0 for r in stress_records]
    esl_scores = [r.esl_score or 0 for r in stress_records]

    stress_level_counts = {'Low': 0, 'Medium': 0, 'High': 0, 'Extremely High': 0}
    for r in stress_records:
        if r.stress_level in stress_level_counts:
            stress_level_counts[r.stress_level] += 1

    workload_impact = {}
    sleep_impact = {}
    for r in stress_records:
        sv = num_map.get(r.stress_level, 0)
        if r.workload not in workload_impact:
            workload_impact[r.workload] = {'count': 0, 'stress_sum': 0}
        workload_impact[r.workload]['count'] += 1
        workload_impact[r.workload]['stress_sum'] += sv

        sh = r.sleep_hours
        if sh < 5:       sr = '< 5 hours'
        elif sh < 6:     sr = '5-6 hours'
        elif sh < 7:     sr = '6-7 hours'
        elif sh <= 8:    sr = '7-8 hours'
        else:            sr = '> 8 hours'
        if sr not in sleep_impact:
            sleep_impact[sr] = {'count': 0, 'stress_sum': 0}
        sleep_impact[sr]['count'] += 1
        sleep_impact[sr]['stress_sum'] += sv

    for c in workload_impact:
        n = workload_impact[c]['count']
        workload_impact[c]['avg_stress'] = workload_impact[c]['stress_sum'] / n if n else 0
    for c in sleep_impact:
        n = sleep_impact[c]['count']
        sleep_impact[c]['avg_stress'] = sleep_impact[c]['stress_sum'] / n if n else 0

    sleep_order = ['< 5 hours', '5-6 hours', '6-7 hours', '7-8 hours', '> 8 hours']
    sorted_sleep = {k: sleep_impact[k] for k in sleep_order if k in sleep_impact}

    return render_template('analysis.html',
        dates=json.dumps(dates), stress_levels=json.dumps(stress_levels_list),
        csi_scores=json.dumps(csi_scores), esl_scores=json.dumps(esl_scores),
        stress_level_counts=stress_level_counts,
        workload_impact=workload_impact, sleep_impact=sorted_sleep)


@app.route('/api/suggestion-feedback', methods=['POST'])
@login_required
def suggestion_feedback():
    data    = request.get_json()
    key     = data.get('key')
    helpful = data.get('helpful')
    if not key:
        return jsonify({'error': 'Missing key'}), 400
    fb = SuggestionFeedback.query.filter_by(
        user_id=current_user.id, suggestion_key=key, stress_after=None
    ).order_by(SuggestionFeedback.shown_at.desc()).first()
    if fb:
        fb.improved = helpful
        db.session.commit()
    return jsonify({'status': 'ok'})

def compute_stress_forecast(user_id):
    """
    PATENTABLE FEATURE 4: Stress Risk Forecasting (SRF)
    Analyses historical records to forecast tomorrow's stress risk.
    Returns forecast_level (Low/Medium/High/Extremely High),
    risk_score (0-100), confidence (%), and reasoning list.
    """
    records = StressRecord.query.filter_by(user_id=user_id)\
        .order_by(StressRecord.date.desc()).limit(30).all()

    if len(records) < 2:
        return None  # Not enough data

    level_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Extremely High': 4}
    scores = [level_map.get(r.stress_level, 2) for r in records]
    csi_vals = [r.csi_score or 0 for r in records]

    reasoning = []

    # --- Signal 1: 7-day rolling average trend ---
    recent = scores[:7]
    older  = scores[7:14] if len(scores) >= 14 else scores
    recent_avg = sum(recent) / len(recent)
    older_avg  = sum(older)  / len(older)
    trend_delta = recent_avg - older_avg
    if trend_delta > 0.3:
        reasoning.append("📈 Your stress has been rising over the past week.")
    elif trend_delta < -0.3:
        reasoning.append("📉 Your stress has been improving recently.")

    # --- Signal 2: Day-of-week pattern ---
    tomorrow = (datetime.utcnow().weekday() + 1) % 7
    day_records = [r for r in records if r.date.weekday() == tomorrow]
    if day_records:
        day_scores = [level_map.get(r.stress_level, 2) for r in day_records]
        day_avg = sum(day_scores) / len(day_scores)
        day_names = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        if day_avg >= 3:
            reasoning.append(f"📅 {day_names[tomorrow]}s are historically high-stress for you.")
        elif day_avg <= 1.5:
            reasoning.append(f"📅 {day_names[tomorrow]}s are usually low-stress for you.")
    else:
        day_avg = recent_avg

    # --- Signal 3: CSI momentum (last 3 records) ---
    if len(csi_vals) >= 3:
        csi_momentum = csi_vals[0] - csi_vals[2]
        if csi_momentum > 15:
            reasoning.append("⚡ Your Compound Stress Index has surged in the last 3 sessions.")
        elif csi_momentum < -15:
            reasoning.append("✅ Your Compound Stress Index has dropped significantly.")

    # --- Signal 4: Consecutive high-stress days ---
    consecutive_high = 0
    for s in scores:
        if s >= 3:
            consecutive_high += 1
        else:
            break
    if consecutive_high >= 3:
        reasoning.append(f"🔴 You've had {consecutive_high} consecutive high-stress sessions — burnout risk elevated.")

    # --- Composite forecast score ---
    w_rolling  = 0.40
    w_day      = 0.30
    w_momentum = 0.20
    w_consec   = 0.10

    momentum_score = min((csi_vals[0] / 100) * 4, 4) if csi_vals else recent_avg
    consec_score   = min(consecutive_high * 0.5 + 1, 4)

    raw = (recent_avg * w_rolling + day_avg * w_day +
           momentum_score * w_momentum + consec_score * w_consec)
    risk_score = round(min((raw / 4) * 100, 100), 1)

    if raw < 1.75:
        forecast_level = 'Low'
    elif raw < 2.5:
        forecast_level = 'Medium'
    elif raw < 3.25:
        forecast_level = 'High'
    else:
        forecast_level = 'Extremely High'

    # Confidence: more records = more confident
    confidence = min(round(50 + len(records) * 1.5, 0), 92)

    if not reasoning:
        reasoning.append("📊 Forecast based on your recent stress patterns.")

    return {
        'forecast_level': forecast_level,
        'risk_score':     risk_score,
        'confidence':     int(confidence),
        'reasoning':      reasoning,
        'records_used':   len(records),
    }


def compute_burnout_risk(user_id):
    """
    PATENTABLE FEATURE 5: Burnout Early Warning System (BEWS)
    Analyses 14-day rolling window to compute Burnout Risk %.
    Returns risk_pct (0-100), alert_level, and signals list.
    """
    records = StressRecord.query.filter_by(user_id=user_id)\
        .order_by(StressRecord.date.desc()).limit(14).all()

    if len(records) < 3:
        return None

    level_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Extremely High': 4}
    scores    = [level_map.get(r.stress_level, 2) for r in records]
    csi_vals  = [r.csi_score or 0 for r in records]
    signals   = []
    risk_components = {}

    # --- Component 1: Sustained High Stress (% of sessions >= High) ---
    high_count = sum(1 for s in scores if s >= 3)
    high_ratio = high_count / len(scores)
    risk_components['sustained_high'] = high_ratio * 100
    if high_ratio >= 0.6:
        signals.append(f"🔴 {int(high_ratio*100)}% of your recent sessions were High or Extremely High stress.")

    # --- Component 2: Upward Trend Slope ---
    if len(scores) >= 5:
        x = list(range(len(scores)))
        x_mean = sum(x) / len(x)
        y_mean = sum(scores) / len(scores)
        slope_num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, scores))
        slope_den = sum((xi - x_mean) ** 2 for xi in x)
        slope = (slope_num / slope_den) if slope_den != 0 else 0
        # Negative slope = getting worse (records are newest-first)
        trend_risk = max(min(-slope * 60, 100), 0)
        risk_components['upward_trend'] = trend_risk
        if slope < -0.15:
            signals.append("📈 Your stress has been steadily increasing over recent sessions.")
    else:
        risk_components['upward_trend'] = 0

    # --- Component 3: No Recovery Sessions (no Low stress in last 7) ---
    recent7 = scores[:7]
    low_sessions = sum(1 for s in recent7 if s == 1)
    recovery_deficit = max(0, 2 - low_sessions)  # expect at least 2 low sessions per week
    recovery_risk = (recovery_deficit / 2) * 100
    risk_components['recovery_deficit'] = recovery_risk
    if low_sessions == 0:
        signals.append("😴 You haven't had a low-stress session in the past 7 records — no recovery detected.")
    elif low_sessions < 2:
        signals.append(f"⚠️ Only {low_sessions} low-stress session(s) recently — recovery is insufficient.")

    # --- Component 4: CSI Elevation (avg CSI above 60 = danger zone) ---
    avg_csi = sum(csi_vals) / len(csi_vals) if csi_vals else 0
    csi_risk = max(0, (avg_csi - 40) / 60 * 100)
    risk_components['csi_elevation'] = round(csi_risk, 1)
    if avg_csi > 65:
        signals.append(f"🧮 Your average Compound Stress Index is {round(avg_csi,1)} — well above safe levels.")

    # --- Weighted Burnout Risk Score ---
    burnout_risk = (
        risk_components['sustained_high']  * 0.35 +
        risk_components['upward_trend']    * 0.25 +
        risk_components['recovery_deficit']* 0.25 +
        risk_components['csi_elevation']   * 0.15
    )
    burnout_risk = round(min(burnout_risk, 100), 1)

    if burnout_risk < 25:
        alert_level = 'Low'
        alert_msg   = 'Your burnout risk is low. Keep it up!'
    elif burnout_risk < 50:
        alert_level = 'Moderate'
        alert_msg   = 'Moderate burnout risk detected. Consider taking restorative breaks.'
    elif burnout_risk < 75:
        alert_level = 'High'
        alert_msg   = 'High burnout risk! Prioritise recovery — reduce workload if possible.'
    else:
        alert_level = 'Critical'
        alert_msg   = 'Critical burnout risk! Please speak to a manager or health professional immediately.'

    if not signals:
        signals.append("✅ No major burnout signals detected in your recent records.")

    return {
        'burnout_risk':    burnout_risk,
        'alert_level':     alert_level,
        'alert_msg':       alert_msg,
        'signals':         signals,
        'components':      risk_components,
        'records_analysed': len(records),
    }


def compute_team_heatmap():
    
 
    records = StressRecord.query.all()
    if not records:
        return None

    level_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Extremely High': 4}

    # Group by shift
    shift_data = {}
    for r in records:
        k = r.working_shift
        if k not in shift_data:
            shift_data[k] = {'count': 0, 'stress_sum': 0, 'csi_sum': 0, 'high_count': 0}
        shift_data[k]['count']      += 1
        shift_data[k]['stress_sum'] += level_map.get(r.stress_level, 2)
        shift_data[k]['csi_sum']    += r.csi_score or 0
        if r.stress_level in ['High', 'Extremely High']:
            shift_data[k]['high_count'] += 1

    for k in shift_data:
        n = shift_data[k]['count']
        shift_data[k]['avg_stress'] = round(shift_data[k]['stress_sum'] / n, 2)
        shift_data[k]['avg_csi']    = round(shift_data[k]['csi_sum'] / n, 1)
        shift_data[k]['high_pct']   = round(shift_data[k]['high_count'] / n * 100, 1)

    # Group by work type
    work_type_data = {}
    for r in records:
        k = r.type_of_work
        if k not in work_type_data:
            work_type_data[k] = {'count': 0, 'stress_sum': 0, 'high_count': 0}
        work_type_data[k]['count']      += 1
        work_type_data[k]['stress_sum'] += level_map.get(r.stress_level, 2)
        if r.stress_level in ['High', 'Extremely High']:
            work_type_data[k]['high_count'] += 1

    for k in work_type_data:
        n = work_type_data[k]['count']
        work_type_data[k]['avg_stress'] = round(work_type_data[k]['stress_sum'] / n, 2)
        work_type_data[k]['high_pct']   = round(work_type_data[k]['high_count'] / n * 100, 1)

    # Group by workload
    workload_data = {}
    for r in records:
        k = r.workload
        if k not in workload_data:
            workload_data[k] = {'count': 0, 'stress_sum': 0, 'high_count': 0}
        workload_data[k]['count']      += 1
        workload_data[k]['stress_sum'] += level_map.get(r.stress_level, 2)
        if r.stress_level in ['High', 'Extremely High']:
            workload_data[k]['high_count'] += 1

    for k in workload_data:
        n = workload_data[k]['count']
        workload_data[k]['avg_stress'] = round(workload_data[k]['stress_sum'] / n, 2)
        workload_data[k]['high_pct']   = round(workload_data[k]['high_count'] / n * 100, 1)

    # Overall stats
    all_scores = [level_map.get(r.stress_level, 2) for r in records]
    overall_avg = round(sum(all_scores) / len(all_scores), 2)
    high_risk_count = sum(1 for r in records if r.stress_level in ['High', 'Extremely High'])
    total_users = db.session.query(StressRecord.user_id).distinct().count()

    return {
        'shift_data':      shift_data,
        'work_type_data':  work_type_data,
        'workload_data':   workload_data,
        'overall_avg':     overall_avg,
        'total_records':   len(records),
        'total_users':     total_users,
        'high_risk_pct':   round(high_risk_count / len(records) * 100, 1),
    }


# ---- New Routes ----

@app.route('/forecast')
@login_required
def forecast():
    if current_user.is_admin:
        flash('Forecast is available for regular users only.', 'info')
        return redirect(url_for('admin_dashboard'))
    forecast_data = compute_stress_forecast(current_user.id)
    burnout_data  = compute_burnout_risk(current_user.id)
    return render_template('forecast.html',
        forecast=forecast_data, burnout=burnout_data)


def compute_trigger_detector(user_id):
    records = StressRecord.query.filter_by(user_id=user_id).all()
    if len(records) < 3:
        return None

    level_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Extremely High': 4}
    factors = {
        'Sleep Deprivation':        [],
        'Heavy Workload':           [],
        'Night Shift':              [],
        'Noisy Environment':        [],
        'Uncomfortable Temperature':[],
        'Demanding Work':           [],
        'Long Working Hours':       [],
        'Abnormal Body Temperature':[],
    }

    for r in records:
        sv = level_map.get(r.stress_level, 2)
        factors['Sleep Deprivation'].append((1 if r.sleep_hours < 6 else 0, sv))
        factors['Heavy Workload'].append((1 if r.workload == 'Heavy' else 0, sv))
        factors['Night Shift'].append((1 if r.working_shift == 'Night shift' else 0, sv))
        factors['Noisy Environment'].append((1 if r.noise_levels == 'Noisy' else 0, sv))
        factors['Uncomfortable Temperature'].append((1 if r.working_area_temperature == 'Uncomfortable' else 0, sv))
        factors['Demanding Work'].append((1 if r.type_of_work == 'Demanding' else 0, sv))
        factors['Long Working Hours'].append((1 if r.working_hours in ['Long Day', 'Extreme Overtime'] else 0, sv))
        factors['Abnormal Body Temperature'].append((1 if r.body_temperature == 'Not Normal' else 0, sv))

    results = {}
    for factor, pairs in factors.items():
        present = [sv for active, sv in pairs if active == 1]
        absent  = [sv for active, sv in pairs if active == 0]
        if len(present) >= 1 and len(absent) >= 1:
            avg_with    = sum(present) / len(present)
            avg_without = sum(absent) / len(absent)
            impact = round((avg_with - avg_without) / 3 * 100, 1)
            frequency   = round(len(present) / len(pairs) * 100, 1)
            results[factor] = {
                'impact':     max(impact, 0),
                'frequency':  frequency,
                'avg_with':   round(avg_with, 2),
                'avg_without':round(avg_without, 2),
                'count':      len(present),
            }
        else:
            results[factor] = {'impact': 0, 'frequency': 0, 'avg_with': 0, 'avg_without': 0, 'count': 0}

    sorted_factors = sorted(results.items(), key=lambda x: x[1]['impact'], reverse=True)
    top_trigger = sorted_factors[0][0] if sorted_factors else None

    return {
        'factors':     sorted_factors,
        'top_trigger': top_trigger,
        'top_impact':  sorted_factors[0][1]['impact'] if sorted_factors else 0,
        'total_records': len(records),
    }

def generate_recovery_plan(user_id):
    records = StressRecord.query.filter_by(
        user_id=user_id).order_by(StressRecord.date.desc()).limit(7).all()
    if not records:
        return None

    burnout = compute_burnout_risk(user_id)
    triggers = compute_trigger_detector(user_id)
    latest = records[0]

    avg_sleep   = sum(r.sleep_hours for r in records) / len(records)
    avg_csi     = sum(r.csi_score or 0 for r in records) / len(records)
    burnout_pct = burnout['burnout_risk'] if burnout else 0
    top_trigger = triggers['top_trigger'] if triggers else None

    # Sleep target
    if avg_sleep < 5.5:    sleep_target = 7.5
    elif avg_sleep < 6.5:  sleep_target = 7.0
    else:                  sleep_target = 6.5

    # Break frequency
    if burnout_pct >= 75:  breaks = 'Every 45 min'
    elif burnout_pct >= 50:breaks = 'Every 60 min'
    else:                  breaks = 'Every 90 min'

    # Workload limit
    if avg_csi >= 70:      workload_cap = 'Light only — delegate heavy tasks'
    elif avg_csi >= 50:    workload_cap = 'Moderate — avoid overtime'
    else:                  workload_cap = 'Normal — maintain boundaries'

    # Recovery activities based on top trigger
    trigger_activity = {
        'Sleep Deprivation':         'Set a fixed 10pm bedtime alarm. Dim lights 1hr before sleep.',
        'Heavy Workload':            'Use time-blocking. Schedule 2 focused work blocks max per day.',
        'Night Shift':               'Keep sleep window fixed. Use blackout curtains and eye mask.',
        'Noisy Environment':         'Use noise-cancelling headphones or white noise for focus blocks.',
        'Uncomfortable Temperature': 'Address workspace temp — personal fan/heater. Dress in layers.',
        'Demanding Work':            'Break demanding tasks into 25-min Pomodoro sessions.',
        'Long Working Hours':        'Hard stop at end of shift. No work after hours. Log off devices.',
        'Abnormal Body Temperature': 'Rest and hydrate. Check with a doctor if persisting.',
    }

    days = []
    day_templates = [
        {'name': 'Monday',    'focus': 'Sleep Reset',       'activity': 'Start your new sleep schedule tonight'},
        {'name': 'Tuesday',   'focus': 'Workspace Audit',   'activity': 'Fix your physical environment factors'},
        {'name': 'Wednesday', 'focus': 'Workload Review',   'activity': 'Delegate or defer non-urgent tasks'},
        {'name': 'Thursday',  'focus': 'Active Recovery',   'activity': '20-min walk + 10-min breathing exercise'},
        {'name': 'Friday',    'focus': 'Social Connection', 'activity': 'Connect with a colleague or friend'},
        {'name': 'Saturday',  'focus': 'Full Rest Day',     'activity': 'No work. Rest, hobby, or nature time'},
        {'name': 'Sunday',    'focus': 'Prepare & Reflect', 'activity': 'Plan next week. Set boundaries in advance'},
    ]

    for i, day in enumerate(day_templates):
        stress_target = 'Low' if i >= 4 else ('Medium' if i >= 2 else 'High → Medium')
        days.append({
            'day':            day['name'],
            'focus':          day['focus'],
            'activity':       day['activity'],
            'sleep_target':   f"{sleep_target}h minimum",
            'break_schedule': breaks,
            'workload_cap':   workload_cap,
            'stress_target':  stress_target,
        })

    return {
        'days':              days,
        'sleep_target':      sleep_target,
        'break_schedule':    breaks,
        'workload_cap':      workload_cap,
        'top_trigger':       top_trigger,
        'trigger_activity':  trigger_activity.get(top_trigger, 'Follow general stress management guidelines.'),
        'burnout_risk':      burnout_pct,
        'avg_csi':           round(avg_csi, 1),
        'avg_sleep':         round(avg_sleep, 1),
    }


@app.route('/trigger-detector')
@login_required
def trigger_detector():
    if current_user.is_admin:
        return redirect(url_for('admin_dashboard'))
    data = compute_trigger_detector(current_user.id)
    return render_template('trigger_detector.html', data=data)


@app.route('/recovery-plan')
@login_required
def recovery_plan():
    if current_user.is_admin:
        return redirect(url_for('admin_dashboard'))
    plan = generate_recovery_plan(current_user.id)
    return render_template('recovery_plan.html', plan=plan)


@app.route('/team-heatmap')
@login_required
@admin_required
def team_heatmap():
    heatmap = compute_team_heatmap()
    return render_template('team_heatmap.html', heatmap=heatmap)




with app.app_context():
    db.create_all()
    admin = User.query.filter_by(username='admin').first()
    if not admin:
        admin = User(
            username='admin', email='admin@stressdetection.com',
            password=generate_password_hash('admin123', method='pbkdf2:sha256'),
            is_admin=True,
        )
        db.session.add(admin)
        db.session.commit()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
   
   
