from flask import Flask, render_template, request, redirect, url_for, session, flash
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import joblib
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# -------------------- MySQL Connection --------------------
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='Devrishi123',
    database='loan_db'
)
cursor = conn.cursor(dictionary=True)

# -------------------- MODEL PATHS --------------------
MODEL_PATH = 'models/loan_model.pkl'
ENCODER_PATH = 'models/label_encoders.pkl'

# -------------------- ROUTES --------------------

# Homepage
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('home'))
    return render_template('index.html')

# Signup
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        role = request.form.get('role', 'user')

        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        existing_user = cursor.fetchone()

        if existing_user:
            flash("Email already exists!", "danger")
        else:
            cursor.execute(
                "INSERT INTO users (name, email, password, role) VALUES (%s, %s, %s, %s)",
                (name, email, password, role)
            )
            conn.commit()
            flash("Signup successful! Please login.", "success")
            return redirect(url_for('login'))

    return render_template('signup.html')

# Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['name'] = user['name']
            session['role'] = user['role']
            return redirect(url_for('home'))
        else:
            flash("Invalid email or password!", "danger")
            return redirect(url_for('login'))

    return render_template('login.html')

# Logout
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# Home / Dashboard
@app.route('/home')
def home():
    if 'user_id' in session:
        cursor.execute("""
            SELECT COUNT(*) AS total, SUM(loan_status) AS approved
            FROM loans WHERE user_id=%s
        """, (session['user_id'],))
        row = cursor.fetchone()
        total_loans = row['total'] if row['total'] else 0
        approved_loans = row['approved'] if row['approved'] else 0
        rejected_loans = total_loans - approved_loans

        stats = {
            'total': total_loans,
            'approved': approved_loans,
            'rejected': rejected_loans
        }

        return render_template('home.html', name=session['name'], role=session['role'], stats=stats)
    return redirect(url_for('login'))

# About page
@app.route('/about')
def about():
    return render_template('about.html')

# Contact page
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Profile page
@app.route('/profile')
def profile():
    if 'user_id' in session:
        return render_template('profile.html', name=session['name'])
    return redirect(url_for('login'))

# Loan History page
@app.route('/loan_history')
def loan_history():
    if 'user_id' in session:
        cursor.execute("SELECT * FROM loans WHERE user_id=%s", (session['user_id'],))
        loans = cursor.fetchall()

        # Pass stats too
        cursor.execute("""
            SELECT COUNT(*) AS total, SUM(loan_status) AS approved
            FROM loans WHERE user_id=%s
        """, (session['user_id'],))
        row = cursor.fetchone()
        total_loans = row['total'] if row['total'] else 0
        approved_loans = row['approved'] if row['approved'] else 0
        rejected_loans = total_loans - approved_loans
        stats = {
            'total': total_loans,
            'approved': approved_loans,
            'rejected': rejected_loans
        }

        return render_template('loan_history.html', loans=loans, stats=stats)
    return redirect(url_for('login'))

# -------------------- Loan Application --------------------
@app.route('/apply_loan', methods=['GET', 'POST'])
def apply_loan():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # Load trained pipeline
    pipeline = joblib.load('models/loan_pipeline.pkl')
    prediction_result = None

    if request.method == 'POST':
        user_data = request.form.to_dict()
        df = pd.DataFrame([user_data])

        # Convert numeric fields safely
        numeric_cols = ['person_age','person_income','person_emp_exp','loan_amnt',
                        'loan_int_rate','loan_percent_income','cb_person_cred_hist_length','credit_score']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df[numeric_cols] = df[numeric_cols].fillna(0)

        # Predict using the pipeline
        prediction = pipeline.predict(df)[0]
        prediction_result = "✅ Approved" if prediction == 1 else "❌ Rejected"

        # Convert numpy types to native Python types for MySQL
        import numpy as np
        row_values = tuple(
            int(x) if isinstance(x, (np.integer, np.int64)) else
            float(x) if isinstance(x, (np.floating, np.float64)) else
            x
            for x in df.iloc[0]
        )
        # Add prediction and user_id
        row_values += (int(prediction), session['user_id'])

        # Insert into MySQL
        cols = ', '.join(df.columns) + ', loan_status, user_id'
        placeholders = ', '.join(['%s'] * (len(df.columns)+2))
        sql = f"INSERT INTO loans ({cols}) VALUES ({placeholders})"
        cursor.execute(sql, row_values)
        conn.commit()

    return render_template('apply_loan.html', prediction=prediction_result)


@app.route('/personal_ai')
def personal_ai():
    if 'user_id' in session:
        return render_template('personal_ai.html')
    return redirect(url_for('login'))


# -------------------- MAIN --------------------
if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    app.run(debug=True)
