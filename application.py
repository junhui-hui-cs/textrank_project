from flask import Flask, request, render_template, redirect, url_for, flash
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin, current_user
from summarizer import textrank_with_params, evaluate_summary
import fitz  # PyMuPDF
import os
from werkzeug.utils import secure_filename

application = Flask(__name__)
application.secret_key = 'supersecretkey'

# === LOGIN SETUP ===
login_manager = LoginManager()
login_manager.init_app(application)
login_manager.login_view = 'login'

# Dummy users
users = {
    "admin": {"password": "password123"}
}

class User(UserMixin):
    def __init__(self, username):
        self.id = username

@login_manager.user_loader
def load_user(user_id):
    if user_id in users:
        return User(user_id)
    return None

# === PDF Handling ===
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(filepath):
    doc = fitz.open(filepath)
    text = "".join([page.get_text() for page in doc])
    doc.close()
    return text

# === ROUTES ===
@application.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username]['password'] == password:
            login_user(User(username))
            return redirect(url_for('index'))
        flash("Invalid credentials")
    return render_template('login.html')

@application.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@application.route('/', methods=['GET', 'POST'])
@login_required
def index():
    summary = None
    metrics = None
    error = None

    if request.method == 'POST':
        try:
            file = request.files.get('file')
            text_input = request.form.get('text', '')

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(application.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                text = extract_text_from_pdf(filepath)
            elif text_input.strip():
                text = text_input
            else:
                raise Exception("No valid text or PDF uploaded.")

            top_n = int(request.form.get('top_n', 3))
            damping = float(request.form.get('damping', 0.85))
            threshold = float(request.form.get('threshold', 0.1))

            summary_sentences = textrank_with_params(
                text,
                top_n=top_n,
                damping_factor=damping,
                similarity_threshold=threshold
            )
            summary = ' '.join(summary_sentences)
            metrics = evaluate_summary(text, summary)

        except Exception as e:
            error = f" Error: {str(e)}"

    return render_template('index.html', summary=summary, metrics=metrics, error=error)

if __name__ == '__main__':
    application.run(debug=True)
