from flask import Flask, request, render_template
from summarizer import textrank_with_params, evaluate_summary
import fitz  # PyMuPDF
import os
from werkzeug.utils import secure_filename

application = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(filepath):
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

@application.route('/', methods=['GET', 'POST'])
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
