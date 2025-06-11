from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Load spaCy model
nlp = spacy.load("en_core_web_lg")

def extract_text_from_pdf(file):
    pdf = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf.pages:
        content = page.extract_text()
        if content:
            text += content
    return text.strip()

def analyze_resume(resume_text, job_desc):
    resume_doc = nlp(resume_text)
    job_doc = nlp(job_desc)

    resume_clean = " ".join([t.lemma_ for t in resume_doc if not t.is_stop])
    job_clean = " ".join([t.lemma_ for t in job_doc if not t.is_stop])

    tfidf = TfidfVectorizer().fit_transform([resume_clean, job_clean])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(score * 100, 2)

@app.route('/analyze', methods=['POST'])
def analyze():
    files = request.files.getlist('resumes')
    job_desc = request.form.get('job_description', '').strip()
    if not files or not job_desc:
        return jsonify({'error': 'Resume(s) and job description are required.'}), 400

    results = []
    for f in files:
        resume_text = extract_text_from_pdf(f)
        score = analyze_resume(resume_text, job_desc)
        results.append({'filename': f.filename, 'match_score': score})

    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True)
