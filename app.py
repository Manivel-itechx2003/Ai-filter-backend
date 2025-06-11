from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os # Import os module to access environment variables
import sys # Import sys module for exiting gracefully

app = Flask(__name__)

# --- IMPORTANT CORS Configuration for Production ---
# This allows requests ONLY from your deployed Vercel frontend URL.
# Replace 'https://ai-filter-frontend-ls1i.vercel.app' with your actual Vercel URL
# if it ever changes (though it's usually stable for production deployments).
# supports_credentials should generally be False unless you are specifically handling
# cookies or HTTP authentication headers with credentials from the frontend.
CORS(app, origins=["https://ai-filter-frontend-ls1i.vercel.app"])

# --- SpaCy Model Loading - Critical Change ---
# 'en_core_web_lg' is too large for most free tiers and will cause memory issues.
# We're switching to 'en_core_web_md' (or 'en_core_web_sm' if 'md' is still too much).
# The model will be downloaded by spaCy CLI or installed via requirements.txt direct link.
# We'll include a robust check for model loading.
SPACY_MODEL_NAME = "en_core_web_md" # Or "en_core_web_sm" for even smaller size

try:
    nlp = spacy.load(SPACY_MODEL_NAME)
    print(f"SpaCy '{SPACY_MODEL_NAME}' model loaded successfully.")
except OSError:
    print(f"SpaCy '{SPACY_MODEL_NAME}' model not found locally. Attempting download...", file=sys.stderr)
    try:
        # Use spacy.cli.download to ensure the model is available
        spacy.cli.download(SPACY_MODEL_NAME)
        nlp = spacy.load(SPACY_MODEL_NAME)
        print(f"SpaCy '{SPACY_MODEL_NAME}' model downloaded and loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to download and load '{SPACY_MODEL_NAME}': {e}", file=sys.stderr)
        print("Please ensure the model is properly installed or referenced in requirements.txt.", file=sys.stderr)
        # Exit the application gracefully if the model is critical and can't be loaded
        sys.exit(1) # This will cause the Render service to fail if model isn't loaded

# Your existing functions remain the same
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
        try:
            resume_text = extract_text_from_pdf(f)
            score = analyze_resume(resume_text, job_desc)
            results.append({'filename': f.filename, 'match_score': score})
        except Exception as e:
            # Add error handling for individual file processing
            print(f"Error processing {f.filename}: {e}", file=sys.stderr)
            results.append({'filename': f.filename, 'error': f"Processing failed: {str(e)}"})

    return jsonify({'results': results})

# --- Production Server Setup - Critical Change ---
if __name__ == '__main__':
    # This block is for local development only. Render will use Gunicorn.
    # It dynamically gets the port from Render's environment variable.
    # debug=False is crucial for production.
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
