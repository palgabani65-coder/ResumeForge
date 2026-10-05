import os
import re
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="ResumeForge AI - Resume Classifier", version="1.0.0")

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
MODELS_DIR = BASE_DIR / "artifacts" / "models"
REPORTS_DIR = BASE_DIR / "artifacts" / "reports"

# Technical token replacements
TECH_TOKEN_REPLACEMENTS = {
    r"\bc\+\+(?!\w)": "cplusplus",
    r"\bc#(?!\\w)": "csharp",
    r"(?<!\w)\.net(?!\w)": "dotnet",
    r"\bnode\.js\b": "nodejs",
    r"\bvue\.js\b": "vuejs",
    r"\breact\.js\b": "reactjs",
    r"\bangular\.js\b": "angularjs",
    r"\bspring\s+boot\b": "springboot",
    r"\bci\/cd\b": "cicd",
    r"\bpower\s+bi\b": "powerbi",
    r"\bms\s+excel\b": "msexcel",
    r"\bms\s+office\b": "msoffice",
    r"\bms\s+word\b": "msword",
    r"\bmachine\s+learning\b": "machinelearning",
    r"\bdeep\s+learning\b": "deeplearning",
    r"\bdata\s+science\b": "datascience",
    r"\bcloud\s+computing\b": "cloudcomputing"
}

DOMAIN_SAFE_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "has", "have", "he", "her",
    "here", "hers", "him", "his", "how", "i", "if", "in", "into", "is", "it",
    "its", "me", "more", "most", "my", "no", "nor", "not", "of", "off", "on",
    "or", "other", "our", "out", "over", "own", "same", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "with", "would", "you", "your", "yours"
}

def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    t = text.lower()
    for pat, rep in TECH_TOKEN_REPLACEMENTS.items():
        t = re.sub(pat, f" {rep} ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"&[a-z]+;", " ", t)
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " ", t)
    t = re.sub(r"\+?\d[\d\s\-\(\)]{7,}\d", " ", t)
    t = re.sub(r"[^a-zA-Z\s]", " ", t)
    words = [w for w in t.split() if w not in DOMAIN_SAFE_STOPWORDS and len(w) >= 2]
    return " ".join(words)

# Model and vectorizer singletons
_model = None
_vectorizer = None
_label_encoder = None

def load_artifacts():
    global _model, _vectorizer, _label_encoder
    if _model is None:
        model_p = MODELS_DIR / "best_model.joblib"
        vec_p = MODELS_DIR / "tfidf_vectorizer.joblib"
        enc_p = MODELS_DIR / "label_encoder.joblib"
        if model_p.exists() and vec_p.exists() and enc_p.exists():
            _model = joblib.load(model_p)
            _vectorizer = joblib.load(vec_p)
            _label_encoder = joblib.load(enc_p)
    return _model, _vectorizer, _label_encoder

class ResumePayload(BaseModel):
    text: str
    top_k: int = 5

SAMPLE_RESUMES = {
    "software_engineer": {
        "title": "Senior Full-Stack Engineer (Information Technology)",
        "category": "INFORMATION-TECHNOLOGY",
        "text": """SENIOR FULL-STACK SOFTWARE ENGINEER
Summary:
Experienced Software Engineer with 6+ years in designing, developing, and deploying scalable web applications and microservices. Proficient in Python, JavaScript, TypeScript, React.js, Node.js, and FastAPI. Strong background in cloud computing with AWS (EC2, S3, Lambda, ECS), Docker containerization, Kubernetes, and CI/CD pipelines.

Technical Skills:
- Languages: Python, JavaScript, TypeScript, C++, SQL, Bash
- Frameworks: Django, FastAPI, Flask, React, Redux, Node.js, Express, Next.js
- Databases: PostgreSQL, MongoDB, Redis, MySQL
- Cloud & DevOps: AWS, Docker, Kubernetes, GitHub Actions, Terraform, Linux
- Data & ML: Pandas, NumPy, Scikit-Learn, PyTorch, RESTful APIs

Professional Experience:
Senior Software Developer | TechCorp Solutions (2021 - Present)
- Architected high-throughput microservices handling 2M+ requests daily with 99.98% uptime.
- Led frontend redesign utilizing React, TypeScript, and modern CSS reducing latency by 40%.
- Automated CI/CD deployments using Docker and GitHub Actions, cutting release cycles from days to hours."""
    },
    "accountant": {
        "title": "Certified Public Accountant (Accountant)",
        "category": "ACCOUNTANT",
        "text": """SENIOR ACCOUNTANT & AUDITOR (CPA)
Professional Profile:
Detail-oriented and results-driven Senior Accountant with 7 years of comprehensive experience in corporate accounting, financial statement preparation, general ledger reconciliation, internal audits, and tax compliance. Advanced mastery of GAAP, IFRS, QuickBooks, SAP ERP, and MS Excel financial modeling.

Core Competencies:
- General Ledger & Journal Entries, Balance Sheet Reconciliations, Accounts Payable/Receivable
- GAAP & IFRS Compliance, SEC Reporting, Tax Filings, Internal Controls & SOX Audit
- Variance Analysis, Cash Flow Forecasting, Cost Accounting, Budgeting & P&L Statements
- Software: SAP ERP, QuickBooks Pro, Oracle NetSuite, Xero, Advanced Excel (Macros, VLOOKUP)

Work History:
Senior Accountant | Summit Financial Advisors (2020 - Present)
- Supervised monthly, quarterly, and annual financial close for 15+ corporate subsidiaries.
- Prepared comprehensive balance sheets, income statements, and cash flow reports in full compliance with GAAP."""
    },
    "hr_manager": {
        "title": "Human Resources & Talent Acquisition Specialist (HR)",
        "category": "HR",
        "text": """HUMAN RESOURCES MANAGER / TALENT ACQUISITION SPECIALIST
Executive Summary:
Dynamic Human Resources professional with 8+ years leading talent acquisition, employee relations, performance management, onboarding, compensation and benefits, and HR compliance. Proven track record in scaling technical organizations, reducing employee turnover, and enhancing workplace culture.

Core Competencies:
- Full-Lifecycle Recruitment & Headhunting, Candidate Sourcing, Interviewing, Offer Negotiation
- HRIS & ATS Administration: Workday, BambooHR, Greenhouse, Lever, ADP Workforce Now
- Employee Engagement, Onboarding, Retention Strategies, Dispute Resolution, Labor Law Compliance
- Performance Appraisals, Diversity & Inclusion (DEI) Programs, Payroll Administration"""
    },
    "chef": {
        "title": "Executive Sous Chef (Chef)",
        "category": "CHEF",
        "text": """EXECUTIVE CHEF & CULINARY DIRECTOR
Profile:
Passionate, classically trained Executive Chef with 10+ years of fine dining and high-volume banquet culinary leadership. Expert in modern French, Mediterranean, and contemporary cuisines. Proven expertise in menu creation, farm-to-table sourcing, kitchen staff training, food cost management, and sanitation compliance (HACCP).

Culinary Skills & Certifications:
- Menu Development, Wine Pairing, Plating & Presentation, Culinary Innovation
- Kitchen Brigade Management, Food Safety & Hygiene (ServSafe Certified), Inventory Control
- Vendor Sourcing, Cost Reduction & Margin Optimization, Catering & Private Dining"""
    }
}

# Static file routes to guarantee CSS and JS always load regardless of path prefix
@app.get("/style.css")
def get_root_css():
    return FileResponse(STATIC_DIR / "style.css", media_type="text/css")

@app.get("/app.js")
def get_root_js():
    return FileResponse(STATIC_DIR / "app.js", media_type="application/javascript")

@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/api/categories")
def get_categories():
    _, _, enc = load_artifacts()
    classes = list(enc.classes_) if enc is not None else []
    return {"categories": classes, "count": len(classes)}

@app.get("/api/sample_resumes")
def get_samples():
    return SAMPLE_RESUMES

@app.get("/api/metrics")
def get_metrics():
    full_report = REPORTS_DIR / "full_model_architectures_comparison.json"
    if full_report.exists():
        with open(full_report, "r") as f:
            return {"architectures": json.load(f)}
    report_file = REPORTS_DIR / "model_comparison.json"
    if report_file.exists():
        with open(report_file, "r") as f:
            return json.load(f)
    return {"message": "Metrics report available in artifacts/reports"}

@app.post("/api/predict")
def predict(payload: ResumePayload):
    model, vectorizer, enc = load_artifacts()
    if model is None or vectorizer is None or enc is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Please ensure artifacts/models exist.")
    
    cleaned = clean_text(payload.text)
    if not cleaned:
        return {
            "predicted_category": "UNKNOWN",
            "confidence": 0.0,
            "top_predictions": [],
            "top_matching_keywords": [],
            "cleaned_text_preview": ""
        }
        
    feat = vectorizer.transform([cleaned])
    classes = list(enc.classes_)
    
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feat)[0]
    elif hasattr(model, "decision_function"):
        d = model.decision_function(feat)[0]
        exp_d = np.exp(d - np.max(d))
        probs = exp_d / exp_d.sum()
    else:
        pred_idx = model.predict(feat)[0]
        probs = np.zeros(len(classes))
        probs[pred_idx] = 1.0
        
    top_indices = np.argsort(probs)[::-1][:payload.top_k]
    top_preds = [
        {
            "category": classes[idx],
            "probability": round(float(probs[idx]), 4),
            "percentage": round(float(probs[idx]) * 100, 2)
        }
        for idx in top_indices
    ]
    
    # Extract top keywords
    csr = feat.tocsr()
    feature_names = np.array(vectorizer.get_feature_names_out())
    top_kw = []
    if len(csr.data) > 0:
        sorted_kw_idx = np.argsort(csr.data)[::-1][:10]
        top_kw = [feature_names[csr.indices[j]] for j in sorted_kw_idx]
        
    return {
        "predicted_category": top_preds[0]["category"],
        "confidence": top_preds[0]["probability"],
        "top_predictions": top_preds,
        "top_matching_keywords": top_kw,
        "cleaned_text_preview": cleaned[:300] + ("..." if len(cleaned) > 300 else "")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web_app.app:app", host="127.0.0.1", port=8000, reload=True)
