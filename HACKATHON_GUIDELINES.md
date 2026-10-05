# SAMATRIX RESUMEFORGE 2026
## RESUME CLASSIFICATION — PARTICIPANT APPROACH GUIDE
**A practical end-to-end NLP workflow:**
`Problem → Data → EDA → Preprocessing → Features → ML/DL → Evaluation → Demo`

> **IMPORTANT**: Do not jump directly to model training. Your solution should show that you understood the resume text, identified the data problems, justified preprocessing and feature choices, built meaningful baselines, and compared models.

---

### 1. Problem Statement
Build an end-to-end Resume Classification system using the provided resume dataset. Given a resume (raw text or extracted text), predict its appropriate resume category/class (24 categories). The final solution should include data understanding, exploratory data analysis (EDA), text preprocessing, feature engineering, model training, evaluation, error analysis, and a simple prediction interface or demo.

---

### 2. Evaluation Rubric (70 Marks Total)

| Step | Stage | What You Should Do | Marks |
| :---: | :--- | :--- | :---: |
| 1 | Problem Understanding | Define input, output, classes, constraints and success metric | 5 |
| 2 | Data Gathering | Load dataset; inspect files, columns, labels and raw resume text | 5 |
| 3 | Data Quality Check | Missing values, duplicates, corrupted/empty resumes, label inconsistencies, leakage | 5 |
| 4 | EDA | Class distribution, text length, vocabulary, WordCloud, unigram/bigram/trigram analysis | 10 |
| 5 | Text Preprocessing | Normalize text carefully; remove noise while preserving meaningful resume info | 10 |
| 6 | Train/Val/Test Split | Split before fitting vocabulary/vectorizers to prevent data leakage | 5 |
| 7 | Feature Engineering | TF-IDF baseline; optionally embeddings; derive useful text statistics | 5 |
| 8 | ML Models | Logistic Regression / Linear SVM / Naive Bayes or another justified baseline | 5 |
| 9 | DL Model | Embedding + Neural/Dense/LSTM/GRU model or justified neural approach | 5 |
| 10 | Evaluation | Accuracy + Precision/Recall/Macro & Weighted F1, confusion matrix, per-class results | 5 |
| 11 | Error Analysis | Inspect wrong predictions and identify data/model weaknesses | 5 |
| 12 | Final Pipeline & Demo | Raw resume → preprocessing → feature extraction → model → predicted category | 5 |
| **Total** | | | **70** |

---

### 3. Key Dataset Details
- **Total Records**: 2,484 resumes
- **Target Classes**: 24 professional categories (e.g., Information-Technology, Business-Development, Finance, Engineering, Healthcare, HR, etc.)
- **Columns**: `ID`, `Resume_str`, `Resume_html`, `Category`
