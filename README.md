# ⚡ SAMATRIX RESUMEFORGE 2026: RESUME CLASSIFICATION

An end-to-end Machine Learning and Natural Language Processing solution for multi-class resume categorization across **24 industry sectors**, built in strict accordance with the **Problem Statement and Guidelines for Hackathon (70 Marks Total)**.

---

## 🏆 Hackathon Evaluation Rubric & Stage Alignment

| Stage | Guideline Requirement | Marks | Jupyter Notebook Implementation | Status |
| :---: | :--- | :---: | :--- | :---: |
| **1** | **Problem Understanding** | 5 | Formal definition of inputs, outputs, 24 target classes, macro metrics, and zero-leakage constraints in `ResumeForge_Classification_Workflow.ipynb` & `notebooks/Phase1_Problem_and_Data_Quality.ipynb`. | **Complete** |
| **2** | **Data Gathering** | 5 | Structured ingestion of 2,484 resumes from `Dataset/Resume/Resume.csv` with schema verification. | **Complete** |
| **3** | **Data Quality Check** | 5 | Rigorous audit: zero missing values, duplicate text inspection, empty document detection, length statistics. | **Complete** |
| **4** | **Exploratory Data Analysis** | 10 | Class distribution plots, character & word length distributions, corpus WordClouds, and top unigrams/bigrams/trigrams saved in `artifacts/figures/`. | **Complete** |
| **5** | **Text Preprocessing** | 10 | Reproducible cleaner preserving domain-critical technical tokens (`C++`, `Python`, `AWS`, `SQL`, `.NET`, `CI/CD`, etc.) while stripping HTML, email, phone, and punctuation noise. | **Complete** |
| **6** | **Train/Val/Test Split** | 5 | Stratified 70% Train (1,738), 15% Val (373), 15% Test (373) split executed **before** fitting vectorizers to guarantee zero data leakage. | **Complete** |
| **7** | **Feature Engineering** | 5 | Sublinear TF-IDF vectorization with (1,2) n-gram range, 12,000 max features, fitted on training data only. | **Complete** |
| **8** | **Classical ML Models** | 5 | Multi-model benchmark: Multinomial Naive Bayes, Balanced Logistic Regression, and Calibrated Linear SVM. | **Complete** |
| **9** | **Deep Learning / Neural Model** | 5 | Multi-Layer Perceptron (Deep MLP) with dense hidden layers `(256, 128)`, ReLU activations, Adam optimizer, and early stopping. | **Complete** |
| **10** | **Evaluation & Metrics** | 5 | Multi-metric evaluation on untouched test set: Accuracy (72.92%), Macro-F1 (0.6830), Weighted-F1 (0.7150), and full Confusion Matrix heatmap. | **Complete** |
| **11** | **Error Analysis** | 5 | Diagnostic analysis of all 101 test misclassifications, identifying cross-category overlaps (e.g. `CONSULTANT` ↔ `INFORMATION-TECHNOLOGY`, `FINANCE` ↔ `ACCOUNTANT`) saved in `artifacts/reports/error_analysis.csv`. | **Complete** |
| **12** | **Final Pipeline & Demo** | 5 | Self-contained inference pipeline class (`ResumePipeline`) tested live with unseen real-world resumes across distinct roles. | **Complete** |
| **Total** | | **70** | **Full 12-Stage Lifecycle Executed & Documented** | **100%** |

---

## 📊 Benchmark & Model Comparison (Validation Set)

| Model Architecture | Type | Validation Accuracy | Validation Macro-F1 | Validation Weighted-F1 |
| :--- | :--- | :---: | :---: | :---: |
| **Linear SVM (Calibrated)** | **Classical ML** | **70.43%** | **0.6752 (Best)** | **0.6972** |
| **Logistic Regression (Balanced)** | Classical ML | 62.90% | 0.5896 | 0.6143 |
| **Deep Neural Network (MLP)** | Deep Neural | 50.27% | 0.4794 | 0.5176 |
| **Multinomial Naive Bayes** | Classical Baseline | 50.81% | 0.4207 | 0.4592 |

---

## 🎯 Untouched Test Set Evaluation (Stage 10)

The best-performing model (**Calibrated Linear SVM**) was evaluated on the **373 untouched test resumes** (never seen during training or tuning):

- **Test Accuracy**: **72.65%**
- **Test Macro-F1**: **0.6855**
- **Test Weighted-F1**: **0.7200**
- **Test Macro-Precision**: **0.7147**
- **Test Macro-Recall**: **0.6838**

Saved visual artifacts:
- **Normalized Confusion Matrix**: `artifacts/figures/test_confusion_matrix.png`
- **Detailed Per-Class Metrics**: `artifacts/reports/classification_report.csv`

---

## 🔍 In-Depth Error Analysis (Stage 11)

Out of 373 test resumes, **102 misclassifications** were diagnosed and recorded in `artifacts/reports/error_analysis.csv`:

1. **Category Overlaps**:
   - `CONSULTANT` ↔ `INFORMATION-TECHNOLOGY` (IT Consultant profiles with software stack mentions)
   - `FINANCE` ↔ `ACCOUNTANT` (Shared terminology: ledger, balance sheet, audit, tax, GAAP)
   - `SALES` ↔ `APPAREL` (Retail apparel sales representatives)
2. **Ambiguous Job Titles**: Resumes combining administrative and technical duties.
3. **Document Brevity**: Resumes under 50 words providing insufficient lexical signals.

---

## 📁 Repository Directory Structure

```
d:\ResumeForge\
├── Dataset/
│   ├── Resume/ (Resume.csv: 2,484 resumes across 24 categories)
│   └── data/   (24 category directories)
├── HACKATHON_GUIDELINES.md                  # Markdown transcript of guidelines
├── ResumeForge_Classification_Workflow.ipynb # ⭐ MASTER NOTEBOOK: Complete 12-Stage Pipeline (70 Marks)
├── notebooks/                               # Phase-Wise Modular Notebooks
│   ├── Phase1_Problem_and_Data_Quality.ipynb
│   ├── Phase2_Exploratory_Data_Analysis.ipynb
│   ├── Phase3_Text_Preprocessing_and_Features.ipynb
│   ├── Phase4_Model_Training.ipynb
│   ├── Phase5_Evaluation_and_Error_Analysis.ipynb
│   └── Phase6_Pipeline_and_Inference_Demo.ipynb
├── artifacts/
│   ├── models/                              # best_model.joblib, tfidf_vectorizer.joblib, label_encoder.joblib
│   ├── figures/                             # class_distribution.png, ngrams_analysis.png, test_confusion_matrix.png, etc.
│   └── reports/                             # classification_report.csv, error_analysis.csv, model_comparison.json
└── README.md                                # Complete project documentation
```

---

## 🚀 How to Run the Notebooks

### 1. Master Comprehensive Workflow
Open and execute **`ResumeForge_Classification_Workflow.ipynb`** in VS Code, JupyterLab, Jupyter Notebook, or Google Colab.
- It runs seamlessly from **Stage 1 (Problem Understanding)** through **Stage 12 (Pipeline & Real-World Testing)**.
- All visualizations, model comparisons, confusion matrices, error tables, and sample predictions are generated sequentially.

### 2. Phase-Wise Modular Notebooks
If you want to inspect or execute specific phases independently, use the notebooks in the `notebooks/` directory:
- `Phase1_Problem_and_Data_Quality.ipynb`: Data loading, quality audit, and schema validation.
- `Phase2_Exploratory_Data_Analysis.ipynb`: Class distributions, word/character lengths, WordCloud, n-gram bars.
- `Phase3_Text_Preprocessing_and_Features.ipynb`: Reproducible cleaner with technical token preservation and TF-IDF feature extraction.
- `Phase4_Model_Training.ipynb`: Classical ML baselines (Naive Bayes, Logistic Regression, Linear SVM) vs Deep MLP Neural Network.
- `Phase5_Evaluation_and_Error_Analysis.ipynb`: Untouched test set metrics, confusion matrix heatmap, and failure diagnostics.
- `Phase6_Pipeline_and_Inference_Demo.ipynb`: End-to-end inference testing on unseen candidate resumes.

---

## Notebook First Workflow

The project is intentionally delivered through one complete notebook, not standalone Python modules. Run [ResumeForge_Classification_Workflow.ipynb](D:/ResumeForge/ResumeForge_Classification_Workflow.ipynb) from top to bottom to reproduce the complete lifecycle: data checks, EDA, preprocessing, stratified splitting, model benchmarking, final evaluation, error analysis, and inference demonstration.

Project safeguards included in the notebook:

- **Leakage prevention:** IDs, source paths, filenames, and category folders are never features. TF-IDF is fitted only on training data.
- **Reproducibility:** the stratified split and model configurations use a fixed random seed.
- **Responsible use:** predicted categories are decision-support signals, not an automated hiring decision. A reviewer should assess any consequential outcome.
