import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score

# 1. Load Data
df = pd.read_csv("Dataset/Resume/Resume.csv")
df['Category'] = df['Category'].astype(str).str.strip().str.upper()

TOKEN_MAP = {
    r"\bc\+\+\b": "cplusplus",
    r"\bc#\b": "csharp",
    r"\b\.net\b": "dotnet",
    r"\bnode\.js\b": "nodejs",
    r"\bvue\.js\b": "vuejs",
    r"\breact\.js\b": "reactjs",
    r"\bci\/cd\b": "cicd",
    r"\bpower\s+bi\b": "powerbi",
    r"\bms\s+excel\b": "msexcel",
    r"\bms\s+office\b": "msoffice",
    r"\bspring\s+boot\b": "springboot",
    r"\bmachine\s+learning\b": "machinelearning",
    r"\bdeep\s+learning\b": "deeplearning",
    r"\bdata\s+science\b": "datascience",
    r"\bcloud\s+computing\b": "cloudcomputing"
}

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "me", "more",
    "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
    "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "we",
    "were", "what", "when", "where", "which", "while", "who", "whom", "why", "with",
    "would", "you", "your", "yours", "yourself", "yourselves"
}

def clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    t = text.lower()
    for pat, rep in TOKEN_MAP.items(): t = re.sub(pat, rep, t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " ", t)
    t = re.sub(r"\+?\d[\d\s\-\(\)]{7,}\d", " ", t)
    t = re.sub(r"[^a-zA-Z0-9\s]", " ", t)
    words = [w for w in t.split() if w not in STOPWORDS and len(w) >= 2 and not w.isdigit()]
    return " ".join(words)

df['Clean_text'] = df['Resume_str'].apply(clean_text)
le = LabelEncoder()
df['Category_Encoded'] = le.fit_transform(df['Category'])

# 70/15/15 Stratified Split
df_train, df_temp = train_test_split(df, test_size=0.30, stratify=df['Category_Encoded'], random_state=42)
df_val, df_test = train_test_split(df_temp, test_size=0.50, stratify=df_temp['Category_Encoded'], random_state=42)

y_train = df_train['Category_Encoded'].values
y_val = df_val['Category_Encoded'].values
y_test = df_test['Category_Encoded'].values

# TF-IDF Features
vec = TfidfVectorizer(ngram_range=(1, 2), max_features=12000, sublinear_tf=True, min_df=2, max_df=0.95)
X_train_tfidf = vec.fit_transform(df_train['Clean_text'])
X_val_tfidf = vec.transform(df_val['Clean_text'])
X_test_tfidf = vec.transform(df_test['Clean_text'])

# Count Features for Naive Bayes
cnt_vec = CountVectorizer(ngram_range=(1, 2), max_features=12000, min_df=2, max_df=0.95)
X_train_cnt = cnt_vec.fit_transform(df_train['Clean_text'])
X_val_cnt = cnt_vec.transform(df_val['Clean_text'])
X_test_cnt = cnt_vec.transform(df_test['Clean_text'])

results = []

def record(name, arch_type, tr_y, tr_p, va_y, va_p, te_y, te_p):
    tr_acc = accuracy_score(tr_y, tr_p)
    va_acc = accuracy_score(va_y, va_p)
    te_acc = accuracy_score(te_y, te_p)
    tr_f1 = f1_score(tr_y, tr_p, average='macro', zero_division=0)
    va_f1 = f1_score(va_y, va_p, average='macro', zero_division=0)
    te_f1 = f1_score(te_y, te_p, average='macro', zero_division=0)
    gap = (tr_acc - va_acc) * 100
    results.append({
        "Model Architecture": name,
        "Input / Representation": arch_type,
        "Train Acc": f"{tr_acc*100:.2f}%",
        "Val Acc": f"{va_acc*100:.2f}%",
        "Test Acc": f"{te_acc*100:.2f}%",
        "Val Macro-F1": f"{va_f1:.4f}",
        "Test Macro-F1": f"{te_f1:.4f}",
        "Overfitting Gap (Train-Val)": f"{gap:+.2f}%"
    })
    print(f"Done: {name:<32} | Train: {tr_acc*100:.1f}% | Val: {va_acc*100:.1f}% | Test: {te_acc*100:.1f}% | Gap: {gap:+.1f}%")

# 1. Multinomial Naive Bayes (Count Features)
nb_cnt = MultinomialNB(alpha=0.5)
nb_cnt.fit(X_train_cnt, y_train)
record("Multinomial Naive Bayes", "Count / BoW Features", y_train, nb_cnt.predict(X_train_cnt), y_val, nb_cnt.predict(X_val_cnt), y_test, nb_cnt.predict(X_test_cnt))

# 2. Multinomial Naive Bayes (TF-IDF)
nb_tfidf = MultinomialNB(alpha=0.1)
nb_tfidf.fit(X_train_tfidf, y_train)
record("Multinomial Naive Bayes", "TF-IDF Features", y_train, nb_tfidf.predict(X_train_tfidf), y_val, nb_tfidf.predict(X_val_tfidf), y_test, nb_tfidf.predict(X_test_tfidf))

# 3. Logistic Regression (Strongly Regularized C=0.5, balanced)
lr = LogisticRegression(C=0.8, max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train_tfidf, y_train)
record("Logistic Regression", "TF-IDF (Regularized)", y_train, lr.predict(X_train_tfidf), y_val, lr.predict(X_val_tfidf), y_test, lr.predict(X_test_tfidf))

# 4. Linear SVM (Calibrated, Regularized C=0.5)
svm = CalibratedClassifierCV(LinearSVC(C=0.5, class_weight='balanced', random_state=42, max_iter=2000), cv=3)
svm.fit(X_train_tfidf, y_train)
record("Linear SVM (Calibrated)", "TF-IDF (Balanced Hyperplane)", y_train, svm.predict(X_train_tfidf), y_val, svm.predict(X_val_tfidf), y_test, svm.predict(X_test_tfidf))

# 5. Ridge Classifier (L2 Regularized Linear)
ridge = RidgeClassifier(alpha=0.4, random_state=42)
ridge.fit(X_train_tfidf, y_train)
record("Ridge Classifier", "TF-IDF (L2 Regularized)", y_train, ridge.predict(X_train_tfidf), y_val, ridge.predict(X_val_tfidf), y_test, ridge.predict(X_test_tfidf))

# ==============================================================================
# 6. Dense NN with Pooled Word Embeddings (EmbeddingBag + MLP + Dropout)
# ==============================================================================
# Build vocab
vocab = {"<PAD>": 0, "<UNK>": 1}
for text in df_train['Clean_text']:
    for word in text.split():
        if word not in vocab and len(vocab) < 15000:
            vocab[word] = len(vocab)

def text_to_bag(text):
    indices = [vocab.get(w, 1) for w in text.split()]
    return indices if indices else [0]

class BagDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = [text_to_bag(t) for t in texts]
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def collate_bag(batch):
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    offsets = [0]
    flat_indices = []
    for item in batch:
        flat_indices.extend(item[0])
        offsets.append(len(flat_indices))
    offsets = torch.tensor(offsets[:-1], dtype=torch.long)
    flat_indices = torch.tensor(flat_indices, dtype=torch.long)
    return flat_indices, offsets, labels

train_bag_ds = BagDataset(df_train['Clean_text'], y_train)
val_bag_ds = BagDataset(df_val['Clean_text'], y_val)
test_bag_ds = BagDataset(df_test['Clean_text'], y_test)

train_bag_loader = DataLoader(train_bag_ds, batch_size=32, shuffle=True, collate_fn=collate_bag)
val_bag_loader = DataLoader(val_bag_ds, batch_size=64, shuffle=False, collate_fn=collate_bag)
test_bag_loader = DataLoader(test_bag_ds, batch_size=64, shuffle=False, collate_fn=collate_bag)

class DensePooledNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_classes=24):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode='mean')
        self.fc1 = nn.Linear(embed_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        x = self.dropout(self.relu(self.bn1(self.fc1(embedded))))
        return self.fc2(x)

dense_model = DensePooledNN(len(vocab), embed_dim=128, num_classes=24)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(dense_model.parameters(), lr=0.005, weight_decay=1e-4)

# Train Dense Pooled NN
for epoch in range(18):
    dense_model.train()
    for indices, offsets, labels in train_bag_loader:
        optimizer.zero_grad()
        out = dense_model(indices, offsets)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

def predict_bag(loader):
    dense_model.eval()
    all_preds = []
    with torch.no_grad():
        for indices, offsets, _ in loader:
            out = dense_model(indices, offsets)
            all_preds.extend(out.argmax(dim=1).cpu().numpy())
    return np.array(all_preds)

record("Dense NN (Pooled Embeddings)", "Pooled Dense Word Vectors (128d)",
       y_train, predict_bag(DataLoader(train_bag_ds, batch_size=64, shuffle=False, collate_fn=collate_bag)),
       y_val, predict_bag(val_bag_loader),
       y_test, predict_bag(test_bag_loader))

# ==============================================================================
# 7. Bidirectional GRU / LSTM Sequence Model (Deep Learning)
# ==============================================================================
MAX_SEQ_LEN = 200

def text_to_seq(text):
    tokens = [vocab.get(w, 1) for w in text.split()][:MAX_SEQ_LEN]
    if len(tokens) < MAX_SEQ_LEN:
        tokens += [0] * (MAX_SEQ_LEN - len(tokens))
    return tokens

class SeqDataset(Dataset):
    def __init__(self, texts, labels):
        self.seqs = torch.tensor([text_to_seq(t) for t in texts], dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx]

train_seq_ds = SeqDataset(df_train['Clean_text'], y_train)
val_seq_ds = SeqDataset(df_val['Clean_text'], y_val)
test_seq_ds = SeqDataset(df_test['Clean_text'], y_test)

train_seq_loader = DataLoader(train_seq_ds, batch_size=32, shuffle=True)
val_seq_loader = DataLoader(val_seq_ds, batch_size=64, shuffle=False)
test_seq_loader = DataLoader(test_seq_ds, batch_size=64, shuffle=False)

class BiGRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=96, hidden_dim=96, num_classes=24):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout_emb = nn.Dropout(0.2)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        embedded = self.dropout_emb(self.embedding(x))
        out, h_n = self.gru(embedded)
        # Max-pool over sequence length
        pooled, _ = torch.max(out, dim=1)
        return self.fc(pooled)

gru_model = BiGRUClassifier(len(vocab))
gru_optimizer = torch.optim.AdamW(gru_model.parameters(), lr=0.004, weight_decay=1e-4)

for epoch in range(12):
    gru_model.train()
    for x_b, y_b in train_seq_loader:
        gru_optimizer.zero_grad()
        logits = gru_model(x_b)
        loss = criterion(logits, y_b)
        loss.backward()
        gru_optimizer.step()

def predict_seq(loader):
    gru_model.eval()
    all_preds = []
    with torch.no_grad():
        for x_b, _ in loader:
            logits = gru_model(x_b)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
    return np.array(all_preds)

record("Bidirectional GRU / LSTM", "Token Sequences + Embeddings",
       y_train, predict_seq(DataLoader(train_seq_ds, batch_size=64, shuffle=False)),
       y_val, predict_seq(val_seq_loader),
       y_test, predict_seq(test_seq_loader))

# Display final comparison table
df_all = pd.DataFrame(results)
print("\n" + "="*80)
print("COMPREHENSIVE MODEL EVALUATION MATRIX (ALL 6 ARCHITECTURES)")
print("="*80)
print(df_all.to_string(index=False))

# Save summary to reports
df_all.to_csv("artifacts/reports/full_model_architectures_comparison.csv", index=False)
with open("artifacts/reports/full_model_architectures_comparison.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved artifacts/reports/full_model_architectures_comparison.csv & .json successfully!")
