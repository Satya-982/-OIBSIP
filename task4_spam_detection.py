# ============================================================
# TASK 4: EMAIL SPAM DETECTION WITH MACHINE LEARNING
# Internship - Data Science | InAmigos Foundation
# ============================================================

# Install required libraries (run once in terminal):
# pip install scikit-learn pandas numpy matplotlib seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# ─── 1. LOAD DATASET ──────────────────────────────────────────
# The dataset is the classic SMS Spam Collection.
# Download from: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
# OR use the built-in URL below (auto-download attempt).

print("=" * 55)
print("   EMAIL SPAM DETECTION")
print("=" * 55)

# Try loading from local file first, else fetch from URL
import os, urllib.request

DATA_FILE = "spam.csv"
DATA_URL   = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

if not os.path.exists(DATA_FILE):
    print(f"[INFO] Downloading dataset...")
    try:
        # Fetch tab-separated version
        urllib.request.urlretrieve(DATA_URL, "sms.tsv")
        df = pd.read_csv("sms.tsv", sep='\t', header=None, names=['label', 'message'])
    except Exception as e:
        print(f"[WARNING] Could not download: {e}")
        print("[INFO] Generating synthetic dataset for demo...")
        spam_msgs = [
            "Congratulations! You've won a £1000 prize. Call NOW!",
            "FREE entry! Win cash prizes. Text WIN to 80085",
            "URGENT: Your account has been compromised. Click here!",
            "You have been selected for a $500 gift card. Claim now!",
            "Hot singles in your area. Click to meet them tonight!",
            "You are a winner! Send your bank details to claim prize.",
            "Exclusive deal! Buy now and get 90% off. Limited time!",
            "Your loan has been approved. No credit check needed.",
            "Make money from home. Earn $5000/week. Sign up free!",
            "WINNER!! You won the national lottery. Text CLAIM to 555",
        ] * 30
        ham_msgs = [
            "Hey, are we still meeting for lunch tomorrow?",
            "Can you please send me the project report by Friday?",
            "I'll be home late tonight. Don't wait up.",
            "Happy birthday! Hope you have a wonderful day.",
            "The meeting has been rescheduled to 3 PM.",
            "Did you watch the game last night? It was amazing!",
            "Please find the attached document for your reference.",
            "Let me know if you need any help with the assignment.",
            "I'm running 10 minutes late. See you soon.",
            "Thanks for the help yesterday, I really appreciate it!",
        ] * 30
        labels = ['spam'] * len(spam_msgs) + ['ham'] * len(ham_msgs)
        msgs   = spam_msgs + ham_msgs
        df = pd.DataFrame({'label': labels, 'message': msgs})
else:
    df = pd.read_csv(DATA_FILE, encoding='latin-1')
    df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})

# Unify label format
df['label'] = df['label'].str.strip().str.lower()
df = df[df['label'].isin(['ham', 'spam'])].reset_index(drop=True)

print(f"\n[INFO] Dataset shape: {df.shape}")
print(f"[INFO] Label distribution:\n{df['label'].value_counts()}")

# ─── 2. EDA ──────────────────────────────────────────────────
df['message_length'] = df['message'].apply(len)
df['word_count']     = df['message'].apply(lambda x: len(x.split()))

print("\n[INFO] Message length stats by label:")
print(df.groupby('label')[['message_length', 'word_count']].mean().round(2))

# Distribution plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col, title in zip(axes, ['message_length', 'word_count'],
                          ['Message Length', 'Word Count']):
    df[df['label'] == 'ham'][col].hist(ax=ax, alpha=0.6, bins=30, color='steelblue', label='Ham')
    df[df['label'] == 'spam'][col].hist(ax=ax, alpha=0.6, bins=30, color='tomato', label='Spam')
    ax.set_title(f'{title} Distribution')
    ax.legend()
plt.suptitle("Spam vs Ham - Message Statistics", fontsize=13)
plt.tight_layout()
plt.savefig("spam_length_dist.png", dpi=100)
plt.show()
print("[SAVED] spam_length_dist.png")

# Label pie chart
plt.figure(figsize=(5, 5))
df['label'].value_counts().plot.pie(
    autopct='%1.1f%%', colors=['#4CAF50', '#F44336'],
    labels=['Ham', 'Spam'], startangle=90, shadow=True
)
plt.title("Spam vs Ham Distribution")
plt.ylabel("")
plt.tight_layout()
plt.savefig("spam_pie_chart.png", dpi=100)
plt.show()
print("[SAVED] spam_pie_chart.png")

# ─── 3. TEXT PREPROCESSING ───────────────────────────────────
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)                         # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()               # remove extra spaces
    return text

df['clean_message'] = df['message'].apply(preprocess_text)

print("\n[INFO] Sample cleaned messages:")
print(df[['message', 'clean_message', 'label']].head(4).to_string())

# ─── 4. FEATURE EXTRACTION ───────────────────────────────────
X = df['clean_message']
y = (df['label'] == 'spam').astype(int)   # 1=spam, 0=ham

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

print(f"\n[INFO] TF-IDF feature matrix: {X_train_tfidf.shape}")

# ─── 5. TRAIN MODELS ─────────────────────────────────────────
models = {
    "Naive Bayes":         MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Linear SVM":          LinearSVC(max_iter=1000, random_state=42),
}

results = {}
print("\n" + "-" * 55)
print("  MODEL COMPARISON")
print("-" * 55)
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name:25s}  Accuracy: {acc * 100:.2f}%")

# ─── 6. BEST MODEL EVALUATION ────────────────────────────────
best_name  = max(results, key=results.get)
best_model = models[best_name]
y_pred_best = best_model.predict(X_test_tfidf)

print(f"\n[BEST MODEL] {best_name}  ({results[best_name]*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=['Ham', 'Spam']))

cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ham', 'Spam'])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("spam_confusion_matrix.png", dpi=100)
plt.show()
print("[SAVED] spam_confusion_matrix.png")

# ─── 7. TOP SPAM KEYWORDS ────────────────────────────────────
if hasattr(best_model, 'coef_'):
    feature_names = tfidf.get_feature_names_out()
    coef = best_model.coef_[0] if best_model.coef_.ndim > 1 else best_model.coef_
    top_spam_idx = np.argsort(coef)[-20:]
    top_spam_words = [feature_names[i] for i in top_spam_idx]
    top_spam_vals  = coef[top_spam_idx]

    plt.figure(figsize=(8, 5))
    plt.barh(top_spam_words, top_spam_vals, color='tomato')
    plt.xlabel("Coefficient Weight")
    plt.title("Top 20 Spam Indicator Words")
    plt.tight_layout()
    plt.savefig("spam_top_keywords.png", dpi=100)
    plt.show()
    print("[SAVED] spam_top_keywords.png")

# ─── 8. TEST CUSTOM MESSAGES ─────────────────────────────────
print("\n" + "=" * 55)
print("  TEST CUSTOM MESSAGES")
print("=" * 55)

test_messages = [
    "Congratulations! You've won a free iPhone. Click here now!",
    "Hey, can we meet for coffee tomorrow at 10?",
    "URGENT: Your bank account needs immediate verification.",
    "Please review the attached report before Monday's meeting.",
]

for msg in test_messages:
    cleaned = preprocess_text(msg)
    vec     = tfidf.transform([cleaned])
    pred    = best_model.predict(vec)[0]
    label   = "🚨 SPAM" if pred == 1 else "✅ HAM "
    print(f"{label}  |  {msg[:65]}")

print("\n[DONE] Task 4 complete!")
