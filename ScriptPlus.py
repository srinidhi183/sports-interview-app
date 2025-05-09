import pandas as pd
import re
import json
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

# NLTK setup
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z']", ' ', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# ---------- Load Data ----------
train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('val.csv')
test_df = pd.read_csv('test.csv')

# Preprocess text
train_df['Clean_Text'] = train_df['Interview Text'].apply(clean_text)
val_df['Clean_Text'] = val_df['Interview Text'].apply(clean_text)
test_df['Clean_Text'] = test_df['Interview Text'].apply(clean_text)

# ---------- Train Model ----------
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train = tfidf.fit_transform(train_df['Clean_Text'])
y_train = train_df['Labels']

X_val = tfidf.transform(val_df['Clean_Text'])
y_val = val_df['Labels']

model = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
model.fit(X_train, y_train)

# ---------- Validation Evaluation ----------
val_preds = model.predict(X_val)

acc = accuracy_score(y_val, val_preds)
f1 = f1_score(y_val, val_preds, average='weighted')
print("Validation Accuracy:", acc)
print("Validation F1 Score:", f1)
print(classification_report(y_val, val_preds))

# Save results.json
results = {
    "f1_score": round(f1, 5),
    "accuracy": round(acc, 5)
}
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved: results.json")

# Confusion Matrix
labels = sorted(y_val.unique())
cm = confusion_matrix(y_val, val_preds, labels=labels)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix - Validation Set')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ---------- Final Prediction on Test Set ----------
X_test = tfidf.transform(test_df['Clean_Text'])
test_preds = model.predict(X_test)

submission = pd.DataFrame({
    'ID': test_df['ID'],
    'Labels': test_preds
})
submission.to_csv('submission.csv', index=False)
print("Saved: submission.csv")

# ✅ Zip it manually as `submission.zip` before uploading to Codabench
