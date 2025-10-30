import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import load
import os

# Load cleaned dataset
df = pd.read_csv("data/chatgpt_reviews_cleaned.csv")

# Map rating → sentiment (same as model training)
def map_sentiment(rating):
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"

df["sentiment"] = df["rating"].apply(map_sentiment)

# Split dataset again (same seed)
from sklearn.model_selection import train_test_split
X = df["clean_review"]
y = df["sentiment"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Load trained model & vectorizer
model = load("models/sentiment_model.pkl")
vectorizer = load("models/tfidf_vectorizer.pkl")

# Transform test data
X_test_tfidf = vectorizer.transform(X_test)
y_pred = model.predict(X_test_tfidf)

# Calculate metrics
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred)

# Save report to CSV
report_df = pd.DataFrame(report).transpose()
os.makedirs("reports", exist_ok=True)
report_df.to_csv("reports/model_performance.csv", index=True)

# Plot confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", xticklabels=["negative","neutral","positive"], yticklabels=["negative","neutral","positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (Accuracy: {acc:.2f})")
plt.tight_layout()
plt.savefig("reports/confusion_matrix.png")
plt.close()

print("✅ Model performance report created!")
print(f"📊 Accuracy: {acc:.2f}")
print("Results saved in /reports folder.")
