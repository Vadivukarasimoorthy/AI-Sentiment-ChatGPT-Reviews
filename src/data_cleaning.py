import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download once
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# 1. Load dataset
df = pd.read_excel("data/chatgpt_style_reviews_dataset.xlsx")

# 2. Fix invalid dates
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['date'] = df['date'].fillna(pd.Timestamp("2024-01-01"))

# 3. Handle missing values
df = df.dropna(subset=['review', 'rating'])
df['language'] = df['language'].fillna('unknown')
df['location'] = df['location'].fillna('unknown')

# 4. Remove duplicates
df = df.drop_duplicates(subset=['review'])

# 5. Prepare for cleaning
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# 6. Apply cleaning
df['clean_review'] = df['review'].apply(clean_text)

# 7. Save cleaned file
df.to_csv("data/chatgpt_reviews_cleaned.csv", index=False)

print("✅ Cleaning done! Cleaned file saved in data/chatgpt_reviews_cleaned.csv")
