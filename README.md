🤖 AI-Powered Sentiment Analysis for ChatGPT Reviews

📊 Domain: Customer Experience & Business Analytics
Analyze real ChatGPT user reviews to uncover satisfaction trends, top concerns, and feedback themes using NLP and ML models.

🧠 Project Overview
This project performs sentiment analysis on ChatGPT user reviews.
It classifies reviews as Positive, Neutral, or Negative, generates insights through EDA,
and visualizes findings using Streamlit dashboards.

🔗 Live App: https://ai-sentiment-chatgpt-reviews-9lxl4hywmim8rbrb2fu5fo.streamlit.app/

🎯 Objectives
• Understand user sentiment from real ChatGPT reviews
• Identify key words and trends in positive vs. negative feedback
• Visualize satisfaction over time, location, and platform
• Deploy an interactive app for sentiment prediction and insights

🧩 Tech Stack
Programming: Python
Data Handling: Pandas, NumPy
NLP: NLTK, Scikit-learn, WordCloud
Visualization: Matplotlib, Seaborn
Model: Logistic Regression / Naive Bayes
Deployment: Streamlit
Version Control: GitHub

⚙️ Project Workflow

Data Preprocessing – clean text, remove stopwords, punctuation, special characters

EDA (Exploratory Data Analysis) – visualize distributions, trends, and patterns

Feature Engineering – TF-IDF vectorization for text representation

Model Training – sentiment classification using ML models

Evaluation – accuracy, precision, recall, F1-score, confusion matrix

Deployment – Streamlit app with prediction and dashboards

📂 Folder Structure
AI-SentimentChatGPT/
│
├── data/ → Datasets
├── models/ → Saved model & vectorizer
├── notebooks/ → Jupyter notebooks for EDA
├── outputs/ → Visuals and word clouds
├── reports/ → Performance reports
├── src/ → Python scripts
│ ├── app.py → Streamlit main file
│ ├── data_cleaning.py
│ ├── data_prep.py
│ ├── model_training.py
│ ├── predict.py
│
├── requirements.txt
├── README.md
└── .gitignore

📊 Key Insights
• Majority reviews rated 4–5 stars → positive sentiment dominates
• Negative reviews mention “bugs”, “slow response”, “wrong answers”
• Some regions show varying satisfaction levels
• Web users rate higher than mobile users
• Verified users are generally happier

🧾 Evaluation Metrics
Accuracy – Measures correct predictions
Precision – Reliability of positive predictions
Recall – How many positives were captured
F1-Score – Balance of precision & recall
Confusion Matrix – Displays classification results

🚀 Deployment
Deployed on Streamlit Cloud for live demo.
App Link: https://ai-sentiment-chatgpt-reviews-9lxl4hywmim8rbrb2fu5fo.streamlit.app/
🏁 Conclusion
This project demonstrates how NLP and ML can extract valuable insights from customer reviews,
helping product teams improve user experience and satisfaction.

👩‍💻 Author
Vadivukarasi Moorthy
AI & Data Science Enthusiast
vadivukarasimoorthy12@gmail.com
