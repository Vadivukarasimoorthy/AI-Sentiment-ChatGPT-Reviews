import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# 🎯 Page setup
st.set_page_config(page_title="AI-Powered Sentiment Analysis for ChatGPT Reviews",
                   page_icon="🤖",
                   layout="wide")

st.title("🤖 AI-Powered Sentiment Analysis for ChatGPT Reviews")
st.markdown("### Explore ChatGPT user feedback through data & sentiment insights")

# 🧭 Sidebar navigation
page = st.sidebar.radio("Go to section:", 
                        ["📝 Predict Sentiment", 
                         "📊 EDA Visualizations",
                         "📈 Insights Summary",
                         "📋 Model Performance"])

# -------------------------------
# 📝 1️⃣ Sentiment Prediction Page
# -------------------------------
if page == "📝 Predict Sentiment":
    st.header("💬 Enter a review to analyze sentiment")

    user_review = st.text_area("Type or paste a ChatGPT review below:")

    if st.button("Analyze Sentiment"):
        from joblib import load
        import os
        model_path = "models/sentiment_model.pkl"
        vectorizer_path = "models/tfidf_vectorizer.pkl"

        if not os.path.exists(model_path):
            st.warning("⚠️ Model not trained yet. Please run model_training.py first.")
        else:
            model = load(model_path)
            vectorizer = load(vectorizer_path)
            X = vectorizer.transform([user_review])
            prediction = model.predict(X)[0]
            st.success(f"Predicted Sentiment: **{prediction.upper()}**")

# -------------------------------
# 📊 2️⃣ EDA Visualizations Page
# -------------------------------
elif page == "📊 EDA Visualizations":
    st.header("📊 Exploratory Data Analysis Dashboard")

    st.markdown("##### Rating Distribution")
    st.image("outputs/rating_distribution.png", use_container_width=True)

    st.markdown("##### Helpful vs Non-Helpful Reviews")
    st.image("outputs/helpful_votes.png", use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Positive Reviews WordCloud")
        st.image("outputs/positive_wordcloud.png", use_container_width=True)
    with col2:
        st.markdown("##### Negative Reviews WordCloud")
        st.image("outputs/negative_wordcloud.png", use_container_width=True)

    st.markdown("##### Average Rating Over Time")
    st.image("outputs/rating_trend.png", use_container_width=True)

    st.markdown("##### Average Rating by Location")
    st.image("outputs/rating_by_location.png", use_container_width=True)

    st.markdown("##### Platform Comparison")
    st.image("outputs/platform_comparison.png", use_container_width=True)

    st.markdown("##### Verified vs Non-Verified Users")
    st.image("outputs/verified_comparison.png", use_container_width=True)

    st.markdown("##### Review Length vs Rating")
    st.image("outputs/review_length.png", use_container_width=True)

    st.markdown("##### Most Common Words in 1-Star Reviews")
    st.image("outputs/one_star_words.png", use_container_width=True)

    st.markdown("##### ChatGPT Versions with Highest Ratings")
    st.image("outputs/version_ratings.png", use_container_width=True)

# -------------------------------
# 📈 3️⃣ Summary & Insights
# -------------------------------
elif page == "📈 Insights Summary":
    st.header("🧠 Key Findings & Insights")
    st.write("""
    - ⭐ **Users mostly give 4–5 star reviews**, showing overall positive sentiment.  
    - 👎 **Low-rated reviews** mention “bugs”, “slow”, “wrong answers”.  
    - ✅ **Verified users** are slightly happier than non-verified.  
    - 🌍 Some regions rate higher than others – important for localization.  
    - 📱 **Mobile users** give slightly lower ratings than web users.  
    - 🕓 Ratings have improved in recent versions → consistent updates are effective.  
    """)

# -------------------------------
# 📋 4️⃣ Model Performance Report
# -------------------------------
elif page == "📋 Model Performance":
    st.header("📋 Model Performance Report")

    st.markdown("### Confusion Matrix")
    st.image("reports/confusion_matrix.png", use_container_width=True)

    st.markdown("### Classification Metrics")
    df_report = pd.read_csv("reports/model_performance.csv")
    st.dataframe(df_report.style.highlight_max(axis=0))

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.info("Built by Vadivu | AI Sentiment Project")
