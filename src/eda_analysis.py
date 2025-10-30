import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

# Load cleaned dataset
df = pd.read_csv("data/chatgpt_reviews_cleaned.csv")

# Create output folder for charts
os.makedirs("outputs", exist_ok=True)

# 1️⃣ Rating distribution
plt.figure(figsize=(6,4))
sns.countplot(x="rating", data=df, palette="coolwarm")
plt.title("Distribution of Review Ratings (1–5 Stars)")
plt.xlabel("Rating")
plt.ylabel("Number of Reviews")
plt.savefig("outputs/rating_distribution.png")
plt.close()

# 2️⃣ Helpful votes distribution
df["helpful_votes"].fillna(0, inplace=True)
helpful_counts = (df["helpful_votes"] > 10).value_counts()
plt.figure(figsize=(5,4))
helpful_counts.plot(kind="pie", labels=["Helpful","Not Helpful"], autopct="%1.1f%%", colors=["lightgreen","lightcoral"])
plt.title("Helpful vs Non-Helpful Reviews")
plt.ylabel("")
plt.savefig("outputs/helpful_votes.png")
plt.close()

# 3️⃣ Wordclouds for positive vs negative reviews
positive_reviews = " ".join(df[df["rating"] >= 4]["clean_review"])
negative_reviews = " ".join(df[df["rating"] <= 2]["clean_review"])

wordcloud_pos = WordCloud(width=800, height=400, background_color="white", colormap="Greens").generate(positive_reviews)
wordcloud_neg = WordCloud(width=800, height=400, background_color="white", colormap="Reds").generate(negative_reviews)

wordcloud_pos.to_file("outputs/positive_wordcloud.png")
wordcloud_neg.to_file("outputs/negative_wordcloud.png")

# 4️⃣ Average rating over time
df["date"] = pd.to_datetime(df["date"], errors="coerce")
rating_trend = df.groupby("date")["rating"].mean().dropna()
plt.figure(figsize=(8,4))
plt.plot(rating_trend.index, rating_trend.values, marker="o", color="blue")
plt.title("Average Rating Over Time")
plt.xlabel("Date")
plt.ylabel("Average Rating")
plt.grid(True)
plt.savefig("outputs/rating_trend.png")
plt.close()

# 5️⃣ Average rating by location
top_locations = df["location"].value_counts().head(10).index
loc_data = df[df["location"].isin(top_locations)].groupby("location")["rating"].mean().sort_values()
plt.figure(figsize=(8,4))
sns.barplot(x=loc_data.values, y=loc_data.index, palette="viridis")
plt.title("Average Rating by Top 10 Locations")
plt.xlabel("Average Rating")
plt.ylabel("Location")
plt.savefig("outputs/rating_by_location.png")
plt.close()

# 6️⃣ Platform comparison
plt.figure(figsize=(6,4))
sns.barplot(x="platform", y="rating", data=df, palette="coolwarm", ci=None)
plt.title("Average Rating by Platform")
plt.savefig("outputs/platform_comparison.png")
plt.close()

# 7️⃣ Verified users vs Non-verified
verified_avg = df.groupby("verified_purchase")["rating"].mean()
verified_avg.plot(kind="bar", color=["lightgreen","coral"])
plt.title("Verified vs Non-Verified User Satisfaction")
plt.ylabel("Average Rating")
plt.savefig("outputs/verified_comparison.png")
plt.close()

# 8️⃣ Average review length per rating
plt.figure(figsize=(6,4))
sns.barplot(x="rating", y="review_length", data=df, palette="coolwarm")
plt.title("Average Review Length per Rating")
plt.savefig("outputs/review_length.png")
plt.close()

# 9️⃣ Most mentioned words in 1-star reviews
one_star = " ".join(df[df["rating"] == 1]["clean_review"])
wordcloud_1star = WordCloud(width=800, height=400, background_color="white", colormap="Reds").generate(one_star)
wordcloud_1star.to_file("outputs/one_star_words.png")

# 🔟 ChatGPT version with highest average rating
version_avg = df.groupby("version")["rating"].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(8,4))
sns.barplot(x=version_avg.values, y=version_avg.index, palette="coolwarm")
plt.title("Top 10 ChatGPT Versions by Average Rating")
plt.xlabel("Average Rating")
plt.ylabel("Version")
plt.savefig("outputs/version_ratings.png")
plt.close()

print("✅ EDA Completed! All visualizations saved inside /outputs folder.")
