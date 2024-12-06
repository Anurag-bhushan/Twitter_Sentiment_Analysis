import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Twitter_Data.csv')

# Preprocessing the dataset
# Dropping rows with missing values
df = df.dropna(subset=['clean_text', 'category'])

# Filter for binary classification (positive and negative sentiments)
df = df[df['category'].isin([-1.0, 1.0])]

# Map categories: -1.0 to 0 (negative), 1.0 to 1 (positive)
df['category'] = df['category'].map({-1.0: 0, 1.0: 1})

# Splitting the dataset
X = df['clean_text']
y = df['category']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Text vectorization using TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test_tfidf)

# Print results
print("Accuracy:", accuracy_score(y_test, y_pred))
classification_rep = classification_report(y_test, y_pred)
print("\nClassification Report:\n", classification_rep)

# Visualization Section
# Plotting Class Distribution
def plot_class_distribution(y):
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y, palette="pastel")
    plt.title("Class Distribution", fontsize=16)
    plt.xlabel("Sentiment", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(ticks=[0, 1], labels=["Negative (0)", "Positive (1)"], fontsize=10)
    plt.show()

# Plotting Confusion Matrix
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Negative (0)", "Positive (1)"], yticklabels=["Negative (0)", "Positive (1)"])
    plt.title("Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.show()

# Plotting Metrics
def plot_metrics(metrics):
    categories = ["Negative (0)", "Positive (1)"]
    plt.figure(figsize=(10, 6))
    bar_width = 0.2
    r1 = range(len(metrics['precision']))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    plt.bar(r1, metrics['precision'], color='skyblue', width=bar_width, label='Precision')
    plt.bar(r2, metrics['recall'], color='lightgreen', width=bar_width, label='Recall')
    plt.bar(r3, metrics['f1-score'], color='salmon', width=bar_width, label='F1-Score')

    plt.title("Metrics by Class", fontsize=16)
    plt.xlabel("Classes", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.xticks([r + bar_width for r in range(len(categories))], categories, fontsize=10)
    plt.legend()
    plt.show()

# Compute Metrics by Class
def get_metrics_by_class(report):
    lines = report.split("\n")[2:4]  # Extract class-specific metrics
    metrics = {'precision': [], 'recall': [], 'f1-score': []}
    for line in lines:
        values = line.split()
        metrics['precision'].append(float(values[1]))
        metrics['recall'].append(float(values[2]))
        metrics['f1-score'].append(float(values[3]))
    return metrics

# Generate Plots
# Plot class distribution
plot_class_distribution(y)

# Plot confusion matrix
plot_confusion_matrix(y_test, y_pred)

# Extract metrics and plot precision, recall, F1-score
metrics_by_class = get_metrics_by_class(classification_rep)
plot_metrics(metrics_by_class)

# Function to classify user-input tweets
def classify_tweet(tweet):
    """
    Preprocess and classify a single tweet.
    
    Args:
        tweet (str): The tweet text to classify.

    Returns:
        str: Sentiment classification ("Positive" or "Negative").
    """
    tweet_tfidf = tfidf.transform([tweet])
    prediction = model.predict(tweet_tfidf)[0]
    return "Positive" if prediction == 1 else "Negative"

# Example usage for classifying user-input tweets
while True:
    user_tweet = input("Enter a tweet (or type 'exit' to quit): ")
    if user_tweet.lower() == 'exit':
        break
    sentiment = classify_tweet(user_tweet)
    print(f"The sentiment of the tweet is: {sentiment}")


import pickle

# Save model
with open("sentiment_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# Save TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "wb") as tfidf_file:
    pickle.dump(tfidf, tfidf_file)