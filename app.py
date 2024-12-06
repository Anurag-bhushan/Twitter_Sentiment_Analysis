from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load pre-trained model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    tweet = data.get("tweet", "")
    if not tweet:
        return jsonify({"error": "No tweet provided"}), 400

    # Transform and predict
    tweet_tfidf = tfidf.transform([tweet])
    prediction = model.predict(tweet_tfidf)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    return jsonify({"sentiment": sentiment})

if __name__ == "__main__":
    app.run(debug=True)
