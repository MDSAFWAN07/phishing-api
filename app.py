from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import re

app = Flask(__name__)

# Load model and tokenizer
model = load_model("phishing_lstm_model.h5")
tokenizer = joblib.load("lstm_tokenizer.pkl")

MAX_LEN = 200

def clean_url(url):
    url = str(url)
    url = re.sub(r'https?:\/\/', '', url)
    url = re.sub(r'www\.', '', url)
    url = re.sub(r'[^a-zA-Z0-9\.]', ' ', url)
    return url.lower().strip()

def preprocess_url(url):
    cleaned = clean_url(url)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    return padded

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "LSTM Phishing API is Running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        url = data.get("url")
        if not url:
            return jsonify({"error": "Missing 'url' in request"}), 400

        X = preprocess_url(url)
        pred = model.predict(X)[0][0]
        result = "phishing" if pred > 0.5 else "legit"
        return jsonify({"url": url, "prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)