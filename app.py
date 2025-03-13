from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Simplified label mapping for a three-class model
id2label = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

# Model path (update if necessary)
model_path = "jaynwanze/interview_sentiment_mode"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model directory '{model_path}' not found!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    top_k=None,  # Return all scores
    device=0 if torch.cuda.is_available() else -1
)

def convert_label(label_str):
    if label_str.startswith("LABEL_"):
        predicted_index = int(label_str.split("_")[1])
        return id2label.get(predicted_index, label_str)
    elif label_str in id2label.values():
        return label_str
    else:
        try:
            index = int(label_str)
            return id2label.get(index, label_str)
        except Exception:
            return label_str

def aggregate_sentiment(feedbacks):
    aggregated_scores = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
    for feedback in feedbacks:
        result = sentiment_analyzer(feedback)
        if result and result[0]:
            for pred in result[0]:
                label = convert_label(pred["label"])
                aggregated_scores[label] += pred["score"]
    num_feedbacks = len(feedbacks)
    for key in aggregated_scores:
        aggregated_scores[key] /= num_feedbacks
    total = sum(aggregated_scores.values())
    if total > 0:
        for key in aggregated_scores:
            aggregated_scores[key] = (aggregated_scores[key] / total) * 100
    return aggregated_scores

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        feedbacks = data.get("feedbacks", [])
        
        if not isinstance(feedbacks, list) or len(feedbacks) == 0:
            return jsonify({"error": "No feedbacks provided"}), 400

        aggregated_scores = aggregate_sentiment(feedbacks)
        predicted_label = max(aggregated_scores, key=aggregated_scores.get)
        predicted_score = aggregated_scores[predicted_label]

        # Adjust prediction based on confidence: if positive but confidence is less than 70, default to neutral.
        # if predicted_label == "positive" and predicted_score < 70:
        #     predicted_label = "neutral"
        #     predicted_score = aggregated_scores["neutral"]
        
        print(f"Aggregated scores: {aggregated_scores}")
        print(f"Predicted label: {predicted_label}")

        return jsonify({
            "aggregated_scores": aggregated_scores,
            "label": predicted_label,
            "score": round(predicted_score, 2)
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
