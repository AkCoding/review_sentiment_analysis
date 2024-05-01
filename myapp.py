import uvicorn
from fastapi import FastAPI, HTTPException
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer
import tensorflow as tf

app = FastAPI()

CLASS_LABELS = ["Negative", "Neutral", "Positive"]
MODEL_PATH = "/home/wappnet-61/Pycharm_project/movie-reviews-distilbert-sentiment-analysis-main/src/tuned_model"
# MODEL_PATH = "../tuned_model"
model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


def preprocess_text(text: str) -> tf.Tensor:
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="tf",
        return_token_type_ids=False,
    )
    input_ids = inputs["input_ids"]
    return input_ids


def predict_sentiment(text: str) -> str:
    preprocessed_text = preprocess_text(text)
    predictions = model.predict(preprocessed_text)[0]
    predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
    predicted_label = CLASS_LABELS[predicted_class]
    return predicted_label


@app.post('/predict_sentiment')
async def predict_sentiment_api(text: dict):
    if 'text' not in text or not text['text']:
        raise HTTPException(status_code=400, detail="Text field cannot be empty")
    input_text = text['text']
    sentiment = predict_sentiment(input_text)
    return {"sentiment": sentiment}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8002)