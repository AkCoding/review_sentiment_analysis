# finetune reviews for sentiment analysis using distilBert

- [Rotten Tomatoes movies and critic reviews dataset](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset)
- [distilbert](https://huggingface.co/docs/transformers/model_doc/distilbert)
- [finetune model drive link](https://drive.google.com/drive/folders/17WU-DDKYcn4ACJVzDJMwdGrbUKaFJLXC?usp=sharing)

## Requirements

- Python 3.10
- FASTAPI

## Installation

`pip install -r requirements.txt`

## Usage

### Train

`python3 src/sentiment_analysis/train.py`

### Predict

`python3 src/sentiment_analysis/myapp.py`

> `{
    "sentiment": "Positive"
}`


## Note : i fine-tuned the DistilBERT model with 1000 datasets due to less time but if we train it with more data and increase the number of epochs, we can expect to see improvements in accuracy and reductions in loss