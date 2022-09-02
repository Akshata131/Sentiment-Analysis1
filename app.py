import pickle
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from keras.preprocessing.text import Tokenizer
from flask import Flask, render_template, request, Response, flash

app = Flask(__name__)

MODEL_PATH = './model.h5'

model = load_model(MODEL_PATH)

# from main import tokenize_pad_sequences
with open('./tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


def predict_class(text):
    '''Function to predict sentiment class of the passed text'''

    sentiment_classes = ['Negative', 'Neutral', 'Positive']
    max_len = 50
    max_words = 500

    # Transforms text to a sequence of integers using a tokenizer object
    xt = tokenizer.texts_to_sequences(text)
    # Pad sequences to the same length
    xt = pad_sequences(xt, padding='post', maxlen=max_len)
    # Do the prediction using the loaded model
    yt = model.predict(xt).argmax(axis=1)
    # Print the predicted sentiment
    print('The predicted sentiment is', sentiment_classes[yt[0]])
    return sentiment_classes[yt[0]]


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/', methods=["POST", "GET"])
def main():
    text = request.form['TEXT']
    print(text)
    answer = predict_class([text])
    return render_template("index.html",sentence='Your Sentiment is {}'.format(answer))


if __name__ == "__main__":
    app.run(debug=True)
