# import library
import string
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from util import JSONParser

def preprocess(chat):
    # konversi ke lowercase
    chat = chat.lower()
    # menghapus tanda baca
    tandabaca = tuple(string.punctuation)
    chat = ''.join(ch for ch in chat if ch not in tandabaca)
    return chat

def bot_response(chat, model_pipe, parser):
    chat = preprocess(chat)
    res = model_pipe.predict_proba([chat])
    max_prob = max(res[0])
    if max_prob < 0.2:
        return "Maaf kak, aku ga ngerti :(" , None
    else:
        max_id = np.argmax(res[0])
        pred_tag = model_pipe.classes_[max_id]
        return parser.get_response(pred_tag), pred_tag

# load data
path = "data/intents.json"
parser = JSONParser()
parser.parse(path)
data = parser.get_dataframe()

# praprocessing
# case folding, transform kapital ke non kapital dan buang tanda baca
data['input_preprocessing'] = data.text_input.apply(preprocess)

# modeling
model_pipe = make_pipeline(CountVectorizer(),
                        MultinomialNB())

# train
print("[INFO] Training Data ...")
model_pipe.fit(data.input_preprocessing, data.intents)

# saving model
with open("chatbot_model.pkl", "wb") as model_file:
    pickle.dump(model_pipe, model_file)

# interaction with bot
print("Selamat Datang di JoFi Chatbot ")
while True:
    chat = input("Anda >> ")
    res, tag = bot_response(chat, model_pipe, parser)
    print(f"Bot >> {res}")
    if tag == 'bye':
        break

