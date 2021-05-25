from django.shortcuts import render
import re
from nltk.corpus import stopwords 
from tensorflow.keras.preprocessing.text import Tokenizer  
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.models import load_model 
# Create your views here.


def index(request):
    context={'a':1}
    return render(request,'index.html',context)

def predict(message):
    import pickle
    token=pickle.load(open("tkn.sav","rb"))
    loaded_model = load_model('LSTM.h5')
    review = str(message)
    english_stops = set(stopwords.words('english'))
    regex = re.compile(r'[^a-zA-Z\s]')
    review = regex.sub('', review)

    words = review.split(' ')
    filtered = [w for w in words if w not in english_stops]
    filtered = ' '.join(filtered)
    filtered = [filtered.lower()]

    tokenize_words = token.texts_to_sequences(filtered)
    tokenize_words = pad_sequences(tokenize_words, maxlen=130, padding='post', truncating='post')

    result = loaded_model.predict(tokenize_words)

    if result >= 0.7:
        return 'positive'
    else:
        return 'negative'


def result(request):
    message = request.GET['message']
    result = predict(message)
    return render(request,'result.html',{'result':result})
        