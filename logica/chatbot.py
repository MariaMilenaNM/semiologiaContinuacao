import random
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from keras.models import load_model

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()

def carregar_modelo(paciente):
    model = load_model(paciente["ml"]["modelo"])
    words = pickle.load(open(paciente["ml"]["words"], "rb"))
    classes = pickle.load(open(paciente["ml"]["classes"], "rb"))

    with open(paciente["ml"]["intents"], encoding="utf-8") as f:
        intents = json.load(f)

    return model, words, classes, intents

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

def predict_class(sentence, model, words, classes):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]), verbose=0)[0]

    ERROR_THRESHOLD = 0.25
    results = [(classes[i], r) for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return results

def get_response(intents, tag):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Pode repetir a pergunta?"

def responder_chat(paciente, frase):
    model, words, classes, intents = carregar_modelo(paciente)

    results = predict_class(frase, model, words, classes)

    if results:
        tag, prob = results[0]
        if prob > 0.8:
            return get_response(intents, tag)

    return "Não entendi. Pode reformular?"

def validar_diagnostico(paciente, diagnostico_usuario):
    """
    Verifica se o diagnóstico digitado pelo usuário é o correto.
    """
    # Normaliza as strings para comparar (minúsculo e sem espaços extras)
    correto = paciente["diagnostico_correto"].lower().strip()
    tentativa = diagnostico_usuario.lower().strip()
    
    return tentativa == correto