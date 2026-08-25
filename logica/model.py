import os
import json
import pickle
import re
import unicodedata
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
from tensorflow import keras
from keras import layers

# Paths
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "saved_model", "intent_model.keras")
VECT_PATH     = os.path.join(BASE_DIR, "saved_model", "vectorizer.pkl")
LABELS_PATH   = os.path.join(BASE_DIR, "saved_model", "labels.json")
INTENTS_JSON  = os.path.join(BASE_DIR, "intentsH5.json")


# Normalização de texto
def _normalize_text(text: str) -> str:
    """
    Normaliza a entrada para reduzir variações que confundem o modelo:
    - minúsculas
    - remove acentos ('úlcera' -> 'ulcera')
    - remove pontuação, mantém só letras/números/espaços
    - colapsa espaços múltiplos

    IMPORTANTE: essa função deve ser usada tanto no treino (train.py /
    build_training_data) quanto na inferência (predict_intent), senão o
    vetorizador vê tokens diferentes em cada etapa.
    """
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _load_intents_json() -> dict:
    if not os.path.exists(INTENTS_JSON):
        raise FileNotFoundError(
            f"Arquivo de intenções não encontrado: {INTENTS_JSON}\n"
            "Copie o intentsH5.json para a pasta logica/."
        )
    with open(INTENTS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["tag"]: item["patterns"] for item in data["intents"]}


_INTENTS_CACHE: dict | None = None

def get_intents() -> dict:
    global _INTENTS_CACHE
    if _INTENTS_CACHE is None:
        _INTENTS_CACHE = _load_intents_json()
    return _INTENTS_CACHE


def build_training_data():
    """Constrói X (textos normalizados) e y (índices de intent)."""
    intents     = get_intents()
    texts       = []
    labels      = []
    label_names = list(intents.keys())

    for idx, (tag, patterns) in enumerate(intents.items()):
        for pattern in patterns:
            texts.append(_normalize_text(pattern))   # <-- usa a normalização
            labels.append(idx)

    return texts, labels, label_names

#Modelo
def build_model(input_dim: int, num_classes: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(input_dim,), name="tfidf_input"),
        layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.005)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(0.005)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax", name="intent_output"),
    ], name="intent_classifier")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def load_artifacts():
    model = keras.models.load_model(MODEL_PATH)
    with open(VECT_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        label_names = json.load(f)
    return model, vectorizer, label_names


# ── Inferência (ALTERADA) ────────────────────────────────────────────────
def predict_intent(text: str, model, vectorizer, label_names: list,
                    threshold: float = 0.68, margin: float = 0.12) -> str:
    """
    threshold: confiança mínima da classe vencedora (era 0.45, agora 0.68).
    margin:    diferença mínima entre a 1ª e a 2ª intent mais prováveis.
               Se as duas estiverem "empatadas" (< margin), o modelo está
               em dúvida entre duas intents parecidas — uma das principais
               causas de resposta "alucinada" — e cai no fallback "naosei".
    """
    vec   = vectorizer.transform([_normalize_text(text)]).toarray()
    probs = model.predict(vec, verbose=0)[0]

    ordered = np.argsort(probs)[::-1]
    top_idx, second_idx = ordered[0], ordered[1]
    top_prob, second_prob = probs[top_idx], probs[second_idx]

    is_confident   = top_prob >= threshold
    is_clear_winner = (top_prob - second_prob) >= margin

    if not (is_confident and is_clear_winner):
        if "naosei" in label_names:
            return "naosei"
        return label_names[-1]

    return label_names[top_idx]