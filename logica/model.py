"""
model.py
--------
Define e expõe o modelo Keras usado para simular respostas do paciente.

Arquitetura:
  Entrada  → vetor TF-IDF da pergunta do estudante (vocabulário fixo)
  Camadas  → Dense + Dropout (regularização)
  Saída    → softmax sobre as categorias de intenção (intent)

As categorias de intenção são lidas do intentsH5.json e mapeiam a pergunta
do estudante para uma chave de resposta no patients.json.
"""

import os
import json
import pickle
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow as tf
from tensorflow import keras
from keras import layers

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "saved_model", "intent_model.keras")
VECT_PATH     = os.path.join(BASE_DIR, "saved_model", "vectorizer.pkl")
LABELS_PATH   = os.path.join(BASE_DIR, "saved_model", "labels.json")

# intentsH5.json deve estar na mesma pasta que model.py (logica/)
INTENTS_JSON  = os.path.join(BASE_DIR, "intentsH5.json")


# ── Carrega intentsH5.json ─────────────────────────────────────────────
def _load_intents_json() -> dict:
    """Lê o intentsH5.json e retorna um dict {tag: [patterns]}."""
    if not os.path.exists(INTENTS_JSON):
        raise FileNotFoundError(
            f"Arquivo de intenções não encontrado: {INTENTS_JSON}\n"
            "Copie o intentsH5.json para a pasta logica/."
        )
    with open(INTENTS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {item["tag"]: item["patterns"] for item in data["intents"]}


# Cache para não reler o arquivo a cada chamada
_INTENTS_CACHE: dict | None = None

def get_intents() -> dict:
    global _INTENTS_CACHE
    if _INTENTS_CACHE is None:
        _INTENTS_CACHE = _load_intents_json()
    return _INTENTS_CACHE


# ── Dados de treino ────────────────────────────────────────────────────
def build_training_data():
    """Constrói X (textos) e y (índices de intent) lendo do intentsH5.json."""
    intents     = get_intents()
    texts       = []
    labels      = []
    label_names = list(intents.keys())

    for idx, (tag, patterns) in enumerate(intents.items()):
        for pattern in patterns:
            texts.append(pattern.lower().strip())
            labels.append(idx)

    return texts, labels, label_names


# ── Modelo ─────────────────────────────────────────────────────────────
def build_model(input_dim: int, num_classes: int) -> keras.Model:
    """
    Cria e compila o modelo de classificação de intenção.

    Arquitetura enxuta (128→64) para evitar overfitting com ~1000 exemplos.
    Dropout maior na primeira camada para regularização mais forte.
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,), name="tfidf_input"),

        layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.005)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(0.005)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(num_classes, activation="softmax", name="intent_output"),
    ], name="intent_classifier")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ── Artefatos salvos ───────────────────────────────────────────────────
def load_artifacts():
    """Carrega modelo, vetorizador e labels salvos em disco."""
    model = keras.models.load_model(MODEL_PATH)

    with open(VECT_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        label_names = json.load(f)

    return model, vectorizer, label_names


# ── Inferência ─────────────────────────────────────────────────────────
def predict_intent(text: str, model, vectorizer, label_names: list,
                   threshold: float = 0.35) -> str:
    """
    Dado um texto de entrada, retorna a intenção prevista.

    Se a confiança máxima for menor que `threshold`, retorna "naosei"
    (ou a última classe do label_names como fallback).
    """
    vec   = vectorizer.transform([text.lower().strip()]).toarray()
    probs = model.predict(vec, verbose=0)[0]
    idx   = int(np.argmax(probs))

    if probs[idx] < threshold:
        if "naosei" in label_names:
            return "naosei"
        return label_names[-1]

    return label_names[idx]