"""
model.py
--------
Define e expõe o modelo Keras usado para simular respostas do paciente.

Arquitetura:
  Entrada  → vetor TF-IDF da pergunta do estudante (vocabulário fixo)
  Camadas  → Dense + Dropout (regularização)
  Saída    → softmax sobre as categorias de intenção (intent)

As categorias de intenção mapeiam uma pergunta do estudante para uma
chave de resposta definida no patients.json (ex.: "febre", "dor", "default").
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
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "saved_model", "intent_model.keras")
VECT_PATH   = os.path.join(BASE_DIR, "saved_model", "vectorizer.pkl")
LABELS_PATH = os.path.join(BASE_DIR, "saved_model", "labels.json")

# ── Intenções e exemplos de treino ─────────────────────────────────────
# Cada intenção corresponde a uma chave nas "responses" do paciente no JSON.
# Adicione mais exemplos por intent para melhorar a acurácia.
INTENTS = {
    "febre": [
        "tem febre", "está com febre", "temperatura", "febre alta", "febril",
        "calafrio", "calafrios", "termômetro", "temperatura corporal",
        "febre baixa", "hipertermia"
    ],
    "dor": [
        "sente dor", "tem dor", "onde dói", "dor em algum lugar", "dores",
        "dor de cabeça", "cefaleia", "dor no corpo", "dor abdominal",
        "artralgia", "mialgia", "dor nas juntas", "dor nos ossos"
    ],
    "cansaco": [
        "está cansado", "fadiga", "fraqueza", "sem energia", "exausto",
        "cansaço", "indisposição", "moleza", "sem disposição", "fraco",
        "prostrado", "astenia"
    ],
    "manchas": [
        "manchas", "hematomas", "equimoses", "petéquias", "roxo", "roxos",
        "mancha roxa", "sangramento na pele", "pele", "lesão na pele",
        "equimose", "púrpura"
    ],
    "sangramento": [
        "sangramento", "sangra", "sangrou", "hemorragia", "sangue",
        "gengiva sangrando", "nariz sangrando", "epistaxe", "sangue na urina",
        "hematúria", "melena", "sangue nas fezes"
    ],
    "historico": [
        "histórico", "doenças anteriores", "já teve", "antecedentes",
        "condição prévia", "problema de saúde", "diagnóstico anterior",
        "doenças crônicas", "comorbidades", "internações anteriores"
    ],
    "remedios": [
        "remédios", "medicamentos", "toma algum remédio", "medicação",
        "faz uso de", "usa algum", "tratamento", "comprimido", "droga",
        "prescrito", "receita"
    ],
    "familia": [
        "família", "familiar", "hereditário", "pai", "mãe", "irmão",
        "parentes", "histórico familiar", "genético", "hereditariedade",
        "avó", "avô"
    ],
    "alimentacao": [
        "alimentação", "dieta", "come", "o que come", "hábitos alimentares",
        "vegetariano", "vegano", "carne", "alimenta", "nutrição", "comida"
    ],
    "urina": [
        "urina", "xixi", "cor da urina", "urina escura", "urinando",
        "hemoglobinúria", "urina avermelhada", "urina marrom", "micção"
    ],
    "transfusao": [
        "transfusão", "recebeu sangue", "bolsa de sangue", "hemoterapia",
        "durante a transfusão", "após a transfusão", "hemocomponente"
    ],
    "fraqueza": [
        "fraqueza muscular", "não consegue andar", "dificuldade de andar",
        "pernas fracas", "força muscular", "paresia", "paralisia",
        "fraqueza nas pernas", "fraqueza nos braços"
    ],
    "movimentos": [
        "movimentos involuntários", "tremor", "espasmo", "convulsão",
        "movimentos anormais", "discinesia", "coreia", "ataxia"
    ],
    "neurologico": [
        "neurológico", "dificuldade de falar", "confusão mental", "memória",
        "formigamento", "parestesia", "dormência", "reflexos", "fala"
    ],
    "visao": [
        "visão", "enxerga", "vista", "olho", "embaçado", "visão turva",
        "perda de visão", "diplopia"
    ],
    "barriga": [
        "barriga", "abdômen inchado", "abdome", "distensão abdominal",
        "barriga crescendo", "ascite", "líquido na barriga"
    ],
    "pele": [
        "pele", "alteração na pele", "mancha escura", "hiperpigmentação",
        "pelos", "hirsutismo", "lesão cutânea", "dermatose"
    ],
    "carocos": [
        "caroço", "nódulo", "inchaço no pescoço", "íngua", "linfadenopatia",
        "linfonodo", "gânglio", "massa palpável", "tumor"
    ],
    "emagrecimento": [
        "emagreceu", "perdeu peso", "perda de peso", "emagreci",
        "emagrecimento", "peso diminuiu", "ficou magro"
    ],
    "hiv": [
        "HIV", "aids", "soropositivo", "retrovírus", "CD4", "antirretroviral",
        "imunidade", "imunodeficiência"
    ],
    "falta_ar": [
        "falta de ar", "dispneia", "dificuldade de respirar", "sem fôlego",
        "ofegante", "taquipneia", "sufocando", "não consegue respirar"
    ],
    "pressao": [
        "pressão alta", "hipertensão", "pressão arterial", "pressão subiu",
        "pressão sanguínea"
    ],
    "melhora": [
        "melhorou", "como melhorou", "o que ajudou", "tratamento ajudou",
        "melhora após", "remissão dos sintomas"
    ],
    "engolir": [
        "engolir", "disfagia", "dificuldade para engolir", "dor ao engolir",
        "deglutição"
    ],
    "alergia": [
        "alergia", "alérgico", "reação alérgica", "intolerância",
        "hipersensibilidade"
    ],
    "infeccoes": [
        "infecções", "infecção", "infecções frequentes", "fica doente sempre",
        "imunidade baixa", "suscetível a infecções"
    ],
    "default": [
        "não entendi", "pode repetir", "outra coisa", "mais alguma coisa",
        "tenho mais sintomas", "o que mais", "qualquer coisa", "?"
    ]
}


def build_training_data():
    """Constrói X (textos) e y (índices de intent) para treino."""
    texts  = []
    labels = []
    label_names = list(INTENTS.keys())

    for idx, (intent, examples) in enumerate(INTENTS.items()):
        for example in examples:
            texts.append(example)
            labels.append(idx)

    return texts, labels, label_names


def build_model(input_dim: int, num_classes: int) -> keras.Model:
    """Cria e compila o modelo de classificação de intenção."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,), name="tfidf_input"),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
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
    """Carrega modelo, vetorizador e labels salvos em disco."""
    model = keras.models.load_model(MODEL_PATH)

    with open(VECT_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        label_names = json.load(f)

    return model, vectorizer, label_names


def predict_intent(text: str, model, vectorizer, label_names: list,
                   threshold: float = 0.35) -> str:
    """
    Dado um texto de entrada, retorna a intenção prevista.

    Se a confiança máxima for menor que `threshold`, retorna "default".
    """
    vec   = vectorizer.transform([text.lower()]).toarray()
    probs = model.predict(vec, verbose=0)[0]
    idx   = int(np.argmax(probs))

    if probs[idx] < threshold:
        return "default"

    return label_names[idx]