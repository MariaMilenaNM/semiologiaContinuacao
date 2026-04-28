"""
train.py
--------
Execute este script UMA VEZ antes de iniciar o servidor Flask.

    python train.py

Ele treina o classificador de intenção, salva o modelo Keras,
o vetorizador TF-IDF e os nomes das classes em ./saved_model/.
"""

import os
import json
import pickle
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras

from model import build_training_data, build_model

# ── Config ─────────────────────────────────────────────────────────────
SAVE_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model")
MODEL_PATH = os.path.join(SAVE_DIR, "intent_model.keras")
VECT_PATH  = os.path.join(SAVE_DIR, "vectorizer.pkl")
LABELS_PATH= os.path.join(SAVE_DIR, "labels.json")
EPOCHS     = 120
BATCH_SIZE = 8

os.makedirs(SAVE_DIR, exist_ok=True)

# ── 1. Dados ────────────────────────────────────────────────────────────
print("[ 1/5 ] Construindo dados de treino...")
texts, labels, label_names = build_training_data()
print(f"       {len(texts)} exemplos  |  {len(label_names)} classes")

from collections import Counter
dist = Counter(labels)
print("       Distribuição de exemplos por classe:")
for idx, name in enumerate(label_names):
    print(f"         {name:35s} → {dist[idx]:3d} exemplos")

# ── 2. Vetorização TF-IDF ───────────────────────────────────────────────
print("[ 2/5 ] Vetorizando com TF-IDF...")
vectorizer = TfidfVectorizer(
    analyzer="char_wb",   # n-gramas de caracteres → lida bem com variações
    ngram_range=(2, 5),
    max_features=5000,
    sublinear_tf=True
)
X = vectorizer.fit_transform(texts).toarray()
y = np.array(labels)
print(f"       Shape X: {X.shape}")

min_count = min(dist.values())
use_stratify = min_count >= 2

# ── 3. Split treino/validação ───────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
print(f"[ 3/5 ] Treino: {len(X_train)}  |  Validação: {len(X_val)}")

classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weight_dict = dict(zip(classes.tolist(), weights.tolist()))
print(f"       Class weights calculados para {len(class_weight_dict)} classes.")

# ── 4. Treino ───────────────────────────────────────────────────────────
print("[ 4/5 ] Treinando o modelo Keras...")
model = build_model(input_dim=X.shape[1], num_classes=len(label_names))
model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=12,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=2
)

# Avaliação final
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\n       Acurácia na validação: {val_acc:.2%}")

# ── 5. Salvar artefatos ─────────────────────────────────────────────────
print("[ 5/5 ] Salvando modelo e artefatos...")
model.save(MODEL_PATH)
print(f"       Modelo salvo em: {MODEL_PATH}")

with open(VECT_PATH, "wb") as f:
    pickle.dump(vectorizer, f)
print(f"       Vetorizador salvo em: {VECT_PATH}")

with open(LABELS_PATH, "w", encoding="utf-8") as f:
    json.dump(label_names, f, ensure_ascii=False, indent=2)
print(f"       Labels salvas em: {LABELS_PATH}")

print("\n✅ Treinamento concluído! Execute: python app.py")