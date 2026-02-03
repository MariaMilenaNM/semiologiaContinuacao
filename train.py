import random
import json
import pickle
import numpy as np
import nltk
import os

from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

# Downloads necessários (executa só na primeira vez)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

def treinar_modelo(caminho_intents, id_paciente):
    """
    Treina um modelo de chatbot para um paciente específico
    e salva os artefatos em backend/modelos/<id_paciente>/
    """

    lemmatizer = WordNetLemmatizer()

    # ===============================
    # 1. Carregar intents
    # ===============================
    with open(caminho_intents, encoding="utf-8") as f:
        intents = json.load(f)

    words = []
    classes = []
    documents = []
    ignore_letters = ['?', '!', '.', ',']

    # ===============================
    # 2. Processar intents
    # ===============================
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            tokens = nltk.word_tokenize(pattern.lower())
            tokens = [w for w in tokens if w not in ignore_letters]

            words.extend(tokens)
            documents.append((tokens, intent["tag"]))

            if intent["tag"] not in classes:
                classes.append(intent["tag"])

    # ===============================
    # 3. Lematização e ordenação
    # ===============================
    words = [lemmatizer.lemmatize(w) for w in words]
    words = sorted(set(words))
    classes = sorted(set(classes))

    # ===============================
    # 4. Criar dados de treino (BoW)
    # ===============================
    training = []
    output_empty = [0] * len(classes)

    for document in documents:
        bag = []
        pattern_words = [lemmatizer.lemmatize(w) for w in document[0]]

        for word in words:
            bag.append(1 if word in pattern_words else 0)

        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1

        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training, dtype=object)

    train_x = np.array(list(training[:, 0]))
    train_y = np.array(list(training[:, 1]))

    # ===============================
    # 5. Criar modelo neural
    # ===============================
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    optimizer = SGD(
        learning_rate=0.001,
        momentum=0.9,
        nesterov=True
    )

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    # ===============================
    # 6. Treinar modelo
    # ===============================
    model.fit(
        train_x,
        train_y,
        epochs=300,
        batch_size=5,
        verbose=1
    )

    # ===============================
    # 7. Salvar artefatos
    # ===============================
    base_dir = f"backend/modelos/{id_paciente}"
    os.makedirs(base_dir, exist_ok=True)

    with open(f"{base_dir}/words.pkl", "wb") as f:
        pickle.dump(words, f)

    with open(f"{base_dir}/classes.pkl", "wb") as f:
        pickle.dump(classes, f)

    model.save(f"{base_dir}/model.keras")

    print(f"Modelo do paciente '{id_paciente}' treinado e salvo com sucesso.")

treinar_modelo("logica\intents\intentsH1.json", "H1")
treinar_modelo("logica\intents\intentsH2.json", "H2")
treinar_modelo("logica\intents\intentsH3.json", "H3")
treinar_modelo("logica\intents\intentsH4.json", "H4")
treinar_modelo("logica\intents\intentsH5.json", "H5")
#treinar_modelo("backend/intents/intentsH6.json", "H6")
#treinar_modelo("backend/intents/intentsH7.json", "H7")
#treinar_modelo("backend/intents/intentsH8.json", "H8")
#treinar_modelo("backend/intents/intentsH9.json", "H9")
#treinar_modelo("backend/intents/intentsH10.json", "H10")