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

INTENTS = {
    "nome": [
        "qual seu nome", "como você se chama", "seu nome", "teu nome",
        "nome do senhor", "como posso chamar", "qual é o seu nome completo",
        "pode me dizer seu nome", "seu nome por favor", "como devo te chamar",
        "qual o nome que prefere", "pode confirmar seu nome", "nome social",
        "qual nome devo anotar", "como está registrado seu nome",
        "qual é o nome que aparece na identidade", "diga seu nome",
        "pode falar seu nome", "confirma seu nome", "como se identifica",
    ],
 
    "idade": [
        "qual a sua idade", "quantos anos você tem", "está com quantos anos",
        "qual a idade", "anos", "idade", "pode me dizer sua idade",
        "sua idade atual", "quantos anos está fazendo", "em que ano nasceu",
        "qual sua data de nascimento", "com que idade está agora",
        "está com mais de 60 anos", "já é maior de idade",
        "qual sua faixa etária", "quantos anos já viveu",
        "está mais perto dos 20 ou dos 30", "qual idade está nos documentos",
    ],
 
    "naturalidade": [
        "qual sua naturalidade", "onde nasceu", "nasceu aqui", "de onde é",
        "qual cidade natal", "é natural de onde", "veio de qual cidade",
        "procedência", "onde mora", "qual o bairro", "qual a cidade",
        "mora aqui há muito tempo", "zona rural", "área urbana", "interior",
        "de qual estado veio", "cresceu onde", "cidade de origem",
        "nascido e criado aqui", "já morou em outra cidade",
    ],
 
    "QPD": [
        "qual o motivo da consulta", "o que está sentindo", "por que veio",
        "qual a sua queixa", "qual a queixa principal", "qual o problema",
        "quais são os seus sintomas", "o que lhe incomoda", "QDP",
        "o que trouxe você ao médico", "o que está acontecendo",
        "qual a razão da consulta", "me fale o motivo", "o que fez você vir",
        "qual sua maior dificuldade", "o que te preocupa mais",
        "como posso ajudar", "o que te trouxe aqui",
        "qual a sua queixa de hoje", "por que veio ao hospital",
    ],
 
    "Inicio": [
        "quando começaram os sintomas", "há quanto tempo sente isso",
        "quando surgiu o problema", "desde quando está assim",
        "quando apareceram os sintomas", "quando começou a se sentir mal",
        "quando foi o início da queixa", "faz quanto tempo que começou",
        "desde quando está sentindo isso", "quando iniciou seu quadro",
        "quando tudo começou", "quando notou pela primeira vez",
        "desde que período sente os sintomas", "quando percebeu",
        "há quanto tempo vem apresentando", "em que momento começou",
    ],
 
    "Detalhe": [
        "pode detalhar melhor", "fale mais sobre seus sintomas",
        "me conte mais", "explique melhor", "conte em detalhes",
        "descreva com mais detalhes", "me explique como está se sentindo",
        "pode descrever melhor", "quais sintomas você tem notado",
        "como você descreveria seu estado", "fale de forma geral",
        "pode aprofundar", "me diga mais sobre seus sintomas",
        "conte os sintomas de forma mais detalhada",
        "pode dar um panorama geral", "explique tudo o que sente",
    ],
 
    "Localizacao": [
        "onde dói", "onde sente a dor", "qual a localização",
        "em que parte do corpo", "dor localizada onde",
        "o sintoma fica em que região", "onde está o problema",
        "qual o local exato", "a dor fica onde",
        "em que lugar sente", "localiza a dor onde",
        "a dor irradia para onde", "em qual região do corpo",
    ],
 
    "Irradiacao": [
        "a dor irradia", "espalha para outro lugar", "a dor se espalha",
        "vai para outro lugar", "sente em outro local também",
        "a dor começa em um lugar e vai para outro",
        "irradia para o braço", "irradia para a perna",
        "irradia para o pescoço", "irradia para as costas",
        "a dor vai de um lugar para o outro",
    ],
 
    "Caracter": [
        "como é a dor", "que tipo de dor", "a dor é em queimação",
        "é uma dor em facada", "a dor é pulsátil", "a dor é constante",
        "é uma dor em peso", "a dor é latejante", "a dor é em cólica",
        "como você descreveria a dor", "é uma dor forte ou fraca",
        "a dor é aguda ou crônica", "qual a intensidade da dor",
        "a dor é contínua ou intermitente", "quanto incomoda de 0 a 10",
    ],
 
    "Fator_melhora": [
        "o que melhora", "quando melhora", "algo que alivia",
        "o que ajuda a melhorar", "algum remédio que melhora",
        "a dor melhora com repouso", "o que faz você se sentir melhor",
        "a dor alivia com alguma posição", "o que ajuda",
        "tem algo que melhora o sintoma", "o que atenua",
    ],
 
    "Fator_piora": [
        "o que piora", "quando piora", "algo que agrava",
        "o que deixa pior", "a dor piora com movimento",
        "a dor piora à noite", "o que agrava os sintomas",
        "tem algo que deixa pior", "o que intensifica a dor",
        "a dor piora com esforço", "piora em algum momento do dia",
    ],
 
    "Sintomas_associados": [
        "outros sintomas", "tem mais sintomas", "além disso sente mais",
        "há outros sintomas associados", "outras manifestações",
        "tem sintomas relacionados", "mais algum sintoma",
        "quais sintomas acompanham", "além da queixa mais sintomas",
        "quais outros sintomas surgem", "sente algo mais além",
        "há sintomas de perda de peso", "você sente mal-estar além",
        "quais sintomas aparecem junto", "tem sintomas extras",
    ],
 
    "Sintomas_negados": [
        "teve febre", "febre nos últimos meses", "está com febre atualmente",
        "apresentou febre alta", "perdeu muito peso",
        "perdeu peso de forma acentuada", "houve emagrecimento",
        "teve sintomas infecciosos", "apresentou gripe ou resfriado",
        "teve contato com doentes", "houve infecções respiratórias",
        "teve sudorese noturna", "houve viroses recentes",
        "teve internações por infecção", "sente dor torácica",
        "teve náusea", "refere vômitos", "alteração intestinal",
        "teve dor nas articulações", "refere artralgia",
        "tem dor muscular", "refere mialgia",
    ],
 
    "historico": [
        "já teve alguma doença importante", "tem algum problema de saúde",
        "quais doenças já teve", "sofreu de alguma condição médica",
        "já foi internado", "precisou de hospitalização", "já passou por cirurgia",
        "teve alguma operação", "doenças crônicas", "comorbidades",
        "tem doença de base", "já foi hospitalizado", "antecedentes médicos",
        "já teve doença grave", "fez tratamento para alguma condição",
        "recebeu diagnóstico de câncer", "já passou por transfusão de sangue",
        "condição prévia", "problema de saúde anterior", "diagnóstico anterior",
    ],
 
    "familia": [
        "alguma doença na família", "seus pais têm algum problema de saúde",
        "há histórico de câncer na família", "seus irmãos têm doenças",
        "existe alguém com problemas de sangue na família",
        "já ouviu falar de anemias hereditárias", "alguém com leucemia",
        "tem parente com doença autoimune", "tem casos de trombose na família",
        "histórico familiar", "família", "familiar", "hereditário",
        "pai", "mãe", "irmão", "parentes", "genético",
        "avó", "avô", "parentes com doença", "doenças hereditárias",
        "há antecedentes de câncer", "alguém teve problema semelhante",
        "há doenças no sangue na família",
    ],
 
    "habitos": [
        "você fuma", "já fumou", "é tabagista", "você bebe",
        "consome álcool", "faz uso de drogas", "já experimentou drogas",
        "pratica atividade física", "você se exercita", "se alimenta bem",
        "segue alguma dieta", "tem vida sedentária", "dorme bem",
        "seu sono é regular", "hábitos saudáveis", "tem rotina saudável",
        "ingere álcool", "bebe bebidas alcoólicas", "faz caminhada",
        "pratica esportes", "faz musculação", "tem hábitos de saúde",
        "cuida da alimentação", "tem sono regular",
    ],
 
    "remedios": [
        "você toma algum medicamento", "usa remédios todos os dias",
        "faz uso contínuo de algum remédio", "está em tratamento",
        "quais comprimidos você usa", "está em uso de algum suplemento",
        "usa vitaminas", "usa anabolizante", "está em tratamento com anticoagulante",
        "usa remédio para pressão", "usa medicação para diabetes",
        "já fez uso de corticoide", "usa antibióticos atualmente",
        "está tomando anti-inflamatório", "toma medicação sem receita",
        "remédios", "medicamentos", "toma algum remédio", "medicação",
        "faz uso de", "usa algum", "comprimido", "prescrito", "receita",
        "usa suplemento", "toma vitaminas", "usa algum tratamento",
    ],
 
    "vacinas": [
        "suas vacinas estão em dia", "tomou todas as vacinas",
        "carteira de vacinação está completa", "está com as vacinas atualizadas",
        "já recebeu todas as vacinas recomendadas", "caderneta de vacinas",
        "tomou vacina contra hepatite", "recebeu vacina de tétano",
        "já tomou vacina de febre amarela", "tomou vacina contra sarampo",
        "vacina contra covid", "esquema completo de poliomielite",
        "vacinado contra gripe", "imunizações em dia", "está imunizado",
        "tem registro de vacinas", "está vacinado",
    ],
 
    "epidemiologia": [
        "qual é a sua profissão", "você trabalha em que área",
        "com o que você trabalha", "seu trabalho envolve esforço físico",
        "já esteve exposto a produtos químicos", "tem contato com metais pesados",
        "lida com radiação", "viaja muito a trabalho", "já morou em outro lugar",
        "tem contato com pacientes doentes", "já trabalhou em hospital",
        "foi exposto a sangue ou secreções", "já sofreu acidente de trabalho",
        "já teve contato com agrotóxicos", "você mora em área urbana",
        "reside em zona rural", "teve contato com enchentes",
        "já fez viagens internacionais", "visitou áreas com malária",
        "esteve em área com febre amarela", "teve contato com animais",
        "profissão", "emprego", "onde trabalha", "área de atuação",
    ],
 
    "cumprimento": [
        "oi", "bom dia", "boa tarde", "boa noite", "olá", "oii", "opa",
        "tudo bem", "tudo certo", "como vai", "prazer",
        "seja bem-vindo", "bem vindo", "como posso ajudar",
        "estou aqui para te ouvir", "pode entrar", "pode sentar",
        "como posso ser útil", "está confortável",
    ],

    "febre": [
        "tem febre", "está com febre", "temperatura", "febre alta", "febril",
        "calafrio", "calafrios", "termômetro", "temperatura corporal",
        "febre baixa", "hipertermia", "está com febre atualmente",
        "houve febre durante o quadro", "apresentou febre alta",
        "teve febre nos últimos meses", "febre vespertina",
        "febre que vai e volta", "episódios febris",
    ],
 
    "dor": [
        "sente dor", "tem dor", "onde dói", "dor em algum lugar", "dores",
        "dor de cabeça", "cefaleia", "dor no corpo", "dor abdominal",
        "artralgia", "mialgia", "dor nas juntas", "dor nos ossos",
        "dor retro-orbital", "dor atrás dos olhos", "dor ao engolir",
        "dor lombar", "dor torácica", "dor no peito",
    ],
 
    "cansaco": [
        "está cansado", "fadiga", "sem energia", "exausto",
        "cansaço", "indisposição", "moleza", "sem disposição",
        "prostrado", "astenia", "se sente fraco", "sem fôlego",
        "cansaço ao subir escadas", "dispneia aos esforços",
        "não consegue se exercitar", "fica sem ar", "falta de ar ao esforço",
    ],
 
    "manchas": [
        "manchas", "hematomas", "equimoses", "petéquias", "roxo", "roxos",
        "mancha roxa", "sangramento na pele", "lesão na pele",
        "equimose", "púrpura", "manchas aparecem do nada",
        "manchas sem trauma", "manchas espontâneas",
        "manchas pelo corpo", "lesão cutânea",
    ],
 
    "sangramento": [
        "sangramento", "sangra", "sangrou", "hemorragia", "sangue",
        "gengiva sangrando", "nariz sangrando", "epistaxe",
        "hematúria", "melena", "sangue nas fezes",
        "sangramento que não para", "sangramento prolongado",
        "sangramento espontâneo", "sangramento ao escovar",
        "menstruação intensa", "menorragia",
    ],
 
    "alimentacao": [
        "alimentação", "dieta", "come", "o que come", "hábitos alimentares",
        "vegetariano", "vegano", "carne", "alimenta", "nutrição", "comida",
        "restrição alimentar", "come pouca carne", "faz dieta",
    ],
 
    "urina": [
        "urina", "xixi", "cor da urina", "urina escura", "urinando",
        "hemoglobinúria", "urina avermelhada", "urina marrom", "micção",
        "urina cor de coca-cola", "urina ao acordar",
        "urina fica mais escura de manhã", "hematúria",
    ],
 
    "transfusao": [
        "transfusão", "recebeu sangue", "bolsa de sangue", "hemoterapia",
        "durante a transfusão", "após a transfusão", "hemocomponente",
        "recebeu transfusão", "quantas bolsas", "já recebeu sangue",
        "reação à transfusão", "durante a infusão do sangue",
    ],
 
    "fraqueza": [
        "fraqueza muscular", "não consegue andar", "dificuldade de andar",
        "pernas fracas", "força muscular", "paresia", "paralisia",
        "fraqueza nas pernas", "fraqueza nos braços",
        "dificuldade de subir escadas", "fraqueza progressiva",
        "perda de força", "fraqueza distal", "fraqueza proximal",
    ],
 
    "movimentos": [
        "movimentos involuntários", "tremor", "espasmo", "convulsão",
        "movimentos anormais", "discinesia", "coreia", "ataxia",
        "não consigo controlar os movimentos", "braços se movem sozinhos",
    ],
 
    "neurologico": [
        "neurológico", "dificuldade de falar", "confusão mental", "memória",
        "formigamento", "parestesia", "dormência", "reflexos", "fala",
        "disartria", "dificuldade de pronunciar", "perda de memória",
        "está esquecendo as coisas", "cognição",
    ],
 
    "visao": [
        "visão", "enxerga", "vista", "olho", "embaçado", "visão turva",
        "perda de visão", "diplopia", "visão embaçada",
        "dificuldade para ver", "papiledema",
    ],
 
    "barriga": [
        "barriga", "abdômen inchado", "abdome", "distensão abdominal",
        "barriga crescendo", "ascite", "líquido na barriga",
        "barriga foi crescendo", "médico disse que tem líquido",
        "baço inchado", "fígado inchado", "hepatomegalia", "esplenomegalia",
    ],
 
    "pele": [
        "pele", "alteração na pele", "mancha escura", "hiperpigmentação",
        "pelos", "hirsutismo", "lesão cutânea", "dermatose",
        "manchas escuras no corpo", "pelos ficaram mais grossos",
        "lesões na pele", "sarcoma de kaposi",
    ],
 
    "carocos": [
        "caroço", "nódulo", "inchaço no pescoço", "íngua", "linfadenopatia",
        "linfonodo", "gânglio", "massa palpável", "tumor",
        "caroço no pescoço", "caroço que cresce", "linfonodos aumentados",
        "íngua que não passa", "gânglios inflamados",
    ],
 
    "emagrecimento": [
        "emagreceu", "perdeu peso", "perda de peso", "emagreci",
        "emagrecimento", "peso diminuiu", "ficou magro",
        "emagreci sem fazer dieta", "perdi apetite", "inapetência",
    ],
 
    "hiv": [
        "HIV", "aids", "soropositivo", "retrovírus", "CD4", "antirretroviral",
        "imunidade", "imunodeficiência", "parou o antirretroviral",
        "faz tratamento para HIV", "carga viral",
    ],
 
    "falta_ar": [
        "falta de ar", "dispneia", "dificuldade de respirar", "sem fôlego",
        "ofegante", "taquipneia", "sufocando", "não consegue respirar",
        "ficou roxo sem ar", "respiração difícil", "falta de ar intensa",
        "não conseguia respirar", "saturação de oxigênio",
    ],
 
    "pressao": [
        "pressão alta", "hipertensão", "pressão arterial", "pressão subiu",
        "pressão sanguínea", "pressão foi de", "pressão elevada",
    ],
 
    "melhora": [
        "melhorou", "como melhorou", "o que ajudou", "tratamento ajudou",
        "melhora após", "remissão dos sintomas", "o que aliviou",
        "melhorou depois que", "o que fez melhorar",
    ],
 
    "engolir": [
        "engolir", "disfagia", "dificuldade para engolir", "dor ao engolir",
        "deglutição", "sente algo na garganta ao engolir",
    ],
 
    "alergia": [
        "alergia", "alérgico", "reação alérgica", "intolerância",
        "hipersensibilidade", "tem alergias conhecidas",
    ],
 
    "infeccoes": [
        "infecções", "infecção", "infecções frequentes", "fica doente sempre",
        "imunidade baixa", "suscetível a infecções",
        "já teve 3 infecções no mês", "infecções recorrentes",
        "pneumonia recente", "amigdalite frequente",
    ],
 
    "naosei": [
        "futebol", "time", "torce para qual time", "presidente", "política",
        "qual o melhor time", "vota em quem",
    ],
 
    "default": [
        "não entendi", "pode repetir", "outra coisa", "mais alguma coisa",
        "tenho mais sintomas", "o que mais", "qualquer coisa", "?",
        "não sei", "não tenho certeza",
    ],
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