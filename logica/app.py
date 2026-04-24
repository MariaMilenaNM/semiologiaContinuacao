"""
app.py
------
Servidor Flask do AnamneseApp.

Rotas:
  GET  /api/patients            → lista todos os pacientes
  GET  /api/patients/<id>       → retorna um paciente pelo ID
  POST /api/chat                → recebe pergunta, retorna resposta do paciente
  POST /api/diagnosis           → valida o diagnóstico do estudante
  GET  /api/exam/<patient_id>/<region> → retorna achados do exame físico

Execute:
  python train.py   (apenas na primeira vez)
  python app.py
"""

import os
import json

from flask import Flask, request, jsonify
from flask_cors import CORS

from model import load_artifacts, predict_intent

# ── App ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # permite requisições do frontend (diferente porta)

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PATIENTS_PATH = os.path.join(BASE_DIR, "patients.json")

# ── Carrega dados e modelo na inicialização ─────────────────────────────
with open(PATIENTS_PATH, "r", encoding="utf-8") as f:
    PATIENTS: list[dict] = json.load(f)

PATIENTS_BY_ID: dict[int, dict] = {p["id"]: p for p in PATIENTS}

print("Carregando modelo de ML...")
try:
    ml_model, vectorizer, label_names = load_artifacts()
    ML_READY = True
    print("✅ Modelo carregado com sucesso.")
except Exception as e:
    ML_READY = False
    print(f"⚠️  Modelo não encontrado: {e}")
    print("   Execute `python train.py` para treinar o modelo antes de usar o chat.")


# ════════════════════════════════════════════════════════════════════════
# ROTAS
# ════════════════════════════════════════════════════════════════════════

@app.route("/api/patients", methods=["GET"])
def get_patients():
    """Retorna a lista de pacientes (sem dados sensíveis de resposta)."""
    safe_fields = ["id", "name", "age", "gender", "color", "emoji",
                   "disease_label", "intro", "symptoms"]
    result = [{k: p[k] for k in safe_fields} for p in PATIENTS]
    return jsonify(result)


@app.route("/api/patients/<int:patient_id>", methods=["GET"])
def get_patient(patient_id: int):
    """Retorna um paciente pelo ID (sem a resposta correta oculta)."""
    patient = PATIENTS_BY_ID.get(patient_id)
    if not patient:
        return jsonify({"error": "Paciente não encontrado."}), 404

    safe_fields = ["id", "name", "age", "gender", "color", "emoji",
                   "disease_label", "intro", "symptoms"]
    return jsonify({k: patient[k] for k in safe_fields})


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Recebe a mensagem do estudante e retorna a resposta do paciente.

    Body JSON:
        {
          "patient_id": 1,
          "message": "O paciente tem febre?"
        }

    Retorna:
        {
          "intent": "febre",
          "response": "Não tenho febre, só essas manchas e o sangramento.",
          "confidence": 0.87
        }
    """
    if not ML_READY:
        return jsonify({
            "error": "Modelo de ML não treinado. Execute `python train.py`."
        }), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Corpo da requisição inválido."}), 400

    patient_id: int = data.get("patient_id")
    message: str    = data.get("message", "").strip()

    if not patient_id or not message:
        return jsonify({"error": "Campos 'patient_id' e 'message' são obrigatórios."}), 400

    patient = PATIENTS_BY_ID.get(patient_id)
    if not patient:
        return jsonify({"error": "Paciente não encontrado."}), 404

    # ── Classificação da intenção via Keras ─────────────────────────────
    intent = predict_intent(message, ml_model, vectorizer, label_names)

    # ── Busca a resposta no JSON do paciente ────────────────────────────
    responses: dict = patient.get("responses", {})
    response_text   = responses.get(intent) or responses.get("default", "...")

    return jsonify({
        "intent":   intent,
        "response": response_text,
    })


@app.route("/api/diagnosis", methods=["POST"])
def check_diagnosis():
    """
    Valida o diagnóstico informado pelo estudante.

    Body JSON:
        {
          "patient_id": 1,
          "diagnosis": "pti"
        }

    Retorna:
        {
          "correct": true,
          "disease": "pti",
          "disease_label": "Púrpura Trombocitopênica Imune (PTI)",
          "feedback": "Parabéns! Diagnóstico correto."
        }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Corpo da requisição inválido."}), 400

    patient_id: int   = data.get("patient_id")
    diagnosis: str    = (data.get("diagnosis") or "").strip().lower()

    if not patient_id or not diagnosis:
        return jsonify({"error": "Campos 'patient_id' e 'diagnosis' são obrigatórios."}), 400

    patient = PATIENTS_BY_ID.get(patient_id)
    if not patient:
        return jsonify({"error": "Paciente não encontrado."}), 404

    correct       = (diagnosis == patient["disease"])
    disease_label = patient["disease_label"]

    feedback = (
        f"✅ Correto! O diagnóstico é {disease_label}."
        if correct else
        "❌ Diagnóstico incorreto. Revise os sintomas e o exame físico."
    )

    return jsonify({
        "correct":       correct,
        "disease":       patient["disease"] if correct else None,
        "disease_label": disease_label      if correct else None,
        "feedback":      feedback,
    })


@app.route("/api/exam/<int:patient_id>/<region>", methods=["GET"])
def get_exam(patient_id: int, region: str):
    """
    Retorna os achados do exame físico de uma região.

    Regiões válidas: head, chest, abdomen, arm, knee, ankle
    """
    patient = PATIENTS_BY_ID.get(patient_id)
    if not patient:
        return jsonify({"error": "Paciente não encontrado."}), 404

    exam    = patient.get("exam", {})
    finding = exam.get(region)

    if finding is None:
        return jsonify({"error": f"Região '{region}' não encontrada."}), 404

    has_finding = "sem alterações" not in finding.lower()

    return jsonify({
        "region":      region,
        "finding":     finding,
        "has_finding": has_finding,
    })


# ── Dev server ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)