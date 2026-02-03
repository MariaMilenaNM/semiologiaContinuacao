from flask import Flask, request, jsonify
from pacientes import PACIENTES
from chatbot import responder_chat, validar_diagnostico

app = Flask(__name__)

@app.route("/pacientes", methods=["GET"])
def listar_pacientes():
    return jsonify([
        {
            "id": p["id"],
            "nome": p["nome"],
            "idade": p["idade"]
        } for p in PACIENTES
    ])


@app.route("/consulta/<int:paciente_id>", methods=["GET"])
def iniciar_consulta(paciente_id):
    paciente = next((p for p in PACIENTES if p["id"] == paciente_id), None)

    if not paciente:
        return jsonify({"erro": "Paciente não encontrado"}), 404

    return jsonify({
        "mensagem": f"Olá, meu nome é {paciente['nome']}. Pode iniciar a consulta."
    })



@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    paciente_id = data["paciente_id"]
    mensagem = data["mensagem"]

    paciente = next((p for p in PACIENTES if p["id"] == paciente_id), None)

    if not paciente:
        return jsonify({"erro": "Paciente não encontrado"}), 404

    resposta = responder_chat(paciente, mensagem)

    return jsonify({"resposta": resposta})



@app.route("/diagnostico", methods=["POST"])
def diagnostico():
    data = request.json
    paciente_id = data["paciente_id"]
    diagnostico_usuario = data["diagnostico"]

    paciente = next((p for p in PACIENTES if p["id"] == paciente_id), None)

    if not paciente:
        return jsonify({"erro": "Paciente não encontrado"}), 404

    correto = validar_diagnostico(paciente, diagnostico_usuario)

    if correto:
        return jsonify({"resultado": "Parabéns, você acertou o diagnóstico!"})
    else:
        return jsonify({"resultado": "Diagnóstico incorreto. Tente novamente."})
    

if __name__ == "__main__":
    app.run(debug=True)

