#logica/test_chat_terminal.py

from chatbot import responder_chat
from pacientes import PACIENTES

print("ARQUIVO CARREGADO", flush=True)

def testar_paciente():
    print("\n[INFO] Carregando sistema... Aguarde.", flush=True) 
    print("=== SIMULADOR DE CONSULTA MÉDICA ===\n", flush=True)

    print("Pacientes disponíveis:")
    for p in PACIENTES:
        print(f"{p['id']} - {p['nome']} ({p['diagnostico_correto']})")

    paciente_id = int(input("\nEscolha o paciente (ID): "))
    paciente = next(p for p in PACIENTES if p["id"] == paciente_id)

    try:
        paciente_id = int(input("\nEscolha o paciente (ID): "))
        paciente = next(p for p in PACIENTES if p["id"] == paciente_id)
    except (ValueError, StopIteration):
        print("Paciente não encontrado!")
        return

    print("Digite 'diagnostico: <doença>' para finalizar\n")

    while True:
        mensagem = input("👨‍⚕️ Você: ")

        if mensagem.lower().startswith("diagnostico:"):
            resposta = responder_chat(paciente, mensagem)
            print(f"🧑‍🦱 Paciente: {resposta}")
            break

        resposta = responder_chat(paciente, mensagem)
        print(f"🧑‍🦱 Paciente: {resposta}")

if __name__ == "__main__":
    testar_paciente()