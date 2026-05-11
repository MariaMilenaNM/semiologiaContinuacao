---
title: HemPocket Backend
emoji: 🩺
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# HemPocket Backend

API Flask para o simulador clínico HemPocket. O frontend (estático) roda no GitHub Pages.

## Endpoints

| Método | Rota | Função |
|---|---|---|
| GET  | `/api/patients`              | Lista pacientes (sem spoilers) |
| GET  | `/api/patients/<id>`         | Detalhes de um paciente |
| POST | `/api/chat`                  | Classifica intenção e devolve resposta do paciente |
| POST | `/api/diagnosis`             | Valida o diagnóstico do estudante |
| GET  | `/api/exam/<id>/<region>`    | Achados do exame físico |

## Variáveis de ambiente

| Nome | Default | Descrição |
|---|---|---|
| `PORT` | `7860` | Porta HTTP exposta pelo gunicorn |
| `FRONTEND_ORIGIN` | `*` | Origin permitido no CORS. Em produção, restrinja para a URL do GitHub Pages |
| `FLASK_DEBUG` | `0` | Liga o modo debug do Flask (NÃO use em produção) |

Configure `FRONTEND_ORIGIN` no painel **Settings → Variables and secrets** do Space.
