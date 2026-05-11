# Atualizando o backend (Hugging Face Space)

Esse projeto tem o backend rodando como um Docker Space no Hugging Face.
O código que o Space executa fica na pasta `logica/` deste repositório. Quando você quer
publicar uma mudança, **empurra essa pasta como se fosse a raiz** de outro repo (o do HF Space)
usando `git subtree`.

> 📌 Os comandos abaixo usam placeholders:
> - `<HF_USER>` → seu username (ou da org) no Hugging Face
> - `<SPACE_NAME>` → o nome do Space (ex: `meuapp-backend`)
>
> Confira esses valores no `frontend/script.js` (linha do `const API = ...`)
> ou pergunte para quem mantém o projeto.

---

## 🛠 Setup (uma vez por máquina)

### 1. Clone o repositório e mude para o branch `live`

```bash
git clone <URL_DO_REPOSITORIO_GITHUB>
cd <NOME_DA_PASTA>
git checkout live
```

### 2. Adicione o remote do Hugging Face

```bash
git remote add hf https://huggingface.co/spaces/<HF_USER>/<SPACE_NAME>
```

Confirme com `git remote -v` — deve listar `origin` (GitHub) **e** `hf` (Hugging Face).

### 3. Crie um access token no HF (precisa permissão **write**)

1. Acesse https://huggingface.co/settings/tokens
2. **Create new token** → tipo **Write** → nome qualquer (ex: `deploy`)
3. Copie o valor (`hf_xxxxx...`) — só aparece uma vez.

> ⚠️ Se você não consegue criar token porque não é dono do Space,
> peça para o dono te adicionar como membro da org dele.
> Spaces sob conta pessoal **não aceitam colaborador direto** — só via organização.

---

## 🚀 Como atualizar o backend

Toda vez que você mexer em algo dentro de `logica/`:

### 1. Teste local com Docker (recomendado)

```bash
docker compose up -d --build
```

Abre `http://localhost:8000` no navegador, exercita o que mudou. Pra derrubar:

```bash
docker compose down
```

> 💡 Se o build local quebrar, o build no HF também vai quebrar.
> Resolva localmente antes de pushar.

### 2. Commit no GitHub

```bash
git add logica/
git commit -m "descreva o que mudou"
git push origin live
```

### 3. Deploy no Hugging Face

```bash
git push hf $(git subtree split --prefix=logica HEAD):main --force
```

Quando aparecer o prompt:
- **Username**: seu usuário HF
- **Password**: cole o token `hf_...` (não vai aparecer enquanto digita)

O HF detecta o push e **rebuilda o container automaticamente** em ~3–8 minutos.

---

## 👀 Acompanhando o build

Abra no navegador:

```
https://huggingface.co/spaces/<HF_USER>/<SPACE_NAME>
```

Aba **Logs** mostra em tempo real:
- `Building` → instalando dependências, rodando `train.py`
- `Running` ✅ → backend respondendo em `https://<HF_USER>-<SPACE_NAME>.hf.space`

Pra testar via terminal:

```bash
curl https://<HF_USER>-<SPACE_NAME>.hf.space/api/patients
```

Se retornar JSON dos pacientes, está no ar.

---

## ❗ Problemas comuns

### "Updates were rejected ... non-fast-forward"
Alguém pushou pelo painel do HF ou outro caminho. Repita o comando da etapa 3 —
ele já usa `--force` e resolve.

### Build falha no HF
Abra a aba **Logs** do Space. Erros frequentes:
- Pacote novo em `logica/requirements.txt` que não existe na PyPI.
- Mudança em `logica/intentsH5.json` que quebrou o `python train.py`.
- Erro de sintaxe em algum arquivo `.py`.

Reforço: rode `docker compose up --build` antes de pushar.

### Push pediu credencial e eu não tenho
O token salvo no Keychain pode ter expirado. Gera outro em
https://huggingface.co/settings/tokens e cola na próxima vez que pedir senha.

### Pushei e o HF não rebuildou
Confirme que o push foi pro remote `hf` (não pro `origin`):

```bash
git remote -v
```

Se faltar a linha do `hf`, refaça o passo 2 do Setup.

### Build passou mas o site não responde
O Space pode estar **dormindo** (free tier dorme após 48h sem tráfego).
A primeira request leva ~30–60s pra acordar; as próximas voltam ao normal.

---

## 📦 O que foi adicionado/modificado para funcionar no HF

Se você for olhar o histórico do projeto, perceberá que o backend original (Flask + Keras)
não tinha nada relacionado a Docker ou Hugging Face. Os arquivos abaixo foram **adicionados**
ou **modificados** especificamente para o deploy no Space funcionar.

### Arquivos novos em `logica/`

| Arquivo | Pra que serve |
|---|---|
| `Dockerfile` | Receita do container. Usa `python:3.11-slim`, cria usuário não-root, instala dependências, **roda `python train.py` no build** (treina o modelo dentro da imagem) e sobe o `gunicorn` na porta 7860. |
| `requirements.txt` | Dependências Python (Flask, flask-cors, TensorFlow, scikit-learn, gunicorn). Sem isso o HF não sabe o que instalar. |
| `README.md` (do Space) | Tem um **frontmatter YAML** no topo (`sdk: docker`, `app_port: 7860`, etc) que diz para o HF: "isso aqui é um Docker Space, escuta na porta 7860". É **obrigatório** pelo HF. |
| `.dockerignore` | Impede que o modelo treinado localmente (`saved_model/*.keras`, `*.pkl`) seja copiado pro container — assim o `train.py` no build sempre gera um modelo fresco. |

### Modificações em `logica/app.py`

| Antes | Depois | Por quê |
|---|---|---|
| `CORS(app)` (libera tudo) | `CORS(app, origins=[FRONTEND_ORIGIN])` lendo de env var | Permitir restringir em produção pra só o domínio do frontend |
| `app.run(debug=True, port=5000)` hardcoded | Lê `FLASK_DEBUG` e `PORT` de env var, default `debug=False` | `debug=True` em produção é falha grave de segurança (expõe console Python). O HF injeta `PORT` no container — o app precisa respeitar |

### Modificações em `frontend/script.js`

| Antes | Depois |
|---|---|
| `const API = "http://localhost:5000/api"` (fixo) | Detecta automaticamente: `localhost` → Flask local; senão → URL pública do HF Space |

Assim o mesmo `script.js` funciona em dev e em prod sem mexer no código.

### Arquivos na raiz do repositório

| Arquivo | Pra que serve |
|---|---|
| `docker-compose.yml` | Orquestra **backend + frontend (nginx)** localmente, simulando o ambiente do HF. Apenas para desenvolvimento — o HF não usa este arquivo. |

### Variáveis de ambiente que o backend respeita

Configurar no painel **Settings → Variables and secrets** do Space:

| Variável | Default | Para que |
|---|---|---|
| `PORT` | `7860` | Porta HTTP onde o gunicorn escuta (o HF normalmente já injeta isso) |
| `FRONTEND_ORIGIN` | `*` | Origin permitido no CORS. Em produção, defina como a URL do frontend |
| `FLASK_DEBUG` | `0` | **NUNCA** ligar em produção |

---

## 🧠 Por que `git subtree` em vez de só `git push`?

O HF Space é **outro repositório git**, com layout próprio: ele espera
`Dockerfile`, `app.py`, `requirements.txt` direto na raiz — não dentro de uma subpasta.

Neste repo, esses arquivos vivem em `logica/`. O comando

```bash
git subtree split --prefix=logica HEAD
```

cria um histórico paralelo onde o conteúdo de `logica/` aparece na raiz, e
`git push hf ... --force` empurra esse histórico para o Space.

Você não precisa entender os detalhes — basta usar a linha de comando da
etapa 3 sempre que for fazer deploy.
