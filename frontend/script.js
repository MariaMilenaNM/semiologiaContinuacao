/* ══════════════════════════════════════
   script.js – AnamneseApp
   Consome a API Flask em http://localhost:5000
══════════════════════════════════════ */

const API = "http://localhost:5000/api";

/* Regiões do exame físico */
const EXAM_LABELS = {
  head:    "Cabeça / Face",
  chest:   "Tórax / Pulmões",
  abdomen: "Abdômen",
  arm:     "Membros Superiores",
  knee:    "Joelhos",
  ankle:   "Tornozelos",
};

/* Perguntas rápidas exibidas no chat */
const QUICK_QUESTIONS = [
  "O paciente tem febre?",
  "Tem alguma dor?",
  "Está cansado ou com fadiga?",
  "Tem manchas ou sangramentos?",
  "Faz uso de algum remédio?",
  "Tem histórico de doenças?",
  "Histórico familiar relevante?",
];

/* ── Estado ── */
let currentPatient = null;
let patients       = [];       // carregado da API

/* ══════════════════════════════════════
   INICIALIZAÇÃO
══════════════════════════════════════ */
async function init() {
  showLoading(true);
  try {
    const res  = await fetch(`${API}/patients`);
    patients   = await res.json();
    renderPatients();
  } catch (err) {
    document.getElementById("patientsGrid").innerHTML =
      `<p style="color:var(--accent);grid-column:1/-1;text-align:center;padding:20px">
        ⚠️ Não foi possível conectar ao servidor.<br>
        <small>Certifique-se de que o Flask está rodando em localhost:5000</small>
      </p>`;
  } finally {
    showLoading(false);
  }
}

/* ══════════════════════════════════════
   SCREEN 1 – Lista de Pacientes
══════════════════════════════════════ */
function renderPatients() {
  const grid = document.getElementById("patientsGrid");
  grid.innerHTML = "";

  patients.forEach((p) => {
    const div = document.createElement("div");
    div.className = "patient-card";

    div.innerHTML = `
      <div class="patient-avatar" style="background:${p.color}">${p.emoji}</div>
      <div class="patient-name">${p.name}</div>
      <div class="patient-tag">${p.age} anos</div>
    `;

    div.onclick = () => openChat(p);
    grid.appendChild(div);
  });
}

/* ══════════════════════════════════════
   NAVEGAÇÃO ENTRE TELAS
══════════════════════════════════════ */
function goTo(screenNumber) {
  document.querySelectorAll(".screen").forEach((s) =>
    s.classList.remove("active")
  );
  document.getElementById("screen" + screenNumber).classList.add("active");
}

/* ══════════════════════════════════════
   SCREEN 2 – Chat
══════════════════════════════════════ */
function openChat(patient) {
  currentPatient = patient;

  /* Cabeçalho */
  document.getElementById("chatTitle").textContent        = patient.name;
  document.getElementById("chatName").textContent         = patient.name;
  document.getElementById("chatAvatar").textContent       = patient.emoji;
  document.getElementById("chatAvatar").style.background  = patient.color;
  document.getElementById("diagnosisSelect").value        = "";

  /* Sintomas como chips */
  const symptomsWrap = document.getElementById("symptomsWrap");
  if (symptomsWrap && patient.symptoms) {
    symptomsWrap.innerHTML = patient.symptoms
      .map((s) => `<span class="symptom-chip">${s}</span>`)
      .join("");
  }

  /* Limpa mensagens e exibe intro */
  document.getElementById("chatArea").innerHTML = "";
  addMessage("patient", patient.intro);

  /* Perguntas rápidas */
  const qo = document.getElementById("quickOptions");
  qo.innerHTML = "";
  QUICK_QUESTIONS.forEach((q) => {
    const btn = document.createElement("button");
    btn.className   = "quick-btn";
    btn.textContent = q;
    btn.onclick = () => {
      document.getElementById("chatInput").value = q;
      sendMsg();
    };
    qo.appendChild(btn);
  });

  goTo(2);
}

/* Adiciona bolha de mensagem na área de chat */
function addMessage(role, text) {
  const area = document.getElementById("chatArea");
  const wrap = document.createElement("div");
  wrap.className = "msg " + role;

  const avatarEl = document.createElement("div");
  avatarEl.className = "msg-avatar";

  if (role === "patient") {
    avatarEl.textContent        = currentPatient.emoji;
    avatarEl.style.background   = currentPatient.color;
  } else {
    avatarEl.textContent        = "🎓";
    avatarEl.style.background   = "#64748b";
  }

  const bubble = document.createElement("div");
  bubble.className   = "msg-bubble";
  bubble.textContent = text;

  wrap.appendChild(avatarEl);
  wrap.appendChild(bubble);
  area.appendChild(wrap);
  area.scrollTop = area.scrollHeight;
}

/* Indicador de "digitando..." */
function showTyping() {
  const area = document.getElementById("chatArea");
  const wrap = document.createElement("div");
  wrap.className = "msg patient";
  wrap.id = "typingIndicator";

  const avatarEl = document.createElement("div");
  avatarEl.className        = "msg-avatar";
  avatarEl.textContent      = currentPatient.emoji;
  avatarEl.style.background = currentPatient.color;

  const bubble = document.createElement("div");
  bubble.className   = "msg-bubble typing-dots";
  bubble.innerHTML   = "<span></span><span></span><span></span>";

  wrap.appendChild(avatarEl);
  wrap.appendChild(bubble);
  area.appendChild(wrap);
  area.scrollTop = area.scrollHeight;
}

function removeTyping() {
  const el = document.getElementById("typingIndicator");
  if (el) el.remove();
}

/* Envia mensagem → API → resposta do paciente */
async function sendMsg() {
  const input = document.getElementById("chatInput");
  const text  = input.value.trim();
  if (!text || !currentPatient) return;

  addMessage("student", text);
  input.value = "";
  showTyping();

  try {
    const res  = await fetch(`${API}/chat`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({
        patient_id: currentPatient.id,
        message:    text,
      }),
    });
    const data = await res.json();
    removeTyping();
    addMessage("patient", data.response || "...");
  } catch (err) {
    removeTyping();
    addMessage("patient", "Erro ao conectar com o servidor.");
  }
}

/* Valida diagnóstico → API */
async function checkDiagnosis() {
  const val = document.getElementById("diagnosisSelect").value;
  if (!val) { showToast("Selecione uma doença primeiro!", ""); return; }
  if (!currentPatient)  { showToast("Nenhum paciente selecionado.", ""); return; }

  try {
    const res  = await fetch(`${API}/diagnosis`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({
        patient_id: currentPatient.id,
        diagnosis:  val,
      }),
    });
    const data = await res.json();

    if (data.correct) {
      showToast("✅ Diagnóstico correto! Parabéns!", "correct");
      addMessage("patient", `(Sistema) ${data.feedback}`);
    } else {
      showToast("❌ Diagnóstico incorreto. Tente novamente.", "wrong");
      addMessage("patient", `(Sistema) ${data.feedback}`);
    }
  } catch (err) {
    showToast("⚠️ Erro ao verificar diagnóstico.", "");
  }
}

/* ══════════════════════════════════════
   SCREEN 3 – Avaliação Física
══════════════════════════════════════ */
async function showExam(region) {
  /* Remove destaques anteriores */
  document.querySelectorAll(".hotspot").forEach((h) =>
    h.classList.remove("active", "finding")
  );

  const panel = document.getElementById("examPanel");
  panel.innerHTML = `<h4>${EXAM_LABELS[region] || region}</h4><p>Carregando...</p>`;

  if (!currentPatient) {
    panel.innerHTML = `
      <h4>Sem paciente</h4>
      <p>Volte e selecione um paciente primeiro.</p>`;
    return;
  }

  try {
    const res  = await fetch(`${API}/exam/${currentPatient.id}/${region}`);
    const data = await res.json();

    if (data.error) throw new Error(data.error);

    panel.innerHTML = `
      <h4>${EXAM_LABELS[region] || region}</h4>
      <p>${data.finding}</p>
      <span class="exam-badge ${data.has_finding ? "alert" : ""}">
        ${data.has_finding ? "⚠ Achado relevante" : "✓ Normal"}
      </span>`;

    /* Ativa hotspots */
    document.querySelectorAll(".hotspot").forEach((h) => {
      const match =
        h.id === "hs-" + region ||
        (region === "arm" && (h.id === "hs-arm-r" || h.id === "hs-arm-l"));

      if (match) {
        h.classList.add("active");
        if (data.has_finding) h.classList.add("finding");
      }
    });
  } catch (err) {
    panel.innerHTML = `
      <h4>${EXAM_LABELS[region] || region}</h4>
      <p style="color:var(--accent)">⚠️ Erro ao carregar achados: ${err.message}</p>`;
  }
}

/* ══════════════════════════════════════
   UTILITÁRIOS
══════════════════════════════════════ */
function showLoading(visible) {
  const grid = document.getElementById("patientsGrid");
  if (visible) {
    grid.innerHTML =
      `<div style="grid-column:1/-1;text-align:center;padding:32px;color:var(--text2)">
        Carregando pacientes...
       </div>`;
  }
}

function showToast(message, type) {
  const t   = document.getElementById("toast");
  t.textContent = message;
  t.className   = "toast show " + type;
  setTimeout(() => { t.className = "toast"; }, 2800);
}

/* ── Start ── */
init();