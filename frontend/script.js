/* 
   script.js – AnamneseApp
   Consome a API Flask em http://localhost:5050
*/

//do jeito q hafif deixou, troquei pra minha url pra testar 
// "https://fikani-semiologiacontinuacao.hf.space/api" 
const API = location.hostname === "localhost" || location.hostname === "127.0.0.1"
  ? "http://localhost:5050/api"
  : "https://mariamilenanm-semiologiacontinuacao.hf.space/api" ;

const EXAM_LABELS = {
  head:    "Cabeça / Face",
  chest:   "Tórax / Pulmões",
  abdomen: "Abdômen",
  arm:     "Membros Superiores",
  knee:    "Joelhos",
  ankle:   "Tornozelos",
};

let currentPatient = null;
let patients       = [];  
let isFirstMessage = true;     

/* 
   INICIALIZAÇÃO E NAVEGAÇÃO
 */
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

function renderPatients() {
  const grid = document.getElementById("patientsGrid");
  grid.innerHTML = "";
  patients.forEach((p) => {
    const div = document.createElement("div");
    div.className = "patient-card";
    div.innerHTML = `
      <div class="patient-avatar" style="background:${p.color}">
        <img src="${p.emoji}" alt="${p.name}">
      </div>
      <div class="patient-name">${p.name}</div>
      <div class="patient-tag">${p.age} anos</div>
    `;
    div.onclick = () => openChat(p);
    grid.appendChild(div);
  });
}

function goTo(screenNumber) {
  document.querySelectorAll(".screen").forEach((s) => s.classList.remove("active"));
  document.getElementById("screen" + screenNumber).classList.add("active");
}

 
//   SCREEN 2 – Chat / Anamnese

function openChat(patient) {
  currentPatient = patient;
  isFirstMessage = true;

  document.getElementById("chatName").textContent         = patient.name;
  document.getElementById("chatAvatar").innerHTML         = `<img src="${patient.emoji}" alt="${patient.name}">`;
  document.getElementById("chatAvatar").style.background  = patient.color;
  
  // Reseta estado da tela de diagnóstico (Screen 5)
  document.getElementById("diagnosisSelect").value        = "";
  document.getElementById("diagnosisResult").style.display = "none";
  document.getElementById("navToPlan").style.display = "none";

  /*
  const symptomsWrap = document.getElementById("symptomsWrap");
  if (symptomsWrap && patient.signs) {
    symptomsWrap.innerHTML = patient.signs.map((s) => `<span class="symptom-chip">${s}</span>`).join("");
  }
*/
  document.getElementById("chatArea").innerHTML = "";
 
  goTo(2);
}

function addMessage(role, text) {
  const area = document.getElementById("chatArea");
  const wrap = document.createElement("div");
  wrap.className = "msg " + role;

  const avatarEl = document.createElement("div");
  avatarEl.className = "msg-avatar";

  if (role === "patient") {
    avatarEl.innerHTML          = `<img src="${currentPatient.emoji}" alt="${currentPatient.name}">`;
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

function showTyping() {
  const area = document.getElementById("chatArea");
  const wrap = document.createElement("div");
  wrap.className = "msg patient";
  wrap.id = "typingIndicator";

  const avatarEl = document.createElement("div");
  avatarEl.className        = "msg-avatar";
  avatarEl.innerHTML        = `<img src="${currentPatient.emoji}" alt="Digitando">`;
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

async function sendMsg() {
  const input = document.getElementById("chatInput");
  const text  = input.value.trim();
  if (!text || !currentPatient) return;

  addMessage("student", text);
  input.value = "";
  showTyping();

  if (isFirstMessage) {
    isFirstMessage = false; // Desativa a trava para as próximas mensagens
    
    // Simula um tempinho de digitação para a intro e retorna
    setTimeout(() => {
      removeTyping();
      addMessage("patient", currentPatient.intro);
    }, 800);
    
    return; // Para a execução aqui, não chama a API
  }

  try {
    const res  = await fetch(`${API}/chat`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ patient_id: currentPatient.id, message: text }),
    });
    const data = await res.json();
    removeTyping();
    addMessage("patient", data.response || "...");
  } catch (err) {
    removeTyping();
    addMessage("patient", "Erro ao conectar com o servidor.");
  }
}


 //  SCREEN 3 – Avaliação Física

async function showExam(region) {
  document.querySelectorAll(".hotspot").forEach((h) => h.classList.remove("active", "finding"));
  const panel = document.getElementById("examPanel");
  panel.innerHTML = `<h4>${EXAM_LABELS[region] || region}</h4><p>Carregando...</p>`;

  if (!currentPatient) {
    panel.innerHTML = `<h4>Sem paciente</h4><p>Volte e selecione um paciente primeiro.</p>`;
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

    document.querySelectorAll(".hotspot").forEach((h) => {
      const match = h.id === "hs-" + region || (region === "arm" && (h.id === "hs-arm-r" || h.id === "hs-arm-l"));
      if (match) {
        h.classList.add("active");
        if (data.has_finding) h.classList.add("finding");
      }
    });
  } catch (err) {
    panel.innerHTML = `<h4>${EXAM_LABELS[region] || region}</h4><p style="color:var(--accent)">⚠️ Erro ao carregar achados: ${err.message}</p>`;
  }
}


//   SCREEN 4 – Exames Laboratoriais

function goToLabScreen() {
  if (!currentPatient) {
    showToast("Selecione um paciente primeiro.", "wrong");
    return;
  }

  const container = document.getElementById("examListContainer");
  container.innerHTML = "";

  if (!currentPatient.lab_display || currentPatient.lab_display.length === 0) {
    container.innerHTML = "<div class='welcome-card'><p>Nenhum exame laboratorial disponível para este paciente.</p></div>";
    goTo(4);
    return;
  }

  const getOrdemCategoria = (nome) => {
    const n = nome.toLowerCase();
    if (n.includes("hemograma")) return 1;
    if (n.includes("esfregaço")) return 2;
    if (n.includes("coagulograma")) return 3;
    if (n.includes("imunologia")) return 4;
    return 5;
  };

  const examesOrdenados = [...currentPatient.lab_display].sort(
    (a, b) => getOrdemCategoria(a.categoria) - getOrdemCategoria(b.categoria)
  );

  examesOrdenados.forEach((categoria, idx) => {
    const hasAlert = categoria.exames.some((e) => e.alterado);

    // Botão clicável da categoria
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "exam-category-btn" + (hasAlert ? " has-alert" : "");
    btn.setAttribute("aria-expanded", "false");
    btn.innerHTML = `
      <span>${categoria.categoria}${hasAlert ? ' <span class="exam-alert-dot"></span>' : ""}</span>
      <span class="exam-toggle-icon">+</span>
    `;

    // Tabela (inicialmente oculta via max-height:0 no CSS)
    const wrapper = document.createElement("div");
    wrapper.className = "exam-collapse";
    wrapper.id = `examCat${idx}`;

    let tableHtml = `
      <table class="lab-table">
        <thead>
          <tr><th>Exame</th><th>Resultado</th><th>Referência</th></tr>
        </thead>
        <tbody>
    `;
    categoria.exames.forEach((exame) => {
      const valorClass = exame.alterado ? "lab-alert" : "";
      tableHtml += `
        <tr>
          <td>${exame.nome}</td>
          <td class="${valorClass}">${exame.valor}</td>
          <td><small>${exame.referencia}</small></td>
        </tr>`;
    });
    tableHtml += `</tbody></table>`;
    wrapper.innerHTML = tableHtml;

    btn.onclick = () => toggleExamCategory(btn, wrapper);

    container.appendChild(btn);
    container.appendChild(wrapper);
  });

  goTo(4);
}

function toggleExamCategory(btn, wrapper) {
  const isOpen = wrapper.classList.contains("open");

  if (isOpen) {
    wrapper.style.maxHeight = null;
    wrapper.classList.remove("open");
    btn.classList.remove("open");
    btn.setAttribute("aria-expanded", "false");
  } else {
    wrapper.classList.add("open");
    btn.classList.add("open");
    btn.setAttribute("aria-expanded", "true");
    wrapper.style.maxHeight = wrapper.scrollHeight + "px";
  }
}

//   SCREEN 5 – Verificação do Diagnóstico

async function checkDiagnosis() {
  const val = document.getElementById("diagnosisSelect").value;
  const resultDiv = document.getElementById("diagnosisResult");
  const navToPlan = document.getElementById("navToPlan");

  if (!val) { showToast("Selecione uma doença primeiro!", ""); return; }
  if (!currentPatient) { showToast("Nenhum paciente selecionado.", ""); return; }

  try {
    const res  = await fetch(`${API}/diagnosis`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ patient_id: currentPatient.id, diagnosis: val }),
    });
    const data = await res.json();

    resultDiv.style.display = "block";

    if (data.correct) {
      // Pega o fechamento que vem do JSON ou usa o feedback da API como fallback
      const textoFechamento = currentPatient.fechamento || data.feedback;

      resultDiv.innerHTML = `
        <div style="background:#d4edda; color:#155724; padding:16px; border-radius:12px; border: 1px solid #c3e6cb;">
          <div style="text-align: center; margin-bottom: 16px; font-size: 16px;">
            <strong>✅ Diagnóstico Correto!</strong>
          </div>
          
          <div style="display: flex; align-items: center; gap: 12px; background: rgba(255,255,255,0.6); padding: 12px; border-radius: 8px;">
            <div style="width: 55px; height: 55px; border-radius: 50%; background: ${currentPatient.color}; flex-shrink: 0; display: flex; justify-content: center; align-items: center; overflow: hidden; border: 2px solid #155724;">
              <img src="${currentPatient.emoji}" alt="${currentPatient.name}" style="width: 100%; height: 100%; object-fit: cover;">
            </div>
            <div>
              <strong style="font-size: 14px; display: block; margin-bottom: 4px;">${currentPatient.name}</strong>
              <p style="margin: 0; font-size: 14px; line-height: 1.4; color: #155724;">"${textoFechamento}"</p>
            </div>
          </div>
        </div>`;
        
      navToPlan.style.display = "flex";
    } else {
      resultDiv.innerHTML = `<div style="background:#f8d7da; color:#721c24; padding:16px; border-radius:12px; border: 1px solid #f5c6cb;">
        <strong>❌ Diagnóstico Incorreto.</strong><br><br>${data.feedback}
      </div>`;
      navToPlan.style.display = "none";
    }
  } catch (err) {
    showToast("Erro ao verificar diagnóstico.", "");
  }
}

// utilitários e modais

function showLoading(visible) {
  const grid = document.getElementById("patientsGrid");
  if (visible) {
    grid.innerHTML = `<div style="grid-column:1/-1;text-align:center;padding:32px;color:var(--text2)">Carregando pacientes...</div>`;
  }
}

function showToast(message, type) {
  const t = document.getElementById("toast");
  t.textContent = message;
  t.className   = "toast show " + type;
  setTimeout(() => { t.className = "toast"; }, 2800);
}

function openAboutModal() { document.getElementById("aboutModal").classList.add("active"); }
function closeAboutModal() { document.getElementById("aboutModal").classList.remove("active"); }

//Tutorial  
/* ═══════════════════════════════════════
   TUTORIAL / ONBOARDING SOB DEMANDA
═══════════════════════════════════════ */
const TUTORIAL_STEPS = [
  {
    screen: 1,
    selector: "#patientsGrid",
    title: "1. Escolha um paciente",
    text: "Toque em um dos cards para iniciar o atendimento simulado.",
  },
  {
    screen: 2,
    selector: "#chatInput",
    title: "2. Faça a anamnese",
    text: "Digite perguntas para o paciente, como em uma consulta real. Ex.: \"Qual o motivo da consulta?\"",
    beforeShow: () => { if (!currentPatient) fakeOpenChatForTutorial(); },
  },
  {
    screen: 2,
    selector: "#screen2 .btn-primary-large",
    title: "3. Avance para o Exame Físico",
    text: "Depois de conversar com o paciente, toque aqui para examiná-lo.",
  },
  {
    screen: 3,
    selector: "#bodyContainer",
    title: "4. Exame Físico",
    text: "Toque nas regiões destacadas do corpo para ver os achados do exame físico daquela área.",
  },
  {
    screen: 4,
    selector: "#examListContainer",
    title: "6. Análise de Exames",
    text: "Nesta tela ficam os exames. Toque nas categorias para abrir e fechar a tabela com os resultados detalhados.",
    beforeShow: () => { goToLabScreen(); }
  },
  {
    screen: 4,
    selector: "#screen4 .btn-primary-large",
    title: "7. Ir para Diagnóstico",
    text: "Com as informações da anamnese, exame físico e laboratorial em mãos, avance para a etapa final.",
  },
  {
    screen: 5,
    selector: "#diagnosisSelect",
    title: "8. Hipótese Diagnóstica",
    text: "Selecione a doença correspondente ao quadro do paciente e confirme para ver o resultado da simulação!",
  }
];

let tutorialIndex = 0;

function startTutorial() {
  tutorialIndex = 0;
  document.getElementById("tutorialOverlay").classList.add("active");
  showTutorialStep();
}

function endTutorial() {
  document.getElementById("tutorialOverlay").classList.remove("active");
}

function tutorialNext() {
  if (tutorialIndex < TUTORIAL_STEPS.length - 1) {
    tutorialIndex++;
    showTutorialStep();
  } else {
    endTutorial();
  }
}

function tutorialPrev() {
  if (tutorialIndex > 0) {
    tutorialIndex--;
    showTutorialStep();
  }
}

function fakeOpenChatForTutorial() {
  // Só para o tutorial poder destacar visualmente o input do chat,
  // sem exigir que o usuário já tenha escolhido um paciente de verdade.
  if (patients.length > 0) openChat(patients[0]);
}

function showTutorialStep() {
  const step = TUTORIAL_STEPS[tutorialIndex];
  if (step.beforeShow) step.beforeShow();

  goTo(step.screen);

  // Espera a troca de tela renderizar antes de medir a posição do elemento
  requestAnimationFrame(() => {
    const target = document.querySelector(step.selector);
    if (!target) { tutorialNext(); return; }

    const rect = target.getBoundingClientRect();
    const spotlight = document.getElementById("tutorialSpotlight");
    const tooltip   = document.getElementById("tutorialTooltip");
    const pad = 8;

    spotlight.style.top    = (rect.top - pad) + "px";
    spotlight.style.left   = (rect.left - pad) + "px";
    spotlight.style.width  = (rect.width + pad * 2) + "px";
    spotlight.style.height = (rect.height + pad * 2) + "px";

    document.getElementById("tutorialTitle").textContent = step.title;
    document.getElementById("tutorialText").textContent  = step.text;
    document.getElementById("tutorialStepCount").textContent =
      `${tutorialIndex + 1}/${TUTORIAL_STEPS.length}`;
    document.getElementById("tutorialPrev").disabled = tutorialIndex === 0;
    document.getElementById("tutorialNext").textContent =
      tutorialIndex === TUTORIAL_STEPS.length - 1 ? "Concluir" : "Próximo ▶";

    let top = rect.bottom + pad + 12;
    if (top + 160 > window.innerHeight) top = rect.top - 180;
    tooltip.style.top  = Math.max(12, top) + "px";
    tooltip.style.left = Math.min(Math.max(12, rect.left), window.innerWidth - 280) + "px";
  });
}

/* Start */
init();