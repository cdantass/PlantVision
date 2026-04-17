const API_ENDPOINT = '/api/analyze';

// DOM Elements
const fileInput = document.getElementById('file-input');
const analyzeBtn = document.getElementById('analyze-btn');
const spinner = document.getElementById('spinner');
const errorDisplay = document.getElementById('error-display');
const resultSection = document.getElementById('result-section');
const uploadCard = document.getElementById('upload-card');

// Result fields
const diagnosisEl = document.getElementById('diagnosis');
const confidenceEl = document.getElementById('confidence');
const reliabilityEl = document.getElementById('reliability');
const conditionEl = document.getElementById('condition'); // ✅ NOVO
const top3Body = document.getElementById('top3-body');
const originalImg = document.getElementById('original-image');
const gradcamImg = document.getElementById('gradcam-image');

function showError(message) {
    errorDisplay.textContent = message;
    errorDisplay.style.display = 'block';
}

function hideError() {
    errorDisplay.style.display = 'none';
}

function setLoading(isLoading) {
    analyzeBtn.disabled = isLoading;
    spinner.style.display = isLoading ? 'block' : 'none';
}

function formatPercent(value) {
    return (value * 100).toFixed(2) + '%';
}

async function submitAnalysis() {
    const file = fileInput.files[0];

    if (!file) {
        showError('Selecione uma imagem primeiro.');
        return;
    }

    hideError();
    setLoading(true);
    resultSection.style.display = 'none';
    uploadCard.style.display = 'block';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(API_ENDPOINT, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Falha na análise.');
        }

        // ===============================
        // RESULTADO PRINCIPAL
        // ===============================
        diagnosisEl.textContent = data.diagnosis;
        confidenceEl.textContent = 'Confiança: ' + formatPercent(data.confidence);

        // ===============================
        // CONFIABILIDADE
        // ===============================
        reliabilityEl.textContent = data.is_reliable
            ? '✓ Confiável'
            : '⚠ Baixa confiança';

        reliabilityEl.className = data.is_reliable
            ? 'reliable'
            : 'unreliable';

        // ===============================
        // ✅ ESTADO DA PLANTA (NOVO)
        // ===============================
        if (data.condition) {
            conditionEl.textContent = data.condition;

            if (data.condition.toLowerCase().includes('saudável')) {
                conditionEl.className = 'condition healthy';
            } else if (data.condition.toLowerCase().includes('atenção')) {
                conditionEl.className = 'condition warning';
            } else {
                conditionEl.className = 'condition danger';
            }
        } else {
            conditionEl.textContent = '';
        }

        // ===============================
        // TOP 3
        // ===============================
        top3Body.innerHTML = data.top3.map(item => `
            <tr>
                <td>${item.class}</td>
                <td>${formatPercent(item.prob)}</td>
            </tr>
        `).join('');

        // ===============================
        // IMAGENS
        // ===============================
        originalImg.src = URL.createObjectURL(file);
        gradcamImg.src = data.heatmap_path;

        resultSection.style.display = 'block';

    } catch (err) {
        showError(err.message || 'Erro ao analisar imagem.');
    } finally {
        setLoading(false);
    }
}