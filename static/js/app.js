const API_ENDPOINT = '/api/analyze';

let currentObjectUrl = null;

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

function resetPreview() {
    if (currentObjectUrl) {
        URL.revokeObjectURL(currentObjectUrl);
        currentObjectUrl = null;
    }
    originalImg.removeAttribute('src');
    gradcamImg.removeAttribute('src');
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
    resetPreview();

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
        // ESTADO DA PLANTA
        // ===============================
        if (data.condition) {
            conditionEl.textContent = data.condition;
            const lower = data.condition.toLowerCase();
            if (lower.includes('saudável')) {
                conditionEl.className = 'condition healthy';
            } else if (lower.includes('inconclus') || lower.includes('atenção')) {
                conditionEl.className = 'condition warning';
            } else {
                conditionEl.className = 'condition danger';
            }
        } else {
            conditionEl.textContent = data.recommendation ? data.recommendation : 'Resumo da condição indisponível.';
            conditionEl.className = 'condition';
        }

        // ===============================
        // TOP 3
        // ===============================
        const top3Items = Array.isArray(data.top3) ? data.top3 : [];
        top3Body.innerHTML = top3Items.map(item => `
            <tr>
                <td>${item.class || 'Classificação'}</td>
                <td>${formatPercent(typeof item.prob === 'number' ? item.prob : 0)}</td>
            </tr>
        `).join('');

        // ===============================
        // IMAGENS
        // ===============================
        currentObjectUrl = URL.createObjectURL(file);
        originalImg.src = currentObjectUrl;
        gradcamImg.src = data.heatmap_path || '/static/outputs/gradcam_placeholder.jpg';

        resultSection.style.display = 'block';

    } catch (err) {
        showError(err.message || 'Erro ao analisar imagem.');
    } finally {
        setLoading(false);
    }
}