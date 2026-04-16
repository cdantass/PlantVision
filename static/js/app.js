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
        showError('Please select an image first.');
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
            throw new Error(data.detail || 'Analysis failed.');
        }

        // Success — populate UI
        diagnosisEl.textContent = data.diagnosis;
        confidenceEl.textContent = 'Confidence: ' + formatPercent(data.confidence);
        reliabilityEl.textContent = data.is_reliable ? '✓ Reliable' : '⚠ Not reliable';
        reliabilityEl.className = data.is_reliable ? 'reliable' : 'unreliable';

        // Top 3 table
        top3Body.innerHTML = data.top3.map(item => `
            <tr>
                <td>${item['class']}</td>
                <td>${formatPercent(item.prob)}</td>
            </tr>
        `).join('');

        // Images
        originalImg.src = URL.createObjectURL(file);
        gradcamImg.src = data.heatmap_path;

        // Show result
        resultSection.style.display = 'block';

    } catch (err) {
        showError(err.message || 'An error occurred. Please try again.');
    } finally {
        setLoading(false);
    }
}
