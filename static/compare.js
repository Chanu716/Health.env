// ===========================
// THEME TOGGLE
// ===========================
const API_BASE_URL = window.API_BASE_URL || '';

function toggleTheme() {
    const html = document.documentElement;
    const currentTheme = html.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    
    html.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    
    const icon = document.getElementById('themeIcon');
    icon.textContent = newTheme === 'light' ? '🌙' : '☀️';
    
    // Reload charts with new theme
    setTimeout(() => initializeCharts(), 100);
}

function loadTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    const icon = document.getElementById('themeIcon');
    icon.textContent = savedTheme === 'light' ? '🌙' : '☀️';
}

// ===========================
// DATA LOADING
// ===========================
let modelsData = [];

async function loadModelData() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/model-comparison`);
        if (!response.ok) {
            throw new Error('Failed to load model comparison data');
        }
        
        const data = await response.json();
        if (data.success && data.models) {
            modelsData = data.models;
            // Sort by accuracy (best first)
            modelsData.sort((a, b) => b.accuracy - a.accuracy);
            return modelsData;
        }
        throw new Error('Invalid data format');
    } catch (error) {
        console.error('Error loading model data:', error);
        // Fallback data
        modelsData = [
            {
                model: 'Random Forest Model',
                model_key: 'random_forest_model',
                accuracy: 0.9911,
                precision: 0.9912,
                recall: 0.9911,
                f1_score: 0.9911
            },
            {
                model: 'Xgboost Model',
                model_key: 'xgboost_model',
                accuracy: 0.9707,
                precision: 0.9719,
                recall: 0.9707,
                f1_score: 0.9708
            },
            {
                model: 'Decision Tree Model',
                model_key: 'decision_tree_model',
                accuracy: 0.9688,
                precision: 0.9699,
                recall: 0.9688,
                f1_score: 0.9689
            },
            {
                model: 'Logistic Regression Model',
                model_key: 'logistic_regression_model',
                accuracy: 0.8998,
                precision: 0.8983,
                recall: 0.8998,
                f1_score: 0.8988
            }
        ];
        return modelsData;
    }
}

// ===========================
// BEST MODEL DISPLAY
// ===========================
function displayBestModel() {
    if (modelsData.length === 0) return;
    
    const best = modelsData[0];
    document.getElementById('bestModelName').textContent = best.model;
    document.getElementById('bestAccuracy').textContent = (best.accuracy * 100).toFixed(2) + '%';
    document.getElementById('bestPrecision').textContent = (best.precision * 100).toFixed(2) + '%';
    document.getElementById('bestRecall').textContent = (best.recall * 100).toFixed(2) + '%';
    document.getElementById('bestF1').textContent = (best.f1_score * 100).toFixed(2) + '%';
}

// ===========================
// CHARTS INITIALIZATION
// ===========================
async function initializeCharts() {
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
    const textColor = currentTheme === 'light' ? '#4a4a4a' : '#b3b3b3';
    const gridColor = currentTheme === 'light' ? 'rgba(150, 150, 150, 0.2)' : 'rgba(255,255,255,0.1)';
    const bgColor = 'rgba(0,0,0,0)';
    
    // Load data if not already loaded
    if (modelsData.length === 0) {
        await loadModelData();
    }
    
    displayBestModel();
    createMetricsTable();
    createInsights();
    
    const modelNames = modelsData.map(m => m.model);
    const accuracies = modelsData.map(m => m.accuracy * 100);
    const precisions = modelsData.map(m => m.precision * 100);
    const recalls = modelsData.map(m => m.recall * 100);
    const f1Scores = modelsData.map(m => m.f1_score * 100);
    
    // Color palette for models
    const modelColors = ['#00ff88', '#00ccff', '#ffcc00', '#ff3366', '#9966ff'];
    
    // ===========================
    // Chart 1: Metrics Bar Chart
    // ===========================
    const metricsBarData = [{
        x: modelNames,
        y: accuracies,
        type: 'bar',
        marker: {
            color: modelColors.slice(0, modelNames.length),
            line: {
                color: textColor,
                width: 1
            }
        },
        text: accuracies.map(v => v.toFixed(2) + '%'),
        textposition: 'outside',
        textfont: {
            color: textColor,
            size: 11
        }
    }];
    
    const metricsBarLayout = {
        paper_bgcolor: bgColor,
        plot_bgcolor: bgColor,
        font: { color: textColor, family: 'Barlow' },
        margin: { l: 60, r: 40, t: 40, b: 120 },
        xaxis: {
            tickangle: -45,
            showgrid: false,
            color: textColor
        },
        yaxis: {
            title: 'Accuracy (%)',
            showgrid: true,
            gridcolor: gridColor,
            color: textColor,
            range: [0, 105]
        }
    };
    
    Plotly.newPlot('metricsBarChart', metricsBarData, metricsBarLayout, {
        responsive: true,
        displayModeBar: false
    });
    
    // ===========================
    // Chart 2: Radar Chart
    // ===========================
    const radarData = modelsData.map((model, idx) => ({
        type: 'scatterpolar',
        r: [
            model.accuracy * 100,
            model.precision * 100,
            model.recall * 100,
            model.f1_score * 100,
            model.accuracy * 100  // Close the loop
        ],
        theta: ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Accuracy'],
        fill: 'toself',
        name: model.model,
        marker: { color: modelColors[idx % modelColors.length] },
        line: { color: modelColors[idx % modelColors.length] }
    }));
    
    const radarLayout = {
        paper_bgcolor: bgColor,
        plot_bgcolor: bgColor,
        font: { color: textColor, family: 'Barlow' },
        polar: {
            radialaxis: {
                visible: true,
                range: [0, 100],
                showgrid: true,
                gridcolor: gridColor,
                color: textColor
            },
            angularaxis: {
                color: textColor
            },
            bgcolor: bgColor
        },
        showlegend: true,
        legend: {
            font: { color: textColor },
            orientation: 'h',
            y: -0.2
        },
        margin: { l: 80, r: 80, t: 40, b: 100 }
    };
    
    Plotly.newPlot('radarChart', radarData, radarLayout, {
        responsive: true,
        displayModeBar: false
    });
    
    // ===========================
    // Chart 3: Grouped Bar Chart
    // ===========================
    const groupedBarData = [
        {
            x: modelNames,
            y: accuracies,
            name: 'Accuracy',
            type: 'bar',
            marker: { color: '#00ff88' }
        },
        {
            x: modelNames,
            y: precisions,
            name: 'Precision',
            type: 'bar',
            marker: { color: '#00ccff' }
        },
        {
            x: modelNames,
            y: recalls,
            name: 'Recall',
            type: 'bar',
            marker: { color: '#ffcc00' }
        },
        {
            x: modelNames,
            y: f1Scores,
            name: 'F1-Score',
            type: 'bar',
            marker: { color: '#ff3366' }
        }
    ];
    
    const groupedBarLayout = {
        paper_bgcolor: bgColor,
        plot_bgcolor: bgColor,
        font: { color: textColor, family: 'Barlow' },
        barmode: 'group',
        margin: { l: 60, r: 40, t: 40, b: 120 },
        xaxis: {
            tickangle: -45,
            showgrid: false,
            color: textColor
        },
        yaxis: {
            title: 'Score (%)',
            showgrid: true,
            gridcolor: gridColor,
            color: textColor,
            range: [0, 105]
        },
        legend: {
            font: { color: textColor },
            orientation: 'h',
            y: 1.1
        }
    };
    
    Plotly.newPlot('groupedBarChart', groupedBarData, groupedBarLayout, {
        responsive: true,
        displayModeBar: false
    });
}

// ===========================
// METRICS TABLE
// ===========================
function createMetricsTable() {
    const tbody = document.getElementById('metricsTableBody');
    tbody.innerHTML = '';
    
    modelsData.forEach((model, idx) => {
        const row = document.createElement('tr');
        
        // Add highlighting for best model
        if (idx === 0) {
            row.classList.add('best-row');
        }
        
        row.innerHTML = `
            <td><strong>${model.model}</strong></td>
            <td>${(model.accuracy * 100).toFixed(2)}%</td>
            <td>${(model.precision * 100).toFixed(2)}%</td>
            <td>${(model.recall * 100).toFixed(2)}%</td>
            <td>${(model.f1_score * 100).toFixed(2)}%</td>
            <td>${idx === 0 ? '🏆 #1' : `#${idx + 1}`}</td>
        `;
        
        tbody.appendChild(row);
    });
}

// ===========================
// INSIGHTS GENERATION
// ===========================
function createInsights() {
    const insightsGrid = document.getElementById('insightsGrid');
    
    if (modelsData.length === 0) return;
    
    const best = modelsData[0];
    const worst = modelsData[modelsData.length - 1];
    const avgAccuracy = modelsData.reduce((sum, m) => sum + m.accuracy, 0) / modelsData.length;
    
    const insights = [
        {
            icon: '🏆',
            title: 'Top Performer',
            text: `${best.model} achieves the highest accuracy at ${(best.accuracy * 100).toFixed(2)}%, making it the recommended model for production deployment.`,
            class: 'success'
        },
        {
            icon: '📊',
            title: 'Performance Spread',
            text: `Models show ${((best.accuracy - worst.accuracy) * 100).toFixed(2)}% performance gap, with an average accuracy of ${(avgAccuracy * 100).toFixed(2)}%.`,
            class: 'info'
        },
        {
            icon: '⚖️',
            title: 'Precision-Recall Balance',
            text: `${best.model} maintains excellent balance with precision at ${(best.precision * 100).toFixed(2)}% and recall at ${(best.recall * 100).toFixed(2)}%.`,
            class: 'balanced'
        },
        {
            icon: '🎯',
            title: 'F1-Score Analysis',
            text: `The best F1-Score of ${(best.f1_score * 100).toFixed(2)}% indicates strong overall performance in dengue risk classification.`,
            class: 'highlight'
        }
    ];
    
    insightsGrid.innerHTML = insights.map(insight => `
        <div class="insight-card ${insight.class}">
            <div class="insight-icon">${insight.icon}</div>
            <h4>${insight.title}</h4>
            <p>${insight.text}</p>
        </div>
    `).join('');
}

// ===========================
// INITIALIZE ON LOAD
// ===========================
window.addEventListener('load', function() {
    loadTheme();
    initializeCharts();
});
