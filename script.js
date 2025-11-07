// ===========================
// THEME TOGGLE
// ===========================
function toggleTheme() {
    const html = document.documentElement;
    const currentTheme = html.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    
    html.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    
    // Update icon
    const icon = document.getElementById('themeIcon');
    icon.textContent = newTheme === 'light' ? '🌙' : '☀️';
    
    // Update charts with new theme
    updateChartsTheme(newTheme);
}

// Load saved theme on page load
function loadTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    const icon = document.getElementById('themeIcon');
    icon.textContent = savedTheme === 'light' ? '🌙' : '☀️';
}

// Update chart colors based on theme
function updateChartsTheme(theme) {
    const textColor = theme === 'light' ? '#4a4a4a' : '#b3b3b3';
    const gridColor = theme === 'light' ? 'rgba(150, 150, 150, 0.2)' : 'rgba(255,255,255,0.1)';
    const bgColor = 'rgba(0,0,0,0)';
    
    // Re-initialize charts with new colors
    setTimeout(() => {
        initializeCharts();
    }, 100);
}

// ===========================
// SLIDER VALUE UPDATES
// ===========================
document.getElementById('temperature').addEventListener('input', function(e) {
    document.getElementById('temp-value').textContent = e.target.value;
});

document.getElementById('rainfall').addEventListener('input', function(e) {
    document.getElementById('rain-value').textContent = e.target.value;
});

document.getElementById('humidity').addEventListener('input', function(e) {
    document.getElementById('humid-value').textContent = e.target.value;
});

document.getElementById('aqi').addEventListener('input', function(e) {
    document.getElementById('aqi-value').textContent = e.target.value;
});

document.getElementById('mosquito').addEventListener('input', function(e) {
    document.getElementById('mosq-value').textContent = e.target.value;
});

// ===========================
// PREDICTION FUNCTION
// ===========================
async function predictRisk() {
    // Get input values
    const city = document.getElementById('city').value;
    const month = document.getElementById('month').value;
    const temperature = parseFloat(document.getElementById('temperature').value);
    const rainfall = parseFloat(document.getElementById('rainfall').value);
    const humidity = parseFloat(document.getElementById('humidity').value);
    const aqi = parseFloat(document.getElementById('aqi').value);
    const mosquito = parseFloat(document.getElementById('mosquito').value);
    const population = parseFloat(document.getElementById('population').value);
    const cases = parseFloat(document.getElementById('cases').value);

    // Show loading state
    const predictionCard = document.getElementById('predictionCard');
    const riskBadge = document.getElementById('riskBadge');
    
    // Display prediction card if hidden
    if (predictionCard) {
        predictionCard.style.display = 'block';
    }
    
    // Show loading state in risk badge
    if (riskBadge) {
        riskBadge.innerHTML = '<span style="font-size: 14px;">🔄 Predicting...</span>';
    }

    try {
        // Call the Python API
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                city: city,
                month: month,
                temperature: temperature,
                rainfall: rainfall,
                humidity: humidity,
                aqi: aqi,
                mosquito: mosquito,
                population: population,
                cases: cases
            })
        });

        if (!response.ok) {
            throw new Error('API request failed');
        }

        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || 'Prediction failed');
        }

        // Extract prediction results
        const prediction = data.prediction;
        const probabilities = data.probabilities;

        let riskLevel = prediction.risk_level.toUpperCase() + ' RISK';
        let riskClass = prediction.risk_level.toLowerCase();
        let riskIcon = prediction.icon;
        let confidence = prediction.confidence / 100;  // Convert back to decimal

        // Update prediction location
        const resultCity = document.getElementById('result-city');
        const resultMonth = document.getElementById('result-month');
        
        if (resultCity) resultCity.textContent = city;
        if (resultMonth) resultMonth.textContent = month;
        
        // Update risk badge - restore proper structure (it was replaced by loading HTML)
        const riskBadge = document.getElementById('riskBadge');
        if (riskBadge) {
            riskBadge.className = `risk-badge ${riskClass}`;
            riskBadge.innerHTML = `
                <span class="risk-icon" id="riskIcon">${riskIcon}</span>
                <span class="risk-text" id="riskText">${riskLevel}</span>
            `;
        }
        
        // Update confidence
        const confidencePercent = Math.round(confidence * 100);
        const confidencePercentElem = document.getElementById('confidencePercent');
        const progressFill = document.getElementById('progressFill');
        
        if (confidencePercentElem) confidencePercentElem.textContent = `${confidencePercent}%`;
        if (progressFill) progressFill.style.width = `${confidencePercent}%`;

        // Update metrics with null checks
        const metricTemp = document.getElementById('metric-temp');
        const metricMosq = document.getElementById('metric-mosq');
        const metricCases = document.getElementById('metric-cases');
        const metricAqi = document.getElementById('metric-aqi');
        
        if (metricTemp) metricTemp.textContent = `${temperature}°C`;
        if (metricMosq) metricMosq.textContent = mosquito.toFixed(2);
        if (metricCases) metricCases.textContent = cases;
        if (metricAqi) metricAqi.textContent = aqi;

        // Update recommendations
        updateRecommendations(riskClass);

        // Scroll to results (with null check)
        const resultPanel = document.querySelector('.result-panel');
        if (resultPanel) {
            resultPanel.scrollIntoView({ behavior: 'smooth' });
        }

    } catch (error) {
        console.error('Prediction error:', error);
        
        // Fallback to rule-based prediction if API fails
        let riskScore = 0;
        
        if (temperature > 28) riskScore += 0.2;
        if (rainfall > 1000) riskScore += 0.25;
        if (humidity > 70) riskScore += 0.2;
        if (mosquito > 0.6) riskScore += 0.2;
        if (cases > 300) riskScore += 0.15;

        let riskLevel, riskClass, riskIcon, confidence;
        
        if (riskScore > 0.7) {
            riskLevel = 'HIGH RISK';
            riskClass = 'high';
            riskIcon = '🔴';
            confidence = 0.75;
        } else if (riskScore > 0.4) {
            riskLevel = 'MODERATE RISK';
            riskClass = 'moderate';
            riskIcon = '🟡';
            confidence = 0.65;
        } else {
            riskLevel = 'LOW RISK';
            riskClass = 'low';
            riskIcon = '🟢';
            confidence = 0.60;
        }

        // Update prediction card (fallback mode) with null checks
        const resultCity = document.getElementById('result-city');
        const resultMonth = document.getElementById('result-month');
        const riskBadgeElem = document.getElementById('riskBadge');
        const confidencePercentElem = document.getElementById('confidencePercent');
        const progressFillElem = document.getElementById('progressFill');
        const metricTemp = document.getElementById('metric-temp');
        const metricMosq = document.getElementById('metric-mosq');
        const metricCases = document.getElementById('metric-cases');
        const metricAqi = document.getElementById('metric-aqi');
        
        if (resultCity) resultCity.textContent = city + ' (Offline Mode)';
        if (resultMonth) resultMonth.textContent = month;
        
        // Update risk badge - restore proper structure
        if (riskBadgeElem) {
            riskBadgeElem.className = `risk-badge ${riskClass}`;
            riskBadgeElem.innerHTML = `
                <span class="risk-icon" id="riskIcon">${riskIcon}</span>
                <span class="risk-text" id="riskText">${riskLevel}</span>
            `;
        }
        
        const confidencePercent = Math.round(confidence * 100);
        if (confidencePercentElem) confidencePercentElem.textContent = `${confidencePercent}%`;
        if (progressFillElem) progressFillElem.style.width = `${confidencePercent}%`;

        if (metricTemp) metricTemp.textContent = `${temperature}°C`;
        if (metricMosq) metricMosq.textContent = mosquito.toFixed(2);
        if (metricCases) metricCases.textContent = cases;
        if (metricAqi) metricAqi.textContent = aqi;

        updateRecommendations(riskClass);

        // Show warning banner
        const predictionCard = document.getElementById('predictionCard');
        if (predictionCard) {
            const existingWarning = predictionCard.querySelector('.offline-warning');
            if (!existingWarning) {
                predictionCard.insertAdjacentHTML('afterbegin', 
                    '<div class="offline-warning" style="background: rgba(255,200,0,0.1); padding: 10px; margin-bottom: 15px; border-radius: 8px; border-left: 4px solid #ffcc00;">⚠️ API unavailable. Using offline prediction mode.</div>'
                );
            }
        }
        
        // Scroll to results (with null check)
        const resultPanel = document.querySelector('.result-panel');
        if (resultPanel) {
            resultPanel.scrollIntoView({ behavior: 'smooth' });
        }
    }
}

// ===========================
// UPDATE RECOMMENDATIONS
// ===========================
function updateRecommendations(riskClass) {
    const recommendations = {
        high: {
            header: '⚠️ Immediate Action Required',
            icon: '⚠️',
            list: [
                '🏥 Increase medical resource allocation',
                '🚫 Eliminate stagnant water sources immediately',
                '🦟 Use mosquito repellents and nets',
                '📢 Launch public awareness campaigns',
                '🌡️ Monitor fever cases closely',
                '💉 Ensure dengue testing availability'
            ]
        },
        moderate: {
            header: '⚡ Preventive Measures Advised',
            icon: '⚡',
            list: [
                '🔍 Regular monitoring of mosquito breeding sites',
                '💊 Keep anti-dengue medication ready',
                '🌊 Check water storage containers weekly',
                '👕 Wear full-sleeve clothing during peak hours',
                '🏠 Use mosquito screens on windows',
                '🧹 Maintain proper sanitation'
            ]
        },
        low: {
            header: '✅ Maintain Standard Precautions',
            icon: '✅',
            list: [
                '🧹 Continue routine sanitation',
                '📊 Monitor environmental conditions',
                '🌿 Maintain clean surroundings',
                '👁️ Stay alert for symptom changes',
                '💧 Prevent water accumulation',
                '🏃 Promote outdoor activities during safe hours'
            ]
        }
    };

    const rec = recommendations[riskClass];
    document.getElementById('recHeader').innerHTML = `
        <span class="rec-icon">${rec.icon}</span>
        <span>${rec.header}</span>
    `;

    const recList = document.getElementById('recList');
    recList.innerHTML = rec.list.map(item => `<li>${item}</li>`).join('');
    
    // Update border color
    const recCard = document.querySelector('.recommendation-card');
    if (riskClass === 'high') {
        recCard.style.borderLeftColor = '#ff3366';
    } else if (riskClass === 'moderate') {
        recCard.style.borderLeftColor = '#ffcc00';
    } else {
        recCard.style.borderLeftColor = '#00ff88';
    }
}

// ===========================
// CHART MODAL FUNCTIONS
// ===========================
let currentChartData = null;
let currentChartLayout = null;

function openChartModal(title, data, layout) {
    currentChartData = data;
    currentChartLayout = {
        ...layout,
        height: 600,
        margin: { l: 60, r: 40, t: 40, b: 60 }
    };
    
    document.getElementById('modalChartTitle').textContent = title;
    document.getElementById('chartModal').classList.add('active');
    
    Plotly.newPlot('modalChartContainer', currentChartData, currentChartLayout, {
        responsive: true,
        displayModeBar: true
    });
}

function closeChartModal() {
    document.getElementById('chartModal').classList.remove('active');
}

// Close modal when clicking outside
document.getElementById('chartModal').addEventListener('click', function(e) {
    if (e.target === this) {
        closeChartModal();
    }
});

// Close modal on Escape key
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        closeChartModal();
    }
});

// ===========================
// INITIALIZE CHARTS
// ===========================

// Cache for API data to avoid repeated requests
let chartDataCache = {
    featureImportance: null,
    trainingStats: null,
    lastFetch: null
};

async function loadRealChartData() {
    // Cache data for 5 minutes to improve performance
    const now = Date.now();
    if (chartDataCache.lastFetch && (now - chartDataCache.lastFetch) < 300000) {
        return; // Use cached data
    }
    
    try {
        // Fetch feature importance
        const featureResponse = await fetch(`${API_BASE_URL}/api/feature-importance`);
        if (featureResponse.ok) {
            const featureData = await featureResponse.json();
            chartDataCache.featureImportance = featureData;
        }
        
        // Fetch training statistics
        const statsResponse = await fetch(`${API_BASE_URL}/api/training-stats`);
        if (statsResponse.ok) {
            const statsData = await statsResponse.json();
            chartDataCache.trainingStats = statsData;
        }
        
        chartDataCache.lastFetch = now;
    } catch (error) {
        console.warn('Failed to load real chart data, using fallback:', error);
    }
}

async function initializeCharts() {
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
    const textColor = currentTheme === 'light' ? '#4a4a4a' : '#b3b3b3';
    const lineColor = currentTheme === 'light' ? '#1a1a1a' : '#ffffff';
    const gridColor = currentTheme === 'light' ? 'rgba(150, 150, 150, 0.2)' : 'rgba(255,255,255,0.1)';
    const fillColor = currentTheme === 'light' ? 'rgba(26, 26, 26, 0.1)' : 'rgba(255, 255, 255, 0.1)';
    
    // Load real data from API (with caching)
    await loadRealChartData();
    
    // Chart 1: Monthly Trend (Real Data from training)
    const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    let trendYValues = [0.3, 0.25, 0.35, 0.45, 0.55, 0.75, 0.85, 0.9, 0.7, 0.5, 0.4, 0.35]; // Fallback
    
    if (chartDataCache.trainingStats && chartDataCache.trainingStats.monthly_risk) {
        const monthlyRisk = chartDataCache.trainingStats.monthly_risk;
        trendYValues = [];
        for (let i = 1; i <= 12; i++) {
            trendYValues.push(monthlyRisk[i] || 0);
        }
    }
    
    const trendData = [{
        x: monthNames,
        y: trendYValues,
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: lineColor, width: 3 },
        marker: { size: 8, color: lineColor },
        fill: 'tozeroy',
        fillcolor: fillColor
    }];

    const trendLayout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: textColor, family: 'Barlow' },
        margin: { l: 40, r: 20, t: 20, b: 40 },
        xaxis: { 
            showgrid: false,
            color: textColor
        },
        yaxis: { 
            showgrid: true, 
            gridcolor: gridColor,
            color: textColor
        },
        showlegend: false
    };

    Plotly.newPlot('trendChart', trendData, trendLayout, { 
        responsive: true, 
        displayModeBar: false 
    });
    
    // Add click event for zoom
    document.getElementById('trendChart').parentElement.addEventListener('click', function() {
        openChartModal('Monthly Risk Trend', trendData, trendLayout);
    });

    // Chart 2: Scatter Plot (Real Data from training)
    let scatterDataLow = { x: [800, 1200, 1500, 2000, 2500, 1800, 1000, 2200], y: [25, 28, 30, 32, 34, 31, 27, 33] };
    let scatterDataMod = { x: [1100, 1600, 2100, 2400, 1900, 1700, 2300], y: [29, 31, 33, 35, 32, 30, 34] };
    let scatterDataHigh = { x: [1800, 2200, 2600, 2800, 2400, 2100], y: [32, 34, 36, 38, 35, 33] };
    
    if (chartDataCache.trainingStats && chartDataCache.trainingStats.scatter_data) {
        const scatterReal = chartDataCache.trainingStats.scatter_data;
        if (scatterReal.low) {
            scatterDataLow = { x: scatterReal.low.rainfall, y: scatterReal.low.temperature };
        }
        if (scatterReal.moderate) {
            scatterDataMod = { x: scatterReal.moderate.rainfall, y: scatterReal.moderate.temperature };
        }
        if (scatterReal.high) {
            scatterDataHigh = { x: scatterReal.high.rainfall, y: scatterReal.high.temperature };
        }
    }
    
    const scatterData = [
        {
            x: scatterDataLow.x,
            y: scatterDataLow.y,
            mode: 'markers',
            type: 'scatter',
            name: 'Low',
            marker: { size: 12, color: '#00ff88', opacity: 0.7 }
        },
        {
            x: scatterDataMod.x,
            y: scatterDataMod.y,
            mode: 'markers',
            type: 'scatter',
            name: 'Moderate',
            marker: { size: 12, color: '#ffcc00', opacity: 0.7 }
        },
        {
            x: scatterDataHigh.x,
            y: scatterDataHigh.y,
            mode: 'markers',
            type: 'scatter',
            name: 'High',
            marker: { size: 12, color: '#ff3366', opacity: 0.7 }
        }
    ];

    const scatterLayout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: textColor, family: 'Barlow', size: 10 },
        margin: { l: 40, r: 20, t: 20, b: 40 },
        xaxis: { 
            title: 'Rainfall (mm)',
            showgrid: true, 
            gridcolor: gridColor,
            color: textColor
        },
        yaxis: { 
            title: 'Temp (°C)',
            showgrid: true, 
            gridcolor: gridColor,
            color: textColor
        },
        legend: {
            orientation: 'h',
            y: 1.1,
            x: 0.5,
            xanchor: 'center',
            font: { size: 10 }
        }
    };

    Plotly.newPlot('scatterChart', scatterData, scatterLayout, { 
        responsive: true, 
        displayModeBar: false 
    });
    
    // Add click event for zoom
    document.getElementById('scatterChart').parentElement.addEventListener('click', function() {
        openChartModal('Rainfall vs Temperature Correlation', scatterData, scatterLayout);
    });

    // Chart 3: Feature Importance (Real Data from XGBoost model)
    const barColors = currentTheme === 'light' 
        ? ['#1a1a1a', '#2a2a2a', '#3a3a3a', '#4a4a4a', '#5a5a5a', '#6a6a6a', '#7a7a7a']
        : ['#ffffff', '#e8e8e8', '#d0d0d0', '#b8b8b8', '#a0a0a0', '#888888', '#707070'];
    
    let importanceX = [0.25, 0.22, 0.18, 0.15, 0.10, 0.08, 0.02]; // Fallback
    let importanceY = ['Mosquito Density', 'Rainfall', 'Humidity', 'Temperature', 'AQI', 'Cases', 'Population'];
    
    if (chartDataCache.featureImportance && chartDataCache.featureImportance.feature_importance) {
        const features = chartDataCache.featureImportance.feature_importance;
        // Take top 7 features
        const topFeatures = features.slice(0, 7);
        importanceX = topFeatures.map(f => f.importance);
        importanceY = topFeatures.map(f => f.feature);
    }
    
    const importanceData = [{
        x: importanceX,
        y: importanceY,
        type: 'bar',
        orientation: 'h',
        marker: {
            color: barColors
        }
    }];

    const importanceLayout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: textColor, family: 'Barlow', size: 10 },
        margin: { l: 120, r: 20, t: 20, b: 40 },
        xaxis: { 
            showgrid: true, 
            gridcolor: gridColor,
            color: textColor
        },
        yaxis: { 
            showgrid: false,
            color: textColor
        },
        showlegend: false
    };

    Plotly.newPlot('importanceChart', importanceData, importanceLayout, { 
        responsive: true, 
        displayModeBar: false 
    });
    
    // Add click event for zoom
    document.getElementById('importanceChart').parentElement.addEventListener('click', function() {
        openChartModal('Feature Importance Analysis', importanceData, importanceLayout);
    });

    // Chart 4: City Risk Distribution (Real Data from training)
    let cityNames = ['Delhi', 'Mumbai', 'Chennai', 'Kolkata', 'Bengaluru', 'Hyderabad', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow', 'Indore', 'Bhopal', 'Nagpur', 'Surat', 'Kanpur'];
    let cityRisks = [0.85, 0.72, 0.68, 0.78, 0.45, 0.55, 0.62, 0.38, 0.70, 0.82, 0.25, 0.30, 0.58, 0.42, 0.75]; // Fallback
    
    if (chartDataCache.trainingStats && chartDataCache.trainingStats.city_risk) {
        const cityRiskData = chartDataCache.trainingStats.city_risk;
        cityNames = Object.keys(cityRiskData);
        cityRisks = Object.values(cityRiskData);
    }
    
    const cityData = [{
        x: cityNames,
        y: cityRisks,
        type: 'bar',
        marker: {
            color: cityRisks,
            colorscale: [[0, '#00ff88'], [0.5, '#ffcc00'], [1, '#ff3366']],
            showscale: true,
            colorbar: {
                title: 'Risk',
                titleside: 'right',
                tickmode: 'linear',
                tick0: 0,
                dtick: 0.25
            }
        }
    }];

    const cityLayout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: textColor, family: 'Barlow' },
        margin: { l: 60, r: 20, t: 40, b: 80 },
        xaxis: { 
            title: 'City',
            showgrid: false,
            tickangle: -45,
            color: textColor
        },
        yaxis: { 
            title: 'Risk Score',
            showgrid: true, 
            gridcolor: gridColor,
            color: textColor
        }
    };

    Plotly.newPlot('cityChart', cityData, cityLayout, { 
        responsive: true, 
        displayModeBar: false 
    });
    
    // Add click event for zoom
    document.getElementById('cityChart').parentElement.addEventListener('click', function() {
        openChartModal('City Risk Distribution Across India', cityData, cityLayout);
    });
}

// ===========================
// INITIALIZE ON LOAD
// ===========================
window.addEventListener('load', function() {
    loadTheme(); // Load saved theme
    initializeCharts();
    predictRisk(); // Run initial prediction
});

// ===========================
// AUTO-UPDATE METRICS ON INPUT CHANGE
// ===========================
const inputs = ['temperature', 'rainfall', 'humidity', 'aqi', 'mosquito', 'population', 'cases'];
inputs.forEach(id => {
    document.getElementById(id).addEventListener('change', function() {
        // Auto-predict on input change (optional)
        // predictRisk();
    });
});
