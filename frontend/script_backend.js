/**
 * CyberShield AI Frontend JavaScript - Backend Connected Version
 * Connects to Flask backend API with 1GB upload support
 */

// API Configuration
const API_BASE_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:5000/api' 
    : '/api';

// Global Variables
let modelsStatus = {
    trained: false,
    available_models: [],
    feature_columns: []
};

let systemHealth = {
    status: 'checking',
    uptime: '99.97%',
    lastUpdate: new Date(),
    threatsDetected: 0,
    totalTransactions: 0,
    activeConnections: 0
};

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    console.log('🛡️ CyberShield AI Frontend Starting...');
    initializeSystem();
    setupEventListeners();
    checkBackendHealth();
});

/**
 * Initialize the system
 */
function initializeSystem() {
    updateSystemStatus('Initializing CyberShield AI...');
    loadSystemStats();
    
    // Start real-time updates
    setInterval(updateRealTimeStats, 2000);
    setInterval(checkBackendHealth, 30000); // Check every 30 seconds
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // Module navigation
    document.querySelectorAll('.module-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const module = this.dataset.module;
            switchModule(module);
            
            // Update active state
            document.querySelectorAll('.module-btn').forEach(b => b.classList.remove('active'));
            this.classList.add('active');
        });
    });
    
    // Generate data button
    const generateBtn = document.getElementById('generateDataBtn');
    if (generateBtn) {
        generateBtn.addEventListener('click', generateSyntheticData);
    }
    
    // File upload
    const uploadInput = document.getElementById('datasetUpload');
    if (uploadInput) {
        uploadInput.addEventListener('change', handleFileUpload);
    }
    
    // Prediction form
    const predictBtn = document.getElementById('predictBtn');
    if (predictBtn) {
        predictBtn.addEventListener('click', predictFraud);
    }
}

/**
 * Check backend health
 */
async function checkBackendHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        if (response.ok) {
            systemHealth.status = 'operational';
            systemHealth.lastUpdate = new Date();
            modelsStatus = {
                trained: data.features.models_trained,
                available_models: data.features.available_models
            };
            
            updateSystemStatus('✅ Backend Connected - All Systems Operational');
            updateModelsStatus();
        } else {
            throw new Error('Backend responded with error');
        }
    } catch (error) {
        console.error('Backend health check failed:', error);
        systemHealth.status = 'offline';
        updateSystemStatus('❌ Backend Offline - Using Demo Mode');
    }
}

/**
 * Update system status display
 */
function updateSystemStatus(message) {
    const statusElement = document.getElementById('systemStatus');
    if (statusElement) {
        statusElement.textContent = message;
        statusElement.className = `alert ${systemHealth.status === 'operational' ? 'alert-success' : 'alert-warning'}`;
    }
}

/**
 * Update models status
 */
function updateModelsStatus() {
    const statusElement = document.getElementById('modelsStatus');
    if (statusElement) {
        if (modelsStatus.trained) {
            statusElement.innerHTML = `
                <div class="alert alert-success">
                    <i class="fas fa-check-circle"></i> 
                    Models Trained: ${modelsStatus.available_models.join(', ')}
                </div>
            `;
        } else {
            statusElement.innerHTML = `
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle"></i> 
                    Models Not Trained - Generate data or upload dataset first
                </div>
            `;
        }
    }
}

/**
 * Generate synthetic data
 */
async function generateSyntheticData() {
    const btn = document.getElementById('generateDataBtn');
    const originalText = btn.textContent;
    
    try {
        btn.textContent = 'Generating...';
        btn.disabled = true;
        
        updateSystemStatus('🔄 Generating synthetic fraud data...');
        
        const response = await fetch(`${API_BASE_URL}/generate-data`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                num_transactions: 10000
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            modelsStatus.trained = true;
            modelsStatus.available_models = Object.keys(data.model_results);
            
            updateSystemStatus('✅ Synthetic data generated and models trained successfully!');
            updateModelsStatus();
            displayModelResults(data.model_results);
            displayDataStats(data.data_stats);
            
            // Show success message
            showNotification('success', 'Models trained successfully with synthetic data!');
        } else {
            throw new Error(data.message);
        }
    } catch (error) {
        console.error('Error generating data:', error);
        updateSystemStatus('❌ Failed to generate synthetic data');
        showNotification('error', `Error: ${error.message}`);
    } finally {
        btn.textContent = originalText;
        btn.disabled = false;
    }
}

/**
 * Handle file upload
 */
async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    // Check file size (1GB limit)
    if (file.size > 1024 * 1024 * 1024) {
        showNotification('error', 'File size exceeds 1GB limit');
        return;
    }
    
    if (!file.name.toLowerCase().endsWith('.csv')) {
        showNotification('error', 'Only CSV files are supported');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        updateSystemStatus(`🔄 Uploading ${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)...`);
        
        const response = await fetch(`${API_BASE_URL}/upload-dataset`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            modelsStatus.trained = true;
            modelsStatus.available_models = Object.keys(data.model_results);
            
            updateSystemStatus('✅ Dataset uploaded and models trained successfully!');
            updateModelsStatus();
            displayModelResults(data.model_results);
            displayDataStats(data.data_stats);
            
            showNotification('success', `Dataset processed: ${data.data_stats.total_transactions} transactions`);
        } else {
            throw new Error(data.message);
        }
    } catch (error) {
        console.error('Error uploading file:', error);
        updateSystemStatus('❌ Failed to upload dataset');
        showNotification('error', `Upload failed: ${error.message}`);
    }
}

/**
 * Predict fraud for single transaction
 */
async function predictFraud() {
    if (!modelsStatus.trained) {
        showNotification('warning', 'Please train models first by generating data or uploading a dataset');
        return;
    }
    
    // Get form data
    const amount = parseFloat(document.getElementById('predAmount').value) || 0;
    const hour = parseInt(document.getElementById('predHour').value) || 0;
    const merchantCategory = parseInt(document.getElementById('predMerchant').value) || 1;
    const countryRisk = parseFloat(document.getElementById('predCountryRisk').value) || 0;
    const model = document.getElementById('predModel').value || 'Random Forest';
    
    const transactionData = {
        amount: amount,
        hour: hour,
        day_of_week: new Date().getDay(),
        merchant_category: merchantCategory,
        transaction_count_1h: 1,
        transaction_count_24h: 5,
        avg_amount_30d: 200,
        card_present: 1,
        country_risk_score: countryRisk,
        velocity_score: countryRisk * 0.8
    };
    
    try {
        updateSystemStatus('🔄 Analyzing transaction...');
        
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                transaction: transactionData,
                model: model
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayPredictionResult(data.prediction, data.model_used);
            updateSystemStatus('✅ Transaction analyzed successfully');
        } else {
            throw new Error(data.message);
        }
    } catch (error) {
        console.error('Error predicting:', error);
        updateSystemStatus('❌ Prediction failed');
        showNotification('error', `Prediction failed: ${error.message}`);
    }
}

/**
 * Display model training results
 */
function displayModelResults(results) {
    const container = document.getElementById('modelResults');
    if (!container) return;
    
    let html = '<h5><i class="fas fa-chart-bar"></i> Model Performance</h5>';
    html += '<div class="row">';
    
    Object.entries(results).forEach(([modelName, metrics]) => {
        const auc = (metrics.auc * 100).toFixed(1);
        const precision = (metrics.precision * 100).toFixed(1);
        const recall = (metrics.recall * 100).toFixed(1);
        const f1 = (metrics.f1_score * 100).toFixed(1);
        
        html += `
            <div class="col-md-6 mb-3">
                <div class="card cyber-card">
                    <div class="card-body">
                        <h6 class="card-title">${modelName}</h6>
                        <div class="metrics">
                            <small>AUC: <span class="text-info">${auc}%</span></small><br>
                            <small>Precision: <span class="text-success">${precision}%</span></small><br>
                            <small>Recall: <span class="text-warning">${recall}%</span></small><br>
                            <small>F1-Score: <span class="text-primary">${f1}%</span></small>
                        </div>
                    </div>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    container.innerHTML = html;
}

/**
 * Display data statistics
 */
function displayDataStats(stats) {
    const container = document.getElementById('dataStats');
    if (!container) return;
    
    const fraudRate = (stats.fraud_rate * 100).toFixed(2);
    
    container.innerHTML = `
        <h5><i class="fas fa-database"></i> Dataset Statistics</h5>
        <div class="row">
            <div class="col-md-3">
                <div class="stat-card">
                    <div class="stat-value">${stats.total_transactions.toLocaleString()}</div>
                    <div class="stat-label">Total Transactions</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card">
                    <div class="stat-value text-danger">${stats.fraud_count.toLocaleString()}</div>
                    <div class="stat-label">Fraud Cases</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card">
                    <div class="stat-value text-warning">${fraudRate}%</div>
                    <div class="stat-label">Fraud Rate</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card">
                    <div class="stat-value">${stats.features.length}</div>
                    <div class="stat-label">Features</div>
                </div>
            </div>
        </div>
    `;
}

/**
 * Display prediction result
 */
function displayPredictionResult(prediction, modelUsed) {
    const container = document.getElementById('predictionResult');
    if (!container) return;
    
    const probability = (prediction.fraud_probability * 100).toFixed(1);
    const riskClass = prediction.risk_level === 'HIGH' ? 'danger' : 
                     prediction.risk_level === 'MEDIUM' ? 'warning' : 'success';
    
    container.innerHTML = `
        <div class="alert alert-${riskClass}">
            <h5><i class="fas fa-shield-alt"></i> Fraud Analysis Result</h5>
            <div class="row">
                <div class="col-md-4">
                    <strong>Fraud Probability:</strong><br>
                    <span class="h4">${probability}%</span>
                </div>
                <div class="col-md-4">
                    <strong>Risk Level:</strong><br>
                    <span class="h4">${prediction.risk_level}</span>
                </div>
                <div class="col-md-4">
                    <strong>Model Used:</strong><br>
                    <span>${modelUsed}</span>
                </div>
            </div>
            <hr>
            <strong>Recommendation:</strong> 
            ${prediction.risk_level === 'HIGH' ? '🚨 Block transaction and investigate' :
              prediction.risk_level === 'MEDIUM' ? '⚠️ Require additional verification' :
              '✅ Transaction appears legitimate'}
        </div>
    `;
}

/**
 * Show notification
 */
function showNotification(type, message) {
    const alertClass = type === 'success' ? 'alert-success' : 
                      type === 'error' ? 'alert-danger' : 
                      type === 'warning' ? 'alert-warning' : 'alert-info';
    
    const notification = document.createElement('div');
    notification.className = `alert ${alertClass} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; max-width: 400px;';
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}

/**
 * Switch between modules
 */
function switchModule(moduleName) {
    // Hide all modules
    document.querySelectorAll('.module-content').forEach(module => {
        module.style.display = 'none';
    });
    
    // Show selected module
    const targetModule = document.getElementById(moduleName);
    if (targetModule) {
        targetModule.style.display = 'block';
    }
}

/**
 * Update real-time stats
 */
function updateRealTimeStats() {
    // Simulate real-time updates
    systemHealth.threatsDetected += Math.floor(Math.random() * 3);
    systemHealth.totalTransactions += Math.floor(Math.random() * 50) + 10;
    systemHealth.activeConnections = 100 + Math.floor(Math.random() * 100);
    
    // Update display
    updateStatsDisplay();
}

/**
 * Update stats display
 */
function updateStatsDisplay() {
    const elements = {
        'totalTransactions': systemHealth.totalTransactions.toLocaleString(),
        'threatsDetected': systemHealth.threatsDetected.toLocaleString(),
        'activeConnections': systemHealth.activeConnections.toLocaleString(),
        'systemUptime': systemHealth.uptime
    };
    
    Object.entries(elements).forEach(([id, value]) => {
        const element = document.getElementById(id);
        if (element) element.textContent = value;
    });
}

/**
 * Load system stats
 */
function loadSystemStats() {
    updateStatsDisplay();
}

// Export functions for global access
window.CyberShieldAI = {
    generateSyntheticData,
    handleFileUpload,
    predictFraud,
    checkBackendHealth,
    switchModule
};
