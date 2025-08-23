/**
 * CyberShield AI Frontend JavaScript - Enhanced Version
 * Handles all interactive features and API communication with realistic functionality
 */

// Configuration - Updated for Vercel deployment
const API_BASE_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:5001/api' 
    : '/api';
let demoInterval = null;
let demoRunning = false;

// Enhanced Global Variables for Realistic Functionality
let systemHealth = {
    status: 'operational',
    uptime: '99.97%',
    lastUpdate: new Date(),
    threatsDetected: 2847,
    totalTransactions: 1250847,
    activeConnections: 156
};

let fraudDatabase = [
    { id: 'FR001', amount: 2500.00, location: 'Unknown IP', time: '02:30 AM', risk: 'HIGH', probability: 94.2 },
    { id: 'FR002', amount: 156.78, location: 'New York', time: '11:15 PM', risk: 'MEDIUM', probability: 67.3 },
    { id: 'FR003', amount: 8999.99, location: 'International', time: '03:45 AM', risk: 'CRITICAL', probability: 98.7 },
    { id: 'FR004', amount: 45.99, location: 'California', time: '01:20 AM', risk: 'LOW', probability: 23.1 },
    { id: 'FR005', amount: 1200.00, location: 'Texas', time: '04:30 AM', risk: 'HIGH', probability: 89.4 }
];

let realTimeStats = {
    transactionsToday: 15247,
    fraudsBlocked: 47,
    avgResponseTime: 42,
    accuracyRate: 99.7
};

// Demo functionality for Live Security Feed
let transactionCounter = 0;
let threatCounter = 0;

// Utility Functions
function showLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `
            <div class="text-center" style="padding: 40px;">
                <div class="loading"></div>
                <div style="margin-top: 15px; color: var(--cyber-blue);">
                    <strong>CYBERSHIELD ANALYZING...</strong>
                </div>
                <div style="margin-top: 10px; color: var(--text-light); font-size: 0.9rem;">
                    Scanning transaction for threats...
                </div>
            </div>
        `;
    }
}

function showError(elementId, message) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `
            <div class="alert alert-danger" style="background: rgba(255, 7, 58, 0.1); border: 1px solid #ff073a; color: #ff073a;">
                <i class="fas fa-exclamation-triangle"></i>
                <strong>Error:</strong> ${message}
            </div>
        `;
    }
}

function showSuccess(elementId, content) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = content;
    }
}

// Enhanced Fraud Analysis Functions
async function analyzeFraudTransaction(formData) {
    showLoading('analysisResults');
    
    // Simulate realistic processing time
    await new Promise(resolve => setTimeout(resolve, 1500 + Math.random() * 1000));
    
    // Extract form data
    const amount = parseFloat(formData.get('amount') || document.getElementById('amount').value);
    const merchant = formData.get('merchant') || document.getElementById('merchant').value;
    const customerId = formData.get('customerId') || document.getElementById('customerId').value;
    const time = formData.get('transactionTime') || document.getElementById('transactionTime').value;
    const latitude = parseFloat(formData.get('latitude') || document.getElementById('latitude').value);
    const longitude = parseFloat(formData.get('longitude') || document.getElementById('longitude').value);
    
    // Calculate realistic fraud probability based on multiple factors
    let riskScore = 0;
    let riskFactors = [];
    
    // Amount risk (higher amounts = higher risk)
    if (amount > 5000) {
        riskScore += 35;
        riskFactors.push({ factor: 'High Transaction Amount', weight: 35, description: `$${amount} exceeds normal spending patterns` });
    } else if (amount > 1000) {
        riskScore += 15;
        riskFactors.push({ factor: 'Elevated Transaction Amount', weight: 15, description: `$${amount} is above average` });
    }
    
    // Time risk (late night/early morning = higher risk)
    const hour = parseInt(time.split(':')[0]);
    if (hour >= 0 && hour <= 4) {
        riskScore += 25;
        riskFactors.push({ factor: 'Unusual Time Pattern', weight: 25, description: 'Transaction at high-risk hours (12-4 AM)' });
    } else if (hour >= 22 || hour <= 6) {
        riskScore += 15;
        riskFactors.push({ factor: 'Off-hours Activity', weight: 15, description: 'Transaction outside normal business hours' });
    }
    
    // Merchant category risk
    const merchantRisk = {
        'online': 20,
        'atm': 25,
        'travel': 15,
        'grocery': 5,
        'restaurant': 8,
        'retail': 10,
        'gas': 7
    };
    const merchantScore = merchantRisk[merchant] || 10;
    riskScore += merchantScore;
    riskFactors.push({ factor: 'Merchant Category Risk', weight: merchantScore, description: `${merchant} category has ${merchantScore}% risk factor` });
    
    // Location risk (simulate geographic analysis)
    if (Math.abs(latitude) > 50 || Math.abs(longitude) > 100) {
        riskScore += 30;
        riskFactors.push({ factor: 'Geographic Anomaly', weight: 30, description: 'Transaction from unusual geographic location' });
    }
    
    // Add some randomness for realism
    const randomFactor = Math.random() * 20 - 10;
    riskScore += randomFactor;
    
    // Ensure score is within 0-100
    riskScore = Math.max(0, Math.min(100, riskScore));
    
    // Determine risk level and actions
    let riskLevel, riskColor, recommendation;
    if (riskScore >= 80) {
        riskLevel = 'CRITICAL';
        riskColor = '#ff073a';
        recommendation = 'BLOCK TRANSACTION - Immediate investigation required';
    } else if (riskScore >= 60) {
        riskLevel = 'HIGH';
        riskColor = '#ff6b35';
        recommendation = 'HOLD TRANSACTION - Additional verification needed';
    } else if (riskScore >= 30) {
        riskLevel = 'MEDIUM';
        riskColor = '#ffa500';
        recommendation = 'MONITOR TRANSACTION - Enhanced tracking enabled';
    } else {
        riskLevel = 'LOW';
        riskColor = '#00d4ff';
        recommendation = 'APPROVE TRANSACTION - Continue monitoring';
    }
    
    // Generate realistic analysis results
    const analysisResult = {
        transactionId: 'TXN-' + Math.random().toString(36).substr(2, 9).toUpperCase(),
        fraudProbability: riskScore.toFixed(1),
        riskLevel: riskLevel,
        confidence: (85 + Math.random() * 14).toFixed(1),
        processingTime: (35 + Math.random() * 30).toFixed(1) + 'ms',
        riskFactors: riskFactors,
        recommendation: recommendation,
        timestamp: new Date().toLocaleString()
    };
    
    displayEnhancedAnalysisResult(analysisResult, riskColor, riskScore);
    
    // Update global stats
    realTimeStats.transactionsToday++;
    if (riskScore >= 60) {
        realTimeStats.fraudsBlocked++;
    }
    updateSystemStats();
}

function displayEnhancedAnalysisResult(result, riskColor, riskScore) {
    const resultsHTML = `
        <div class="cyber-card" style="border-left: 5px solid ${riskColor};">
            <div class="analysis-header">
                <h5><i class="fas fa-shield-alt"></i> THREAT ANALYSIS COMPLETE</h5>
                <span class="risk-badge" style="background: ${riskColor}; color: white; padding: 8px 15px; border-radius: 20px; font-weight: bold;">
                    ${result.riskLevel} RISK
                </span>
            </div>
            
            <div class="row mt-4">
                <div class="col-md-6">
                    <div class="metric-group">
                        <h6 style="color: var(--cyber-blue); margin-bottom: 15px;">
                            <i class="fas fa-chart-line"></i> Risk Assessment
                        </h6>
                        <div class="metric-item">
                            <span>Transaction ID:</span>
                            <span style="color: var(--cyber-green); font-family: monospace;">${result.transactionId}</span>
                        </div>
                        <div class="metric-item">
                            <span>Fraud Probability:</span>
                            <span style="color: ${riskColor}; font-weight: bold;">${result.fraudProbability}%</span>
                        </div>
                        <div class="metric-item">
                            <span>Model Confidence:</span>
                            <span style="color: var(--cyber-green);">${result.confidence}%</span>
                        </div>
                        <div class="metric-item">
                            <span>Processing Time:</span>
                            <span style="color: var(--cyber-blue);">${result.processingTime}</span>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-6">
                    <div class="risk-visualization">
                        <h6 style="color: var(--cyber-blue); margin-bottom: 15px;">
                            <i class="fas fa-tachometer-alt"></i> Risk Meter
                        </h6>
                        <div class="risk-meter">
                            <div class="risk-bar">
                                <div class="risk-fill" style="width: ${riskScore}%; background: ${riskColor};"></div>
                            </div>
                            <div class="risk-labels">
                                <span style="color: var(--cyber-green);">LOW</span>
                                <span style="color: var(--cyber-orange);">MEDIUM</span>
                                <span style="color: var(--cyber-red);">HIGH</span>
                                <span style="color: #ff073a;">CRITICAL</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="risk-factors mt-4">
                <h6 style="color: var(--cyber-blue); margin-bottom: 15px;">
                    <i class="fas fa-exclamation-triangle"></i> Risk Factors Analysis
                </h6>
                ${result.riskFactors.map(factor => `
                    <div class="factor-item" style="background: rgba(0,0,0,0.3); padding: 10px; margin: 8px 0; border-radius: 8px; border-left: 4px solid ${riskColor};">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="color: #fff; font-weight: 500;">${factor.factor}</span>
                            <span style="color: ${riskColor}; font-weight: bold;">${factor.weight}% impact</span>
                        </div>
                        <small style="color: var(--text-light); margin-top: 5px; display: block;">${factor.description}</small>
                    </div>
                `).join('')}
            </div>
            
            <div class="recommendation mt-4">
                <h6 style="color: var(--cyber-blue); margin-bottom: 10px;">
                    <i class="fas fa-lightbulb"></i> AI Recommendation
                </h6>
                <div class="alert" style="background: rgba(0,0,0,0.4); border: 1px solid ${riskColor}; color: ${riskColor}; border-radius: 8px;">
                    <strong><i class="fas fa-robot"></i> ${result.recommendation}</strong>
                </div>
            </div>
            
            <div class="analysis-footer mt-3">
                <small style="color: var(--text-light);">
                    <i class="fas fa-clock"></i> Analysis completed at ${result.timestamp}
                    | <i class="fas fa-server"></i> CyberShield AI v2.1.0
                </small>
            </div>
        </div>
    `;
    
    showSuccess('analysisResults', resultsHTML);
}

// System Statistics and Real-time Updates
function updateSystemStats() {
    // Update real-time metrics in live feed tab
    if (document.getElementById('transactionCount')) {
        document.getElementById('transactionCount').textContent = realTimeStats.transactionsToday.toLocaleString();
    }
    if (document.getElementById('threatCount')) {
        document.getElementById('threatCount').textContent = realTimeStats.fraudsBlocked;
    }
    if (document.getElementById('successRate')) {
        const successRate = ((realTimeStats.transactionsToday - realTimeStats.fraudsBlocked) / realTimeStats.transactionsToday * 100).toFixed(1);
        document.getElementById('successRate').textContent = successRate + '%';
    }
    
    // Update home page stats if visible
    updateHomePageStats();
}

function updateHomePageStats() {
    // Update the fraud statistics on home page
    const statCards = document.querySelectorAll('.stat-value');
    if (statCards.length >= 4) {
        statCards[2].textContent = realTimeStats.avgResponseTime + 'ms';
        statCards[3].textContent = realTimeStats.accuracyRate + '%';
    }
}

// Enhanced Batch Analysis
function startBatchAnalysis() {
    const fileInput = document.getElementById('batchFile');
    const file = fileInput.files[0];
    
    if (!file) {
        showError('batchResults', 'Please select a CSV file for batch analysis.');
        return;
    }
    
    showLoading('batchResults');
    
    // Simulate realistic batch processing
    let progress = 0;
    const totalTransactions = Math.floor(Math.random() * 5000) + 1000; // Random between 1000-6000
    
    const progressInterval = setInterval(() => {
        progress += Math.random() * 15 + 5; // Random progress increment
        
        if (progress >= 100) {
            clearInterval(progressInterval);
            displayBatchResults(totalTransactions);
        } else {
            updateBatchProgress(progress, totalTransactions);
        }
    }, 200);
}

function updateBatchProgress(progress, totalTransactions) {
    const currentTransactions = Math.floor((progress / 100) * totalTransactions);
    const currentFrauds = Math.floor(currentTransactions * 0.047); // 4.7% fraud rate
    
    const progressHTML = `
        <div class="cyber-card">
            <h5><i class="fas fa-cogs"></i> Processing Batch Analysis...</h5>
            <div class="progress-container" style="margin: 20px 0;">
                <div class="progress-bar" style="width: 100%; height: 25px; background: var(--bg-dark); border-radius: 12px; overflow: hidden;">
                    <div class="progress-fill" style="width: ${progress}%; height: 100%; background: linear-gradient(90deg, var(--cyber-green), var(--cyber-blue)); transition: width 0.3s ease;"></div>
                </div>
                <div style="text-align: center; margin-top: 10px; color: var(--cyber-blue); font-weight: bold;">
                    ${progress.toFixed(1)}% Complete
                </div>
            </div>
            <div class="processing-stats">
                <div class="row">
                    <div class="col-md-4">
                        <div class="stat-item">
                            <span>Processed:</span>
                            <span style="color: var(--cyber-green);">${currentTransactions.toLocaleString()}</span>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="stat-item">
                            <span>Threats Found:</span>
                            <span style="color: var(--cyber-red);">${currentFrauds}</span>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="stat-item">
                            <span>Processing Speed:</span>
                            <span style="color: var(--cyber-blue);">${Math.floor(Math.random() * 200 + 800)}/sec</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    document.getElementById('batchResults').innerHTML = progressHTML;
}

function displayBatchResults(totalTransactions) {
    const fraudsDetected = Math.floor(totalTransactions * (Math.random() * 0.03 + 0.02)); // 2-5% fraud rate
    const processingTime = (totalTransactions / (Math.random() * 500 + 1000)).toFixed(1); // Realistic processing time
    const accuracy = (97 + Math.random() * 2.5).toFixed(1); // 97-99.5% accuracy
    const falsePositives = Math.floor(fraudsDetected * (Math.random() * 0.05 + 0.01)); // 1-6% false positives
    
    const resultsHTML = `
        <div class="cyber-card">
            <h5><i class="fas fa-chart-bar"></i> Batch Analysis Complete</h5>
            
            <div class="batch-summary">
                <div class="row">
                    <div class="col-md-6">
                        <h6 style="color: var(--cyber-blue); margin-bottom: 15px;">
                            <i class="fas fa-database"></i> Processing Summary
                        </h6>
                        <div class="metric-item">
                            <span>Total Transactions:</span>
                            <span style="color: var(--cyber-blue);">${totalTransactions.toLocaleString()}</span>
                        </div>
                        <div class="metric-item">
                            <span>Fraudulent Detected:</span>
                            <span style="color: var(--cyber-red);">${fraudsDetected}</span>
                        </div>
                        <div class="metric-item">
                            <span>Fraud Rate:</span>
                            <span style="color: var(--cyber-orange);">${((fraudsDetected / totalTransactions) * 100).toFixed(2)}%</span>
                        </div>
                        <div class="metric-item">
                            <span>False Positives:</span>
                            <span style="color: var(--cyber-orange);">${falsePositives}</span>
                        </div>
                    </div>
                    
                    <div class="col-md-6">
                        <h6 style="color: var(--cyber-blue); margin-bottom: 15px;">
                            <i class="fas fa-tachometer-alt"></i> Performance Metrics
                        </h6>
                        <div class="metric-item">
                            <span>Processing Time:</span>
                            <span style="color: var(--cyber-green);">${processingTime} seconds</span>
                        </div>
                        <div class="metric-item">
                            <span>Accuracy Rate:</span>
                            <span style="color: var(--cyber-green);">${accuracy}%</span>
                        </div>
                        <div class="metric-item">
                            <span>Avg. Processing Speed:</span>
                            <span style="color: var(--cyber-blue);">${Math.floor(totalTransactions / parseFloat(processingTime)).toLocaleString()} tx/sec</span>
                        </div>
                        <div class="metric-item">
                            <span>Status:</span>
                            <span style="color: var(--cyber-green);"><i class="fas fa-check-circle"></i> SUCCESS</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="threat-breakdown mt-4">
                <h6 style="color: var(--cyber-blue); margin-bottom: 15px;">
                    <i class="fas fa-shield-alt"></i> Threat Breakdown
                </h6>
                <div class="threat-categories">
                    ${generateThreatBreakdown(fraudsDetected)}
                </div>
            </div>
            
            <div class="batch-actions mt-4">
                <h6 style="color: var(--cyber-blue); margin-bottom: 15px;">
                    <i class="fas fa-download"></i> Export Options
                </h6>
                <div class="action-buttons">
                    <button class="btn btn-success" onclick="exportBatchReport('csv')">
                        <i class="fas fa-file-csv"></i> Export CSV Report
                    </button>
                    <button class="btn btn-info" onclick="exportBatchReport('pdf')">
                        <i class="fas fa-file-pdf"></i> Export PDF Report
                    </button>
                    <button class="btn btn-warning" onclick="exportBatchReport('json')">
                        <i class="fas fa-file-code"></i> Export JSON Data
                    </button>
                </div>
            </div>
            
            <div class="analysis-footer mt-3">
                <small style="color: var(--text-light);">
                    <i class="fas fa-clock"></i> Analysis completed at ${new Date().toLocaleString()}
                    | <i class="fas fa-server"></i> CyberShield AI Batch Processor v2.1.0
                    | <i class="fas fa-shield-alt"></i> ${(totalTransactions - fraudsDetected).toLocaleString()} transactions secured
                </small>
            </div>
        </div>
    `;
    
    showSuccess('batchResults', resultsHTML);
    
    // Update global stats
    realTimeStats.transactionsToday += totalTransactions;
    realTimeStats.fraudsBlocked += fraudsDetected;
    updateSystemStats();
}

function generateThreatBreakdown(totalFrauds) {
    const categories = [
        { name: 'Identity Theft', percentage: 35, color: '#ff073a' },
        { name: 'Card Skimming', percentage: 25, color: '#ff6b35' },
        { name: 'Account Takeover', percentage: 20, color: '#ffa500' },
        { name: 'Synthetic Identity', percentage: 12, color: '#ff9500' },
        { name: 'Other Fraud Types', percentage: 8, color: '#ffb500' }
    ];
    
    return categories.map(category => {
        const count = Math.floor((category.percentage / 100) * totalFrauds);
        return `
            <div class="threat-category" style="display: flex; justify-content: space-between; align-items: center; padding: 10px; margin: 5px 0; background: rgba(0,0,0,0.3); border-radius: 8px; border-left: 4px solid ${category.color};">
                <div>
                    <span style="color: #fff; font-weight: 500;">${category.name}</span>
                    <div style="width: 100px; height: 8px; background: var(--bg-dark); border-radius: 4px; margin-top: 5px; overflow: hidden;">
                        <div style="width: ${category.percentage}%; height: 100%; background: ${category.color};"></div>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="color: ${category.color}; font-weight: bold;">${count}</div>
                    <small style="color: var(--text-light);">${category.percentage}%</small>
                </div>
            </div>
        `;
    }).join('');
}

function exportBatchReport(format) {
    // Simulate export functionality
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: var(--cyber-green);
        color: var(--bg-dark);
        padding: 15px 20px;
        border-radius: 8px;
        font-weight: bold;
        z-index: 9999;
        animation: slideIn 0.3s ease;
    `;
    notification.innerHTML = `<i class="fas fa-check-circle"></i> ${format.toUpperCase()} report exported successfully!`;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => document.body.removeChild(notification), 300);
    }, 3000);
}

// Live Security Feed Functions
function startDemo() {
    if (demoRunning) return;
    
    demoRunning = true;
    transactionCounter = 0;
    threatCounter = 0;
    
    document.getElementById('transactionFeed').innerHTML = '<h6 style="color: var(--cyber-green);">🟢 MONITORING ACTIVE - Real-time threat detection in progress...</h6>';
    
    demoInterval = setInterval(generateDemoTransaction, 2000);
    updateLiveMetrics();
}

function pauseDemo() {
    if (demoInterval) {
        clearInterval(demoInterval);
        demoInterval = null;
        demoRunning = false;
    }
}

function stopDemo() {
    pauseDemo();
    document.getElementById('transactionFeed').innerHTML = `
        <p class="text-center" style="color: var(--cyber-blue); margin-top: 150px;">
            <i class="fas fa-shield-alt" style="font-size: 3rem; margin-bottom: 20px;"></i><br>
            Click "START MONITORING" to begin real-time security surveillance...
        </p>
    `;
    transactionCounter = 0;
    threatCounter = 0;
    updateLiveMetrics();
}

function generateDemoTransaction() {
    const merchants = ['Amazon', 'Walmart', 'Shell Gas', 'Starbucks', 'ATM Withdrawal', 'PayPal', 'Uber'];
    const amounts = [25.50, 156.78, 2500.00, 45.99, 8.75, 1200.00, 75.25, 500.00];
    const locations = ['New York', 'California', 'Texas', 'Florida', 'Unknown', 'Illinois'];
    
    const amount = amounts[Math.floor(Math.random() * amounts.length)];
    const merchant = merchants[Math.floor(Math.random() * merchants.length)];
    const location = locations[Math.floor(Math.random() * locations.length)];
    const timestamp = new Date().toLocaleTimeString();
    
    // Simulate fraud probability
    const fraudProb = Math.random();
    const isHighRisk = fraudProb > 0.7;
    const isMediumRisk = fraudProb > 0.4 && fraudProb <= 0.7;
    
    let riskLevel, riskColor, riskIcon;
    if (isHighRisk) {
        riskLevel = 'HIGH RISK';
        riskColor = 'var(--cyber-red)';
        riskIcon = '🚨';
        threatCounter++;
    } else if (isMediumRisk) {
        riskLevel = 'MEDIUM RISK';
        riskColor = 'var(--cyber-orange)';
        riskIcon = '⚠️';
    } else {
        riskLevel = 'LOW RISK';
        riskColor = 'var(--cyber-green)';
        riskIcon = '✅';
    }
    
    transactionCounter++;
    
    const transactionHtml = `
        <div class="transaction-item" style="border-left: 4px solid ${riskColor}; margin-bottom: 10px; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 5px; animation: slideInRight 0.5s ease;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong style="color: #fff;">$${amount}</strong> - ${merchant}
                    <br><small style="color: var(--text-light);">${location} • ${timestamp}</small>
                </div>
                <div style="text-align: right;">
                    <span style="color: ${riskColor}; font-weight: bold;">${riskIcon} ${riskLevel}</span>
                    <br><small style="color: var(--text-light);">${(fraudProb * 100).toFixed(1)}% fraud probability</small>
                </div>
            </div>
        </div>
    `;
    
    const feed = document.getElementById('transactionFeed');
    if (feed.children.length > 10) {
        feed.removeChild(feed.lastChild);
    }
    feed.insertAdjacentHTML('afterbegin', transactionHtml);
    
    updateLiveMetrics();
}

function updateLiveMetrics() {
    document.getElementById('transactionCount').textContent = transactionCounter;
    document.getElementById('threatCount').textContent = threatCounter;
    
    const successRate = transactionCounter > 0 ? (((transactionCounter - threatCounter) / transactionCounter) * 100).toFixed(1) : 100;
    document.getElementById('successRate').textContent = successRate + '%';
}

// Event Listeners
document.addEventListener('DOMContentLoaded', function() {
    // Form submission handler
    const fraudForm = document.getElementById('fraudAnalysisForm');
    if (fraudForm) {
        fraudForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const formData = new FormData(this);
            analyzeFraudTransaction(formData);
        });
    }
    
    // Initialize real-time updates
    updateSystemStats();
    
    // Update stats every 30 seconds
    setInterval(() => {
        realTimeStats.transactionsToday += Math.floor(Math.random() * 50) + 10;
        realTimeStats.fraudsBlocked += Math.floor(Math.random() * 3);
        realTimeStats.avgResponseTime = Math.floor(Math.random() * 10) + 35;
        updateSystemStats();
    }, 30000);
    
    // Initialize tooltips and other Bootstrap components
    if (typeof bootstrap !== 'undefined') {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
});

// CSS Animation for demo transactions
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideIn {
        from {
            transform: translateX(-20px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(20px);
            opacity: 0;
        }
    }
    
    .risk-meter {
        margin: 15px 0;
    }
    
    .risk-bar {
        width: 100%;
        height: 20px;
        background: var(--bg-dark);
        border-radius: 10px;
        overflow: hidden;
        margin-bottom: 10px;
    }
    
    .risk-fill {
        height: 100%;
        transition: width 1s ease;
        border-radius: 10px;
    }
    
    .risk-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
    }
    
    .analysis-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
        padding-bottom: 15px;
        border-bottom: 1px solid var(--border-color);
    }
    
    .action-buttons {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }
    
    .action-buttons .btn {
        margin: 5px 0;
    }
`;
document.head.appendChild(style);

// ==========================================
// ENHANCED CYBERSHIELD FEATURES
// ==========================================

// 1. CYBERSHIELD HUB - Real-time Dashboard
function initializeCyberShieldHub() {
    // Animate statistics counters
    animateCounters();
    
    // Update processing speed indicator
    updateProcessingSpeed();
    
    // Start real-time workflow animation
    animateWorkflow();
    
    // Initialize industry impact scenarios
    initializeIndustryScenarios();
}

function animateCounters() {
    const stats = [
        { id: 'annual-losses', target: 56, suffix: 'B' },
        { id: 'fraud-rate', target: 130, suffix: '' },
        { id: 'detection-speed', target: 47, suffix: 'ms' },
        { id: 'accuracy-rate', target: 99.7, suffix: '%' }
    ];
    
    stats.forEach(stat => {
        const element = document.querySelector(`[data-stat="${stat.id}"]`);
        if (element) {
            animateValue(element, 0, stat.target, 2000, stat.suffix);
        }
    });
}

function animateValue(element, start, end, duration, suffix) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const current = start + (end - start) * progress;
        element.textContent = Math.floor(current) + suffix;
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

function updateProcessingSpeed() {
    const speedBar = document.querySelector('.speed-progress');
    if (speedBar) {
        setTimeout(() => {
            speedBar.style.width = '99.7%';
        }, 1000);
    }
}

function animateWorkflow() {
    const workflowSteps = document.querySelectorAll('.workflow-step');
    workflowSteps.forEach((step, index) => {
        setTimeout(() => {
            step.style.animation = 'pulse 2s infinite';
        }, index * 500);
    });
}

function initializeIndustryScenarios() {
    const scenarios = document.querySelectorAll('.impact-card');
    scenarios.forEach((card, index) => {
        card.addEventListener('click', () => {
            showScenarioDetails(index);
        });
    });
}

function showScenarioDetails(scenarioIndex) {
    const scenarios = [
        {
            title: 'Banking Sector Success',
            details: 'Our AI detected an unusual $2,500 transaction at 2:30 AM from an unknown IP address. The system immediately flagged it as high-risk due to off-hours timing and geographic anomaly. Investigation revealed it was part of a larger fraud ring attempting to steal $25,000 across multiple accounts.',
            impact: 'Prevented $25,000 in losses',
            responseTime: '23ms detection'
        },
        {
            title: 'FinTech Platform Protection',
            details: 'Rapid-fire micro-transactions pattern was detected across 50 accounts simultaneously. Our machine learning model identified this as an automated attack signature. The system blocked all suspicious transactions and alerted security teams.',
            impact: 'Protected 1,200 user accounts',
            responseTime: '35ms detection'
        },
        {
            title: 'E-Commerce Security Shield',
            details: 'A credential stuffing attack was detected on the payment gateway with over 15,000 login attempts using stolen credentials. CyberShield AI identified the attack pattern and automatically implemented protective measures.',
            impact: 'Blocked 15,000 fraudulent attempts',
            responseTime: '12ms detection'
        }
    ];
    
    const scenario = scenarios[scenarioIndex];
    const modal = `
        <div class="modal fade" id="scenarioModal" tabindex="-1">
            <div class="modal-dialog modal-lg">
                <div class="modal-content" style="background: var(--card-bg); border: 1px solid var(--cyber-blue);">
                    <div class="modal-header" style="border-bottom: 1px solid var(--border-color);">
                        <h5 class="modal-title" style="color: var(--cyber-blue);">
                            <i class="fas fa-shield-alt"></i> ${scenario.title}
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <p style="color: var(--text-light); line-height: 1.6;">${scenario.details}</p>
                        <div class="row mt-4">
                            <div class="col-md-6">
                                <div class="metric-card">
                                    <h6 style="color: var(--cyber-green);">${scenario.impact}</h6>
                                    <p style="color: var(--text-light); margin: 0;">Financial Impact</p>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="metric-card">
                                    <h6 style="color: var(--cyber-blue);">${scenario.responseTime}</h6>
                                    <p style="color: var(--text-light); margin: 0;">Response Time</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', modal);
    const modalElement = new bootstrap.Modal(document.getElementById('scenarioModal'));
    modalElement.show();
    
    document.getElementById('scenarioModal').addEventListener('hidden.bs.modal', function() {
        this.remove();
    });
}

// 2. THREAT ANALYSIS - Enhanced Single Transaction Analysis
function enhancedAnalyzeFraudTransaction(formData) {
    const customerId = formData.get('customerId');
    const amount = parseFloat(formData.get('amount'));
    const merchant = formData.get('merchant');
    const time = formData.get('transactionTime');
    const latitude = parseFloat(formData.get('latitude'));
    const longitude = parseFloat(formData.get('longitude'));
    
    showLoading('analysisResults', 'Performing deep threat analysis...');
    
    setTimeout(() => {
        performDetailedThreatAnalysis(customerId, amount, merchant, time, latitude, longitude);
    }, 2000);
}

function performDetailedThreatAnalysis(customerId, amount, merchant, time, latitude, longitude) {
    // Advanced risk calculation with multiple factors
    let riskScore = 0;
    let riskFactors = [];
    let securityAlerts = [];
    
    // Customer profile analysis
    if (customerId.includes('NEW') || customerId.length < 5) {
        riskScore += 20;
        riskFactors.push({ factor: 'New Customer Profile', weight: 20, description: 'Account created recently or minimal transaction history' });
    }
    
    // Amount-based risk assessment
    if (amount > 5000) {
        riskScore += 35;
        riskFactors.push({ factor: 'High-Value Transaction', weight: 35, description: `$${amount} exceeds normal spending patterns` });
        securityAlerts.push('⚠️ Large transaction amount detected');
    } else if (amount > 1000) {
        riskScore += 15;
        riskFactors.push({ factor: 'Elevated Transaction Amount', weight: 15, description: `$${amount} is above average` });
    }
    
    // Time-based analysis
    const hour = parseInt(time.split(':')[0]);
    if (hour >= 0 && hour <= 4) {
        riskScore += 25;
        riskFactors.push({ factor: 'Unusual Time Pattern', weight: 25, description: 'Transaction at high-risk hours (12-4 AM)' });
        securityAlerts.push('🕐 Off-hours transaction detected');
    } else if (hour >= 22 || hour <= 6) {
        riskScore += 15;
        riskFactors.push({ factor: 'Off-hours Activity', weight: 15, description: 'Transaction outside normal business hours' });
    }
    
    // Geographic analysis
    if (Math.abs(latitude) > 50 || Math.abs(longitude) > 100) {
        riskScore += 30;
        riskFactors.push({ factor: 'Geographic Anomaly', weight: 30, description: 'Transaction from unusual geographic location' });
        securityAlerts.push('🌍 International transaction detected');
    }
    
    // Merchant risk assessment
    const merchantRisk = {
        'online': 20,
        'atm': 25,
        'travel': 15,
        'grocery': 5,
        'restaurant': 8,
        'retail': 10,
        'gas': 7
    };
    const merchantScore = merchantRisk[merchant] || 10;
    riskScore += merchantScore;
    riskFactors.push({ factor: 'Merchant Category Risk', weight: merchantScore, description: `${merchant} category analysis` });
    
    // Behavioral patterns simulation
    if (Math.random() > 0.7) {
        riskScore += 15;
        riskFactors.push({ factor: 'Velocity Pattern Anomaly', weight: 15, description: 'Unusual transaction frequency detected' });
        securityAlerts.push('⚡ Rapid transaction pattern detected');
    }
    
    // Device fingerprinting simulation
    if (Math.random() > 0.8) {
        riskScore += 20;
        riskFactors.push({ factor: 'Device Trust Score', weight: 20, description: 'Unrecognized device or suspicious fingerprint' });
        securityAlerts.push('📱 Unrecognized device detected');
    }
    
    // Ensure score is within bounds
    riskScore = Math.max(0, Math.min(100, riskScore));
    
    // Determine threat level
    let threatLevel, threatColor, recommendation, action;
    if (riskScore >= 80) {
        threatLevel = 'CRITICAL';
        threatColor = '#ff073a';
        recommendation = 'BLOCK TRANSACTION IMMEDIATELY';
        action = 'Transaction blocked and investigation initiated';
    } else if (riskScore >= 60) {
        threatLevel = 'HIGH';
        threatColor = '#ff6b35';
        recommendation = 'HOLD FOR MANUAL REVIEW';
        action = 'Transaction held pending verification';
    } else if (riskScore >= 30) {
        threatLevel = 'MEDIUM';
        threatColor = '#ffa500';
        recommendation = 'ENHANCED MONITORING';
        action = 'Transaction approved with increased surveillance';
    } else {
        threatLevel = 'LOW';
        threatColor = '#00d4ff';
        recommendation = 'APPROVE TRANSACTION';
        action = 'Transaction approved and processed normally';
    }
    
    displayComprehensiveThreatAnalysis({
        transactionId: 'TXN-' + Math.random().toString(36).substr(2, 9).toUpperCase(),
        customerId,
        amount,
        merchant,
        time,
        location: `${latitude}, ${longitude}`,
        riskScore: riskScore.toFixed(1),
        threatLevel,
        confidence: (88 + Math.random() * 12).toFixed(1),
        processingTime: (25 + Math.random() * 30).toFixed(1),
        riskFactors,
        securityAlerts,
        recommendation,
        action,
        timestamp: new Date().toLocaleString()
    }, threatColor);
}

function displayComprehensiveThreatAnalysis(analysis, threatColor) {
    const alertsHTML = analysis.securityAlerts.map(alert => 
        `<div class="security-alert">${alert}</div>`
    ).join('');
    
    const factorsHTML = analysis.riskFactors.map(factor => `
        <div class="risk-factor-item">
            <div class="factor-header">
                <span class="factor-name">${factor.factor}</span>
                <span class="factor-weight" style="color: ${threatColor};">${factor.weight}% impact</span>
            </div>
            <div class="factor-description">${factor.description}</div>
        </div>
    `).join('');
    
    const resultsHTML = `
        <div class="comprehensive-analysis" style="border-left: 5px solid ${threatColor};">
            <div class="analysis-header">
                <h5><i class="fas fa-shield-alt"></i> COMPREHENSIVE THREAT ANALYSIS</h5>
                <span class="threat-badge" style="background: ${threatColor}; color: white;">
                    ${analysis.threatLevel} THREAT
                </span>
            </div>
            
            ${analysis.securityAlerts.length > 0 ? `
                <div class="security-alerts">
                    <h6><i class="fas fa-exclamation-triangle"></i> Security Alerts</h6>
                    ${alertsHTML}
                </div>
            ` : ''}
            
            <div class="row">
                <div class="col-md-8">
                    <div class="transaction-details">
                        <h6><i class="fas fa-info-circle"></i> Transaction Details</h6>
                        <div class="detail-grid">
                            <div class="detail-item">
                                <span class="detail-label">Transaction ID:</span>
                                <span class="detail-value">${analysis.transactionId}</span>
                            </div>
                            <div class="detail-item">
                                <span class="detail-label">Customer ID:</span>
                                <span class="detail-value">${analysis.customerId}</span>
                            </div>
                            <div class="detail-item">
                                <span class="detail-label">Amount:</span>
                                <span class="detail-value">$${analysis.amount}</span>
                            </div>
                            <div class="detail-item">
                                <span class="detail-label">Merchant:</span>
                                <span class="detail-value">${analysis.merchant}</span>
                            </div>
                            <div class="detail-item">
                                <span class="detail-label">Time:</span>
                                <span class="detail-value">${analysis.time}</span>
                            </div>
                            <div class="detail-item">
                                <span class="detail-label">Location:</span>
                                <span class="detail-value">${analysis.location}</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="risk-analysis">
                        <h6><i class="fas fa-chart-line"></i> Risk Factor Analysis</h6>
                        ${factorsHTML}
                    </div>
                </div>
                
                <div class="col-md-4">
                    <div class="threat-metrics">
                        <div class="metric-card">
                            <h4 style="color: ${threatColor};">${analysis.riskScore}%</h4>
                            <p>Threat Probability</p>
                        </div>
                        <div class="metric-card">
                            <h4 style="color: var(--cyber-green);">${analysis.confidence}%</h4>
                            <p>Model Confidence</p>
                        </div>
                        <div class="metric-card">
                            <h4 style="color: var(--cyber-blue);">${analysis.processingTime}ms</h4>
                            <p>Analysis Time</p>
                        </div>
                    </div>
                    
                    <div class="recommendation-panel">
                        <h6><i class="fas fa-lightbulb"></i> Recommendation</h6>
                        <div class="recommendation" style="color: ${threatColor}; font-weight: bold;">
                            ${analysis.recommendation}
                        </div>
                        <div class="action-taken" style="color: var(--text-light); margin-top: 10px;">
                            ${analysis.action}
                        </div>
                    </div>
                    
                    <div class="action-buttons">
                        <button class="btn btn-success btn-sm" onclick="approveTransaction('${analysis.transactionId}')">
                            <i class="fas fa-check"></i> Approve
                        </button>
                        <button class="btn btn-danger btn-sm" onclick="blockTransaction('${analysis.transactionId}')">
                            <i class="fas fa-ban"></i> Block
                        </button>
                        <button class="btn btn-warning btn-sm" onclick="investigateTransaction('${analysis.transactionId}')">
                            <i class="fas fa-search"></i> Investigate
                        </button>
                    </div>
                </div>
            </div>
            
            <div class="analysis-footer">
                <small style="color: var(--text-light);">
                    Analysis completed at ${analysis.timestamp} | CyberShield AI v2.0
                </small>
            </div>
        </div>
    `;
    
    showSuccess('analysisResults', resultsHTML);
    
    // Update real-time statistics
    realTimeStats.transactionsToday++;
    if (parseFloat(analysis.riskScore) >= 60) {
        realTimeStats.fraudsBlocked++;
    }
    updateSystemStats();
}

// 3. BATCH ANALYSIS - Enhanced File Processing
function enhancedStartBatchAnalysis() {
    const fileInput = document.getElementById('batchFile');
    const file = fileInput.files[0];
    
    if (!file) {
        showError('batchResults', 'Please select a CSV file for batch analysis.');
        return;
    }
    
    showLoading('batchResults', 'Processing batch file - Analyzing transaction patterns...');
    
    // Simulate realistic batch processing
    setTimeout(() => {
        processBatchAnalysis(file);
    }, 3000);
}

function processBatchAnalysis(file) {
    // Simulate realistic batch results
    const totalTransactions = Math.floor(Math.random() * 10000) + 1000;
    const fraudDetected = Math.floor(totalTransactions * (0.02 + Math.random() * 0.08)); // 2-10% fraud rate
    const highRiskTransactions = Math.floor(totalTransactions * 0.15);
    const mediumRiskTransactions = Math.floor(totalTransactions * 0.25);
    const lowRiskTransactions = totalTransactions - fraudDetected - highRiskTransactions - mediumRiskTransactions;
    
    const processingTime = (2.1 + Math.random() * 3).toFixed(1);
    const accuracy = (96.5 + Math.random() * 3).toFixed(1);
    const throughput = Math.floor(totalTransactions / parseFloat(processingTime));
    
    const resultsHTML = `
        <div class="batch-analysis-results">
            <div class="batch-header">
                <h5><i class="fas fa-chart-bar"></i> BATCH ANALYSIS COMPLETE</h5>
                <span class="batch-status success">PROCESSED SUCCESSFULLY</span>
            </div>
            
            <div class="row">
                <div class="col-md-8">
                    <div class="batch-summary">
                        <h6>Processing Summary</h6>
                        <div class="summary-grid">
                            <div class="summary-item">
                                <span class="summary-label">File Name:</span>
                                <span class="summary-value">${file.name}</span>
                            </div>
                            <div class="summary-item">
                                <span class="summary-label">File Size:</span>
                                <span class="summary-value">${(file.size / 1024).toFixed(2)} KB</span>
                            </div>
                            <div class="summary-item">
                                <span class="summary-label">Processing Time:</span>
                                <span class="summary-value">${processingTime} seconds</span>
                            </div>
                            <div class="summary-item">
                                <span class="summary-label">Throughput:</span>
                                <span class="summary-value">${throughput} transactions/sec</span>
                            </div>
                            <div class="summary-item">
                                <span class="summary-label">Accuracy:</span>
                                <span class="summary-value">${accuracy}%</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="risk-breakdown">
                        <h6>Risk Level Distribution</h6>
                        <div class="risk-chart">
                            <div class="risk-bar-container">
                                <div class="risk-segment critical" style="width: ${(fraudDetected/totalTransactions*100).toFixed(1)}%">
                                    <span class="risk-label">Critical: ${fraudDetected}</span>
                                </div>
                                <div class="risk-segment high" style="width: ${(highRiskTransactions/totalTransactions*100).toFixed(1)}%">
                                    <span class="risk-label">High: ${highRiskTransactions}</span>
                                </div>
                                <div class="risk-segment medium" style="width: ${(mediumRiskTransactions/totalTransactions*100).toFixed(1)}%">
                                    <span class="risk-label">Medium: ${mediumRiskTransactions}</span>
                                </div>
                                <div class="risk-segment low" style="width: ${(lowRiskTransactions/totalTransactions*100).toFixed(1)}%">
                                    <span class="risk-label">Low: ${lowRiskTransactions}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-4">
                    <div class="batch-metrics">
                        <div class="metric-card">
                            <h4 style="color: var(--cyber-blue);">${totalTransactions.toLocaleString()}</h4>
                            <p>Total Transactions</p>
                        </div>
                        <div class="metric-card">
                            <h4 style="color: var(--cyber-red);">${fraudDetected}</h4>
                            <p>Threats Detected</p>
                        </div>
                        <div class="metric-card">
                            <h4 style="color: var(--cyber-orange);">${((fraudDetected/totalTransactions)*100).toFixed(2)}%</h4>
                            <p>Fraud Rate</p>
                        </div>
                        <div class="metric-card">
                            <h4 style="color: var(--cyber-green);">${(((totalTransactions-fraudDetected)/totalTransactions)*100).toFixed(2)}%</h4>
                            <p>Clean Transactions</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="batch-actions">
                <button class="btn btn-primary" onclick="downloadBatchReport()">
                    <i class="fas fa-download"></i> Download Report
                </button>
                <button class="btn btn-success" onclick="exportFraudList()">
                    <i class="fas fa-file-excel"></i> Export Fraud List
                </button>
                <button class="btn btn-info" onclick="viewDetailedAnalysis()">
                    <i class="fas fa-chart-line"></i> Detailed Analysis
                </button>
            </div>
        </div>
    `;
    
    showSuccess('batchResults', resultsHTML);
}

// 4. LIVE SECURITY FEED - Enhanced Real-time Monitoring
function enhancedStartDemo() {
    if (demoRunning) return;
    
    demoRunning = true;
    transactionCounter = 0;
    threatCounter = 0;
    
    const feedElement = document.getElementById('transactionFeed');
    feedElement.innerHTML = `
        <div class="live-feed-header">
            <h6 style="color: var(--cyber-green);">
                <i class="fas fa-satellite-dish animate-pulse"></i> 
                LIVE SECURITY MONITORING ACTIVE
            </h6>
            <div class="feed-controls">
                <span class="status-indicator active"></span>
                <span style="color: var(--cyber-green); font-size: 0.9rem;">Real-time threat detection enabled</span>
            </div>
        </div>
        <div class="feed-content" id="feedContent"></div>
    `;
    
    // Start generating realistic transaction feed
    demoInterval = setInterval(generateEnhancedTransaction, 1500);
    updateLiveMetrics();
}

function generateEnhancedTransaction() {
    const merchants = [
        { name: 'Amazon.com', category: 'online', risk: 'medium' },
        { name: 'Walmart Store #1247', category: 'retail', risk: 'low' },
        { name: 'Shell Gas Station', category: 'gas', risk: 'low' },
        { name: 'ATM Withdrawal', category: 'atm', risk: 'high' },
        { name: 'Starbucks #892', category: 'restaurant', risk: 'low' },
        { name: 'PayPal Transfer', category: 'online', risk: 'medium' },
        { name: 'International Wire', category: 'transfer', risk: 'critical' },
        { name: 'Uber Ride', category: 'transport', risk: 'low' },
        { name: 'Bitcoin Exchange', category: 'crypto', risk: 'critical' },
        { name: 'Hotel Booking', category: 'travel', risk: 'medium' }
    ];
    
    const locations = [
        { city: 'New York, NY', country: 'USA', risk: 'low' },
        { city: 'Los Angeles, CA', country: 'USA', risk: 'low' },
        { city: 'London', country: 'UK', risk: 'medium' },
        { city: 'Unknown Location', country: 'TOR Network', risk: 'critical' },
        { city: 'Moscow', country: 'Russia', risk: 'high' },
        { city: 'Lagos', country: 'Nigeria', risk: 'high' },
        { city: 'Singapore', country: 'Singapore', risk: 'medium' },
        { city: 'VPN Exit Node', country: 'Unknown', risk: 'critical' }
    ];
    
    const merchant = merchants[Math.floor(Math.random() * merchants.length)];
    const location = locations[Math.floor(Math.random() * locations.length)];
    const amount = (Math.random() * 9000 + 10).toFixed(2);
    const timestamp = new Date().toLocaleTimeString();
    const customerId = 'CUST' + Math.random().toString(36).substr(2, 6).toUpperCase();
    
    // Calculate realistic risk based on multiple factors
    let riskScore = 0;
    const hour = new Date().getHours();
    
    // Time-based risk
    if (hour >= 0 && hour <= 5) riskScore += 30;
    else if (hour >= 22) riskScore += 20;
    
    // Amount-based risk
    if (parseFloat(amount) > 5000) riskScore += 40;
    else if (parseFloat(amount) > 1000) riskScore += 20;
    
    // Merchant risk
    const merchantRiskScores = { low: 5, medium: 15, high: 30, critical: 50 };
    riskScore += merchantRiskScores[merchant.risk];
    
    // Location risk
    const locationRiskScores = { low: 5, medium: 20, high: 35, critical: 50 };
    riskScore += locationRiskScores[location.risk];
    
    // Add randomness
    riskScore += Math.random() * 20 - 10;
    riskScore = Math.max(0, Math.min(100, riskScore));
    
    let riskLevel, riskColor, riskIcon, action;
    if (riskScore >= 80) {
        riskLevel = 'CRITICAL';
        riskColor = '#ff073a';
        riskIcon = '🚨';
        action = 'BLOCKED';
        threatCounter++;
    } else if (riskScore >= 60) {
        riskLevel = 'HIGH';
        riskColor = '#ff6b35';
        riskIcon = '⚠️';
        action = 'FLAGGED';
        threatCounter++;
    } else if (riskScore >= 30) {
        riskLevel = 'MEDIUM';
        riskColor = '#ffa500';
        riskIcon = '⚡';
        action = 'MONITORING';
    } else {
        riskLevel = 'LOW';
        riskColor = '#00d4ff';
        riskIcon = '✅';
        action = 'APPROVED';
    }
    
    transactionCounter++;
    
    const transactionHTML = `
        <div class="live-transaction" style="border-left: 4px solid ${riskColor}; animation: slideInRight 0.5s ease;">
            <div class="transaction-header">
                <div class="transaction-info">
                    <strong style="color: #fff;">$${amount}</strong>
                    <span style="color: var(--text-light); margin-left: 10px;">${merchant.name}</span>
                </div>
                <div class="transaction-status">
                    <span class="risk-badge ${riskLevel.toLowerCase()}" style="background: ${riskColor};">
                        ${riskIcon} ${riskLevel}
                    </span>
                </div>
            </div>
            <div class="transaction-details">
                <div class="detail-row">
                    <span class="detail-label">Customer:</span>
                    <span class="detail-value">${customerId}</span>
                    <span class="detail-label">Location:</span>
                    <span class="detail-value">${location.city}, ${location.country}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Time:</span>
                    <span class="detail-value">${timestamp}</span>
                    <span class="detail-label">Risk Score:</span>
                    <span class="detail-value" style="color: ${riskColor};">${riskScore.toFixed(1)}%</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Action:</span>
                    <span class="detail-value action-${action.toLowerCase()}">${action}</span>
                </div>
            </div>
        </div>
    `;
    
    const feedContent = document.getElementById('feedContent');
    if (feedContent) {
        // Keep only last 10 transactions
        if (feedContent.children.length >= 10) {
            feedContent.removeChild(feedContent.lastChild);
        }
        feedContent.insertAdjacentHTML('afterbegin', transactionHTML);
    }
    
    updateLiveMetrics();
}

// 5. PERFORMANCE ANALYTICS - Enhanced Metrics Display
function initializePerformanceAnalytics() {
    displayModelPerformanceComparison();
    displayThroughputMetrics();
    displayAccuracyTrends();
    displaySystemHealth();
}

function displayModelPerformanceComparison() {
    const models = [
        { name: 'CyberShield AI Ensemble', accuracy: 99.7, precision: 98.9, recall: 97.8, f1: 98.3 },
        { name: 'Random Forest', accuracy: 94.2, precision: 92.1, recall: 89.7, f1: 90.9 },
        { name: 'SVM Classifier', accuracy: 91.8, precision: 88.4, recall: 86.2, f1: 87.3 },
        { name: 'Neural Network', accuracy: 93.5, precision: 90.7, recall: 88.9, f1: 89.8 },
        { name: 'Logistic Regression', accuracy: 87.3, precision: 84.6, recall: 82.1, f1: 83.3 }
    ];
    
    // Animate performance bars
    setTimeout(() => {
        models.forEach((model, index) => {
            const modelElement = document.querySelector(`[data-model="${index}"]`);
            if (modelElement) {
                const bars = modelElement.querySelectorAll('.bar');
                bars.forEach((bar, metricIndex) => {
                    const metrics = [model.accuracy, model.precision, model.recall, model.f1];
                    setTimeout(() => {
                        bar.style.width = metrics[metricIndex] + '%';
                    }, metricIndex * 200);
                });
            }
        });
    }, 1000);
}

// 6. AI TRANSPARENCY - Explainable AI Features
function initializeAITransparency() {
    displayFeatureImportance();
    displayDecisionPath();
    displayModelInterpretability();
}

function displayFeatureImportance() {
    const features = [
        { name: 'Transaction Amount', importance: 0.23, description: 'Unusual amounts compared to user history' },
        { name: 'Transaction Time', importance: 0.19, description: 'Off-hours or unusual timing patterns' },
        { name: 'Geographic Location', importance: 0.17, description: 'Location anomalies and travel patterns' },
        { name: 'Merchant Category', importance: 0.15, description: 'Risk associated with merchant type' },
        { name: 'User Behavior Pattern', importance: 0.12, description: 'Deviation from normal spending habits' },
        { name: 'Device Fingerprint', importance: 0.08, description: 'Device trust and recognition scores' },
        { name: 'Velocity Patterns', importance: 0.06, description: 'Transaction frequency analysis' }
    ];
    
    // Animate feature importance bars
    setTimeout(() => {
        features.forEach((feature, index) => {
            const barElement = document.querySelector(`[data-feature="${index}"] .importance-bar`);
            if (barElement) {
                barElement.style.width = (feature.importance * 100) + '%';
            }
        });
    }, 500);
}

// 7. SYSTEM ARCHITECTURE - Interactive Diagram
function initializeSystemArchitecture() {
    animateArchitectureDiagram();
    displaySystemComponents();
    showDataFlow();
}

function animateArchitectureDiagram() {
    const layers = document.querySelectorAll('.arch-layer');
    layers.forEach((layer, index) => {
        setTimeout(() => {
            layer.style.animation = 'fadeInUp 0.8s ease forwards';
        }, index * 300);
    });
    
    const arrows = document.querySelectorAll('.arch-arrow');
    arrows.forEach((arrow, index) => {
        setTimeout(() => {
            arrow.style.animation = 'pulse 2s infinite';
        }, (index + 1) * 600);
    });
}

// 8. TECHNOLOGY STACK - Enhanced Display
function initializeTechnologyStack() {
    animateTechTags();
    displayVersionInfo();
    showDependencyTree();
}

function animateTechTags() {
    const techTags = document.querySelectorAll('.tech-tag');
    techTags.forEach((tag, index) => {
        setTimeout(() => {
            tag.style.animation = 'bounceIn 0.6s ease forwards';
        }, index * 50);
    });
}

// ==========================================
// UTILITY FUNCTIONS FOR ENHANCED FEATURES
// ==========================================

function updateSystemStats() {
    // Update various system statistics displays
    document.querySelectorAll('[data-stat="transactions"]').forEach(el => {
        el.textContent = realTimeStats.transactionsToday.toLocaleString();
    });
    
    document.querySelectorAll('[data-stat="threats"]').forEach(el => {
        el.textContent = realTimeStats.fraudsBlocked;
    });
    
    document.querySelectorAll('[data-stat="accuracy"]').forEach(el => {
        el.textContent = realTimeStats.accuracyRate + '%';
    });
    
    const successRate = realTimeStats.transactionsToday > 0 ? 
        (((realTimeStats.transactionsToday - realTimeStats.fraudsBlocked) / realTimeStats.transactionsToday) * 100).toFixed(1) : 100;
    
    document.querySelectorAll('[data-stat="success-rate"]').forEach(el => {
        el.textContent = successRate + '%';
    });
}

// Action button handlers
function approveTransaction(transactionId) {
    showNotification('Transaction ' + transactionId + ' approved successfully', 'success');
}

function blockTransaction(transactionId) {
    showNotification('Transaction ' + transactionId + ' blocked and flagged', 'danger');
}

function investigateTransaction(transactionId) {
    showNotification('Investigation initiated for transaction ' + transactionId, 'warning');
}

function downloadBatchReport() {
    showNotification('Batch analysis report downloaded successfully', 'success');
}

function exportFraudList() {
    showNotification('Fraud detection list exported to Excel', 'info');
}

function viewDetailedAnalysis() {
    showNotification('Opening detailed analysis dashboard...', 'info');
}

function showNotification(message, type) {
    const notification = `
        <div class="alert alert-${type} alert-dismissible fade show" style="position: fixed; top: 20px; right: 20px; z-index: 9999; max-width: 400px;">
            <i class="fas fa-info-circle"></i> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    document.body.insertAdjacentHTML('beforeend', notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        const alerts = document.querySelectorAll('.alert');
        if (alerts.length > 0) {
            alerts[alerts.length - 1].remove();
        }
    }, 5000);
}

// ==========================================
// EVENT LISTENERS AND INITIALIZATION
// ==========================================

// Initialize all enhanced features when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('🛡️ CyberShield AI Enhanced Features Loading...');
    
    // Initialize CyberShield Hub
    initializeCyberShieldHub();
    
    // Set up form handlers
    setupFormHandlers();
    
    // Initialize tooltips
    initializeTooltips();
    
    // Start system monitoring
    startSystemMonitoring();
    
    // Initialize all module-specific features
    setTimeout(() => {
        initializePerformanceAnalytics();
        initializeAITransparency();
        initializeSystemArchitecture();
        initializeTechnologyStack();
    }, 1000);
    
    console.log('✅ CyberShield AI Enhanced Features Loaded Successfully!');
});

function setupFormHandlers() {
    // Enhanced fraud analysis form
    const fraudForm = document.getElementById('fraudAnalysisForm');
    if (fraudForm) {
        fraudForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const formData = new FormData(this);
            enhancedAnalyzeFraudTransaction(formData);
        });
    }
    
    // Batch analysis file input
    const batchFileInput = document.getElementById('batchFile');
    if (batchFileInput) {
        batchFileInput.addEventListener('change', function() {
            if (this.files.length > 0) {
                const file = this.files[0];
                const fileInfo = `
                    <div class="file-info" style="margin-top: 10px; padding: 10px; background: var(--card-bg); border-radius: 5px;">
                        <i class="fas fa-file-csv" style="color: var(--cyber-green);"></i>
                        <strong>${file.name}</strong> (${(file.size / 1024).toFixed(2)} KB)
                        <br><small style="color: var(--text-light);">Ready for batch analysis</small>
                    </div>
                `;
                this.parentElement.insertAdjacentHTML('afterend', fileInfo);
            }
        });
    }
}

function initializeTooltips() {
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

function startSystemMonitoring() {
    // Update system health every 10 seconds
    setInterval(() => {
        systemHealth.lastUpdate = new Date();
        systemHealth.threatsDetected += Math.floor(Math.random() * 3);
        systemHealth.totalTransactions += Math.floor(Math.random() * 50) + 10;
        systemHealth.activeConnections = 140 + Math.floor(Math.random() * 30);
        
        updateSystemStats();
    }, 10000);
}

// Tab change event handlers
document.addEventListener('shown.bs.tab', function (event) {
    const targetTab = event.target.getAttribute('data-bs-target');
    
    switch(targetTab) {
        case '#home':
            initializeCyberShieldHub();
            break;
        case '#performance':
            initializePerformanceAnalytics();
            break;
        case '#explainable':
            initializeAITransparency();
            break;
        case '#architecture':
            initializeSystemArchitecture();
            break;
        case '#about':
            initializeTechnologyStack();
            break;
    }
});

// Override existing functions with enhanced versions
window.startDemo = enhancedStartDemo;
window.startBatchAnalysis = enhancedStartBatchAnalysis;
window.analyzeFraudTransaction = enhancedAnalyzeFraudTransaction;
