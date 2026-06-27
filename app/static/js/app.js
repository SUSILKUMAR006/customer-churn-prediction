// RetinaAI Client-Side Dashboard Application

// Outreach Table State
let currentPage = 1;
const recordsPerPage = 10;
let currentSearch = '';

// Initialize app on DOM content loaded
document.addEventListener('DOMContentLoaded', () => {
    // Load initial analytics
    fetchAnalytics();
    // Load outreach table
    loadOutreachTable();
});

// 1. Tab Switching Logic
function switchTab(tabName) {
    // Remove active class from all tabs and contents
    document.querySelectorAll('.nav-item').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    
    // Add active class to selected tab and content
    if (tabName === 'analytics') {
        document.getElementById('btn-analytics').classList.add('active');
        document.getElementById('tab-content-analytics').classList.add('active');
        fetchAnalytics(); // Refresh analytics when switching back
    } else if (tabName === 'predict') {
        document.getElementById('btn-predict').classList.add('active');
        document.getElementById('tab-content-predict').classList.add('active');
    } else if (tabName === 'outreach') {
        document.getElementById('btn-outreach').classList.add('active');
        document.getElementById('tab-content-outreach').classList.add('active');
        loadOutreachTable(); // Refresh table
    }
}

// 2. Fetch and Render Analytics (Tab 1)
async function fetchAnalytics() {
    try {
        const response = await fetch('/api/analytics');
        if (!response.ok) throw new Error('Failed to fetch analytics data');
        
        const data = await response.json();
        
        // Update KPIs
        document.getElementById('kpi-total-customers').textContent = data.total_customers.toLocaleString();
        document.getElementById('kpi-revenue-risk').textContent = '$' + data.monthly_revenue_at_risk.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
        document.getElementById('kpi-revenue-risk-pct').textContent = `${data.monthly_revenue_at_risk_pct}% of total billing`;
        document.getElementById('kpi-potential-savings').textContent = '$' + data.potential_monthly_savings.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
        
        // Update Cohorts
        const highCohort = data.cohorts['High Risk (>=70%)'] || {count: 0, pct: 0, revenue: 0};
        const medCohort = data.cohorts['Medium Risk (30-70%)'] || {count: 0, pct: 0, revenue: 0};
        const lowCohort = data.cohorts['Low Risk (<30%)'] || {count: 0, pct: 0, revenue: 0};
        
        // High Risk
        document.getElementById('cohort-high-count').textContent = highCohort.count.toLocaleString();
        document.getElementById('cohort-high-pct').textContent = `${highCohort.pct}%`;
        document.getElementById('cohort-high-bar').style.width = `${highCohort.pct}%`;
        document.getElementById('cohort-high-rev').textContent = '$' + highCohort.revenue.toLocaleString();
        
        // Medium Risk
        document.getElementById('cohort-medium-count').textContent = medCohort.count.toLocaleString();
        document.getElementById('cohort-medium-pct').textContent = `${medCohort.pct}%`;
        document.getElementById('cohort-medium-bar').style.width = `${medCohort.pct}%`;
        document.getElementById('cohort-medium-rev').textContent = '$' + medCohort.revenue.toLocaleString();
        
        // Low Risk
        document.getElementById('cohort-low-count').textContent = lowCohort.count.toLocaleString();
        document.getElementById('cohort-low-pct').textContent = `${lowCohort.pct}%`;
        document.getElementById('cohort-low-bar').style.width = `${lowCohort.pct}%`;
        document.getElementById('cohort-low-rev').textContent = '$' + lowCohort.revenue.toLocaleString();
        
    } catch (error) {
        console.error('Error loading analytics:', error);
    }
}

// 3. Predictor Form Input Controls
function updateTenureValue(value) {
    document.getElementById('tenure-val').textContent = value;
}

function togglePhoneService() {
    const phoneService = document.getElementById('PhoneService').value;
    const groupMultipleLines = document.getElementById('group-multiple-lines');
    const multipleLinesSelect = document.getElementById('MultipleLines');
    
    if (phoneService === 'No') {
        groupMultipleLines.style.opacity = '0.5';
        groupMultipleLines.style.pointerEvents = 'none';
        multipleLinesSelect.value = 'No phone service';
    } else {
        groupMultipleLines.style.opacity = '1';
        groupMultipleLines.style.pointerEvents = 'auto';
        if (multipleLinesSelect.value === 'No phone service') {
            multipleLinesSelect.value = 'No';
        }
    }
}

function toggleInternetService() {
    const internetService = document.getElementById('InternetService').value;
    const sectionAddons = document.getElementById('section-addons');
    const addonSelects = sectionAddons.querySelectorAll('select');
    
    if (internetService === 'No') {
        sectionAddons.style.opacity = '0.4';
        sectionAddons.style.pointerEvents = 'none';
        addonSelects.forEach(select => {
            select.value = 'No internet service';
        });
    } else {
        sectionAddons.style.opacity = '1';
        sectionAddons.style.pointerEvents = 'auto';
        addonSelects.forEach(select => {
            if (select.value === 'No internet service') {
                select.value = 'No';
            }
        });
    }
}

// 4. Submit Prediction Form (Tab 2)
async function submitPrediction(event) {
    event.preventDefault();
    
    const submitBtn = document.getElementById('btn-predict-submit');
    submitBtn.textContent = 'Analyzing...';
    submitBtn.disabled = true;
    
    // Construct payload
    const payload = {
        gender: document.getElementById('gender').value,
        SeniorCitizen: parseInt(document.getElementById('SeniorCitizen').value),
        Partner: document.getElementById('Partner').value,
        Dependents: document.getElementById('Dependents').value,
        tenure: parseInt(document.getElementById('tenure').value),
        PhoneService: document.getElementById('PhoneService').value,
        MultipleLines: document.getElementById('MultipleLines').value,
        InternetService: document.getElementById('InternetService').value,
        OnlineSecurity: document.getElementById('OnlineSecurity').value,
        OnlineBackup: document.getElementById('OnlineBackup').value,
        DeviceProtection: document.getElementById('DeviceProtection').value,
        TechSupport: document.getElementById('TechSupport').value,
        StreamingTV: document.getElementById('StreamingTV').value,
        StreamingMovies: document.getElementById('StreamingMovies').value,
        Contract: document.getElementById('Contract').value,
        PaperlessBilling: document.getElementById('PaperlessBilling').value,
        PaymentMethod: document.getElementById('PaymentMethod').value,
        MonthlyCharges: parseFloat(document.getElementById('MonthlyCharges').value),
    };
    
    const totalChargesInput = document.getElementById('TotalCharges').value;
    if (totalChargesInput.trim() !== '') {
        payload.TotalCharges = parseFloat(totalChargesInput);
    }
    
    try {
        const response = await requestApi('/api/predict', 'POST', payload);
        
        // Hide placeholder, show display
        document.getElementById('result-placeholder').classList.add('d-none');
        const display = document.getElementById('result-display');
        display.classList.remove('d-none');
        
        // Update dial
        const prob = data.probability;
        const probPct = (prob * 100).toFixed(1) + '%';
        document.getElementById('result-probability').textContent = probPct;
        
        // SVG stroke offset calculation: circumference = 2 * pi * r = 2 * 3.14159 * 40 = 251.2
        const strokeOffset = 251.2 - (251.2 * prob);
        const dialFill = document.getElementById('result-dial-fill');
        dialFill.style.strokeDashoffset = strokeOffset;
        
        // Update dial color based on cohort
        if (prob >= 0.70) {
            dialFill.style.stroke = 'var(--accent-red)';
        } else if (prob >= 0.30) {
            dialFill.style.stroke = 'var(--accent-orange)';
        } else {
            dialFill.style.stroke = 'var(--accent-green)';
        }
        
        // Update badge
        const badge = document.getElementById('result-cohort-badge');
        badge.className = 'cohort-badge ' + data.color_class;
        badge.textContent = data.cohort;
        
        // Update mini metrics
        document.getElementById('res-avg-spend').textContent = '$' + data.engineered_features.AvgMonthlySpend.toFixed(2);
        document.getElementById('res-service-count').textContent = data.engineered_features.ServiceCount + ' / 6';
        document.getElementById('res-bill-ratio').textContent = data.engineered_features.ChargeIncreaseRatio.toFixed(2);
        document.getElementById('res-tenure-group').textContent = data.engineered_features.TenureGroup;
        
        // Render playbooks
        const playbooksList = document.getElementById('result-playbooks');
        playbooksList.innerHTML = '';
        
        data.playbooks.forEach(playbook => {
            const card = document.createElement('div');
            card.className = 'playbook-card';
            
            // Set dynamic border color based on risk level
            if (prob >= 0.70) {
                card.style.borderLeft = '3px solid var(--accent-red)';
            } else if (prob >= 0.30) {
                card.style.borderLeft = '3px solid var(--accent-orange)';
            } else {
                card.style.borderLeft = '3px solid var(--accent-cyan)';
            }
            
            card.innerHTML = `
                <h5>${playbook.title}</h5>
                <p class="playbook-detail"><strong>Trigger:</strong> ${playbook.trigger}</p>
                <p class="playbook-detail"><strong>Recommended Action:</strong> ${playbook.action}</p>
                <p class="playbook-detail"><strong>Business Benefit:</strong> ${playbook.benefit}</p>
            `;
            playbooksList.appendChild(card);
        });
        
    } catch (error) {
        alert('Error scoring prediction: ' + error.message);
    } finally {
        submitBtn.textContent = 'Analyze Churn Risk';
        submitBtn.disabled = false;
    }
}

// Helper wrapper for request
let data = null; // Store globally for form retrieval scope
async function requestApi(url, method = 'GET', body = null) {
    const options = {
        method,
        headers: {
            'Content-Type': 'application/json'
        }
    };
    if (body) {
        options.body = JSON.stringify(body);
    }
    const res = await fetch(url, options);
    data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Server error');
    return data;
}

// 5. Load and Render Outreach Table (Tab 3)
async function loadOutreachTable() {
    try {
        const url = `/api/customers?page=${currentPage}&limit=${recordsPerPage}&search=${currentSearch}`;
        const response = await fetch(url);
        if (!response.ok) throw new Error('Failed to load customer list');
        
        const data = await response.json();
        
        const tbody = document.getElementById('outreach-table-body');
        tbody.innerHTML = '';
        
        if (data.records.length === 0) {
            tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-muted); padding: 3rem 0;">No matching customer records found.</td></tr>`;
            document.getElementById('pagination-info').textContent = 'Showing 0 records';
            document.getElementById('btn-page-prev').disabled = true;
            document.getElementById('btn-page-next').disabled = true;
            return;
        }
        
        data.records.forEach(customer => {
            const tr = document.createElement('tr');
            
            // Format Risk level badge
            const prob = customer.ChurnProbability;
            const probPct = (prob * 100).toFixed(1) + '%';
            
            let riskBadge = '';
            if (prob >= 0.70) {
                riskBadge = `<span class="cohort-badge risk-high">High</span>`;
            } else if (prob >= 0.30) {
                riskBadge = `<span class="cohort-badge risk-medium">Medium</span>`;
            } else {
                riskBadge = `<span class="cohort-badge risk-low">Low</span>`;
            }
            
            // Stringify customer object to pass safely to onclick
            const customerStr = JSON.stringify(customer).replace(/"/g, '&quot;');
            
            tr.innerHTML = `
                <td class="td-id">${customer.customerID}</td>
                <td>${customer.tenure} m</td>
                <td>${customer.Contract}</td>
                <td>${customer.InternetService}</td>
                <td>$${customer.MonthlyCharges.toFixed(2)}</td>
                <td style="font-weight: 700;">${probPct}</td>
                <td>${riskBadge}</td>
                <td>
                    <button class="btn-outreach-action" onclick="openPlaybookModal('${customerStr}')">Outreach Playbook</button>
                </td>
            `;
            tbody.appendChild(tr);
        });
        
        // Update pagination UI
        const startRecord = (data.page - 1) * data.limit + 1;
        const endRecord = Math.min(startRecord + data.records.length - 1, data.total_records);
        document.getElementById('pagination-info').textContent = `Showing ${startRecord.toLocaleString()}-${endRecord.toLocaleString()} of ${data.total_records.toLocaleString()} customers`;
        
        document.getElementById('btn-page-prev').disabled = data.page <= 1;
        document.getElementById('btn-page-next').disabled = data.page >= data.total_pages;
        
        currentPage = data.page; // Align
        
    } catch (error) {
        console.error('Error loading outreach table:', error);
    }
}

function changePage(direction) {
    currentPage += direction;
    loadOutreachTable();
}

function handleSearch(value) {
    currentSearch = value.trim();
    currentPage = 1; // Reset to page 1
    loadOutreachTable();
}

// 6. Outreach Playbook Modal Dialog
function openPlaybookModal(customerJSONStr) {
    const customer = JSON.parse(customerJSONStr.replace(/&quot;/g, '"'));
    const prob = customer.ChurnProbability;
    const probPct = (prob * 100).toFixed(1) + '%';
    
    let riskBadge = '';
    if (prob >= 0.70) {
        riskBadge = `<span class="cohort-badge risk-high">High Risk</span>`;
    } else if (prob >= 0.30) {
        riskBadge = `<span class="cohort-badge risk-medium">Medium Risk</span>`;
    } else {
        riskBadge = `<span class="cohort-badge risk-low">Low Risk</span>`;
    }
    
    // Gather active triggers for playbooks
    const playbooks = [];
    
    if (customer.Contract === 'Month-to-month') {
        playbooks.push({
            name: "Playbook A: Proactive Contract Conversion",
            trigger: "Contract is Month-to-month (structural instability).",
            script: `"Hello ${customer.Partner === 'Yes' ? 'Mr/Mrs' : ''} ${customer.customerID}, I notice you have been with us for ${customer.tenure} months. We value your loyalty, and I am authorized to offer you an immediate 15% discount on your monthly bill if you transition to our 1-year stability package today. This locks in your rate and saves you $${(customer.MonthlyCharges * 0.15).toFixed(2)} every single month!"`
        });
    }
    
    if (customer.InternetService === 'Fiber optic') {
        playbooks.push({
            name: "Playbook B: Quality-of-Service Check & Add-on Bundle",
            trigger: "Customer is subscribed to Fiber Optic internet (high-risk tier).",
            script: `"Hi there, I am calling from our Customer Success team. We are performing quality diagnostics for our premium Fiber lines in your area. I want to ensure you are receiving the fastest speeds. As a thank you for your time, I would like to attach a complimentary 3-month Tech Support and Online Security shield to your account today, entirely free of charge, to keep your connection fully protected."`
        });
    }
    
    if (customer.PaymentMethod === 'Electronic check') {
        playbooks.push({
            name: "Playbook C: Auto-Pay Conversion Incentive",
            trigger: "Customer pays via manual Electronic Check (billing friction).",
            script: `"Hello, I see that you pay manually via Electronic Check every month. To make your life easier and prevent any accidental late fees, we are running an Auto-Pay enrollment drive. If you register your card or bank account for automatic monthly payments today, I can apply a one-time $10 credit to your next bill immediately!"`
        });
    }
    
    if (customer.tenure <= 6) {
        playbooks.push({
            name: "Playbook D: Onboarding Welcome & Integration Review",
            trigger: "Customer is within their critical first 6 months (tenure = " + customer.tenure + ").",
            script: `"Welcome to the family! I wanted to check in personally to see how your first few months of service have been. My goal is to ensure you have zero issues. Are your speeds satisfying, and do you have any questions about your billing or features? Let's verify everything is set up perfectly for you."`
        });
    }
    
    if (playbooks.length === 0) {
        playbooks.push({
            name: "Passive Engagement & Standard Nurture",
            trigger: "Customer is stable (Low Risk). No high-priority triggers detected.",
            script: `"Hello, we just wanted to reach out to say thank you for being a valued customer! Your loyalty is highly appreciated. Is there anything at all we can assist you with today to make your experience even better?"`
        });
    }
    
    // Construct HTML for modal content
    let playbooksHTML = '';
    playbooks.forEach((p, idx) => {
        playbooksHTML += `
            <div class="playbook-card" style="margin-top: 1rem; border-left: 3px solid ${prob >= 0.70 ? 'var(--accent-red)' : (prob >= 0.30 ? 'var(--accent-orange)' : 'var(--accent-cyan)')}">
                <h5 style="font-size: 0.95rem; margin-bottom: 0.25rem;">${p.name}</h5>
                <p style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.5rem;"><strong>Trigger:</strong> ${p.trigger}</p>
                <div style="background: rgba(15, 23, 42, 0.4); padding: 0.85rem; border-radius: 6px; font-style: italic; font-size: 0.8rem; border: 1px dashed rgba(255,255,255,0.05); color: var(--text-main);">
                    ${p.script}
                </div>
            </div>
        `;
    });

    const modalContent = document.getElementById('modal-content');
    modalContent.innerHTML = `
        <div class="modal-profile-summary">
            <div>
                <h4 style="font-size: 1.15rem; font-family: var(--font-heading); font-weight: 700;">Customer ID: ${customer.customerID}</h4>
                <p style="font-size: 0.8rem; color: var(--text-muted);">Demographics: ${customer.gender} | ${customer.SeniorCitizen === 1 ? 'Senior' : 'Non-Senior'} | ${customer.HasFamily === 1 ? 'Family Account' : 'Individual'}</p>
            </div>
            <div class="modal-score-box">
                <span class="modal-prob-large" style="color: ${prob >= 0.70 ? 'var(--accent-red)' : (prob >= 0.30 ? 'var(--accent-orange)' : 'var(--accent-green)')};">${probPct}</span>
                ${riskBadge}
            </div>
        </div>
        
        <div class="modal-profile-grid">
            <div class="modal-profile-item">
                <p>Tenure with Telco</p>
                <strong>${customer.tenure} Months</strong>
            </div>
            <div class="modal-profile-item">
                <p>Contract Agreement</p>
                <strong>${customer.Contract}</strong>
            </div>
            <div class="modal-profile-item">
                <p>Internet Service</p>
                <strong>${customer.InternetService}</strong>
            </div>
            <div class="modal-profile-item">
                <p>Monthly Charge</p>
                <strong>$${customer.MonthlyCharges.toFixed(2)}</strong>
            </div>
            <div class="modal-profile-item">
                <p>Total Billing</p>
                <strong>$${customer.TotalCharges.toFixed(2)}</strong>
            </div>
            <div class="modal-profile-item">
                <p>Payment Method</p>
                <strong>${customer.PaymentMethod}</strong>
            </div>
        </div>
        
        <div class="playbooks-container">
            <h4 style="font-size: 1.05rem; font-family: var(--font-heading); font-weight: 600; border-bottom: 1px solid var(--border-light); padding-bottom: 0.5rem;">Agent Customer Call Script</h4>
            <div class="playbooks-list" style="max-height: 280px; overflow-y: auto;">
                ${playbooksHTML}
            </div>
        </div>
    `;
    
    // Open modal
    document.getElementById('playbook-modal').classList.add('active');
}

function closeModal(event) {
    // Close only if clicking outside card or clicking close button
    if (event === null || event.target === document.getElementById('playbook-modal')) {
        document.getElementById('playbook-modal').classList.remove('active');
    }
}
