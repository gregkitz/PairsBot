/**
 * Main JavaScript for the Intraday Statistical Arbitrage System
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Handle confirmation dialogs
    const confirmButtons = document.querySelectorAll('[data-confirm]');
    confirmButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            if (!confirm(this.getAttribute('data-confirm'))) {
                e.preventDefault();
            }
        });
    });

    // Auto-refresh data on pages with [data-auto-refresh] attribute
    const autoRefreshElements = document.querySelectorAll('[data-auto-refresh]');
    autoRefreshElements.forEach(element => {
        const interval = element.getAttribute('data-auto-refresh');
        if (interval) {
            setInterval(() => {
                refreshData(element);
            }, parseInt(interval) * 1000);
        }
    });

    // Date range picker initialization
    const dateRangePickers = document.querySelectorAll('.date-range-picker');
    dateRangePickers.forEach(picker => {
        // This is a placeholder - implement with your date range picker library of choice
        console.log('Date range picker initialized');
    });

    // JSON editor initialization
    const jsonEditors = document.querySelectorAll('.json-editor');
    jsonEditors.forEach(editor => {
        // This is a placeholder - implement with your JSON editor library of choice
        console.log('JSON editor initialized');
    });

    // Initialize strategy controls
    initializeStrategyControls();
});

/**
 * Refresh data in an element using AJAX
 * @param {Element} element - The element to refresh
 */
function refreshData(element) {
    const url = element.getAttribute('data-refresh-url');
    const targetId = element.getAttribute('data-refresh-target');
    
    if (!url || !targetId) return;
    
    const target = document.getElementById(targetId);
    if (!target) return;
    
    fetch(url)
        .then(response => response.json())
        .then(data => {
            // Update element with new data
            updateElement(target, data);
        })
        .catch(error => {
            console.error('Error refreshing data:', error);
        });
}

/**
 * Update an element with new data
 * @param {Element} element - The element to update
 * @param {Object} data - The data to update with
 */
function updateElement(element, data) {
    // This is a simple implementation - expand as needed
    if (element.tagName === 'TABLE') {
        updateTable(element, data);
    } else if (element.classList.contains('chart-container')) {
        updateChart(element, data);
    } else {
        // For other elements, try to update based on data properties
        for (const key in data) {
            const targetElement = element.querySelector(`[data-field="${key}"]`);
            if (targetElement) {
                targetElement.textContent = data[key];
            }
        }
    }
}

/**
 * Update a table with new data
 * @param {Element} table - The table element
 * @param {Object} data - The data to update with
 */
function updateTable(table, data) {
    // Assume data is an array of objects
    if (!Array.isArray(data)) return;
    
    const tbody = table.querySelector('tbody');
    if (!tbody) return;
    
    // Clear existing rows
    tbody.innerHTML = '';
    
    // Add new rows
    data.forEach(item => {
        const row = document.createElement('tr');
        
        // Create cells based on table headers
        const headers = table.querySelectorAll('th');
        headers.forEach(header => {
            const field = header.getAttribute('data-field');
            if (field && item[field] !== undefined) {
                const cell = document.createElement('td');
                cell.textContent = item[field];
                row.appendChild(cell);
            }
        });
        
        tbody.appendChild(row);
    });
}

/**
 * Initialize strategy controls
 */
function initializeStrategyControls() {
    // Strategy start/stop buttons
    const strategyButtons = document.querySelectorAll('.strategy-control');
    strategyButtons.forEach(button => {
        button.addEventListener('click', function() {
            const action = this.getAttribute('data-action');
            const strategyId = this.getAttribute('data-strategy-id');
            
            if (action === 'start') {
                startStrategy(strategyId);
            } else if (action === 'stop') {
                stopStrategy(strategyId);
            }
        });
    });

    // Strategy status updates
    const strategyStatuses = document.querySelectorAll('.strategy-status');
    strategyStatuses.forEach(status => {
        const strategyId = status.getAttribute('data-strategy-id');
        
        // Update status every 10 seconds
        setInterval(() => {
            updateStrategyStatus(status, strategyId);
        }, 10000);
    });
}

/**
 * Start a trading strategy
 * @param {string} strategyId - The ID of the strategy to start
 */
function startStrategy(strategyId) {
    const form = document.querySelector(`#start-strategy-${strategyId}`);
    if (form) {
        form.submit();
    }
}

/**
 * Stop a trading strategy
 * @param {string} strategyId - The ID of the strategy to stop
 */
function stopStrategy(strategyId) {
    const form = document.querySelector(`#stop-strategy-${strategyId}`);
    if (form) {
        form.submit();
    }
}

/**
 * Update strategy status display
 * @param {Element} statusElement - The element to update
 * @param {string} strategyId - The ID of the strategy
 */
function updateStrategyStatus(statusElement, strategyId) {
    fetch(`/strategy/status/${strategyId}`)
        .then(response => response.json())
        .then(data => {
            // Update status display
            statusElement.classList.remove('bg-success', 'bg-danger', 'bg-warning');
            
            if (data.running) {
                statusElement.textContent = 'Running';
                statusElement.classList.add('bg-success');
            } else {
                statusElement.textContent = 'Stopped';
                statusElement.classList.add('bg-danger');
            }
        })
        .catch(error => {
            console.error('Error updating strategy status:', error);
            statusElement.textContent = 'Unknown';
            statusElement.classList.add('bg-warning');
        });
} 