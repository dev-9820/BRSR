document.addEventListener('DOMContentLoaded', () => {
    fetch('analysis_results.json')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            populateSummary(data.summary);
            populateTable(data.results);
        })
        .catch(error => {
            console.error('Error fetching or parsing analysis_results.json:', error);
            document.getElementById('analysis-table').querySelector('tbody').innerHTML = 
                '<tr><td colspan="6" style="text-align: center; color: red;">Failed to load analysis data. Check console for details.</td></tr>';
        });
});

/**
 * Populates the overall summary cards.
 * @param {object} summary 
 */
function populateSummary(summary) {
    document.getElementById('total-principles').textContent = summary.total_principles;
    document.getElementById('exact-match').textContent = summary.exact_match;
    document.getElementById('major-gaps').textContent = summary.moderate_gap + summary.vague;
    
    const avgScoreEl = document.getElementById('avg-score');
    avgScoreEl.textContent = summary.average_score.toFixed(2);
    
    // Set color based on score (Lower is better)
    if (summary.average_score <= 1) {
        avgScoreEl.style.color = 'var(--color-primary)'; // Green
    } else if (summary.average_score <= 2) {
        avgScoreEl.style.color = 'var(--color-warning)'; // Yellow/Orange
    } else {
        avgScoreEl.style.color = 'var(--color-danger)'; // Red
    }
}

/**
 * Populates the detailed analysis table.
 * @param {array} results 
 */
function populateTable(results) {
    const tbody = document.getElementById('analysis-table').querySelector('tbody');
    tbody.innerHTML = ''; // Clear existing content

    results.forEach(item => {
        const row = document.createElement('tr');
        const analysis = item.analysis;
        const requirement = item.requirement;
        const scoreClass = `score-${analysis.faithfulness_score}`;

        row.innerHTML = `
            <td>${requirement.id}</td>
            <td><strong>${requirement.title}</strong></td>
            <td>${requirement.requirement}</td>
            <td class="${scoreClass}">${analysis.drift_category}</td>
            <td class="${scoreClass}">${analysis.faithfulness_score}</td>
            <td>
                <p><strong>Justification:</strong> ${analysis.justification}</p>
                <p class="citation"><strong>Company Disclosure:</strong> 
                    ${analysis.company_disclosure.substring(0, 150)}... 
                    <br/>(Source: ${analysis.company_citation})
                </p>
            </td>
        `;
        tbody.appendChild(row);
    });
}