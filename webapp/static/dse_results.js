/* DSE results page — Top-N table, Pareto plot, radar, rerank. */
(function () {
    'use strict';

    let allResults = [];
    let candidatesMeta = {};
    let currentTopN = [];
    let currentPareto = [];

    document.addEventListener('DOMContentLoaded', async () => {
        await loadResults();
        wireSliders();
        document.getElementById('btn-rerank').addEventListener('click', rerank);
        document.getElementById('pareto-x').addEventListener('change', drawPareto);
        document.getElementById('pareto-y').addEventListener('change', drawPareto);
    });

    async function loadResults() {
        const r = await fetch(`/api/dse/jobs/${JOB_ID}/results`);
        const j = await r.json();
        allResults = j.all_candidates || [];
        currentTopN = j.top_n || [];
        currentPareto = j.pareto || [];
        candidatesMeta = j.candidates_meta || {};
        renderTopN(currentTopN);
        drawPareto();
        drawRadar(currentTopN);
    }

    function fmtHw(label) {
        const m = candidatesMeta[label];
        if (!m) return '—';
        return Object.entries(m.hw_distribution || {}).map(([h, c]) => `${c}×${h}`).join(' + ');
    }

    function fmtPar(label) {
        const m = candidatesMeta[label];
        if (!m) return '—';
        const p = m.parallelism || {};
        return `tp${p.tp}_pp${p.pp}_dp${p.dp}`;
    }

    function renderTopN(rows) {
        const tbody = document.getElementById('top-n-body');
        tbody.innerHTML = '';
        rows.forEach((r, i) => {
            const m = r.metrics || {};
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${i + 1}</td>
                <td title="${r.label}">${r.label}</td>
                <td>${fmtHw(r.label)}</td>
                <td>${fmtPar(r.label)}</td>
                <td>${candidatesMeta[r.label]?.pd_layout || '—'}</td>
                <td>${fmt(m.p99_ttft_ms)}</td>
                <td>${fmt(m.p99_tpot_ms)}</td>
                <td>${fmt(m.total_token_tp)}</td>
                <td>${fmt(m.total_energy_wh, 4)}</td>
                <td>${r.score != null ? r.score.toFixed(3) : '—'}</td>`;
            tbody.appendChild(tr);
        });
    }
    function fmt(v, d=2) { return v == null ? '—' : Number(v).toFixed(d); }

    function drawPareto() {
        const xKey = document.getElementById('pareto-x').value;
        const yKey = document.getElementById('pareto-y').value;
        const paretoLabels = new Set(currentPareto.map(r => r.label));
        const pts = allResults
            .filter(r => r.state === 'done' && r.metrics?.[xKey] != null && r.metrics?.[yKey] != null)
            .map(r => ({label: r.label, x: r.metrics[xKey], y: r.metrics[yKey],
                        pareto: paretoLabels.has(r.label)}));
        const inside  = pts.filter(p => !p.pareto);
        const outside = pts.filter(p => p.pareto);
        Plotly.newPlot('pareto-plot', [
            {
                x: inside.map(p => p.x), y: inside.map(p => p.y), mode: 'markers',
                marker: {size: 8, color: '#888'}, text: inside.map(p => p.label),
                hovertemplate: '%{text}<br>%{x:.2f}, %{y:.2f}<extra></extra>',
                name: 'Dominated',
            },
            {
                x: outside.map(p => p.x), y: outside.map(p => p.y), mode: 'markers',
                marker: {size: 12, color: '#d62728'}, text: outside.map(p => p.label),
                hovertemplate: '%{text}<br>%{x:.2f}, %{y:.2f}<extra>Pareto</extra>',
                name: 'Pareto-optimal',
            },
        ], {
            xaxis: {title: xKey}, yaxis: {title: yKey},
            margin: {l: 60, r: 30, t: 20, b: 50}, paper_bgcolor: 'rgba(0,0,0,0)',
        }, {responsive: true});
    }

    function drawRadar(rows) {
        if (!rows.length) {
            Plotly.purge('radar-plot');
            return;
        }
        const metrics = [
            ['ttft', 'p99_ttft_ms', 'min'],
            ['tpot', 'p99_tpot_ms', 'min'],
            ['tp', 'total_token_tp', 'max'],
            ['power', 'total_energy_wh', 'min'],
            ['itl', 'p99_itl_ms', 'min'],
        ];
        // Normalize across the Top-N
        const norms = {};
        for (const [name, key, dir] of metrics) {
            const vals = rows.map(r => r.metrics?.[key]).filter(v => v != null);
            if (!vals.length) { norms[name] = null; continue; }
            const lo = Math.min(...vals), hi = Math.max(...vals);
            const span = hi - lo;
            norms[name] = rows.map(r => {
                const v = r.metrics?.[key];
                if (v == null) return 0;
                if (span === 0) return 1;
                return dir === 'min' ? (hi - v) / span : (v - lo) / span;
            });
        }
        const traces = rows.map((r, i) => ({
            type: 'scatterpolar',
            r: metrics.map(([name]) => norms[name] ? norms[name][i] : 0),
            theta: metrics.map(([name]) => name),
            fill: 'toself',
            name: r.label,
        }));
        Plotly.newPlot('radar-plot', traces, {
            polar: {radialaxis: {visible: true, range: [0, 1]}},
            paper_bgcolor: 'rgba(0,0,0,0)',
            margin: {l: 40, r: 40, t: 40, b: 40},
        }, {responsive: true});
    }

    function wireSliders() {
        ['ttft','tpot','tp','power','tokwh'].forEach(id => {
            const s = document.getElementById('rw-' + id);
            const v = document.getElementById('rw-' + id + '-v');
            s.addEventListener('input', () => v.textContent = s.value);
        });
    }

    async function rerank() {
        const weights = {
            ttft:       parseFloat(document.getElementById('rw-ttft').value),
            tpot:       parseFloat(document.getElementById('rw-tpot').value),
            throughput: parseFloat(document.getElementById('rw-tp').value),
            power:      parseFloat(document.getElementById('rw-power').value),
            tokwh:      parseFloat(document.getElementById('rw-tokwh').value),
        };
        const topN = parseInt(document.getElementById('rerank-top-n').value, 10) || 5;
        const r = await fetch(`/api/dse/jobs/${JOB_ID}/rerank`, {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({weights, top_n: topN}),
        });
        const j = await r.json();
        if (!r.ok) { alert('rerank failed: ' + (j.detail || 'error')); return; }
        currentTopN = j.top_n;
        currentPareto = j.pareto;
        renderTopN(currentTopN);
        drawPareto();
        drawRadar(currentTopN);
    }
})();
