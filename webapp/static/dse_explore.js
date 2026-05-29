/* DSE explore page — gather spec, hit /api/dse/dry-run + /api/dse/jobs. */
(function () {
    'use strict';

    let catalog = null;     // /api/dse/catalog response
    let datasets = [];

    document.addEventListener('DOMContentLoaded', async () => {
        await loadCatalog();
        await loadDatasets();
        // Seed one resource pool row
        addHwRow();
        document.getElementById('btn-add-hw').addEventListener('click', addHwRow);
        document.getElementById('btn-dry-run').addEventListener('click', dryRun);
        document.getElementById('btn-start').addEventListener('click', startJob);
        wirePriorityRows();
    });

    async function loadCatalog() {
        const r = await fetch('/api/dse/catalog');
        catalog = await r.json();
        // Populate model dropdown with intersect of catalog availability
        const sel = document.getElementById('model-select');
        const models = Object.keys(catalog.models);
        for (const m of models) {
            const opt = document.createElement('option');
            opt.value = m; opt.textContent = m;
            sel.appendChild(opt);
        }
    }

    async function loadDatasets() {
        // /api/datasets returns a list of {path, name, family, compatible_models}
        // (NOT wrapped in {datasets: ...}). Use `path` as the value (workload
        // spec expects the relative path) and `name` as the display text,
        // with family annotation when known.
        try {
            const r = await fetch('/api/datasets');
            datasets = await r.json();
        } catch (e) { datasets = []; }
        const sel = document.getElementById('dataset-select');
        for (const ds of datasets) {
            const opt = document.createElement('option');
            opt.value = ds.path;
            opt.textContent = ds.family && ds.family !== 'unknown'
                ? `${ds.name}  [${ds.family}]`
                : ds.name;
            sel.appendChild(opt);
        }
    }

    function addHwRow() {
        if (!catalog) return;
        const tbody = document.getElementById('resource-pool-body');
        const tr = document.createElement('tr');
        const select = document.createElement('select');
        for (const hw of Object.keys(catalog.hardware)) {
            const opt = document.createElement('option');
            opt.value = hw; opt.textContent = hw;
            select.appendChild(opt);
        }
        const minI = mkInput('number', '0', 0);
        const maxI = mkInput('number', '2', 0);
        const rm = document.createElement('button');
        rm.type = 'button'; rm.textContent = '✕';
        rm.className = 'btn-secondary'; rm.style.padding = '4px 10px';
        rm.addEventListener('click', () => tr.remove());

        tr.appendChild(td(select));
        tr.appendChild(td(minI));
        tr.appendChild(td(maxI));
        tr.appendChild(td(rm));
        tbody.appendChild(tr);
    }
    function mkInput(type, value, min) {
        const i = document.createElement('input');
        i.type = type; i.value = value;
        if (min !== undefined) i.min = String(min);
        i.style.maxWidth = '80px';
        return i;
    }
    function td(el) { const t = document.createElement('td'); t.appendChild(el); return t; }

    function collectSpec() {
        const items = [...document.querySelectorAll('#resource-pool-body tr')].map(tr => {
            const cells = tr.cells;
            return {
                hw: cells[0].querySelector('select').value,
                min: parseInt(cells[1].querySelector('input').value, 10),
                max: parseInt(cells[2].querySelector('input').value, 10),
            };
        });
        const totalMax = parseInt(document.getElementById('total-max').value, 10);
        const constraints = {};

        // Derive weights from objective checkboxes + Low/Med/High priority.
        // Unchecked objectives get weight 0 (excluded from scoring).
        // Priority values: Low=1, Medium=3, High=9.
        const priVal = (name) => {
            const sel = document.querySelector(`input[name="${name}"]:checked`);
            return sel ? Number(sel.value) : 3;
        };
        const weights = {
            ttft:       document.getElementById('obj-ttft').checked  ? priVal('pri-ttft')  : 0,
            tpot:       document.getElementById('obj-tpot').checked  ? priVal('pri-tpot')  : 0,
            throughput: document.getElementById('obj-tp').checked    ? priVal('pri-tp')    : 0,
            power:      document.getElementById('obj-power').checked ? priVal('pri-power') : 0,
            tokwh:      document.getElementById('obj-tokwh').checked ? priVal('pri-tokwh') : 0,
        };
        // Ensure at least one weight > 0 (fall back to equal if nothing checked)
        if (Object.values(weights).every(v => v === 0)) {
            weights.ttft = weights.tpot = weights.throughput = weights.power = weights.tokwh = 1;
        }

        return {
            resource_pool: {
                items: items,
                total_max_npus: Number.isFinite(totalMax) ? totalMax : null,
            },
            model: {
                name: document.getElementById('model-select').value,
                fp: parseInt(document.getElementById('fp-select').value, 10),
            },
            workload: {
                dataset: document.getElementById('dataset-select').value,
                num_req: parseInt(document.getElementById('num-req').value, 10) || 100,
                timeout_s: parseInt(document.getElementById('timeout-s').value, 10) || 120,
            },
            constraints: constraints,
            features: {
                allow_pd_disagg: document.getElementById('feat-pd').checked,
                prefix_caching: document.getElementById('feat-prefix').checked,
                attn_offloading: document.getElementById('feat-attn-off').checked,
            },
            search: {
                max_combinations: parseInt(document.getElementById('search-max').value, 10) || 20,
                sampling_strategy: document.getElementById('search-sampling').value,
                random_seed: parseInt(document.getElementById('search-seed').value, 10) || 0,
            },
            weights: weights,
            top_n: parseInt(document.getElementById('top-n').value, 10) || 5,
        };
    }

    async function dryRun() {
        const spec = collectSpec();
        const status = document.getElementById('dse-status');
        status.textContent = 'Estimating…';
        try {
            const r = await fetch('/api/dse/dry-run', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(spec),
            });
            const j = await r.json();
            if (!r.ok) {
                status.textContent = '❌ ' + (j.detail || 'error');
                document.getElementById('dry-run-results').style.display = 'none';
                return;
            }
            const unique = j.estimated_candidates;
            const sim = j.simulated_candidates ?? Math.min(unique, spec.search.max_combinations);
            status.textContent = sim < unique
                ? `≈ ${unique} candidates found → ${sim} will be simulated (sampled, cap=${spec.search.max_combinations})`
                : `≈ ${sim} candidates will be simulated (all found)`;
            renderDryRunList(j.candidates || [], unique, sim);
        } catch (e) {
            status.textContent = '❌ ' + e.message;
            document.getElementById('dry-run-results').style.display = 'none';
        }
    }

    function renderDryRunList(candidates, unique, sim) {
        const section = document.getElementById('dry-run-results');
        const label = document.getElementById('dry-run-count-label');
        const tbody = document.getElementById('dry-run-list-body');

        label.textContent = `(${unique} total, ${sim} to simulate)`;
        tbody.innerHTML = '';

        for (const c of candidates) {
            const hw = Object.entries(c.hw_distribution || {})
                .filter(([, n]) => n > 0)
                .map(([hw, n]) => `${n}×${hw}`)
                .join(' + ') || '—';
            const par = c.parallelism || {};
            const willSim = c.will_simulate;
            const tr = document.createElement('tr');
            if (!willSim) tr.style.opacity = '0.45';
            tr.innerHTML = `
                <td>${c.label}</td>
                <td>${hw}</td>
                <td>${par.tp ?? '—'}</td>
                <td>${par.pp ?? '—'}</td>
                <td>${par.dp ?? '—'}</td>
                <td>${c.pd_layout || '—'}</td>
                <td>${willSim ? '✓' : '—'}</td>`;
            tbody.appendChild(tr);
        }
        section.style.display = candidates.length ? '' : 'none';
    }

    async function startJob() {
        const spec = collectSpec();
        const status = document.getElementById('dse-status');
        const btn = document.getElementById('btn-start');
        btn.disabled = true;
        status.textContent = 'Starting job…';
        try {
            const r = await fetch('/api/dse/jobs', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(spec),
            });
            const j = await r.json();
            if (!r.ok) {
                status.textContent = '❌ ' + (j.detail || 'create failed');
                btn.disabled = false;
                return;
            }
            window.location = `/dse/jobs/${j.job_id}`;
        } catch (e) {
            status.textContent = '❌ ' + e.message;
            btn.disabled = false;
        }
    }

    function wirePriorityRows() {
        document.querySelectorAll('.obj-row').forEach(row => {
            const cb = row.querySelector('input[type="checkbox"]');
            const priSpan = row.querySelector('.obj-pri');
            if (!cb || !priSpan) return;
            const update = () => {
                priSpan.style.opacity = cb.checked ? '1' : '0.3';
                priSpan.querySelectorAll('input').forEach(i => i.disabled = !cb.checked);
            };
            cb.addEventListener('change', update);
            update();
        });
    }
})();
