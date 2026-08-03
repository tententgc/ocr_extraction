/* DocuFlow front end — vanilla JS, no build step. */
'use strict';

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const state = {
  status: null,
  models: { ocr: [], llm: [] },
  samples: [],
  schemas: {},
  settings: null,       // business policy and the rule catalogue
  customRules: [],      // rules the user wrote
  draft: null,          // a compiled rule awaiting approval
  currentDoc: null,     // document being reviewed
  edits: {},            // pending reviewer edits
  page: 1,
};

/* ---------------------------------------------------------------- helpers */

const esc = (value) => String(value ?? '').replace(/[&<>"']/g,
  (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const pct = (value) => `${Math.round((value ?? 0) * 100)}%`;

const money = (value, currency) => value == null || value === ''
  ? '—'
  : `${Number(value).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}${currency ? ' ' + currency : ''}`;

const DECISION = {
  auto_post: { label: 'Posted automatically', icon: '✓', chip: 'ok' },
  needs_review: { label: 'Needs human review', icon: '⚑', chip: 'warn' },
  rejected: { label: 'Rejected', icon: '✕', chip: 'bad' },
};

function toast(message, kind = '') {
  const node = document.createElement('div');
  node.className = `toast ${kind}`;
  node.textContent = message;
  $('#toast-host').append(node);
  setTimeout(() => node.remove(), kind === 'error' ? 8000 : 4500);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const isJson = (response.headers.get('content-type') || '').includes('json');
  const body = isJson ? await response.json() : await response.text();
  if (!response.ok) throw new Error(body?.detail || body?.message || `Request failed (${response.status})`);
  return body;
}

function confClass(value) {
  if (value >= 0.9) return '';
  if (value >= 0.75) return 'mid';
  return 'low';
}

/* ------------------------------------------------------------------ tabs */

$$('.tab').forEach((tab) => tab.addEventListener('click', () => {
  $$('.tab').forEach((t) => t.classList.toggle('active', t === tab));
  $$('.view').forEach((v) => v.classList.toggle('active', v.id === `view-${tab.dataset.view}`));
  if (tab.dataset.view === 'queue') loadQueue();
  if (tab.dataset.view === 'records') loadRecords();
  if (tab.dataset.view === 'insights') loadInsights();
  if (tab.dataset.view === 'settings') { loadSettings(); loadCustomRules(); }
}));

/* ---------------------------------------------------------------- status */

async function loadStatus() {
  try {
    state.status = await api('/api/status');
  } catch (error) {
    toast(`Could not load status: ${error.message}`, 'error');
    return;
  }
  $('#threshold-chip').innerHTML =
    `<i class="dot"></i>Auto-post&nbsp;<b>≥ ${pct(state.status.thresholds.auto_post_min_confidence)}</b>`;
  updateBadges(state.status.stats);
}

/* -------------------------------------------------------- model pickers */

async function loadModels() {
  let payload;
  try { payload = await api('/api/models'); }
  catch (error) { toast(`Could not list models: ${error.message}`, 'error'); return; }

  state.models = payload;
  fillPicker($('#ocr-picker'), payload.ocr, payload.selection.ocr_key);
  fillPicker($('#llm-picker'), payload.llm, payload.selection.llm_key);
}

function fillPicker(select, options, activeKey) {
  const groups = { available: [], unavailable: [] };
  options.forEach((option) => groups[option.available ? 'available' : 'unavailable'].push(option));

  const renderOption = (option) => {
    const tags = option.tags?.length ? ` — ${option.tags.join(', ')}` : '';
    const star = option.recommended ? '★ ' : '';
    return `<option value="${esc(option.key)}" ${option.available ? '' : 'disabled'}
      title="${esc(option.reason || option.label)}">${star}${esc(option.label)}${esc(tags)}</option>`;
  };

  select.innerHTML = `
    ${groups.available.length ? `<optgroup label="Ready">${groups.available.map(renderOption).join('')}</optgroup>` : ''}
    ${groups.unavailable.length ? `<optgroup label="Not configured">${groups.unavailable.map(renderOption).join('')}</optgroup>` : ''}`;

  // Match the active key; fall back to the first ready option.
  const match = [...select.options].find((option) => option.value === activeKey);
  select.value = match ? activeKey : (groups.available[0]?.key ?? '');
}

async function selectModel(kind, key) {
  const picker = $(kind === 'ocr' ? '#ocr-picker' : '#llm-picker');
  picker.parentElement.classList.add('busy');
  try {
    const result = await api('/api/models/select', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(kind === 'ocr' ? { ocr_key: key } : { llm_key: key }),
    });
    const available = kind === 'ocr' ? result.ocr_available : result.llm_available;
    const reason = kind === 'ocr' ? result.ocr_reason : result.llm_reason;
    if (available) {
      toast(`${kind === 'ocr' ? 'OCR' : 'Agent'} model switched to ${key.split(':').slice(1).join(':') || key}`, 'success');
    } else {
      toast(`Selected, but not usable: ${reason}`, 'error');
    }
    loadStatus();
  } catch (error) {
    toast(error.message, 'error');
    loadModels();
  } finally {
    picker.parentElement.classList.remove('busy');
  }
}

$('#ocr-picker').addEventListener('change', (event) => selectModel('ocr', event.target.value));
$('#llm-picker').addEventListener('change', (event) => selectModel('llm', event.target.value));

function updateBadges(stats) {
  if (!stats) return;
  $('#queue-badge').textContent = stats.by_decision?.needs_review ?? 0;
  $('#records-badge').textContent = (stats.by_decision?.auto_post ?? 0);
}

/* --------------------------------------------------------------- samples */

async function loadSamples() {
  try {
    state.samples = await api('/api/samples');
    state.schemas = await api('/api/schemas');
  } catch (error) {
    toast(`Could not load samples: ${error.message}`, 'error');
    return;
  }
  $('#sample-list').innerHTML = state.samples
    .filter((sample) => sample.available)
    .map((sample) => `
      <button class="sample" data-sample="${esc(sample.doc_id)}">
        <span class="stype">${esc(sample.doc_type.replace('_', ' '))}</span>
        <span>${esc(sample.title)}
          <small>${esc(sample.demonstrates)}</small></span>
      </button>`).join('');

  $$('#sample-list .sample').forEach((button) => button.addEventListener('click',
    () => ingest(`/api/documents/sample/${button.dataset.sample}`, { method: 'POST' },
      `Processing sample: ${button.textContent.trim().split('\n')[0]}`)));
}

/* -------------------------------------------------------------- ingestion */

let busy = false;

async function ingest(path, options, label) {
  if (busy) { toast('Still processing the previous document.', 'error'); return; }
  busy = true;
  $('#progress-card').classList.remove('hidden');
  $('#progress-title').textContent = label || 'Processing…';
  $('#progress-detail').textContent = 'Reading the document with Typhoon OCR, then extracting and validating.';
  $('#result-host').innerHTML = '';

  try {
    const data = await api(path, options);
    renderResult(data, $('#result-host'), false);
    const decision = data.document.decision;
    toast(`${DECISION[decision].label} — ${pct(data.document.confidence)} confidence`,
      decision === 'auto_post' ? 'success' : decision === 'rejected' ? 'error' : '');
    loadStatus();
  } catch (error) {
    $('#result-host').innerHTML = `<div class="card"><h3>Could not process this document</h3>
      <p class="muted" style="margin-top:6px">${esc(error.message)}</p></div>`;
    toast(error.message, 'error');
  } finally {
    busy = false;
    $('#progress-card').classList.add('hidden');
  }
}

function uploadFile(file) {
  const form = new FormData();
  form.append('file', file);
  ingest('/api/documents/upload', { method: 'POST', body: form }, `Processing ${file.name}`);
}

$('#file-input').addEventListener('change', (event) => {
  const file = event.target.files?.[0];
  if (file) uploadFile(file);
  event.target.value = '';
});

const dropzone = $('#dropzone');
['dragenter', 'dragover'].forEach((type) => dropzone.addEventListener(type, (event) => {
  event.preventDefault(); dropzone.classList.add('drag');
}));
['dragleave', 'drop'].forEach((type) => dropzone.addEventListener(type, (event) => {
  event.preventDefault(); dropzone.classList.remove('drag');
}));
dropzone.addEventListener('drop', (event) => {
  const file = event.dataTransfer?.files?.[0];
  if (file) uploadFile(file);
});

$('#url-form').addEventListener('submit', (event) => {
  event.preventDefault();
  const url = $('#url-input').value.trim();
  if (!url) return;
  ingest('/api/documents/url', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url }),
  }, 'Fetching document from URL');
});

/* ---------------------------------------------------------------- result */

function renderResult(data, host, editable) {
  const doc = data.document;
  state.currentDoc = doc;
  state.edits = {};
  state.page = 1;

  const decision = DECISION[doc.decision];
  const schema = state.schemas[doc.doc_type] || { fields: [], display_name: doc.doc_type };

  host.innerHTML = `
    <div class="verdict ${doc.decision}">
      <span class="icon">${decision.icon}</span>
      <div>
        <h2>${decision.label}</h2>
        <div class="sub">${esc(doc.source.filename)} · ${esc(schema.display_name)}
          ${doc.record_id ? ` · record <b>${esc(doc.record_id)}</b>` : ''}</div>
      </div>
      <div class="conf"><b>${pct(doc.confidence)}</b><span>confidence</span></div>
    </div>

    ${doc.decision_reasons?.length ? `<div class="card"><h3>Why</h3>
      <ul style="margin:8px 0 0;padding-left:18px;font-size:13px">
        ${doc.decision_reasons.map((r) => `<li>${esc(r)}</li>`).join('')}</ul></div>` : ''}

    ${renderAnalyst(data)}

    <div class="result-grid">
      <div>
        <div class="card page-viewer" data-page-viewer></div>
        <div class="card"><h3>Pipeline</h3><div class="trace">${renderTrace(doc.trace)}</div></div>
        ${renderVendorCard(data)}
      </div>
      <div>
        <div class="card">
          <div class="card-head">
            <h3>Extracted fields</h3>
            <span class="muted">${doc.fields.filter((f) => f.value != null && f.value !== '').length}
              of ${doc.fields.length} found</span>
          </div>
          <div class="table-wrap">${renderFields(doc, schema, editable)}</div>
          ${editable ? renderReviewActions(doc) : ''}
        </div>
        ${renderLineItems(doc)}
        <div class="card"><h3>Business rules</h3>${renderFindings(doc.findings)}</div>
      </div>
    </div>`;

  renderPageViewer(doc, host);
  if (editable) wireReview(doc, schema, host);
}

function renderTrace(trace = []) {
  if (!trace.length) return '<p class="muted">No steps recorded.</p>';
  return trace.map((step) => `
    <div class="trace-step ${step.status}">
      <i class="tdot"></i>
      <div><b>${esc(step.step.replace('_', ' '))}</b> — ${esc(step.summary)}</div>
      ${step.duration_ms ? `<span class="ms">${step.duration_ms.toLocaleString()} ms</span>` : ''}
    </div>`).join('');
}

/** `resultHost` scopes the lookup: the Process pane and the Review pane can
 *  each hold a rendered result, so a document-wide query would fill the wrong
 *  one and leave the visible pane blank. */
function renderPageViewer(doc, resultHost) {
  const viewer = $('[data-page-viewer]', resultHost);
  if (!viewer) return;
  const total = doc.page_images?.length || 0;
  if (!total) { viewer.innerHTML = '<p class="muted">No page image available.</p>'; return; }
  // No loading="lazy": the panel is inserted while its tab may still be
  // hidden, and the lazy observer then never re-evaluates once it is shown —
  // leaving a blank card. There is one image and it is the point of the panel.
  viewer.innerHTML = `
    <img src="/api/documents/${encodeURIComponent(doc.document_id)}/page/${state.page}"
         alt="Page ${state.page} of ${esc(doc.source.filename)}" decoding="async" />
    ${total > 1 ? `<div class="page-nav">
      <button class="btn small" data-page="prev" ${state.page === 1 ? 'disabled' : ''}>◀</button>
      Page ${state.page} of ${total}
      <button class="btn small" data-page="next" ${state.page === total ? 'disabled' : ''}>▶</button>
    </div>` : ''}`;

  $('img', viewer).addEventListener('error', function onImageError() {
    this.replaceWith(Object.assign(document.createElement('p'), {
      className: 'muted',
      textContent: 'Page image could not be loaded.',
    }));
  });

  $$('[data-page]', viewer).forEach((button) => button.addEventListener('click', () => {
    state.page += button.dataset.page === 'next' ? 1 : -1;
    renderPageViewer(doc, resultHost);
  }));
}

function renderVendorCard(data) {
  const match = data.vendor_match;
  if (!match) return '';
  if (!match.matched) {
    return `<div class="card"><h3>Vendor match</h3>
      <p class="muted" style="margin-top:6px">No master-data record matched
        (best score ${pct(match.score)}).</p>
      ${match.candidates?.length ? `<p class="muted" style="margin-top:6px">Closest:
        ${match.candidates.map((c) => esc(c)).join(', ')}</p>` : ''}</div>`;
  }
  const vendor = match.vendor;
  return `<div class="card"><h3>Vendor match</h3>
    <p style="margin-top:6px"><b>${esc(vendor.legal_name)}</b>
      <span class="chip accent">${esc(vendor.vendor_id)}</span>
      <span class="chip ${vendor.status === 'active' ? 'ok' : 'warn'}">${esc(vendor.status)}</span></p>
    <p class="muted" style="margin-top:5px">Matched by ${esc(match.method)} (${pct(match.score)}) ·
      tax ID ${esc(vendor.tax_id || '—')} · account ${esc(vendor.bank_account || '—')}</p></div>`;
}

function renderAnalyst(data) {
  const rec = data.reconciliation;
  if (!rec || (!rec.assessment && !rec.corrections?.length)) return '';
  return `<div class="card analyst">
    <h3>Discrepancy analyst</h3>
    <p style="font-size:13px">${esc(rec.assessment)}</p>
    ${rec.reviewer_note ? `<p class="muted" style="margin-top:6px"><b>Reviewer:</b> ${esc(rec.reviewer_note)}</p>` : ''}
    ${rec.root_cause && rec.root_cause !== 'none'
      ? `<p style="margin-top:6px"><span class="chip">${esc(rec.root_cause.replace('_', ' '))}</span></p>` : ''}
    ${renderToolCalls(rec.tool_calls)}
    ${(rec.corrections || []).map((c) => `
      <div class="suggestion">
        <span class="chip ${c.grounded ? 'ok' : 'bad'}">${c.grounded ? 'verified on page' : 'not on page'}</span>
        <span><b>${esc(c.field)}</b> → <code>${esc(c.suggested_value)}</code></span>
        <span class="muted" style="flex:1 1 100%">${esc(c.reason)}</span>
        ${c.grounded ? `<button class="btn small" data-apply="${esc(c.field)}"
            data-value="${esc(c.suggested_value)}">Apply</button>` : ''}
      </div>`).join('')}
  </div>`;
}

/** The agent's investigation: which tools it called and what came back. */
function renderToolCalls(calls) {
  if (!calls?.length) return '';
  const rows = calls.map((call) => {
    const args = Object.entries(call.arguments || {})
      .map(([key, value]) => `${key}=${JSON.stringify(value)}`).join(', ');
    const result = JSON.stringify(call.result ?? {});
    return `<div class="tool-call">
      <code class="tname">${esc(call.name)}</code>
      <code class="targs">(${esc(args)})</code>
      <div class="tresult">${esc(result.length > 220 ? result.slice(0, 220) + '…' : result)}</div>
    </div>`;
  }).join('');
  return `<details class="tool-log"><summary>Investigation — ${calls.length} tool call${calls.length === 1 ? '' : 's'}</summary>
    ${rows}</details>`;
}

function renderFields(doc, schema, editable) {
  const specs = new Map((schema.fields || []).map((spec) => [spec.name, spec]));
  const rows = doc.fields.map((field) => {
    const spec = specs.get(field.name) || {};
    const empty = field.value == null || field.value === '';
    const value = empty ? '' : String(field.value);
    const cls = confClass(field.confidence);

    const valueCell = editable
      ? `<input class="finput" data-field="${esc(field.name)}" value="${esc(value)}"
           placeholder="${empty ? 'not found' : ''}" />`
      : `<span class="fval ${empty ? 'empty' : ''}">${empty ? 'not found' : esc(value)}</span>`;

    const flags = [
      spec.required ? '<span class="chip">required</span>' : '',
      spec.critical ? '<span class="chip accent">critical</span>' : '',
      !empty && !field.grounded ? '<span class="chip bad">unverified</span>' : '',
      field.source === 'human' ? '<span class="chip ok">edited</span>' : '',
      field.source === 'derived' ? '<span class="chip warn">inferred</span>' : '',
    ].filter(Boolean).join(' ');

    return `<tr>
      <td class="fname">${esc(spec.label || field.name)}<small>${esc(field.name)}</small></td>
      <td>${valueCell}
        ${field.evidence ? `<div class="evidence">${esc(String(field.evidence).slice(0, 180))}</div>` : ''}
        ${field.notes?.length ? `<div class="evidence" style="border-left-color:var(--warn)">${esc(field.notes[0])}</div>` : ''}
      </td>
      <td>${flags}</td>
      <td class="conf-cell">
        <span class="conf-val">${empty ? '—' : pct(field.confidence)}</span>
        ${empty ? '' : `<span class="conf-bar ${cls}"><i style="width:${Math.max(3, field.confidence * 100)}%"></i></span>`}
      </td>
    </tr>`;
  }).join('');

  return `<table><thead><tr><th>Field</th><th>Value &amp; evidence</th><th></th><th>Confidence</th></tr></thead>
    <tbody>${rows}</tbody></table>`;
}

function renderLineItems(doc) {
  if (!doc.line_items?.length) return '';
  return `<div class="card"><h3>Line items</h3>
    <div class="table-wrap"><table>
      <thead><tr><th>Description</th><th class="num">Qty</th><th class="num">Unit</th><th class="num">Amount</th></tr></thead>
      <tbody>${doc.line_items.map((item) => `<tr>
        <td>${esc(item.description)}</td>
        <td class="num">${item.quantity ?? '—'}</td>
        <td class="num">${money(item.unit_price)}</td>
        <td class="num">${money(item.amount)}</td></tr>`).join('')}</tbody>
    </table></div></div>`;
}

function renderFindings(findings = []) {
  if (!findings.length) return '<p class="muted">No rules were evaluated.</p>';
  // Free mode passes by definition, so it would hide inside the collapsed
  // "checks passed" list - exactly where nobody looks. It is a mode, not a
  // result, so it is pulled out and shown as a standing warning instead.
  const freeMode = findings.find((f) => f.rule_id === 'business_rules_disabled');
  const rest = findings.filter((f) => f !== freeMode);
  const failed = rest.filter((f) => !f.passed);
  const passed = rest.filter((f) => f.passed);
  const render = (finding) => `
    <div class="finding ${finding.passed ? 'passed' : finding.severity}">
      <span>${finding.passed ? '✓' : finding.severity === 'blocking' ? '✕' : '⚠'}</span>
      <div>${esc(finding.message)}<span class="rid">${esc(finding.rule_id)}</span></div>
    </div>`;
  return `${freeMode ? `<div class="finding warning"><span>⚠</span>
      <div>${esc(freeMode.message)}<span class="rid">${esc(freeMode.rule_id)}</span></div></div>` : ''}
    ${failed.map(render).join('')}
    ${passed.length ? `<details style="margin-top:8px"><summary class="muted" style="cursor:pointer;font-size:12.5px">
      ${passed.length} check${passed.length === 1 ? '' : 's'} passed</summary>
      <div style="margin-top:8px">${passed.map(render).join('')}</div></details>` : ''}`;
}

/* ---------------------------------------------------------------- review */

function renderReviewActions(doc) {
  return `<div class="review-actions">
    <span class="grow muted" id="edit-summary">No changes yet.</span>
    <button class="btn" id="btn-save">Save corrections</button>
    <button class="btn danger" id="btn-reject">Reject</button>
    <button class="btn primary" id="btn-approve">Approve &amp; post</button>
  </div>`;
}

function wireReview(doc, schema, host) {
  const summary = $('#edit-summary', host);

  const refresh = () => {
    const count = Object.keys(state.edits).length;
    summary.textContent = count
      ? `${count} field${count === 1 ? '' : 's'} changed: ${Object.keys(state.edits).join(', ')}`
      : 'No changes yet.';
  };

  $$('.finput', host).forEach((input) => {
    const original = input.value;
    input.addEventListener('input', () => {
      if (input.value === original) delete state.edits[input.dataset.field];
      else state.edits[input.dataset.field] = input.value;
      input.classList.toggle('changed', input.value !== original);
      refresh();
    });
  });

  $$('[data-apply]', host).forEach((button) => button.addEventListener('click', () => {
    const input = $(`.finput[data-field="${button.dataset.apply}"]`, host);
    if (!input) return;
    input.value = button.dataset.value;
    input.dispatchEvent(new Event('input'));
    input.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }));

  const submit = async (action) => {
    const note = action === 'reject' ? (prompt('Reason for rejecting this document:') ?? '') : '';
    if (action === 'reject' && note === '') return;
    $$('.review-actions .btn', host).forEach((button) => (button.disabled = true));
    try {
      const result = await api(`/api/documents/${encodeURIComponent(doc.document_id)}/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ edits: state.edits, action, reviewer: 'reviewer', note }),
      });
      if (result.rejected_edits?.length) {
        toast(`Some edits were not accepted: ${result.rejected_edits.join('; ')}`, 'error');
      }
      toast(result.message, action === 'reject' ? '' : 'success');
      await loadQueue();
      await openDocument(doc.document_id);
      loadStatus();
    } catch (error) {
      toast(error.message, 'error');
      $$('.review-actions .btn', host).forEach((button) => (button.disabled = false));
    }
  };

  $('#btn-approve', host).addEventListener('click', () => submit('approve'));
  $('#btn-save', host).addEventListener('click', () => submit('save'));
  $('#btn-reject', host).addEventListener('click', () => submit('reject'));
}

/* ----------------------------------------------------------------- queue */

async function loadQueue() {
  const filter = $('#queue-filter').value;
  let rows;
  try {
    rows = await api(`/api/documents${filter ? `?decision=${encodeURIComponent(filter)}` : ''}`);
  } catch (error) { toast(error.message, 'error'); return; }

  const body = $('#queue-table tbody');
  if (!rows.length) {
    body.innerHTML = '<tr><td colspan="7"><div class="empty-state">No documents yet — process one from the Process tab.</div></td></tr>';
    return;
  }
  body.innerHTML = rows.map((row) => `<tr>
    <td><b>${esc(row.filename)}</b><small style="display:block;color:var(--muted)">${esc(row.document_id)}</small></td>
    <td>${esc(row.doc_type.replace('_', ' '))}</td>
    <td>${esc(row.source_kind)}</td>
    <td>${pct(row.confidence)}</td>
    <td><span class="chip ${DECISION[row.decision]?.chip || ''}">${DECISION[row.decision]?.label || row.decision}</span></td>
    <td>${row.record_id ? esc(row.record_id) : '—'}</td>
    <td><button class="btn small" data-open="${esc(row.document_id)}">Open</button></td>
  </tr>`).join('');

  $$('[data-open]').forEach((button) => button.addEventListener('click',
    () => openDocument(button.dataset.open)));
}

async function openDocument(documentId) {
  try {
    const data = await api(`/api/documents/${encodeURIComponent(documentId)}`);
    const editable = data.document.decision !== 'auto_post' && !data.document.record_id;
    renderResult({ ...data, reconciliation: extractReconciliation(data.document) },
      $('#review-host'), editable);
    $('#review-host').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (error) { toast(error.message, 'error'); }
}

/** The reconciliation result is persisted inside the run's trace. */
function extractReconciliation(doc) {
  const step = (doc.trace || []).find((s) => s.step === 'reconcile');
  if (!step) return null;
  return {
    ran: true,
    assessment: step.detail?.assessment || step.summary,
    root_cause: step.detail?.root_cause,
    reviewer_note: step.detail?.reviewer_note,
    corrections: step.detail?.corrections || [],
    tool_calls: step.detail?.tool_calls || [],
  };
}

$('#queue-filter').addEventListener('change', loadQueue);
$('#queue-refresh').addEventListener('click', loadQueue);

/* --------------------------------------------------------------- records */

async function loadRecords() {
  let rows;
  try { rows = await api('/api/records'); }
  catch (error) { toast(error.message, 'error'); return; }

  const body = $('#records-table tbody');
  if (!rows.length) {
    body.innerHTML = '<tr><td colspan="7"><div class="empty-state">Nothing posted yet.</div></td></tr>';
    return;
  }
  body.innerHTML = rows.map((row) => `<tr>
    <td><b>${esc(row.record_id)}</b></td>
    <td><span class="chip">${esc(row.record_kind)}</span></td>
    <td>${esc(row.reference || '—')}</td>
    <td>${esc(row.counterparty || '—')}</td>
    <td class="num">${money(row.amount, row.currency)}</td>
    <td>${esc(row.issue_date || '—')}</td>
    <td>${esc(row.posted_by)}</td>
  </tr>`).join('');
}

$('#records-refresh').addEventListener('click', loadRecords);

/* -------------------------------------------------------------- insights */

async function loadInsights() {
  let metrics, vendors, orders;
  try {
    [metrics, vendors, orders] = await Promise.all([
      api('/api/metrics'), api('/api/master/vendors'), api('/api/master/purchase-orders'),
    ]);
  } catch (error) { toast(error.message, 'error'); return; }

  const decisions = metrics.by_decision || {};
  $('#metric-grid').innerHTML = `
    ${metric('Documents processed', metrics.total_documents, 'since the database was created')}
    ${metric('Straight-through rate', pct(metrics.auto_post_rate),
      `${decisions.auto_post || 0} posted without a human`)}
    ${metric('Awaiting review', decisions.needs_review || 0, 'held back by a rule or low confidence')}
    ${metric('Average confidence', pct(metrics.average_confidence), 'across all documents')}
    ${metric('Reviewer corrections', metrics.corrections, 'field edits made by humans')}`;

  const total = metrics.total_documents || 1;
  const byType = Object.entries(metrics.by_type || {});
  if (byType.length) {
    $('#metric-grid').insertAdjacentHTML('afterend', '');
  }

  $('#vendors-table tbody').innerHTML = vendors.map((vendor) => `<tr>
    <td>${esc(vendor.vendor_id)}</td><td>${esc(vendor.legal_name)}</td>
    <td class="fval">${esc(vendor.tax_id || '—')}</td>
    <td class="fval">${esc(vendor.bank_account || '—')}</td>
    <td><span class="chip ${vendor.status === 'active' ? 'ok' : 'warn'}">${esc(vendor.status)}</span></td>
  </tr>`).join('');

  $('#po-table tbody').innerHTML = orders.map((order) => `<tr>
    <td>${esc(order.po_number)}</td><td>${esc(order.vendor_id)}</td>
    <td class="num">${money(order.amount, order.currency)}</td>
    <td><span class="chip">${esc(order.status)}</span></td>
  </tr>`).join('');

  const hotspots = metrics.error_hotspots || [];
  const max = Math.max(1, ...hotspots.map((h) => h.corrections));
  $('#hotspots').innerHTML = hotspots.length
    ? hotspots.map((spot) => `<div class="bar-row">
        <span class="lbl">${esc(spot.doc_type)} · <b>${esc(spot.field_name)}</b></span>
        <span class="track"><i style="width:${(spot.corrections / max) * 100}%"></i></span>
        <span class="cnt">${spot.corrections}</span></div>`).join('')
    : '<p class="muted">No corrections recorded yet — review a document and edit a field to populate this.</p>';
}

const metric = (label, value, sub) => `<div class="metric">
  <div class="label">${esc(label)}</div><div class="value">${esc(value)}</div>
  <div class="sub">${esc(sub)}</div></div>`;

/* -------------------------------------------------------------- settings */

async function loadSettings() {
  try {
    state.settings = await api('/api/settings');
  } catch (error) {
    toast(`Could not load settings: ${error.message}`, 'error');
    return;
  }
  renderSettings();
}

/** Send only what changed; the server merges it over the stored policy. */
async function saveSettings(patch) {
  try {
    state.settings = await api('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    });
  } catch (error) {
    toast(error.message, 'error');
    renderSettings();          // snap the controls back to the stored truth
    return;
  }
  renderSettings();
  loadStatus();                // the header threshold chip reads from status
}

function renderSettings() {
  const { policy, rules } = state.settings;

  $('#policy-warnings').innerHTML = (policy.warnings || [])
    .map((text) => `<div class="notice"><span>⚠</span><div>${esc(text)}</div></div>`).join('');

  const slider = $('#threshold-slider');
  slider.value = Math.round(policy.auto_post_min_confidence * 100);
  $('#threshold-out').textContent = pct(policy.auto_post_min_confidence);
  $('#threshold-note').textContent =
    `Fields below ${pct(policy.field_review_confidence)} are flagged for review individually.`;

  $('#free-mode-toggle').checked = policy.rules_free_mode;
  $('#settings-origin').textContent = policy.overridden.length
    ? `${policy.overridden.length} setting(s) changed from .env`
    : 'Matching the configured defaults';
  $('#settings-reset').disabled = policy.overridden.length === 0;

  const docTypes = [...new Set(rules.flatMap((r) => r.doc_types))].filter((t) => t !== '*').sort();
  const filter = $('#rule-doc-filter');
  if (filter.options.length <= 1) {
    filter.innerHTML = '<option value="">All document types</option>' +
      docTypes.map((t) => `<option value="${esc(t)}">${esc(t.replace('_', ' '))}</option>`).join('');
  }
  renderRuleList();
}

function renderRuleList() {
  const { policy, rules } = state.settings;
  const term = $('#rule-search').value.trim().toLowerCase();
  const docType = $('#rule-doc-filter').value;

  const visible = rules.filter((rule) => {
    if (docType && !rule.applies_to_all && !rule.doc_types.includes(docType)) return false;
    if (!term) return true;
    return `${rule.rule_id} ${rule.summary}`.toLowerCase().includes(term);
  });

  // Free mode already turns everything off; leaving the boxes live would invite
  // someone to toggle a rule that cannot run either way.
  const frozen = policy.rules_free_mode;
  $('#rule-list').innerHTML = visible.length
    ? visible.map((rule) => `
      <label class="rule-row ${rule.enabled && !frozen ? '' : 'off'}">
        <input type="checkbox" data-rule="${esc(rule.rule_id)}"
          ${rule.enabled ? 'checked' : ''} ${frozen ? 'disabled' : ''} />
        <span class="grow">
          <b class="rid">${esc(rule.rule_id)}</b>
          <span class="desc">${esc(rule.summary || 'No description.')}</span>
          <span class="types">
            ${rule.applies_to_all
              ? '<span class="chip">every document</span>'
              : rule.doc_types.map((t) => `<span class="chip">${esc(t.replace('_', ' '))}</span>`).join('')}
            ${rule.aliases.map((a) => `<span class="chip">also reports as ${esc(a)}</span>`).join('')}
          </span>
        </span>
      </label>`).join('')
    : '<p class="muted">No rule matches that filter.</p>';
}

$('#threshold-slider').addEventListener('input', (event) => {
  $('#threshold-out').textContent = `${event.target.value}%`;
});
$('#threshold-slider').addEventListener('change', (event) => {
  saveSettings({ auto_post_min_confidence: Number(event.target.value) / 100 });
});
$('#free-mode-toggle').addEventListener('change', (event) => {
  saveSettings({ rules_free_mode: event.target.checked });
});
$('#rule-search').addEventListener('input', renderRuleList);
$('#rule-doc-filter').addEventListener('change', renderRuleList);
$('#settings-reset').addEventListener('click', async () => {
  try {
    state.settings = await api('/api/settings/reset', { method: 'POST' });
  } catch (error) {
    toast(error.message, 'error');
    return;
  }
  renderSettings();
  loadStatus();
  toast('Back to the configured defaults.');
});
$('#rule-list').addEventListener('change', (event) => {
  const box = event.target.closest('input[data-rule]');
  if (!box) return;
  // The API replaces the whole set, so send every rule currently switched off.
  const disabled = state.settings.rules
    .filter((rule) => (rule.rule_id === box.dataset.rule ? !box.checked : !rule.enabled))
    .map((rule) => rule.rule_id);
  saveSettings({ disabled_rules: disabled });
});

/* ------------------------------------------------------- authoring rules */

async function loadCustomRules() {
  let payload;
  try {
    payload = await api('/api/rules/custom');
  } catch (error) {
    toast(`Could not load your rules: ${error.message}`, 'error');
    return;
  }
  state.customRules = payload.rules;

  const picker = $('#rule-types');
  if (!picker.options.length) {
    // Invoices are the common case; starting on whatever sorts first would have
    // people writing invoice rules against the contract schema.
    picker.innerHTML = payload.doc_types
      .filter((t) => t !== 'unknown')
      .map((t) => `<option value="${esc(t)}"${t === 'invoice' ? ' selected' : ''}>${esc(t.replace('_', ' '))}</option>`)
      .join('');
  }
  renderCustomRules();
}

function renderCustomRules() {
  const rules = state.customRules || [];
  $('#custom-count').textContent = rules.length ? `${rules.length} rule(s)` : '';
  $('#custom-rule-list').innerHTML = rules.length
    ? rules.map((rule) => `
      <div class="custom-rule ${rule.enabled ? '' : 'off'}">
        <input type="checkbox" data-custom="${esc(rule.rule_id)}" ${rule.enabled ? 'checked' : ''} />
        <span class="grow">
          <b class="rid">${esc(rule.rule_id)}</b>
          <span class="chip ${rule.severity === 'blocking' ? 'bad' : 'warn'}">${esc(rule.severity)}</span>
          <span class="desc">${esc(rule.explanation)}</span>
          ${rule.prompt ? `<span class="said">you wrote: “${esc(rule.prompt)}”</span>` : ''}
          <span class="types">${rule.doc_types.map((t) => `<span class="chip">${esc(t.replace('_', ' '))}</span>`).join('')}</span>
        </span>
        <button class="btn small danger" data-delete="${esc(rule.rule_id)}">Delete</button>
      </div>`).join('')
    : '<p class="muted">No rules of your own yet. Describe one above.</p>';
}

const selectedDocTypes = () => [...$('#rule-types').selectedOptions].map((o) => o.value);

function renderDraft(draft) {
  state.draft = draft;
  if (!draft) { $('#rule-draft').innerHTML = ''; return; }

  const { rule, preview } = draft;
  const evaluated = preview.passed + preview.failed;
  // A rule nothing passes, or nothing fails, is the shape of an inverted
  // condition — worth showing loudly rather than burying in the counts.
  const suspect = evaluated > 0 && (preview.passed === 0 || preview.failed === 0);

  $('#rule-draft').innerHTML = `
    <div class="draft">
      <h4>${esc(rule.title)} <span class="chip ${rule.severity === 'blocking' ? 'bad' : 'warn'}">${esc(rule.severity)}</span></h4>
      <div class="reads">${esc(rule.explanation)}</div>
      <p class="muted">Reviewers will see: “${esc(rule.message)}”</p>
      <div class="tally">
        <span>Tried on <b>${preview.considered}</b> document(s)</span>
        <span>would pass <b class="hi">${preview.passed}</b></span>
        <span>would fail <b class="lo">${preview.failed}</b></span>
        <span>not applicable <b>${preview.not_applicable}</b></span>
      </div>
      <p class="${suspect ? '' : 'muted'}" style="font-size:13px">${suspect ? '⚠ ' : ''}${esc(preview.note)}</p>
      ${preview.examples.length ? `<pre>${esc(preview.examples.map((e) =>
        `${e.outcome.toUpperCase().padEnd(5)} ${e.filename}  ${JSON.stringify(e.observed || {})}`).join('\n'))}</pre>` : ''}
      <details><summary class="muted" style="cursor:pointer;font-size:12.5px">Show the compiled condition</summary>
        <pre>${esc(JSON.stringify({ when: rule.when, assert: rule['assert'] }, null, 2))}</pre></details>
      <div class="actions">
        <button class="btn primary" id="draft-save">Save this rule</button>
        <button class="btn" id="draft-discard">Discard</button>
      </div>
    </div>`;

  $('#draft-save').addEventListener('click', saveDraft);
  $('#draft-discard').addEventListener('click', () => renderDraft(null));
}

async function saveDraft() {
  const button = $('#draft-save');
  button.disabled = true;
  try {
    await api('/api/rules/custom', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rule: state.draft.rule, prompt: state.draft.rule.prompt }),
    });
  } catch (error) {
    toast(error.message, 'error');
    button.disabled = false;
    return;
  }
  renderDraft(null);
  $('#rule-prompt').value = '';
  await loadCustomRules();
  toast('Rule saved. It runs on the next document.');
}

$('#rule-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const docTypes = selectedDocTypes();
  if (!docTypes.length) { toast('Pick at least one document type.', 'error'); return; }

  const button = $('#rule-compile');
  button.disabled = true;
  button.textContent = 'Compiling…';
  try {
    const draft = await api('/api/rules/draft', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: $('#rule-prompt').value, doc_types: docTypes }),
    });
    renderDraft(draft);
  } catch (error) {
    toast(error.message, 'error');
  } finally {
    button.disabled = false;
    button.textContent = 'Compile';
  }
});

$('#custom-rule-list').addEventListener('change', async (event) => {
  const box = event.target.closest('input[data-custom]');
  if (!box) return;
  try {
    await api(`/api/rules/custom/${encodeURIComponent(box.dataset.custom)}/enabled?enabled=${box.checked}`,
      { method: 'POST' });
  } catch (error) {
    toast(error.message, 'error');
  }
  loadCustomRules();
});

$('#custom-rule-list').addEventListener('click', async (event) => {
  const button = event.target.closest('button[data-delete]');
  if (!button) return;
  const ruleId = button.dataset.delete;
  if (!confirm(`Delete the rule "${ruleId}"? It will stop running on new documents.`)) return;
  try {
    await api(`/api/rules/custom/${encodeURIComponent(ruleId)}`, { method: 'DELETE' });
  } catch (error) {
    toast(error.message, 'error');
    return;
  }
  loadCustomRules();
  toast(`Deleted ${ruleId}.`);
});

/* ------------------------------------------------------------------ init */

loadStatus();
loadModels();
loadSamples();
