/* DocuFlow front end — vanilla JS, no build step. */
'use strict';

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const state = {
  status: null,
  models: { ocr: [], llm: [] },
  samples: [],
  schemas: {},
  settings: null,       // business policy and the rule catalogue, as stored
  policyDraft: {},      // policy edits staged on screen, not yet saved
  customRules: [],      // rules the user wrote
  recordSummary: null,  // ledger totals per record kind
  recordKind: null,     // which kind the records table is filtered to
  insights: null,       // per-document-type reading quality
  insightType: null,    // which document type the insight breakdowns describe
  agenda: null,         // dated items and stuck counterparties needing attention
  masterTab: 'vendors', // which master file is on screen
  pages: { queue: 1, records: 1, master: 1 },   // current page per list
  draft: null,          // a compiled rule awaiting approval
  currentDoc: null,     // document being reviewed
  edits: {},            // pending reviewer edits
  autoOpen: null,       // job whose result should open itself when it lands
  page: 1,
};

/* ---------------------------------------------------------------- helpers */

const esc = (value) => String(value ?? '').replace(/[&<>"']/g,
  (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const pct = (value) => `${Math.round((value ?? 0) * 100)}%`;

const money = (value, currency) => value == null || value === ''
  ? '—'
  : `${Number(value).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}${currency ? ' ' + currency : ''}`;

/* ---------------------------------------------------------------- paging */

/* Every list here grows without bound, and the failure mode of not paging is
   silent: the table renders, it is just missing the rows nobody scrolled to.
   One pager, driven by the envelope the API returns. */
const PER_PAGE = 10;

function renderPager(host, meta, onGo) {
  const element = typeof host === 'string' ? $(host) : host;
  if (!element) return;
  const { page = 1, pages = 1, total = 0, per_page = PER_PAGE } = meta || {};
  if (total <= per_page) { element.innerHTML = ''; return; }

  const first = (page - 1) * per_page + 1;
  const last = Math.min(page * per_page, total);
  element.innerHTML = `
    <div class="pager">
      <span class="muted">${first}–${last} of ${total}</span>
      <span class="grow"></span>
      <button class="btn small" data-page="1" ${page === 1 ? 'disabled' : ''}>« First</button>
      <button class="btn small" data-page="${page - 1}" ${page === 1 ? 'disabled' : ''}>Previous</button>
      <span class="muted">Page ${page} of ${pages}</span>
      <button class="btn small" data-page="${page + 1}" ${page >= pages ? 'disabled' : ''}>Next</button>
      <button class="btn small" data-page="${pages}" ${page >= pages ? 'disabled' : ''}>Last »</button>
    </div>`;

  element.querySelectorAll('button[data-page]').forEach((button) =>
    button.addEventListener('click', () => onGo(Number(button.dataset.page))));
}

/** Add page/per_page to a path that may already carry a query string. */
const paged = (path, page) =>
  `${path}${path.includes('?') ? '&' : '?'}page=${page}&per_page=${PER_PAGE}`;

const DECISION = {
  auto_post: { label: 'Posted automatically', icon: '✓', chip: 'ok' },
  needs_review: { label: 'Needs human review', icon: '⚑', chip: 'warn' },
  rejected: { label: 'Rejected', icon: '✕', chip: 'bad' },
};

/* Saving a correction re-runs the rules, so a document can clear every check
   without being posted - "save" deliberately stops short of the ledger. Calling
   that "Posted automatically" claims something that has not happened; the
   document is cleared and waiting for someone to approve it. */
function decisionOf(doc) {
  if (doc.decision === 'auto_post' && !doc.record_id) {
    return { label: 'Cleared — ready to post', icon: '✓', chip: 'ok' };
  }
  return DECISION[doc.decision] || { label: doc.decision, icon: '•', chip: '' };
}

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

/* Named so that a link on one page can send the reader to another - following an
   insight to the document it came from is a normal move here, not an edge case. */
function showView(name) {
  $$('.tab').forEach((tab) => tab.classList.toggle('active', tab.dataset.view === name));
  $$('.view').forEach((view) => view.classList.toggle('active', view.id === `view-${name}`));
  // Each view reloads its own list, so the counts on the tabs beside it must not
  // be left showing what was true when the page was opened.
  loadStatus();
  if (name === 'queue') loadQueue();
  if (name === 'records') loadRecords();
  if (name === 'insights') loadInsights();
  if (name === 'settings') { loadSettings(); loadCustomRules(); }
}

$$('.tab').forEach((tab) => tab.addEventListener('click', () => showView(tab.dataset.view)));

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
  // Entries in the ledger, not documents that once produced one. Deleting a
  // record leaves its document behind on purpose, and counting those here made
  // the tab claim 15 records over an empty page.
  $('#records-badge').textContent = stats.posted_records ?? 0;
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
  // Each sample exists as the PDF it was drawn as and as a photograph of it.
  // A PDF can carry a text layer; an image cannot, so only the image puts the
  // OCR model under real load — worth being able to pick.
  $('#sample-list').innerHTML = state.samples
    .filter((sample) => sample.available)
    .map((sample) => `
      <div class="sample">
        <span class="stype">${esc(sample.doc_type.replace('_', ' '))}</span>
        <span class="grow">${esc(sample.title)}
          <small>${esc(sample.demonstrates)}</small></span>
        <span class="sample-formats">
          ${(sample.formats || ['pdf']).map((format) => `
            <button class="btn small" data-sample="${esc(sample.doc_id)}" data-format="${esc(format)}"
              title="${format === 'image' ? 'Process a photograph of this page — no text layer, OCR only'
                                          : 'Process the PDF, text layer included'}"
              >${format === 'image' ? 'Image' : 'PDF'}</button>`).join('')}
        </span>
      </div>`).join('');

  $$('#sample-list button[data-sample]').forEach((button) => button.addEventListener('click', () => {
    const title = button.closest('.sample').querySelector('.grow').textContent.trim().split('\n')[0];
    ingest(`/api/documents/sample/${button.dataset.sample}?format=${button.dataset.format}&queue=true`,
      { method: 'POST' }, `${title} (${button.dataset.format})`);
  }));
}

/* -------------------------------------------------------------- ingestion */

/* Everything goes through the queue — one file, twenty files, a sample, a URL.
   Processing one document inline read nicely until you dropped a second file
   while the first was still going: the second was refused outright, and with a
   local OCR model the queue's single worker is the only thing keeping two
   documents from fighting over one resident model. */

async function ingest(path, options, label) {
  let result;
  try {
    result = await api(path, options);
  } catch (error) {
    toast(error.message, 'error');
    return;
  }
  (result.rejected || []).forEach((row) => toast(`${row.filename}: ${row.reason}`, 'error'));
  const queued = result.queued || [];
  if (!queued.length) return;

  // A single document is usually something you want to look at, so its result
  // opens by itself when it lands. A batch does not: twenty cards fighting over
  // one panel helps nobody, and the queue list is the view of a batch.
  state.autoOpen = queued.length === 1 ? queued[0].job_id : null;
  toast(queued.length === 1 ? `Queued ${label || queued[0].filename}.`
                            : `Queued ${queued.length} documents.`);
  startJobPolling();
}

function handleFiles(files) {
  const list = [...(files || [])];
  if (!list.length) return;
  const form = new FormData();
  list.forEach((file) => form.append('files', file));
  ingest('/api/documents/batch', { method: 'POST', body: form }, list[0].name);
}

$('#file-input').addEventListener('change', (event) => {
  handleFiles(event.target.files);
  event.target.value = '';
});

const dropzone = $('#dropzone');
['dragenter', 'dragover'].forEach((type) => dropzone.addEventListener(type, (event) => {
  event.preventDefault(); dropzone.classList.add('drag');
}));
['dragleave', 'drop'].forEach((type) => dropzone.addEventListener(type, (event) => {
  event.preventDefault(); dropzone.classList.remove('drag');
}));
dropzone.addEventListener('drop', (event) => handleFiles(event.dataTransfer?.files));

$('#url-form').addEventListener('submit', (event) => {
  event.preventDefault();
  const url = $('#url-input').value.trim();
  if (!url) return;
  ingest('/api/documents/url?queue=true', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url }),
  }, url);
});

/* ---------------------------------------------------------------- result */

function renderResult(data, host, editable) {
  const doc = data.document;
  state.currentDoc = doc;
  state.edits = {};
  state.page = 1;

  const decision = decisionOf(doc);
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
    ${renderRegister(data)}

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

  // Wired outside the editable branch: a posted document can still be the one
  // that tells you a contract belongs in the register.
  $$('[data-register]', host).forEach((button) => button.addEventListener(
    'click', () => submitRegistration(doc.document_id, button.dataset.register, host)));
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

/* A document is not only something to check against the master file — often it
   is the thing asking to be *in* it. The values are already read and already
   carry their evidence, so registering is a matter of confirming them, not of
   retyping them somewhere else. A person presses the button; nothing here is
   automatic, which is the entire point of `vendor_known`. */
const REGISTER_LABELS = {
  legal_name: 'Legal name', tax_id: 'Tax ID', bank_account: 'Bank account',
  currency: 'Currency', payment_terms: 'Payment terms', po_number: 'PO number',
  po_date: 'PO date', amount: 'Amount', contract_number: 'Contract number',
  party_a: 'Party A', party_b: 'Party B', effective_date: 'Effective', expiry_date: 'Expires',
  value: 'Value', vendor_id: 'Vendor',
};

function renderRegister(data) {
  const offers = data.register || [];
  if (!offers.length) return '';

  return offers.map((offer) => {
    if (!offer.available) {
      return `<div class="card"><h3>${esc(offer.title)}</h3>
        <p class="muted" style="margin-top:6px">${esc(offer.blocked)}</p></div>`;
    }
    const fields = Object.entries(offer.values).map(([name, value]) => `
      <label class="reg-field">
        <span class="lbl">${esc(REGISTER_LABELS[name] || name)}
          ${offer.ungrounded.includes(name) ? '<span class="chip warn">unverified</span>' : ''}</span>
        <input class="reg-input" data-reg-field="${esc(name)}" value="${esc(value)}" />
        ${offer.evidence[name]
          ? `<span class="evidence">“${esc(offer.evidence[name])}”</span>`
          : '<span class="evidence">no quote on the page supports this</span>'}
      </label>`).join('');

    return `<div class="card" data-register-card="${esc(offer.kind)}">
      <div class="card-head"><h3>${esc(offer.title)}</h3>
        <button class="btn primary" data-register="${esc(offer.kind)}">Add to master data</button></div>
      <p class="muted">Read from this document. Check each value against the image before saving —
        once it is in the master file, every future document is judged against it.</p>
      ${offer.warnings.map((text) => `<div class="notice"><span>⚠</span><div>${esc(text)}</div></div>`).join('')}
      <div class="reg-grid">${fields}</div>
    </div>`;
  }).join('');
}

async function submitRegistration(documentId, kind, host) {
  const card = $(`[data-register-card="${kind}"]`, host);
  const values = Object.fromEntries(
    $$('.reg-input', card).map((input) => [input.dataset.regField, input.value.trim()]));

  const button = $(`[data-register="${kind}"]`, host);
  button.disabled = true;
  button.textContent = 'Saving…';
  try {
    const result = await api(`/api/documents/${encodeURIComponent(documentId)}/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ kind, values }),
    });
    toast(result.message, 'success');
    // The document was held because this record was missing. Now that it
    // exists, re-judge it rather than leaving the reviewer to work out that
    // the reason has gone.
    await api(`/api/documents/${encodeURIComponent(documentId)}/reevaluate`, { method: 'POST' });
    await loadQueue();
    await openDocument(documentId);
    loadStatus();
  } catch (error) {
    toast(error.message, 'error');
    button.disabled = false;
    button.textContent = 'Add to master data';
  }
}

function renderAnalyst(data) {
  const rec = data.reconciliation;
  if (!rec || (!rec.assessment && !rec.corrections?.length)) return '';
  // Prose cannot be checked the way a value can, so when every value it offered
  // was refused by the page, the paragraph gets the same warning rather than
  // being read as a finding.
  const label = (c) => c.contradicts_evidence ? 'the page says otherwise'
    : c.grounded ? 'verified on page' : 'not on page';

  return `<div class="card analyst${rec.unreliable ? ' doubted' : ''}">
    <h3>Discrepancy analyst</h3>
    ${rec.unreliable ? `<p class="doubt">⚠ Nothing this agent proposed held up against the page.
      Read the note below as a suggestion to check, not as a finding.</p>` : ''}
    <p style="font-size:13px">${esc(rec.assessment)}</p>
    ${rec.reviewer_note ? `<p class="muted" style="margin-top:6px"><b>Reviewer:</b> ${esc(rec.reviewer_note)}</p>` : ''}
    ${rec.root_cause && rec.root_cause !== 'none'
      ? `<p style="margin-top:6px"><span class="chip">${esc(rec.root_cause.replace('_', ' '))}</span></p>` : ''}
    ${renderToolCalls(rec.tool_calls)}
    ${(rec.corrections || []).map((c) => `
      <div class="suggestion">
        <span class="chip ${c.grounded ? 'ok' : 'bad'}">${label(c)}</span>
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

const VERDICT = {
  pass: { icon: '✓', label: 'satisfied' },
  fail: { icon: '✕', label: 'not satisfied' },
  unsure: { icon: '?', label: 'could not be decided' },
};
const STEP_MARK = { ok: '✓', problem: '✕', unknown: '?' };

// An agent rule reached its answer somehow, and a reviewer who cannot see how
// has no way to tell a good verdict from a confident one. The plan is what the
// author approved; the steps and tools are what actually happened.
function renderAgentWork(detail) {
  if (!detail || detail.kind !== 'agent') return '';
  const verdict = VERDICT[detail.verdict] || VERDICT.unsure;
  const steps = detail.checked?.length
    ? detail.checked
    : (detail.plan || []).map((step) => ({ step, result: 'unknown' }));
  return `<details class="agent-work">
    <summary>${verdict.icon} agent · ${esc(verdict.label)}${detail.model ? ` · ${esc(detail.model)}` : ''}</summary>
    <div class="agent-body">
      <p class="asked">Asked to decide: ${esc(detail.instruction || '')}</p>
      <ol>${steps.map((row) => `<li class="${esc(row.result || 'unknown')}">
        <span>${STEP_MARK[row.result] || '·'}</span> ${esc(row.step)}</li>`).join('')}</ol>
      ${detail.evidence?.length ? `<p class="lbl">What it saw</p>
        <ul>${detail.evidence.map((item) => `<li>${esc(item)}</li>`).join('')}</ul>` : ''}
      ${detail.tools_used?.length ? `<p class="lbl">Tools used</p>
        <p>${detail.tools_used.map((name) => `<span class="chip">${esc(name)}</span>`).join(' ')}</p>` : ''}
      ${detail.note ? `<p class="muted">${esc(detail.note)}</p>` : ''}
    </div></details>`;
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
      <div>${esc(finding.message)}<span class="rid">${esc(finding.rule_id)}</span>
        ${renderAgentWork(finding.detail)}</div>
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
    <button class="btn" id="btn-recheck" title="Apply the current rules and threshold to this document without reading it again">Re-check</button>
    <button class="btn" id="btn-save" title="Keep the corrected values. The document stays where it is until you approve or re-check it">Save</button>
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

  /* Unsaved edits are not part of a re-check: this asks what the *current*
     policy makes of the document as it stands, so a pending edit would quietly
     change the answer. */
  $('#btn-recheck', host).addEventListener('click', async () => {
    if (Object.keys(state.edits).length
        && !confirm('Re-check ignores unsaved edits. Save them first if they should count.')) return;
    $$('.review-actions .btn', host).forEach((button) => (button.disabled = true));
    try {
      const result = await api(
        `/api/documents/${encodeURIComponent(doc.document_id)}/reevaluate`, { method: 'POST' });
      toast(result.message, result.changed ? 'success' : '');
      await loadQueue();
      await openDocument(doc.document_id);
      loadStatus();
    } catch (error) {
      toast(error.message, 'error');
      $$('.review-actions .btn', host).forEach((button) => (button.disabled = false));
    }
  });
}

/* ----------------------------------------------------------------- queue */

async function loadQueue(page = state.pages.queue) {
  const filter = $('#queue-filter').value;
  state.pages.queue = page;
  let payload;
  try {
    payload = await api(paged(
      `/api/documents${filter ? `?decision=${encodeURIComponent(filter)}` : ''}`, page));
  } catch (error) { toast(error.message, 'error'); return; }

  const rows = payload.rows;
  renderPager('#queue-pager', payload, loadQueue);
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
    <td><span class="chip ${decisionOf(row).chip}">${esc(decisionOf(row).label)}</span></td>
    <td>${row.record_id ? esc(row.record_id) : '—'}</td>
    <td class="row-actions">
      <button class="btn small" data-open="${esc(row.document_id)}">Open</button>
      ${row.record_id ? '' : `<button class="btn small danger" data-delete="${esc(row.document_id)}"
        data-name="${esc(row.filename)}" title="Remove this document and its files">Delete</button>`}
    </td>
  </tr>`).join('');

  $$('[data-open]').forEach((button) => button.addEventListener('click',
    () => openDocument(button.dataset.open)));
}

/* A posted document has no Delete button at all: the ledger refers to it, and
   offering an action the server will refuse is worse than not offering it. */
$('#queue-table').addEventListener('click', async (event) => {
  const button = event.target.closest('button[data-delete]');
  if (!button) return;
  const { delete: documentId, name } = button.dataset;
  if (!confirm(`Delete ${name}?\n\nThe document, its rendered pages and its stored file are removed. This cannot be undone.`)) return;

  button.disabled = true;
  try {
    const result = await api(`/api/documents/${encodeURIComponent(documentId)}`, { method: 'DELETE' });
    toast(result.message, 'success');
    if (state.currentDoc?.document_id === documentId) $('#review-host').innerHTML = '';
    await loadQueue();
    loadStatus();
  } catch (error) {
    toast(error.message, 'error');
    button.disabled = false;
  }
});

/* Re-judging the queue after a policy change. No page is read again - only the
   rules, the threshold and the master data are applied afresh. */
$('#queue-recheck').addEventListener('click', async () => {
  const scope = $('#queue-filter').value || 'needs_review';
  if (!confirm(`Re-check every document with the decision "${scope.replace('_', ' ')}" against the policy in force now?\n\nDocuments that now clear every check will post themselves, exactly as they would have on upload.`)) return;

  const button = $('#queue-recheck');
  button.disabled = true;
  button.textContent = 'Re-checking…';
  try {
    const result = await api('/api/documents/reevaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ decision: scope }),
    });
    toast(result.summary, result.changed ? 'success' : '');
    await loadQueue();
    loadStatus();
  } catch (error) {
    toast(error.message, 'error');
  } finally {
    button.disabled = false;
    button.textContent = 'Re-check under current policy';
  }
});

async function openDocument(documentId) {
  try {
    const data = await api(`/api/documents/${encodeURIComponent(documentId)}`);
    // Posted is what closes a document to editing, not the routing decision.
    // Judging by decision hid the Approve button on exactly the documents a save
    // had just cleared, leaving them stuck: cleared, unposted, unapprovable.
    const editable = !data.document.record_id;
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

$('#queue-filter').addEventListener('change', () => loadQueue(1));
$('#queue-refresh').addEventListener('click', () => loadQueue());

/* --------------------------------------------------------------- records */

async function loadRecords(page = state.pages.records) {
  state.pages.records = page;
  let payload, summary;
  try {
    // The summary drives the cards; the list is filtered to whichever kind is
    // selected, so the two calls always describe the same ledger.
    [summary, payload] = await Promise.all([
      api('/api/records/summary'),
      api(paged(state.recordKind
        ? `/api/records?record_kind=${encodeURIComponent(state.recordKind)}` : '/api/records', page)),
    ]);
  } catch (error) { toast(error.message, 'error'); return; }

  const rows = payload.rows;
  renderPager('#records-pager', payload, loadRecords);
  state.recordSummary = summary;
  renderKindCards(summary);
  renderKindInsight(summary);

  const body = $('#records-table tbody');
  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="8"><div class="empty-state">${
      state.recordKind ? 'Nothing of this kind yet.' : 'Nothing posted yet.'}</div></td></tr>`;
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
    <td class="row-actions">
      ${row.document_id
        ? `<button class="btn small" data-open-record="${esc(row.document_id)}">View</button>` : ''}
      <button class="btn small danger" data-delete-record="${esc(row.record_id)}"
        data-label="${esc(`${row.record_id} · ${row.counterparty || ''} ${money(row.amount, row.currency)}`)}"
        title="Remove this ledger entry and release the document behind it">Delete</button>
    </td>
  </tr>`).join('');
}

$('#records-refresh').addEventListener('click', () => loadRecords());

/* Each kind of record answers a different question, so the ledger is shown as
   one card per kind rather than one undifferentiated table. Clicking a card
   filters the table and the insight strip below it. */
function renderKindCards(summary) {
  const labels = summary.labels || {};
  const cards = summary.kinds.map((kind) => {
    const meta = labels[kind.record_kind] || {};
    const money = kind.totals.length
      ? kind.totals.map((t) => money_(t.total, t.currency)).join(' · ')
      : '—';
    return `
      <button class="kind-card ${state.recordKind === kind.record_kind ? 'active' : ''}"
              data-kind="${esc(kind.record_kind)}">
        <div class="k-title">${esc(meta.title || kind.record_kind)}</div>
        <div class="k-count">${kind.count}</div>
        <div class="k-money">${esc(money)}</div>
        <div class="k-blurb">${esc(meta.blurb || '')}</div>
      </button>`;
  }).join('');

  $('#kind-grid').innerHTML = cards
    || '<p class="muted">Nothing has been posted yet. Process a document to fill the ledger.</p>';
}

function renderKindInsight(summary) {
  const selected = state.recordKind
    ? summary.kinds.find((k) => k.record_kind === state.recordKind)
    : null;
  const meta = (summary.labels || {})[state.recordKind] || {};

  $('#kind-title').textContent = selected ? (meta.title || state.recordKind) : 'All records';
  $('#kind-blurb').textContent = selected ? (meta.blurb || '') : `${summary.total_records} record(s)`;
  $('#kind-clear').hidden = !state.recordKind;

  if (!selected) { $('#kind-insight').innerHTML = ''; return; }

  const parties = (selected.top_counterparties || []).slice(0, 4)
    .map((p) => `<div class="party">${esc(p.counterparty || '—')}
      <b>${p.count}×</b> ${esc(money_(p.total, p.currency))}</div>`).join('');

  $('#kind-insight').innerHTML = `
    <div class="insight-row">
      ${insightItem('Records', selected.count)}
      ${insightItem('Posted automatically', `${selected.auto_posted} of ${selected.count}`)}
      ${insightItem('Counterparties', selected.counterparties)}
      ${insightItem('Period', `${selected.earliest || '—'} → ${selected.latest || '—'}`)}
      ${selected.totals.map((t) => insightItem(`Total ${t.currency}`, money_(t.total, t.currency))).join('')}
    </div>
    ${parties ? `<div style="padding:8px 0"><div class="lbl muted"
      style="font-size:11px;text-transform:uppercase;letter-spacing:.04em">Most frequent counterparties</div>
      ${parties}</div>` : ''}`;
}

const insightItem = (label, value) =>
  `<div class="item"><div class="lbl">${esc(label)}</div><div class="val">${esc(value)}</div></div>`;

// The table's `money` helper takes (value, currency); this keeps the call sites short.
const money_ = (value, currency) => money(value, currency);

$('#kind-grid').addEventListener('click', (event) => {
  const card = event.target.closest('button[data-kind]');
  if (!card) return;
  // Clicking the selected kind again clears the filter.
  state.recordKind = state.recordKind === card.dataset.kind ? null : card.dataset.kind;
  $('#record-detail-host').innerHTML = '';
  loadRecords(1);
});

$('#kind-clear').addEventListener('click', () => {
  state.recordKind = null;
  $('#record-detail-host').innerHTML = '';
  loadRecords(1);
});


/* Deleting a ledger entry releases the document that produced it: the document
   goes back to the queue rather than being removed, because the page is still
   the evidence for the entry that used to exist. */
$('#records-table').addEventListener('click', async (event) => {
  const button = event.target.closest('button[data-delete-record]');
  if (!button) return;
  if (!confirm(`Delete ${button.dataset.label}?\n\nThe ledger entry is removed and its document returns to the review queue. This cannot be undone.`)) return;

  button.disabled = true;
  try {
    const result = await api(`/api/records/${encodeURIComponent(button.dataset.deleteRecord)}`,
      { method: 'DELETE' });
    toast(result.message, 'success');
    $('#record-detail-host').innerHTML = '';
    await loadRecords();
    loadStatus();
  } catch (error) {
    toast(error.message, 'error');
    button.disabled = false;
  }
});

/* A posted record is only as trustworthy as the page it came from, so opening
   one shows the same panel the reviewer saw: the document image beside every
   extracted field, its evidence and every rule that ran. Read-only - this
   document is already in the ledger. */
$('#records-table').addEventListener('click', async (event) => {
  const button = event.target.closest('button[data-open-record]');
  if (!button) return;
  const host = $('#record-detail-host');
  host.innerHTML = '<div class="card"><p class="muted">Loading document…</p></div>';
  try {
    const data = await api(`/api/documents/${encodeURIComponent(button.dataset.openRecord)}`);
    renderResult(data, host, false);
    host.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (error) {
    host.innerHTML = '';
    toast(error.message, 'error');
  }
});

/* -------------------------------------------------------------- insights */

async function loadInsights() {
  let metrics, insights, agenda;
  try {
    [metrics, insights, agenda] = await Promise.all([
      api('/api/metrics'), api('/api/insights'), api('/api/attention'),
    ]);
  } catch (error) { toast(error.message, 'error'); return; }

  state.insights = insights;
  renderAttention(agenda);
  const decisions = metrics.by_decision || {};
  $('#metric-grid').innerHTML = `
    ${metric('Documents processed', metrics.total_documents, 'since the database was created')}
    ${metric('Straight-through rate', pct(metrics.auto_post_rate),
      `${decisions.auto_post || 0} posted without a human`)}
    ${metric('Awaiting review', decisions.needs_review || 0, 'held back by a rule or low confidence')}
    ${metric('Average confidence', pct(metrics.average_confidence), 'across all documents')}
    ${metric('Reviewer corrections', metrics.corrections, 'field edits made by humans')}`;

  renderTypeCards(insights);
  renderTypeDetail(insights);
  loadMaster(state.masterTab);
}

const metric = (label, value, sub) => `<div class="metric">
  <div class="label">${esc(label)}</div><div class="value">${esc(value)}</div>
  <div class="sub">${esc(sub)}</div></div>`;

/* ------------------------------------------------------------- attention */

/* Every line says where it came from and opens that document. A dashboard
   figure nobody can trace back to a page is just a confident guess. */
const URGENCY = {
  overdue: { chip: 'bad', label: 'overdue' },
  soon: { chip: 'warn', label: 'due soon' },
  watch: { chip: '', label: 'watch' },
};

function renderAttention(agenda) {
  state.agenda = agenda;
  $('#attention-asof').textContent = `as of ${agenda.as_of}`;

  const { overdue = 0, soon = 0, stuck_parties = 0 } = agenda.counts || {};
  if (!agenda.items.length && !agenda.parties.length) {
    $('#attention-list').innerHTML =
      '<p class="muted">Nothing is due and nothing is held. Process a document to fill this in.</p>';
    $('#stuck-parties').innerHTML = '';
    return;
  }

  $('#attention-list').innerHTML = agenda.items.map((item) => {
    const urgency = URGENCY[item.urgency] || URGENCY.watch;
    return `<div class="attn ${esc(item.urgency)}">
      <div class="attn-main">
        <div class="attn-head">
          <span class="chip ${urgency.chip}">${esc(urgency.label)}</span>
          <b>${esc(item.headline)}</b>
        </div>
        <div class="attn-detail">${esc(item.detail)}</div>
        <div class="attn-source">from ${esc(item.source)}</div>
      </div>
      <div class="attn-actions">
        ${item.days_left === null ? '' : `<div class="attn-days">${daysLabel(item.days_left)}</div>`}
        ${item.document_id
          ? `<button class="btn small" data-open-source="${esc(item.document_id)}">Read it</button>`
          : '<span class="muted small">no document on file</span>'}
      </div>
    </div>`;
  }).join('');

  $('#stuck-parties').innerHTML = agenda.parties.length ? `
    <h4 class="sub-head" style="margin-top:16px">Who is stuck</h4>
    <p class="muted small">One held document is an incident. The same counterparty held
      repeatedly for the same rule is a problem with the record, the policy or how they
      issue documents.</p>
    ${agenda.parties.map((party) => `<div class="attn">
      <div class="attn-main">
        <div class="attn-head"><b>${esc(party.counterparty)}</b>
          <span class="chip">${party.documents} held</span></div>
        <div class="attn-detail">${party.rules.length
          ? `Held by ${party.rules.map((rule) => `<code>${esc(rule)}</code>`).join(', ')}.`
          : 'No rule failed — held on confidence alone.'}
          Oldest has waited ${party.oldest_days} day(s).</div>
      </div>
      <div class="attn-actions">${party.document_ids.slice(0, 3).map((id) =>
        `<button class="btn small" data-open-source="${esc(id)}">${esc(id.slice(-6))}</button>`).join('')}</div>
    </div>`).join('')}` : '';
}

const daysLabel = (days) => days < 0 ? `${Math.abs(days)}d late` : days === 0 ? 'today' : `in ${days}d`;

/* The agent writes "[CTR-2025-018]" the way a person writes a reference number.
   Turn each one into the thing it refers to: the counterparty's name, clickable
   through to the document. An id printed raw is a worse sentence and a dead end.
   Escaping happens first, so nothing from the model reaches the DOM as markup. */
function citeLinks(text) {
  const items = state.agenda?.items || [];
  return esc(text).replace(/\[([A-Za-z0-9][\w.\-]*)\]/g, (whole, id) => {
    const item = items.find((candidate) => candidate.item_id === id);
    if (!item) return whole;
    const label = esc(item.label || item.headline);
    return item.document_id
      ? `<button class="linkish" data-open-source="${esc(item.document_id)}">${label}</button>`
      : `<b>${label}</b>`;
  });
}

/* The agent's note. It ranks and explains; it never supplies a number, and any
   item id it cited that is not on the agenda was dropped before this rendered. */
function renderBriefing(briefing) {
  if (!briefing) { $('#briefing').innerHTML = ''; return; }
  const highlights = (briefing.highlights || []).map((highlight) => {
    const item = (state.agenda?.items || []).find((i) => i.item_id === highlight.item_id);
    if (!item) return '';
    return `<li><b>${esc(item.headline)}</b> — ${esc(highlight.note)}
      ${item.document_id ? `<button class="linkish" data-open-source="${esc(item.document_id)}">read it</button>` : ''}</li>`;
  }).join('');

  $('#briefing').innerHTML = `<div class="briefing">
    <div class="briefing-head">
      <span class="chip accent">agent</span>
      <span class="muted small">written by ${esc(briefing.written_by)} over ${
        state.agenda?.items.length || 0} computed item(s) — it ranks them, it does not compute them</span>
    </div>
    <p class="briefing-text">${citeLinks(briefing.summary)}</p>
    ${highlights ? `<ul class="briefing-list">${highlights}</ul>` : ''}
    ${briefing.watch ? `<p class="muted small">Worth noticing: ${esc(briefing.watch)}</p>` : ''}
    ${briefing.dropped?.length
      ? `<p class="muted small">Dropped ${briefing.dropped.length} reference(s) to items that are not on the
         agenda: ${esc(briefing.dropped.join(', '))}.</p>` : ''}
    ${briefing.note ? `<p class="muted small">${esc(briefing.note)}</p>` : ''}
  </div>`;
}

$('#attention-brief').addEventListener('click', async () => {
  const button = $('#attention-brief');
  button.disabled = true;
  button.textContent = 'Reading the agenda…';
  try {
    const payload = await api('/api/attention/briefing', { method: 'POST' });
    renderAttention(payload);
    renderBriefing(payload.briefing);
  } catch (error) {
    toast(error.message, 'error');
  } finally {
    button.disabled = false;
    button.textContent = 'Summarise with the agent';
  }
});

/* One handler for every "open the source" control on this page. */
$('#attention-card').addEventListener('click', async (event) => {
  const button = event.target.closest('[data-open-source]');
  if (!button) return;
  showView('queue');
  await openDocument(button.dataset.openSource);
});

/* One card per document type. The straight-through rate is the headline because
   it is the number the whole system exists to move. */
function renderTypeCards(insights) {
  $('#type-grid').innerHTML = insights.types.map((type) => `
    <button class="kind-card ${state.insightType === type.doc_type ? 'active' : ''}"
            data-type="${esc(type.doc_type)}">
      <div class="k-title">${esc(type.title)}</div>
      <div class="k-count">${pct(type.auto_post_rate)}</div>
      <div class="k-money">${type.auto_posted} of ${type.total} posted on their own</div>
      <div class="k-blurb">${esc(heldNote(type))}</div>
    </button>`).join('')
    || '<p class="muted">No documents processed yet. Upload one to fill this in.</p>';
}

/* Held by a rule and held by low confidence are different problems with
   different fixes, so the card says which one this type has. */
function heldNote({ held, blocked }) {
  if (!held) return 'Nothing held.';
  if (!blocked) return `${held} held, none by a rule`;
  if (blocked === held) return `${held} held, all by a blocking rule`;
  return `${held} held, ${blocked} by a blocking rule`;
}

/* With a type selected the three breakdowns describe that type alone; with none
   selected they describe everything, each row carrying the type it came from. */
function renderTypeDetail(insights) {
  const selected = insights.types.find((type) => type.doc_type === state.insightType) || null;
  const scope = selected ? [selected] : insights.types;

  $('#type-title').textContent = selected ? selected.title : 'All document types';
  $('#type-blurb').textContent = selected
    ? selected.blurb
    : `${insights.total_documents} document(s) across ${insights.types.length} type(s)`;
  $('#type-clear').hidden = !state.insightType;

  $('#type-insight').innerHTML = selected ? `
    <div class="insight-row">
      ${insightItem('Processed', selected.total)}
      ${insightItem('Posted automatically', `${selected.auto_posted} of ${selected.total}`)}
      ${insightItem('Held for review', selected.held)}
      ${insightItem('Blocked by a rule', selected.blocked)}
      ${insightItem('Average confidence', pct(selected.average_confidence))}
      ${insightItem('Period', `${selected.earliest || '—'} → ${selected.latest || '—'}`)}
      ${insightItem('Posts into', selected.record_kind)}
    </div>` : '';

  // Failing rules, most-blocking first: a warning that fires often still matters
  // less than a blocking rule that fires at all.
  const rules = flatten(scope, 'failing_rules', (rule) => ({
    label: rule.rule_id,
    title: (insights.rules || {})[rule.rule_id] || '',
    count: rule.documents,
    severity: rule.severity,
  })).sort((a, b) => (b.severity === 'blocking') - (a.severity === 'blocking') || b.count - a.count);

  $('#type-rules').innerHTML = bars(rules, 'Nothing failed — every rule passed on these documents.',
    (row) => `<span class="chip ${row.severity === 'blocking' ? 'bad' : 'warn'}">${esc(row.severity)}</span>`);

  const fields = flatten(scope, 'field_gaps', (gap) => ({
    label: gap.label || gap.field_name,
    title: `${gap.missing} empty · ${gap.unverified} unverified of ${gap.seen} read`,
    count: gap.missing + gap.unverified,
    critical: gap.critical, required: gap.required,
  })).sort((a, b) => (b.critical - a.critical) || (b.required - a.required) || b.count - a.count);

  $('#type-fields').innerHTML = bars(fields, 'Every field came back filled and verified.',
    (row) => row.critical ? '<span class="chip bad">critical</span>'
      : row.required ? '<span class="chip warn">required</span>' : '');

  const corrections = flatten(scope, 'corrections', (spot) => ({
    label: spot.field_name, count: spot.corrections,
  })).sort((a, b) => b.count - a.count);

  $('#type-corrections').innerHTML = bars(corrections,
    'No corrections recorded yet — review a document and edit a field to populate this.');
}

/* Pull one list off every type in scope, tagging each row with its type when
   more than one is shown so an unlabelled row never means two different things. */
function flatten(types, key, shape) {
  return types.flatMap((type) => (type[key] || []).map((item) => ({
    ...shape(item),
    doc_type: types.length > 1 ? type.doc_type : '',
  })));
}

function bars(rows, emptyMessage, chip = () => '') {
  if (!rows.length) return `<p class="muted small">${esc(emptyMessage)}</p>`;
  const max = Math.max(1, ...rows.map((row) => row.count));
  return rows.slice(0, 10).map((row) => `<div class="bar-row" ${row.title ? `title="${esc(row.title)}"` : ''}>
    <span class="lbl">${row.doc_type ? `<i class="dim">${esc(row.doc_type)}</i> ` : ''}<b>${esc(row.label)}</b>
      ${chip(row)}</span>
    <span class="track"><i style="width:${(row.count / max) * 100}%"></i></span>
    <span class="cnt">${row.count}</span></div>`).join('');
}

$('#type-grid').addEventListener('click', (event) => {
  const card = event.target.closest('button[data-type]');
  if (!card) return;
  state.insightType = state.insightType === card.dataset.type ? null : card.dataset.type;
  renderTypeCards(state.insights);
  renderTypeDetail(state.insights);
});

$('#type-clear').addEventListener('click', () => {
  state.insightType = null;
  renderTypeCards(state.insights);
  renderTypeDetail(state.insights);
});

$('#insights-refresh').addEventListener('click', loadInsights);

/* ------------------------------------------------------------ master data */

/* Three different files with three different shapes, so each gets its own
   columns rather than a lowest-common-denominator table. */
const MASTER_VIEWS = {
  'vendors': {
    path: '/api/master/vendors',
    empty: 'No vendors registered. Documents from any counterparty will be held as unknown.',
    head: ['ID', 'Legal name', 'Tax ID', 'Bank account', 'Status'],
    row: (v) => `<td>${esc(v.vendor_id)}</td><td>${esc(v.legal_name)}</td>
      <td class="fval">${esc(v.tax_id || '—')}</td><td class="fval">${esc(v.bank_account || '—')}</td>
      <td><span class="chip ${v.status === 'active' ? 'ok' : 'warn'}">${esc(v.status)}</span></td>`,
  },
  'purchase-orders': {
    path: '/api/master/purchase-orders',
    empty: 'No purchase orders on file. Invoices cannot be matched against one.',
    head: ['PO', 'Vendor', 'Date', 'Amount', 'Status'],
    row: (o) => `<td>${esc(o.po_number)}</td><td>${esc(o.vendor_id)}</td>
      <td>${esc(o.po_date || '—')}</td><td class="num">${money(o.amount, o.currency)}</td>
      <td><span class="chip ${o.status === 'open' ? 'ok' : ''}">${esc(o.status)}</span></td>`,
  },
  'contracts': {
    path: '/api/master/contracts',
    empty: 'No contracts on file.',
    head: ['Contract', 'Vendor', 'Effective', 'Expires', 'Value'],
    row: (c) => `<td>${esc(c.contract_number)}</td><td>${esc(c.vendor_id)}</td>
      <td>${esc(c.effective_date || '—')}</td><td>${esc(c.expiry_date || '—')}</td>
      <td class="num">${money(c.value, c.currency)}</td>`,
  },
};

/* The API names these with underscores; the tabs use hyphens because they are
   also URL fragments. One place to convert. */
const masterKind = (tab) => tab.replace('-', '_');

async function loadMaster(name, page = 1) {
  const view = MASTER_VIEWS[name];
  if (!view) return;
  if (name !== state.masterTab) page = 1;      // a new file starts at its own page one
  state.masterTab = name;
  state.pages.master = page;
  $$('#master-tabs .chip-tab').forEach((tab) => tab.classList.toggle('active', tab.dataset.master === name));
  $('#master-template').href = `/api/master/template?kind=${masterKind(name)}`;
  $('#master-template').textContent = `Template CSV`;

  let payload;
  try { payload = await api(paged(view.path, page)); }
  catch (error) { toast(error.message, 'error'); return; }

  const rows = payload.rows;
  renderPager('#master-pager', payload, (next) => loadMaster(name, next));
  $('#master-host').innerHTML = rows.length
    ? `<table><thead><tr>${view.head.map((h) => `<th>${esc(h)}</th>`).join('')}</tr></thead>
       <tbody>${rows.map((row) => `<tr>${view.row(row)}</tr>`).join('')}</tbody></table>`
    : `<p class="muted">${esc(view.empty)}</p>`;
}

$('#master-tabs').addEventListener('click', (event) => {
  const tab = event.target.closest('button[data-master]');
  if (tab) loadMaster(tab.dataset.master);
});

/* Import loads whichever file the visible tab is showing, so there is no way to
   put purchase orders into the vendor table by picking the wrong option. */
$('#master-import').addEventListener('click', () => $('#master-file').click());

$('#master-file').addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  event.target.value = '';                    // so the same file can be picked twice

  const body = new FormData();
  body.append('file', file);
  $('#import-report').innerHTML = '<p class="muted">Reading the file…</p>';
  try {
    const report = await api(`/api/master/import?kind=${masterKind(state.masterTab)}`, { method: 'POST', body });
    renderImportReport(report);
    toast(report.summary, report.imported ? 'success' : 'error');
    await loadMaster(state.masterTab);
    loadInsights();                           // the agenda reads the register
  } catch (error) {
    $('#import-report').innerHTML = `<div class="notice"><span>⚠</span><div>${esc(error.message)}</div></div>`;
  }
});

/* A rejected row is only useful if the reader can find and fix it, so each one
   is reported by the row number they will see in their spreadsheet. */
function renderImportReport(report) {
  if (!report.errors.length) {
    $('#import-report').innerHTML = report.imported
      ? `<div class="notice ok"><span>✓</span><div>${esc(report.summary)}</div></div>` : '';
    return;
  }
  $('#import-report').innerHTML = `<div class="notice"><span>⚠</span><div>
    ${esc(report.summary)}
    <ul class="import-errors">${report.errors.slice(0, 12)
      .map((error) => `<li><b>row ${error.row}</b> — ${esc(error.message)}</li>`).join('')}</ul>
    ${report.errors.length > 12 ? `<span class="muted small">…and ${report.errors.length - 12} more.</span>` : ''}
  </div></div>`;
}

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

/* Changes are staged, not sent. Switching off a dozen rules used to be a dozen
   POSTs, each one immediately live, with no moment to look at the set before it
   took effect - and policy is exactly the thing worth reading back before it
   starts deciding what posts to the ledger. */

/** The policy as the screen currently shows it: stored, with pending edits over it. */
function draftPolicy() {
  return { ...state.settings.policy, ...state.policyDraft };
}

/** The rule list as the screen shows it, with pending toggles and re-grades. */
function draftRules() {
  const off = state.policyDraft.disabled_rules;
  const severity = state.policyDraft.rule_severity;
  return state.settings.rules.map((rule) => ({
    ...rule,
    enabled: off ? !off.includes(rule.rule_id) : rule.enabled,
    severity: severity?.[rule.rule_id] ?? (severity ? rule.default_severity : rule.severity),
  }));
}

function stage(patch) {
  state.policyDraft = { ...state.policyDraft, ...patch };
  renderSettings();
}

function describePending() {
  const stored = state.settings.policy;
  const draft = state.policyDraft;
  const changes = [];
  if (draft.auto_post_min_confidence !== undefined
      && draft.auto_post_min_confidence !== stored.auto_post_min_confidence) {
    changes.push(`auto-post threshold ${pct(stored.auto_post_min_confidence)} → ${pct(draft.auto_post_min_confidence)}`);
  }
  if (draft.rules_free_mode !== undefined && draft.rules_free_mode !== stored.rules_free_mode) {
    changes.push(draft.rules_free_mode ? 'free mode on — no rule will run' : 'free mode off');
  }
  if (draft.builtin_rules_disabled !== undefined
      && draft.builtin_rules_disabled !== stored.builtin_rules_disabled) {
    changes.push(draft.builtin_rules_disabled
      ? 'built-in rules off — only your own rules will run'
      : 'built-in rules back on');
  }
  if (draft.disabled_rules) {
    const before = new Set(state.settings.rules.filter((rule) => !rule.enabled).map((rule) => rule.rule_id));
    const after = new Set(draft.disabled_rules);
    const off = [...after].filter((id) => !before.has(id)).length;
    const on = [...before].filter((id) => !after.has(id)).length;
    if (off) changes.push(`${off} rule(s) switched off`);
    if (on) changes.push(`${on} rule(s) switched back on`);
  }
  if (draft.rule_severity) {
    const regraded = Object.keys(draft.rule_severity).length;
    const stored_count = state.settings.rules.filter((rule) => rule.regraded).length;
    if (regraded !== stored_count) changes.push(`${regraded} rule(s) re-graded`);
  }
  return changes;
}

/** Commit every staged change in one request. */
async function commitSettings() {
  const button = $('#policy-save');
  button.disabled = true;
  const held = await heldCount();
  try {
    state.settings = await api('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(state.policyDraft),
    });
  } catch (error) {
    toast(error.message, 'error');
    button.disabled = false;
    return;
  }
  state.policyDraft = {};
  button.disabled = false;
  renderSettings();
  loadStatus();                // the header threshold chip reads from status
  toast('Policy saved.', 'success');

  // The documents already in the queue were judged by the policy that just
  // changed. Saying so here is the difference between a setting and an effect.
  $('#policy-followup').innerHTML = held
    ? `<div class="notice"><span>↻</span><div>${held} document(s) in the queue were judged under the
         previous policy. <button class="btn small" id="policy-recheck">Re-check them now</button></div></div>`
    : '';
}

async function heldCount() {
  try {
    return (await api('/api/documents?decision=needs_review&per_page=1')).total;
  } catch { return 0; }
}

function renderSettings() {
  const policy = draftPolicy();
  const rules = draftRules();
  const pending = describePending();

  $('#policy-save-bar').hidden = pending.length === 0;
  $('#policy-pending').textContent = pending.length
    ? `Not saved yet: ${pending.join(' · ')}.` : '';

  $('#policy-warnings').innerHTML = (policy.warnings || [])
    .map((text) => `<div class="notice"><span>⚠</span><div>${esc(text)}</div></div>`).join('');

  const slider = $('#threshold-slider');
  slider.value = Math.round(policy.auto_post_min_confidence * 100);
  $('#threshold-out').textContent = pct(policy.auto_post_min_confidence);
  $('#threshold-note').textContent =
    `Fields below ${pct(policy.field_review_confidence)} are flagged for review individually.`;

  $('#free-mode-toggle').checked = policy.rules_free_mode;
  $('#builtin-off-toggle').checked = policy.builtin_rules_disabled;
  // Free mode already covers this one; leaving it live would invite someone to
  // toggle a setting that changes nothing.
  $('#builtin-off-toggle').disabled = policy.rules_free_mode;
  // `overridden` describes what is stored, so it is read from the saved policy
  // rather than the draft: a change nobody has saved has not overridden anything.
  const overridden = state.settings.policy.overridden || [];
  $('#settings-origin').textContent = overridden.length
    ? `${overridden.length} setting(s) changed from .env`
    : 'Matching the configured defaults';
  $('#settings-reset').disabled = overridden.length === 0;

  const docTypes = [...new Set(rules.flatMap((r) => r.doc_types))].filter((t) => t !== '*').sort();
  const filter = $('#rule-doc-filter');
  if (filter.options.length <= 1) {
    filter.innerHTML = '<option value="">All document types</option>' +
      docTypes.map((t) => `<option value="${esc(t)}">${esc(t.replace('_', ' '))}</option>`).join('');
  }
  renderRuleList();
}

function renderRuleList() {
  const policy = draftPolicy();
  const rules = draftRules();
  const term = $('#rule-search').value.trim().toLowerCase();
  const docType = $('#rule-doc-filter').value;

  const visible = rules.filter((rule) => {
    if (docType && !rule.applies_to_all && !rule.doc_types.includes(docType)) return false;
    if (!term) return true;
    return `${rule.rule_id} ${rule.summary}`.toLowerCase().includes(term);
  });

  // Free mode already turns everything off; leaving the boxes live would invite
  // someone to toggle a rule that cannot run either way.
  const frozen = policy.rules_free_mode || policy.builtin_rules_disabled;
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
        <select class="sev" data-severity="${esc(rule.rule_id)}" ${frozen || !rule.enabled ? 'disabled' : ''}
          title="How a failure of this rule is reported">
          ${['blocking', 'warning', 'info'].map((level) => `
            <option value="${level}" ${rule.severity === level ? 'selected' : ''}>${level}</option>`).join('')}
        </select>
      </label>`).join('')
    : '<p class="muted">No rule matches that filter.</p>';
}

$('#threshold-slider').addEventListener('input', (event) => {
  $('#threshold-out').textContent = `${event.target.value}%`;
});
$('#threshold-slider').addEventListener('change', (event) => {
  stage({ auto_post_min_confidence: Number(event.target.value) / 100 });
});
$('#free-mode-toggle').addEventListener('change', (event) => {
  stage({ rules_free_mode: event.target.checked });
});
$('#builtin-off-toggle').addEventListener('change', (event) => {
  stage({ builtin_rules_disabled: event.target.checked });
});
$('#rule-search').addEventListener('input', renderRuleList);
$('#rule-doc-filter').addEventListener('change', renderRuleList);

$('#policy-save').addEventListener('click', commitSettings);

/* Staged policy lives only on this page. Closing the tab with changes on screen
   should not silently discard them. */
window.addEventListener('beforeunload', (event) => {
  if (Object.keys(state.policyDraft).length) event.preventDefault();
});
$('#policy-discard').addEventListener('click', () => {
  state.policyDraft = {};
  $('#policy-followup').innerHTML = '';
  renderSettings();
});

/* Offered after a save, because the queue was filled by the policy that just
   changed. It re-judges without re-reading anything. */
$('#policy-followup').addEventListener('click', async (event) => {
  if (!event.target.closest('#policy-recheck')) return;
  const button = $('#policy-recheck');
  button.disabled = true;
  button.textContent = 'Re-checking…';
  try {
    const result = await api('/api/documents/reevaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ decision: 'needs_review' }),
    });
    toast(result.summary, result.changed ? 'success' : '');
    $('#policy-followup').innerHTML = '';
    loadStatus();
  } catch (error) {
    toast(error.message, 'error');
    button.disabled = false;
    button.textContent = 'Re-check them now';
  }
});

$('#settings-reset').addEventListener('click', async () => {
  try {
    state.settings = await api('/api/settings/reset', { method: 'POST' });
  } catch (error) {
    toast(error.message, 'error');
    return;
  }
  state.policyDraft = {};   // the stored policy just changed under any draft
  $('#policy-followup').innerHTML = '';
  renderSettings();
  loadStatus();
  toast('Back to the configured defaults.');
});

$('#wipe-documents').addEventListener('click', async () => {
  // Asked fresh, not read from the copy of /api/status taken at page load. That
  // copy goes stale the moment anything is processed, and gating a button on it
  // meant the button did nothing at all while the page still showed data.
  let stats;
  try { stats = (await api('/api/status')).stats; }
  catch (error) { toast(error.message, 'error'); return; }
  const total = stats.total_documents ?? 0;
  if (!total) { toast('There is nothing to remove.'); return; }

  // Naming the numbers, because "are you sure?" is a question nobody reads.
  const held = stats.by_decision?.needs_review ?? 0;
  if (!confirm(`Delete all ${total} document(s) and ${stats.posted_records ?? 0} ledger entr` +
      `${stats.posted_records === 1 ? 'y' : 'ies'}?\n\n` +
      `${held} of them are still waiting for review. Page images and archived copies go too. ` +
      `Master data and the audit log are kept.\n\nThis cannot be undone.`)) return;

  const button = $('#wipe-documents');
  button.disabled = true;
  button.textContent = 'Deleting…';
  let result;
  try {
    result = await api('/api/documents/wipe?confirm=true', { method: 'POST' });
  } catch (error) {
    toast(error.message, 'error');
    return;
  } finally {
    button.disabled = false;
    button.textContent = 'Delete all documents & records';
  }
  $('#wipe-summary').textContent = result.message;
  loadStatus();
  toast(result.message);
});

$('#rule-list').addEventListener('change', (event) => {
  const shown = draftRules();

  // Re-grading keeps a check running while stopping it from blocking. Switching
  // a rule off to get past it would lose the warning as well.
  const picker = event.target.closest('select[data-severity]');
  if (picker) {
    // The API replaces the whole map, so it carries every rule not at its default.
    stage({
      rule_severity: Object.fromEntries(
        shown
          .map((rule) => [rule.rule_id,
            rule.rule_id === picker.dataset.severity ? picker.value : rule.severity])
          .filter(([id, level]) => level !== shown.find((r) => r.rule_id === id).default_severity)),
    });
    return;
  }

  const box = event.target.closest('input[data-rule]');
  if (!box) return;
  // Likewise for the disabled set: it is replaced, not merged.
  stage({
    disabled_rules: shown
      .filter((rule) => (rule.rule_id === box.dataset.rule ? !box.checked : !rule.enabled))
      .map((rule) => rule.rule_id),
  });
});

/* ------------------------------------------------------------ job queue */

const JOB_ICON = { queued: '○', running: '◐', done: '✓', failed: '✕' };
let jobTimer = null;

function startJobPolling() {
  $('#job-card').classList.remove('hidden');
  refreshJobs();
  if (jobTimer) return;
  jobTimer = setInterval(refreshJobs, 2000);
}

async function refreshJobs() {
  let payload;
  try {
    payload = await api('/api/jobs');
  } catch (error) {
    clearInterval(jobTimer); jobTimer = null;
    toast(`Lost track of the queue: ${error.message}`, 'error');
    return;
  }

  const { jobs, summary } = payload;
  $('#job-active').textContent = summary.active;
  $('#job-workers').textContent =
    `${summary.workers} worker${summary.workers === 1 ? '' : 's'} · ${summary.done} done`
    + (summary.failed ? ` · ${summary.failed} failed` : '');

  $('#job-list').innerHTML = jobs.length
    ? jobs.map((job) => `
      <div class="job-row ${job.status}">
        <span class="icon">${JOB_ICON[job.status] || '·'}</span>
        <span>
          <span class="name">${esc(job.filename)}</span>
          <span class="sub">${esc(jobDetail(job))}</span>
        </span>
        <span class="conf ${job.confidence != null ? confClass(job.confidence) : ''}">
          ${job.confidence != null ? pct(job.confidence) : ''}</span>
        ${job.document_id
          ? `<button class="btn small" data-open-job="${esc(job.document_id)}">Open</button>`
          : '<span></span>'}
      </div>`).join('')
    : '<p class="muted">Nothing queued.</p>';

  await openWhenReady(jobs);

  // Stop polling once the queue drains; the list stays until it is cleared.
  if (summary.active === 0 && jobTimer) {
    clearInterval(jobTimer); jobTimer = null;
    loadStatus();
  }
}

/* One document submitted on its own opens by itself when it finishes — the
   thing a single upload used to do by holding the request open. Dropping a
   second file while the first is reading replaces the target, because the one
   you just dropped is the one you are waiting to see. */
async function openWhenReady(jobs) {
  const job = state.autoOpen && jobs.find((row) => row.job_id === state.autoOpen);
  if (!job || job.status === 'queued' || job.status === 'running') return;
  state.autoOpen = null;

  if (job.status === 'failed' || !job.document_id) {
    $('#result-host').innerHTML = `<div class="card"><h3>Could not process this document</h3>
      <p class="muted" style="margin-top:6px">${esc(job.error || 'Processing failed.')}</p></div>`;
    return;
  }
  try {
    renderResult(await api(`/api/documents/${encodeURIComponent(job.document_id)}`),
      $('#result-host'), false);
  } catch (error) {
    toast(error.message, 'error');
    return;
  }
  toast(`${DECISION[job.decision]?.label || job.decision} — ${pct(job.confidence)} confidence`,
    job.decision === 'auto_post' ? 'success' : job.decision === 'rejected' ? 'error' : '');
}

function jobDetail(job) {
  if (job.status === 'failed') return job.error || 'failed';
  if (job.status === 'running') return job.detail || job.stage || 'processing';
  if (job.status === 'done') {
    const label = DECISION[job.decision]?.label || job.decision || 'processed';
    return job.record_id ? `${label} · ${job.record_id}` : label;
  }
  return 'waiting';
}

$('#job-clear').addEventListener('click', async () => {
  try { await api('/api/jobs/clear', { method: 'POST' }); }
  catch (error) { toast(error.message, 'error'); return; }
  refreshJobs();
});

$('#job-list').addEventListener('click', async (event) => {
  const button = event.target.closest('button[data-open-job]');
  if (!button) return;
  try {
    const data = await api(`/api/documents/${encodeURIComponent(button.dataset.openJob)}`);
    renderResult(data, $('#result-host'), false);
    $('#result-host').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (error) {
    toast(error.message, 'error');
  }
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
          ${rule.kind === 'agent' ? '<span class="chip agent">agent</span>' : ''}
          <span class="desc">${esc(rule.explanation)}</span>
          ${rule.prompt ? `<span class="said">you wrote: “${esc(rule.prompt)}”</span>` : ''}
          <span class="types">${rule.doc_types.map((t) => `<span class="chip">${esc(t.replace('_', ' '))}</span>`).join('')}</span>
        </span>
        <button class="btn small danger" data-delete="${esc(rule.rule_id)}">Delete</button>
      </div>`).join('')
    : '<p class="muted">No rules of your own yet. Describe one above.</p>';
}

const selectedDocTypes = () => [...$('#rule-types').selectedOptions].map((o) => o.value);

// How the draft describes itself: an expression shows the condition it compiled
// to, an agent shows the plan it will follow. Both are the thing to check before
// saving, and neither is readable as the other.
function renderDraftMachinery(rule) {
  if (rule.kind !== 'agent') {
    return `<details><summary class="muted" style="cursor:pointer;font-size:12.5px">Show the compiled condition</summary>
      <pre>${esc(JSON.stringify({ when: rule.when, assert: rule['assert'] }, null, 2))}</pre></details>`;
  }
  return `<div class="agent-plan">
    <p class="lbl">The agent will work through</p>
    <ol>${(rule.plan || []).map((step) => `<li>${esc(step)}</li>`).join('')}</ol>
    <p class="lbl">Using</p>
    <p>${(rule.tools || []).length
      ? rule.tools.map((name) => `<span class="chip">${esc(name)}</span>`).join(' ')
      : '<span class="muted">every tool available</span>'}</p>
    <p class="muted">It runs a model on each document this rule applies to. If it cannot
      reach an answer the document is held for a person — never passed.</p>
  </div>`;
}

function renderDraft(draft) {
  state.draft = draft;
  if (!draft) { $('#rule-draft').innerHTML = ''; return; }

  const { rule, preview } = draft;
  const unsure = preview.unsure || 0;
  const evaluated = preview.passed + preview.failed + unsure;
  // A rule nothing passes, or nothing fails, is the shape of an inverted
  // condition — worth showing loudly rather than burying in the counts. For an
  // agent, deciding nothing at all is the equivalent shape.
  const suspect = evaluated > 0 && (preview.passed === 0 || preview.failed + unsure === 0);

  $('#rule-draft').innerHTML = `
    <div class="draft">
      <h4>${esc(rule.title)} <span class="chip ${rule.severity === 'blocking' ? 'bad' : 'warn'}">${esc(rule.severity)}</span>
        ${rule.kind === 'agent' ? '<span class="chip agent">agent</span>' : ''}</h4>
      <div class="reads">${esc(rule.explanation)}</div>
      <p class="muted">Reviewers will see: “${esc(rule.message)}”</p>
      <div class="tally">
        <span>Tried on <b>${preview.considered}</b> document(s)</span>
        <span>would pass <b class="hi">${preview.passed}</b></span>
        <span>would fail <b class="lo">${preview.failed}</b></span>
        ${unsure ? `<span>undecided <b class="lo">${unsure}</b></span>` : ''}
        <span>not applicable <b>${preview.not_applicable}</b></span>
      </div>
      <p class="${suspect ? '' : 'muted'}" style="font-size:13px">${suspect ? '⚠ ' : ''}${esc(preview.note)}</p>
      ${preview.examples.length ? `<pre>${esc(preview.examples.map((e) =>
        `${e.outcome.toUpperCase().padEnd(7)} ${e.filename}  ${e.reason || JSON.stringify(e.observed || {})}`).join('\n'))}</pre>` : ''}
      ${renderDraftMachinery(rule)}
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
      body: JSON.stringify({
        prompt: $('#rule-prompt').value,
        doc_types: docTypes,
        mode: $('#rule-mode').value,
      }),
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
