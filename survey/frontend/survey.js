/* No analytics, external scripts, model calls, or document-wide key capture. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const clone = x => JSON.parse(JSON.stringify(x));
  const uuid = () => crypto.randomUUID();
  const terminal = s => ['submitted', 'ended'].includes(s.status);
  let schema, state, args, initialized = false, revision = 0, seq = 0;
  let events = [], inflight = null, saveQueue = [], cacheKey, pausedView = false;
  let clientId = uuid(), lastSent = 0, lastSaved = '', storageAvailable = true;
  let saveError = '', lastAck = '', activeVideo = null;
  let lastConsoleStatus = '';
  let lastFrameHeight = 0;
  // Two hosts: inside Streamlit (an iframe component), or a standalone web page
  // (html-survey/) that calls window.startSurvey() and saves through its own API.
  let standalone = null, requestOpen = false;

  function bridge(type, extra = {}) {
    if (standalone) return;
    window.parent.postMessage({isStreamlitMessage: true, type, ...extra}, '*');
  }
  function resize() {
    if (standalone) return;
    let height = window.innerHeight;
    // The host CSS supplies a viewport-sized frame. When same-origin access is
    // available, also follow the visual viewport (including an on-screen keyboard).
    try {
      const frame = window.frameElement, viewport = window.parent.visualViewport;
      if (frame && viewport) {
        height = Math.max(160, Math.floor(viewport.height + viewport.offsetTop - frame.getBoundingClientRect().top - 8));
        frame.style.setProperty('height', height + 'px', 'important');
      }
    } catch (_) { /* Cross-origin hosts retain the CSS-sized frame. */ }
    if (height !== lastFrameHeight) {
      lastFrameHeight = height;
      bridge('streamlit:setFrameHeight', {height});
    }
  }
  function cache() {
    try {
      sessionStorage.setItem(cacheKey, JSON.stringify({state, events, inflight, saveQueue, revision, seq, clientId, lastSaved}));
    } catch (_) { storageAvailable = false; }
  }
  function record(type, field, details = {}, event) {
    // Only called by survey controls. Resume codes and other applications are excluded.
    const entry = {event_id: uuid(), client_id: clientId, sequence: ++seq,
      client_utc: new Date().toISOString(), elapsed_ms: performance.now(), time_origin_ms: performance.timeOrigin,
      browser_event_ms: event ? event.timeStamp : null, page_id: state.page,
      field_id: field || state.page, type, ...details};
    events.push(entry);
    cache();
    setStatus();
  }
  function setStatus() {
    const el = $('save-status');
    const saving = inflight || saveQueue.length;
    const pending = events.length || saving;
    const warning = !!saveError || (!storageAvailable && !!pending);
    el.classList.toggle('hidden', !warning);
    el.classList.toggle('warning', warning);
    el.textContent = saveError ? 'Your latest changes have not been saved yet. Please keep this tab open while we try again.'
      : warning ? 'Please choose Next or Save and take a break before closing this tab. Your changes are only held in memory until saved.' : '';
    const diagnostic = saveError ? 'Save failed; retry pending.' : saving ? 'Saving requested survey changes.' : events.length ? 'Changes buffered until navigation or save.' : 'Survey changes saved.';
    if (diagnostic !== lastConsoleStatus) {
      console.debug('[Survey]', diagnostic); lastConsoleStatus = diagnostic;
    }
  }
  function requestSave() {
    if (!initialized) return;
    // Freeze exactly what the participant requested to save. Edits on the next
    // question must not be swept into this save when an earlier batch completes.
    saveQueue.push({batch_id: uuid(), state: clone(state), events: events.splice(0)});
    cache();
    if (!inflight) sendPending();
    setStatus();
  }
  function sendPending() {
    if (!initialized) return;
    if (!inflight) {
      if (!saveQueue.length) return;
      inflight = {...saveQueue.shift(), base_revision: revision};
      cache();
    }
    lastSent = Date.now();
    if (standalone) sendStandalone({...inflight, attempt: uuid()});
    else bridge('streamlit:setComponentValue', {value: {...inflight, attempt: uuid()}, dataType: 'json'});
    setStatus();
  }
  function sendStandalone(packet) {
    // One request at a time; the retry timer resends the identical batch, and the
    // server returns the already-saved record when it has seen this batch ID.
    if (requestOpen) return;
    requestOpen = true;
    standalone.save(packet).then(saved => {
      requestOpen = false;
      receive({...args, ack: packet.batch_id, revision: saved.revision, saved_at: saved.saved_at, error: ''});
    }, error => {
      requestOpen = false;
      receive({...args, error: (error && error.message) || 'The storage service could not be reached.'});
    });
  }
  function answer(id) { return state.answers[id]; }
  // The choice that has an optional text field: "Other" unless a question names another.
  function otherOf(p) { return p.other_option || 'Other'; }
  function condition(c) {
    if (!c) return true;
    if (c.all) return c.all.every(condition);
    if (c.any) return c.any.some(condition);
    if (c.answered) return answer(c.answered)?.status === 'answered';
    return (answer(c.question)?.choices || []).some(v => c.values.includes(v));
  }
  function questionItems(p) {
    return p.rating_group ? schema.pages.filter(item => item.rating_group === p.rating_group) : [p];
  }
  function visiblePages() {
    return schema.pages.filter(p => condition(p.when) && questionItems(p)[0].id === p.id);
  }
  function currentPage() {
    // Sessions saved on a removed screen resume on its replacement.
    const id = schema.pages.some(p => p.id === state.page) ? state.page : schema.retired_pages?.[state.page];
    const p = schema.pages.find(p => p.id === id);
    // Old saved sessions can resume on any item in a now-combined question.
    return p && questionItems(p)[0];
  }
  function prune() {
    // Document order is parent-before-child, so deleting a hidden parent also hides its descendants.
    for (const p of schema.pages) {
      if (!condition(p.when) && state.answers[p.id]) {
        const previous = state.answers[p.id];
        delete state.answers[p.id];
        record('answer_invalidated', p.id, {source: 'branch_change', previous});
      }
    }
  }
  function button(text, fn, cls = '') {
    const b = document.createElement('button'); b.type = 'button'; b.textContent = text;
    b.className = cls; b.addEventListener('click', fn); return b;
  }
  function text(tag, content, cls = '') {
    const el = document.createElement(tag); el.textContent = content; el.className = cls; return el;
  }
  function focusHeading() {
    requestAnimationFrame(() => { const h = $('question-title'); h?.focus({preventScroll: true}); $('page').scrollTop = 0; resize(); });
  }
  function go(id, type = 'next') {
    record('navigation', state.page, {action: type, destination: id});
    state.page = id; cache(); render();
    record('page_view', id); requestSave(); focusHeading();
  }
  function next() {
    const list = visiblePages(), i = list.findIndex(p => p.id === currentPage()?.id);
    if (i >= 0 && i < list.length - 1) go(list[i + 1].id);
  }
  function skip() {
    for (const p of questionItems(currentPage())) {
      const previous = answer(p.id) || null;
      state.answers[p.id] = {status: 'skipped'};
      record('skip', p.id, {previous});
    }
    prune(); next();
  }
  function choicesChanged(p, group, value, checked, event, source = 'participant') {
    const old = clone(answer(p.id) || {choices: [], other: {}, groups: {}});
    const a = clone(old); a.status = 'answered';
    if (group) {
      a.groups ||= {}; a.groups[group] = value;
    } else if (p.kind === 'multi') {
      let selections = [...(a.choices || [])];
      if (checked && (p.exclusive || []).includes(value)) selections = [];
      else if (checked) selections = selections.filter(v => !(p.exclusive || []).includes(v));
      a.choices = checked ? [...new Set([...selections, value])] : selections.filter(v => v !== value);
      if (!a.choices.length) a.status = 'unanswered';
    } else a.choices = [value];
    a.other ||= {};
    state.answers[p.id] = a;
    const before = group ? [old.groups?.[group]].filter(Boolean) : old.choices || [];
    const after = group ? [value] : a.choices;
    for (const v of before.filter(v => !after.includes(v)))
      record('option_deselected', group ? `${p.id}.${group}` : p.id,
        {option: v, source: v === value ? 'participant' : 'exclusive_choice'}, event);
    for (const v of after.filter(v => !before.includes(v)))
      record('option_selected', group ? `${p.id}.${group}` : p.id, {option: v, source}, event);
    // Other explanations are final answers only while Other is selected; edits remain in the event log.
    const otherKey = group || 'other';
    if (!after.includes(otherOf(p)) && a.other[otherKey]) {
      record('other_cleared', `${p.id}.${otherKey}`, {previous_text: a.other[otherKey], source: 'option_change'});
      delete a.other[otherKey];
      const ta = $('text-' + p.id + '.' + otherKey);
      if (ta) { ta.value = ''; ta.dispatchEvent(new Event('survey-text-reset')); }
    }
    prune(); cache();
    // Update controls in place; do not destroy keyboard focus on the chosen option.
    document.querySelectorAll('input[data-question]').forEach(input => {
      if (input.dataset.question !== p.id) return;
      input.checked = input.dataset.group ? a.groups?.[input.dataset.group] === input.value : (a.choices || []).includes(input.value);
    });
    updateOther(p, group); setStatus(); resize();
    if (p.rating_group) document.querySelector('#label-' + p.id)?.parentElement.querySelector('.legacy-rating')?.remove();
  }
  function delta(before, after) {
    let start = 0;
    while (start < before.length && start < after.length && before[start] === after[start]) start++;
    let endBefore = before.length, endAfter = after.length;
    while (endBefore > start && endAfter > start && before[endBefore - 1] === after[endAfter - 1]) {endBefore--; endAfter--;}
    return {start_utf16: start, deleted: before.slice(start, endBefore), inserted: after.slice(start, endAfter)};
  }
  function textarea(p, field, initial, onValue, label) {
    const wrap = document.createElement('div'); wrap.className = 'other-wrap';
    const id = 'text-' + field;
    const l = text('label', label); l.htmlFor = id; wrap.append(l);
    const ta = document.createElement('textarea'); ta.id = id; ta.value = initial || '';
    ta.maxLength = 10000; ta.setAttribute('aria-describedby', 'typing-help-' + field);
    const help = text('p', 'Optional. A few words are enough.', 'help'); help.id = 'typing-help-' + field;
    let previous = ta.value;
    ta.addEventListener('survey-text-reset', () => { previous = ta.value; });
    for (const kind of ['keydown', 'keyup']) ta.addEventListener(kind, e => {
      record(kind, field, {key: e.key, code: e.code, repeat: e.repeat, is_composing: e.isComposing,
        ctrl: e.ctrlKey, alt: e.altKey, shift: e.shiftKey, meta: e.metaKey,
        selection_start: ta.selectionStart, selection_end: ta.selectionEnd}, e);
    });
    ta.addEventListener('beforeinput', e => record('before_input', field, {
      input_type: e.inputType || null, data: e.data, is_composing: e.isComposing,
      selection_start: ta.selectionStart, selection_end: ta.selectionEnd}, e));
    ta.addEventListener('input', e => {
      const change = delta(previous, ta.value); previous = ta.value;
      onValue(ta.value, e);
      record('text_input', field, {...change, input_type: e.inputType || 'unknown', is_composing: e.isComposing || false,
        value_after: ta.value, selection_start: ta.selectionStart, selection_end: ta.selectionEnd}, e);
    });
    for (const kind of ['compositionstart', 'compositionend']) ta.addEventListener(kind, e => record(kind, field, {data: e.data}, e));
    for (const kind of ['focus', 'blur']) ta.addEventListener(kind, e => record(kind, field, {}, e));
    wrap.append(ta, help); return wrap;
  }
  function updateOther(p, group) {
    const key = group || 'other', host = $('other-' + p.id + '-' + key);
    if (!host) return;
    const a = answer(p.id);
    if (host.firstChild) return;
    const field = textarea(p, `${p.id}.${key}`, a?.other?.[key], (value, event) => {
      const current = answer(p.id);
      const selected = group ? current?.groups?.[group] === otherOf(p) : current?.choices?.includes(otherOf(p));
      if (!selected && !value.trim()) return;
      if (!selected && value.trim()) choicesChanged(p, group, otherOf(p), true, event, 'other_text');
      if (!state.answers[p.id]) return;
      state.answers[p.id].other ||= {}; state.answers[p.id].other[key] = value;
    }, 'Other details (optional)');
    field.querySelector('label').className = 'other-text-label';
    field.querySelector('.help').remove();
    const input = field.querySelector('textarea');
    input.removeAttribute('aria-describedby');
    input.rows = 2; input.placeholder = 'Please specify';
    host.append(field);
  }
  function optionList(p, options, group) {
    const container = document.createElement('div');
    container.className = 'answer-options';
    const list = document.createElement('div'); list.className = 'options' + (options.every(v => v.length < 27) ? ' short' : '');
    if (!group) list.setAttribute('aria-labelledby', 'question-title');
    const otherRow = document.createElement('div'); otherRow.className = 'other-row';
    const choice = v => {
      const label = document.createElement('label'); label.className = 'choice';
      const input = document.createElement('input'); input.type = p.kind === 'multi' ? 'checkbox' : 'radio';
      input.name = group ? p.id + '-' + group : p.id; input.value = v;
      input.dataset.question = p.id; if (group) input.dataset.group = group;
      input.checked = group ? answer(p.id)?.groups?.[group] === v : !!answer(p.id)?.choices?.includes(v);
      input.addEventListener('change', e => choicesChanged(p, group, v, input.checked, e));
      label.append(input, text('span', v));
      if (v === otherOf(p)) {
        input.setAttribute('aria-label', v);
        const optional = text('small', '(optional details)');
        optional.setAttribute('aria-hidden', 'true'); label.append(optional);
      }
      return label;
    };
    if (p.option_groups && !group) {
      // One line per group of related actions (schema option_groups).
      const groups = document.createElement('div'); groups.className = 'option-groups';
      groups.setAttribute('role', 'radiogroup'); groups.setAttribute('aria-labelledby', 'question-title');
      p.option_groups.forEach((g, i) => {
        const shown = g.options.filter(v => v !== otherOf(p) && options.includes(v));
        if (!shown.length) return;
        const row = document.createElement('div'); row.className = 'option-group' + (g.label ? '' : ' unlabeled');
        if (g.label) {
          const heading = text('p', g.label, 'option-group-label'); heading.id = `group-${p.id}-${i}`;
          row.setAttribute('role', 'group'); row.setAttribute('aria-labelledby', heading.id); row.append(heading);
        }
        const rowList = document.createElement('div'); rowList.className = 'options';
        shown.forEach(v => rowList.append(choice(v)));
        row.append(rowList); groups.append(row);
      });
      container.append(groups);
      // The unlabeled last line holds "Not sure" and Other together.
      const last = groups.querySelector('.option-group.unlabeled .options');
      if (last && options.includes(otherOf(p))) { otherRow.classList.add('in-group'); last.append(otherRow); }
    } else {
      options.filter(v => v !== otherOf(p)).forEach(v => list.append(choice(v)));
      container.append(list);
    }
    if (options.includes(otherOf(p))) {
      otherRow.append(choice(otherOf(p)));
      const other = document.createElement('div'); other.id = 'other-' + p.id + '-' + (group || 'other');
      otherRow.append(other); if (!otherRow.parentNode) container.append(otherRow);
    }
    return container;
  }
  // ---- Word candidates on the post-demonstration edit screens ----
  // Words are compared by their letters; surrounding punctuation stays in place.
  function wordTokens(value) {
    // Letters/digits with internal apostrophes, so dashes and punctuation separate words.
    return [...value.matchAll(/[\p{L}\p{N}]+(?:[’'][\p{L}\p{N}]+)*/gu)]
      .map(m => ({start: m.index, end: m.index + m[0].length, word: m[0]}));
  }
  function similarity(a, b) {
    // Dice coefficient on letter pairs, 0 (unrelated) to 1 (same).
    const pairs = w => { const out = []; for (let i = 0; i < w.length - 1; i++) out.push(w.slice(i, i + 2)); return out; };
    const x = pairs(a.toLowerCase()), y = pairs(b.toLowerCase());
    if (!x.length || !y.length) return a.toLowerCase() === b.toLowerCase() ? 1 : 0;
    let shared = 0; const rest = [...y];
    for (const pair of x) { const i = rest.indexOf(pair); if (i >= 0) { shared++; rest.splice(i, 1); } }
    return 2 * shared / (x.length + y.length);
  }
  function alignSlots(words, entries) {
    // Weighted word-level edit distance. Each current word maps to the original word
    // it matches or replaced; inserted words map to -1 and get no candidates. A
    // replacement is cheap when the new word is one of that slot's candidates or is
    // spelled similarly, so deletions elsewhere do not shift words to the wrong slot.
    const n = words.length, m = entries.length;
    const sub = (w, e) => w.toLowerCase() === e.word.toLowerCase() ? 0
      : e.options.some(o => o.toLowerCase() === w.toLowerCase()) ? 0.2 : 1 - 0.5 * similarity(w, e.word);
    const d = Array.from({length: n + 1}, (_, i) => Array.from({length: m + 1}, (_, j) => i + j));
    for (let i = 1; i <= n; i++) for (let j = 1; j <= m; j++)
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + sub(words[i - 1], entries[j - 1]));
    const slot = new Array(n).fill(-1), close = (a, b) => Math.abs(a - b) < 1e-9;
    for (let i = n, j = m; i > 0 && j > 0;) {
      if (close(d[i][j], d[i - 1][j - 1] + sub(words[i - 1], entries[j - 1]))) { slot[i - 1] = j - 1; i--; j--; }
      else if (close(d[i][j], d[i - 1][j] + 1)) i--;
      else j--;
    }
    return slot;
  }
  function candidateOptions(entry, word) {
    // Same six positions; a chosen candidate is swapped for the original word.
    const list = [...entry.options];
    if (word.toLowerCase() !== entry.word.toLowerCase()) {
      const i = list.findIndex(o => o.toLowerCase() === word.toLowerCase());
      if (i >= 0) list[i] = entry.word;
    }
    return list;
  }
  function wordRect(ta, start, end) {
    // Measure a word inside the text box with an invisible copy of its text layout.
    const cs = getComputedStyle(ta), mirror = document.createElement('div');
    for (const k of ['fontFamily', 'fontSize', 'fontWeight', 'fontStyle', 'fontVariant', 'letterSpacing', 'wordSpacing',
      'lineHeight', 'textTransform', 'textIndent', 'tabSize', 'paddingTop', 'paddingLeft', 'paddingRight']) mirror.style[k] = cs[k];
    Object.assign(mirror.style, {position: 'absolute', visibility: 'hidden', top: '0', left: '0', whiteSpace: 'pre-wrap',
      overflowWrap: 'break-word', boxSizing: 'content-box', border: '0',
      width: (ta.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight)) + 'px'});
    const span = document.createElement('span'); span.textContent = ta.value.slice(start, end);
    mirror.append(ta.value.slice(0, start), span, ta.value.slice(end));
    document.body.append(mirror);
    const rect = {left: ta.offsetLeft + parseFloat(cs.borderLeftWidth) + span.offsetLeft,
      top: ta.offsetTop + parseFloat(cs.borderTopWidth) + span.offsetTop - ta.scrollTop,
      width: span.offsetWidth, height: span.offsetHeight};
    mirror.remove();
    return rect;
  }
  function mountCandidates(p, input, setText) {
    const box = document.createElement('div'); box.className = 'transcript-box';
    input.before(box); box.append(input);
    const highlight = document.createElement('div'); highlight.className = 'word-highlight hidden';
    const rows = ['above', 'below'].map(side => {
      const row = document.createElement('div'); row.className = `candidate-row ${side} hidden`;
      row.setAttribute('role', 'group'); return row;
    });
    box.append(highlight, ...rows);
    let open = null;
    const hide = () => { open = null; highlight.classList.add('hidden'); rows.forEach(r => r.classList.add('hidden')); };
    const close = reason => {
      if (!open) return;
      record('candidates_closed', p.id, {position: open.slot + 1, word: open.word, reason});
      hide();
    };
    const choose = (option, index) => {
      const o = open; if (!o) return;
      record('candidate_selected', p.id, {position: o.slot + 1, from: o.word, to: option, option_index: index, options: o.options});
      hide();
      const before = input.value;
      setText(before.slice(0, o.start) + option + before.slice(o.end), 'candidateSelected');
      input.setSelectionRange(o.start + option.length, o.start + option.length);
    };
    const show = () => {
      if (input.selectionStart !== input.selectionEnd) return close('text_selected');
      const pos = input.selectionStart, toks = wordTokens(input.value);
      const i = toks.findIndex(t => t.start <= pos && pos <= t.end);
      const slot = i < 0 ? -1 : alignSlots(toks.map(t => t.word), p.candidates)[i];
      if (slot < 0) return close('no_candidates');
      const tok = toks[i];
      if (open && open.start === tok.start && open.word === tok.word) return;
      close('other_word');
      const options = candidateOptions(p.candidates[slot], tok.word);
      open = {slot, start: tok.start, end: tok.end, word: tok.word, options};
      const r = wordRect(input, tok.start, tok.end);
      Object.assign(highlight.style, {left: r.left - 4 + 'px', top: r.top - 2 + 'px', width: r.width + 8 + 'px', height: r.height + 4 + 'px'});
      highlight.classList.remove('hidden');
      rows.forEach((row, k) => {
        row.replaceChildren(...options.slice(k * 3, k * 3 + 3).map((option, j) => {
          const b = button(option, () => choose(option, k * 3 + j), 'candidate');
          b.addEventListener('mousedown', e => e.preventDefault()); // keep the text box focused
          return b;
        }));
        row.setAttribute('aria-label', `Suggestions for ${tok.word}`);
        row.classList.remove('hidden');
        row.style.top = (k ? r.top + r.height + 8 : r.top - 8) + 'px';
        row.style.left = Math.max(0, Math.min(box.clientWidth - row.offsetWidth, r.left + r.width / 2 - row.offsetWidth / 2)) + 'px';
      });
      record('candidates_shown', p.id, {position: slot + 1, word: tok.word, options});
      // Keep both suggestion rows inside the scrollable question area.
      const page = $('page'), pr = page.getBoundingClientRect();
      const above = rows[0].getBoundingClientRect(), below = rows[1].getBoundingClientRect();
      if (above.top < pr.top) page.scrollTop -= pr.top - above.top + 4;
      else if (below.bottom > pr.bottom) page.scrollTop += Math.min(below.bottom - pr.bottom + 4, above.top - pr.top);
    };
    input.addEventListener('click', show);
    input.addEventListener('input', () => close('typing'));
    input.addEventListener('scroll', () => close('scroll'));
    input.addEventListener('keydown', e => { if (e.key === 'Escape') close('escape'); });
    new ResizeObserver(() => close('resize')).observe(input);
    return close;
  }
  function render() {
    activeVideo = null;
    const page = $('page-content'), nav = $('navigation'), foot = $('footer');
    page.replaceChildren(); nav.replaceChildren(); foot.replaceChildren();
    $('page').scrollTop = 0;
    const p = currentPage() || schema.pages[0], list = visiblePages();
    $('section').textContent = p.section ? `Part ${p.section} of 4 · ${schema.sections[p.section]}` : 'Welcome';
    const pos = list.findIndex(x => x.id === p.id);
    $('progress').style.width = `${Math.round(100 * pos / Math.max(1, list.length - 1))}%`;
    if (terminal(state)) {
      page.className = 'complete';
      const pending = events.length || inflight || saveQueue.length;
      page.append(text('h1', pending ? 'Saving your survey…' : 'Thank you for sharing your experiences.'));
      page.append(text('p', pending ? 'Please keep this tab open until saving is complete.' : 'Your responses have been saved. You can now close this tab.'));
      setStatus(); resize(); return;
    }
    page.className = '';
    if (pausedView || state.status === 'paused') {
      page.append(text('h1', 'Your survey is paused'), text('p', 'Take as much time as you need. To return later, choose “Return to your survey” and enter your participant ID.'));
      const pending = events.length || inflight || saveQueue.length;
      page.append(text('p', pending ? 'Please wait a moment before closing this tab.' : 'You can now close this tab and return later using your participant ID.'));
      nav.append(button('Continue survey', () => {pausedView = false; state.status = 'active'; record('resume', state.page); render(); requestSave();}, 'primary'));
      setStatus(); resize(); return;
    }
    page.classList.toggle('with-scenario', !!p.scenario);
    page.classList.toggle('with-groups', p.kind === 'group');
    page.classList.toggle('with-ratings', !!p.rating_group);
    page.classList.toggle('with-edit', p.kind === 'edit');
    if (p.group) page.append(text('p', p.group, 'group-label'));
    if (p.scenario) {
      const box = document.createElement('section'); box.className = 'scenario';
      box.append(text('h2', p.scenario.title), text('p', p.scenario.situation));
      for (const [label, value] of [['You meant', p.scenario.meant], ['The device shows', p.scenario.shown]]) {
        const line = document.createElement('div'); line.className = 'sentence';
        line.append(text('strong', label), text('p', value)); box.append(line);
      }
      page.append(box);
    }
    const heading = text('h1', p.title); heading.id = 'question-title'; heading.tabIndex = -1; page.append(heading);
    (p.paragraphs || []).forEach(s => page.append(text('p', s)));
    // Structured text (the introduction): bold headings, paragraphs, bullet lists.
    for (const block of p.blocks || []) {
      const section = document.createElement('section'); section.className = 'text-block' + (block.notice ? ' notice' : '');
      if (block.heading) section.append(text('h2', block.heading));
      (block.paragraphs || []).forEach(s => section.append(text('p', s)));
      if (block.bullets) {
        const list = document.createElement('ul');
        block.bullets.forEach(s => list.append(text('li', s)));
        section.append(list);
      }
      page.append(section);
    }
    if (p.item && !p.rating_group) page.append(text('p', p.item, 'item'));
    if (p.help) page.append(text('p', p.help, 'help'));
    if (p.rating_group) {
      page.append(text('p', 'Choose one rating for each statement.', 'help'));
      for (const item of questionItems(p)) {
        const row = document.createElement('div'); row.className = 'rating-row';
        row.setAttribute('role', 'group'); row.setAttribute('aria-labelledby', 'label-' + item.id);
        const label = text('p', item.item, 'rating-statement'); label.id = 'label-' + item.id;
        const options = optionList(item, item.options);
        options.querySelector('.options').setAttribute('aria-labelledby', label.id);
        row.append(label, options);
        // Preserve historical N/A answers without silently recoding research data.
        if (answer(item.id)?.choices?.includes('N/A')) row.append(text('p', 'Previously answered: N/A. Choose a rating to change it.', 'help legacy-rating'));
        page.append(row);
      }
    } else if (['single', 'multi'].includes(p.kind)) {
      page.append(text('p', p.kind === 'multi' ? 'Choose all that apply.' : 'Choose one.', 'help'));
      let options = p.options;
      if (p.exclude_selected_from) options = options.filter(v => !answer(p.exclude_selected_from)?.choices?.includes(v));
      page.append(optionList(p, options)); updateOther(p);
    } else if (p.kind === 'group') {
      for (const field of p.fields) {
        const fs = document.createElement('fieldset'); fs.append(text('legend', field.title), optionList(p, field.options, field.id));
        page.append(fs); updateOther(p, field.id);
      }
    } else if (p.kind === 'text') {
      page.append(textarea(p, p.id, answer(p.id)?.text, value => {
        state.answers[p.id] = {text: value, status: value.trim() ? 'answered' : 'unanswered'};
      }, 'Your answer'));
    } else if (p.kind === 'video') {
      const host = document.createElement('div'); host.id = 'video-host'; page.append(host); mountVideo();
    } else if (p.kind === 'edit') {
      // Post-demonstration example: the participant may edit the transcript.
      // The shared textarea helper records keys, input deltas, focus and composition.
      const task = document.createElement('section'); task.className = 'edit-task';
      task.append(text('h2', 'Conversation context'), text('p', p.situation, 'context'));
      const meant = document.createElement('div'); meant.className = 'meant';
      meant.append(text('h3', 'What you meant to say'), text('p', p.meant));
      task.append(meant);
      const field = textarea(p, p.id, answer(p.id)?.text ?? p.transcript, value => {
        state.answers[p.id] = {status: 'answered', decision: 'edited', text: value, original: p.transcript, edited: value !== p.transcript};
        task.querySelectorAll('.decisions button').forEach(b => b.setAttribute('aria-pressed', 'false'));
      }, 'Transcript');
      field.className = 'transcript-wrap';
      const label = field.querySelector('label'); label.className = 'transcript-label';
      const help = field.querySelector('.help');
      help.textContent = p.candidates
        ? 'Click a word to see suggestions, or type to change the text. Then choose Next. You don’t need to match the sentence exactly.'
        : 'Change the text the way you would fix it, then choose Next. You don’t need to match the sentence exactly.';
      label.after(help);
      const input = field.querySelector('textarea');
      // Tool buttons change the text; the shared text_input handler logs the change
      // like any other edit. Browsers blank non-standard inputType values on
      // synthetic events, so the source is set directly.
      const setText = (value, inputType) => {
        input.focus(); input.value = value;
        const event = new InputEvent('input', {bubbles: true});
        Object.defineProperty(event, 'inputType', {value: inputType});
        input.dispatchEvent(event);
      };
      const tools = document.createElement('div'); tools.className = 'edit-tools';
      tools.append(button('Delete all', () => {
        if (!input.value) return;
        record('delete_all', p.id, {previous_text: input.value});
        setText('', 'deleteAllButton');
      }), button('Reset', () => {
        if (input.value === p.transcript) return;
        record('transcript_reset', p.id, {previous_text: input.value});
        setText(p.transcript, 'resetButton');
      }));
      // Decisions record what they would do instead of fixing the text, then move on.
      const decisions = document.createElement('div'); decisions.className = 'decisions';
      decisions.setAttribute('role', 'group'); decisions.setAttribute('aria-labelledby', 'decisions-' + p.id);
      const decisionsLabel = text('p', 'Instead of fixing the text:', 'decisions-label'); decisionsLabel.id = 'decisions-' + p.id;
      decisions.append(decisionsLabel);
      for (const [labelText, decision] of [
        ['Keep as is', 'kept'], ['Say it again', 'say_again'], ['Switch to my AAC', 'switch_aac'],
        ['Ask for help', 'ask_help'], ['Abandon', 'abandoned'], ['Not sure', 'not_sure']]) {
        const b = button(labelText, () => {
          state.answers[p.id] = {status: 'answered', decision, text: input.value, original: p.transcript, edited: input.value !== p.transcript};
          record('decision', p.id, {decision, text: input.value});
          next();
        }, 'decision');
        b.setAttribute('aria-pressed', String(answer(p.id)?.decision === decision));
        decisions.append(b);
      }
      // Under the text box: the text tools, then the decisions on their own line.
      field.append(tools, decisions);
      if (p.candidates) {
        const closeCandidates = mountCandidates(p, input, setText);
        task.addEventListener('pointerdown', e => { if (!e.target.closest('.transcript-box')) closeCandidates('outside'); });
      }
      task.append(field); page.append(task);
    }
    if (p.kind === 'edit') nav.append(button('Stop the exercise', () => {
      // Leave the optional exercise: unanswered remaining examples are skipped.
      const list = visiblePages(), edits = list.filter(q => q.kind === 'edit');
      const skipped = [];
      for (const q of edits.slice(edits.findIndex(q => q.id === p.id))) {
        if (answer(q.id)?.status !== 'answered') { state.answers[q.id] = {status: 'skipped', stopped: true}; skipped.push(q.id); }
      }
      record('exercise_stopped', p.id, {skipped});
      const after = list[list.indexOf(edits.at(-1)) + 1];
      if (after) go(after.id);
    }, 'small stop-exercise'));
    if (['single', 'multi', 'group', 'text'].includes(p.kind)) nav.append(button('Clear answer', () => {
      for (const item of questionItems(p)) {
        const previous = answer(item.id) || null; delete state.answers[item.id];
        record('clear_answer', item.id, {previous});
      }
      prune(); render();
    }, 'small'));
    if (p.scenario) {
      const scenario = page.querySelector('.scenario');
      const response = document.createElement('div'); response.className = 'scenario-response';
      response.append(...Array.from(page.children).filter(el => el !== scenario));
      page.append(response);
    }
    if (pos > 0) nav.prepend(button('Back', () => go(list[pos - 1].id, 'back')));
    if (p.kind === 'finish') {
      const submit = button('Submit survey', () => finish('submitted'), 'primary'); submit.id = 'submit-survey'; nav.append(submit);
    } else {
      if (p.kind !== 'info') nav.append(button(p.kind === 'video' ? 'Skip demonstration' : 'Skip', skip));
      if (p.counter) nav.append(text('span', p.counter, 'example-counter'));
      nav.append(button(p.next_label || 'Next', () => {
        if (p.kind === 'video') {
          if (!activeVideo || activeVideo.readyState < 2 || activeVideo.error) return;
          state.answers[p.id] = {status: 'answered', choices: ['watched']};
          record('demo_confirmed', p.id);
        } else if (p.kind === 'edit') {
          // Next saves an edit; with no edit and no decision the example is unanswered.
          const a = answer(p.id);
          if (!a || (a.decision === 'edited' && !a.edited)) {
            state.answers[p.id] = {status: 'unanswered'};
            record('unanswered_next', p.id);
          }
        } else if (['single','multi','text','group'].includes(p.kind)) {
          for (const item of questionItems(p)) if (!answer(item.id)) {
            state.answers[item.id] = {status:'unanswered'};
            record('unanswered_next',item.id);
          }
        }
        next();
      }, 'primary'));
      if (p.kind === 'video') nav.lastChild.disabled = !activeVideo || activeVideo.readyState < 2 || !!activeVideo.error;
    }
    foot.append(button('Save and take a break', () => {
      state.status = 'paused'; pausedView = true; record('pause', state.page); render(); requestSave();
    }));
    foot.append(button('End my survey now', () => {
      const heading = text('h1', 'End your survey now?'); heading.tabIndex = -1;
      page.className = '';
      page.replaceChildren(heading, text('p', 'Your responses so far will be saved and submitted. You will not be able to return to answer more questions. If you want to return later, choose “Keep going,” then “Save and take a break.”'));
      nav.replaceChildren(button('Keep going', render),button('End and submit survey', () => finish('ended'), 'primary'));
      foot.replaceChildren(); $('page').scrollTop = 0; resize(); heading.focus({preventScroll:true});
    }));
    setStatus(); resize();
  }
  function finish(status) {
    state.status = status; record('survey_' + status, state.page); render(); requestSave();
  }
  function mediaUrl(path) {
    if (standalone) return new URL(path, window.location.href).href;
    // Local components are served under <external app prefix>/component/...
    // That prefix includes both baseUrlPath and any hosting proxy route. A
    // root-relative URL or document.referrer can discard the latter (and the
    // referrer can be origin-only). Resolve against our own component route.
    const here = new URL(window.location.href);
    const component = here.pathname.lastIndexOf('/component/');
    const base = component >= 0
      ? here.origin + here.pathname.slice(0, component + 1)
      : new URL('.', document.referrer || here.href).href;
    return new URL(path.replace(/^\/(?!\/)/, ''), base).href;
  }
  function mountVideo() {
    const host = $('video-host'); if (!host || activeVideo) return;
    host.replaceChildren();
    if (!args.demo_url) {
      host.append(text('p', args.demo_error || 'Loading the demonstration…', 'help')); return;
    }
    activeVideo = document.createElement('video'); activeVideo.controls = true; activeVideo.preload = 'auto';
    activeVideo.src = mediaUrl(args.demo_url);
    activeVideo.setAttribute('aria-label', 'Speech recognition and correction demonstration');
    if (args.demo_captions) {
      const track = document.createElement('track'); track.kind = 'captions'; track.srclang = 'en'; track.label = 'English'; track.default = true;
      track.src = 'data:text/vtt;base64,' + args.demo_captions; activeVideo.append(track);
    }
    for (const kind of ['play','pause','seeked','ended']) activeVideo.addEventListener(kind, e => record('video_' + kind, 'demo_video', {video_seconds: activeVideo.currentTime}, e));
    const status = text('p', 'Loading the demonstration… You can skip it if you prefer.', 'help');
    status.setAttribute('role', 'status');
    const video = activeVideo;
    const setReady = () => {
      if (activeVideo !== video) return;
      const ready = video.readyState >= 2 && !video.error;
      status.classList.toggle('hidden', ready);
      const nav = $('navigation'); if (nav.lastChild) nav.lastChild.disabled = !ready;
    };
    video.addEventListener('loadeddata', setReady);
    video.addEventListener('canplay', setReady);
    video.addEventListener('error', () => {
      if (activeVideo !== video) return;
      status.textContent = 'The video could not be loaded. You can skip it or go Back and try again.';
      setReady();
    });
    host.append(video, status);
    if (args.demo_transcript) {
      const details = document.createElement('details'); details.append(text('summary','Read the demonstration transcript'),text('p',args.demo_transcript)); host.append(details);
    }
    resize();
  }
  // Keyboard navigation and activations on choice/navigation controls are also
  // recorded. Text-area keys have their own handler with selection positions.
  for (const kind of ['keydown', 'keyup']) $('survey').addEventListener(kind, e => {
    if (!initialized || state.status !== 'active' || !e.target.matches('input,button,summary')) return;
    const field = e.target.dataset.question || e.target.id || state.page;
    record(kind, field, {key:e.key, code:e.code, repeat:e.repeat, is_composing:e.isComposing,
      ctrl:e.ctrlKey, alt:e.altKey, shift:e.shiftKey, meta:e.metaKey,
      control:e.target.tagName.toLowerCase()}, e);
  });
  $('survey').addEventListener('click', e => {
    if (!initialized || state.status !== 'active') return;
    const control = e.target.closest('button,input');
    if (control) record('control_click', control.dataset.question || state.page,
      {control:control.tagName.toLowerCase(), label:control.value || control.textContent}, e);
  }, true);
  window.addEventListener('message', e => {
    if (standalone || e.source !== window.parent || e.data?.type !== 'streamlit:render') return;
    receive(e.data.args);
  });
  // Standalone page entry point: options = {schema, record, session_key, demo_url,
  // demo_transcript, demo_captions, save(packet) -> Promise<saved record>}.
  window.startSurvey = options => {
    if (initialized) return;
    standalone = {save: options.save};
    receive(options);
  };
  function receive(a) {
    args = a;
    if (!initialized) {
      schema = args.schema; state = clone(args.record.state); revision = args.record.revision;
      cacheKey = 'communication-survey:' + args.session_key;
      try {
        const cached = JSON.parse(sessionStorage.getItem(cacheKey) || 'null');
        // A different server revision is authoritative unless it is precisely our pending batch.
        if (cached && (cached.revision === revision || cached.inflight?.batch_id === args.record.batch_id)) {
          ({state, events, inflight, seq, clientId} = cached);
          saveQueue = cached.saveQueue || [];
          revision = cached.revision;
          lastSaved = cached.lastSaved || '';
          if (inflight?.batch_id === args.record.batch_id) {
            revision = args.record.revision; inflight = null;
          }
        }
      } catch (_) { storageAvailable = false; }
      initialized = true; render();
      if (state.status === 'active') record('session_open', state.page, {schema_version:schema.version, time_origin_ms: performance.timeOrigin});
      else if (terminal(state) && !events.length && !inflight && !saveQueue.length) {
        try {sessionStorage.removeItem(cacheKey);} catch (_) {}
      }
      // Restore only previously requested saves; ordinary drafts stay local.
      // Older cached pause/submit records can have an unsent final event buffer.
      if (state.status !== 'active' && events.length) requestSave();
      else sendPending();
    }
    if (args.ack && args.ack !== lastAck) {
      lastAck = args.ack;
      if (inflight && inflight.batch_id === args.ack) {
        revision = args.revision; inflight = null; saveError = ''; lastSaved = args.saved_at;
        sendPending();
        cache(); setStatus();
        if (terminal(state) || state.status === 'paused') render();
        if (terminal(state) && !events.length && !inflight && !saveQueue.length) {
          try {sessionStorage.removeItem(cacheKey);} catch (_) {}
        }
      }
    }
    if (args.error) { saveError = args.error; setStatus(); }
    if (state.page === 'demo_video') mountVideo();
  }
  document.addEventListener('visibilitychange', () => {
    if (!initialized || state.status !== 'active') return;
    record('visibility_change', state.page, {visibility: document.visibilityState});
  });
  window.addEventListener('beforeunload', e => {
    if (events.length || inflight || saveQueue.length) {cache(); e.preventDefault(); e.returnValue = '';}
  });
  window.addEventListener('pagehide', () => {if (initialized) cache();});
  // Retry the identical in-flight batch; deduplication is performed before server acknowledgement.
  setInterval(() => {if (inflight && Date.now() - lastSent > 12000) sendPending();}, 3000);
  new ResizeObserver(resize).observe(document.body);
  window.addEventListener('resize', resize);
  try {
    const viewport = window.parent.visualViewport;
    viewport?.addEventListener('resize', resize);
    viewport?.addEventListener('scroll', resize);
    window.addEventListener('pagehide', () => {
      viewport?.removeEventListener('resize', resize);
      viewport?.removeEventListener('scroll', resize);
    }, {once:true});
  } catch (_) { /* The host's viewport CSS remains the fallback. */ }
  if (!window.SURVEY_STANDALONE) {
    bridge('streamlit:componentReady', {apiVersion: 1});
    resize();
  }
})();
