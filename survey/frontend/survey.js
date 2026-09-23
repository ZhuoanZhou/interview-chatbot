/* No analytics, external scripts, model calls, or document-wide key capture. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const clone = x => JSON.parse(JSON.stringify(x));
  const uuid = () => crypto.randomUUID();
  const terminal = s => ['submitted', 'ended'].includes(s.status);
  let schema, state, args, initialized = false, revision = 0, seq = 0;
  let events = [], inflight = null, cacheKey, timer, pausedView = false;
  let clientId = uuid(), lastSent = 0, lastSaved = '', storageAvailable = true;
  let saveError = '', lastAck = '', activeVideo = null;
  let lastConsoleStatus = '';

  function bridge(type, extra = {}) {
    window.parent.postMessage({isStreamlitMessage: true, type, ...extra}, '*');
  }
  function resize() {
    bridge('streamlit:setFrameHeight', {height: Math.ceil(document.documentElement.scrollHeight)});
  }
  function cache() {
    try {
      sessionStorage.setItem(cacheKey, JSON.stringify({state, events, inflight, revision, seq, clientId, lastSaved}));
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
    schedule();
  }
  function schedule() {
    clearTimeout(timer);
    timer = setTimeout(flush, 1200);
  }
  function setStatus() {
    const el = $('save-status');
    const pending = events.length || inflight;
    const warning = !!saveError || (!storageAvailable && !!pending);
    el.classList.toggle('hidden', !warning);
    el.classList.toggle('warning', warning);
    el.textContent = saveError ? 'Your latest changes have not been saved yet. Please keep this tab open while we try again.'
      : warning ? 'Please keep this tab open while your changes are saved.' : '';
    const diagnostic = saveError ? 'Save failed; retry pending.' : pending ? 'Saving survey changes.' : 'Survey changes saved.';
    if (diagnostic !== lastConsoleStatus) {
      console.debug('[Survey]', diagnostic); lastConsoleStatus = diagnostic;
    }
    const submit = $('submit-survey');
    if (submit) submit.disabled = !!pending;
  }
  function flush() {
    clearTimeout(timer);
    if (!initialized || (!inflight && !events.length)) return;
    if (!inflight) {
      // Freeze the batch. Later edits stay in events until this exact batch is acknowledged.
      inflight = {batch_id: uuid(), base_revision: revision, state: clone(state), events: events.splice(0)};
      cache();
    }
    lastSent = Date.now();
    bridge('streamlit:setComponentValue', {value: {...inflight, attempt: uuid()}, dataType: 'json'});
    setStatus();
  }
  function answer(id) { return state.answers[id]; }
  function condition(c) {
    if (!c) return true;
    if (c.all) return c.all.every(condition);
    if (c.any) return c.any.some(condition);
    if (c.answered) return answer(c.answered)?.status === 'answered';
    return (answer(c.question)?.choices || []).some(v => c.values.includes(v));
  }
  function visiblePages() { return schema.pages.filter(p => condition(p.when)); }
  function currentPage() { return schema.pages.find(p => p.id === state.page); }
  function prune() {
    // Document order is parent-before-child, so deleting a hidden parent also hides its descendants.
    for (const p of schema.pages) {
      if (!condition(p.when) && state.answers[p.id]) {
        const previous = state.answers[p.id];
        delete state.answers[p.id];
        record('answer_invalidated', p.id, {source: 'branch_change', previous});
      }
    }
    const different = answer('different_repair');
    const first = answer('first_repair')?.choices?.[0];
    if (different?.choices?.includes(first)) {
      delete state.answers.different_repair;
      record('answer_invalidated', 'different_repair', {source: 'excluded_previous_method', previous: different});
      prune();
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
    requestAnimationFrame(() => { const h = $('question-title'); h?.focus({preventScroll: true}); window.scrollTo(0, 0); resize(); });
  }
  function go(id, type = 'next') {
    record('navigation', state.page, {action: type, destination: id});
    state.page = id; cache(); render();
    record('page_view', id); flush(); focusHeading();
  }
  function next() {
    const list = visiblePages(), i = list.findIndex(p => p.id === state.page);
    if (i >= 0 && i < list.length - 1) go(list[i + 1].id);
  }
  function skip() {
    const p = currentPage(), previous = answer(p.id) || null;
    state.answers[p.id] = {status: 'skipped'};
    record('skip', p.id, {previous}); prune(); next();
  }
  function choicesChanged(p, group, value, checked, event) {
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
      record('option_selected', group ? `${p.id}.${group}` : p.id, {option: v, source: 'participant'}, event);
    // Other explanations are final answers only while Other is selected; edits remain in the event log.
    const otherKey = group || 'other';
    if (!after.includes('Other') && a.other[otherKey]) {
      record('other_hidden', `${p.id}.${otherKey}`, {previous_text: a.other[otherKey], source: 'option_change'});
      delete a.other[otherKey];
    }
    prune(); cache();
    // Update controls in place; do not destroy keyboard focus on the chosen option.
    document.querySelectorAll('input[data-question]').forEach(input => {
      input.checked = input.dataset.group ? a.groups?.[input.dataset.group] === input.value : a.choices.includes(input.value);
    });
    updateOther(p, group); setStatus(); resize();
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
      onValue(ta.value);
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
    const selected = group ? a?.groups?.[group] === 'Other' : a?.choices?.includes('Other');
    if (!selected) { host.replaceChildren(); return; }
    if (host.firstChild) return;
    host.append(textarea(p, `${p.id}.${key}`, a?.other?.[key], value => {
      state.answers[p.id].other ||= {}; state.answers[p.id].other[key] = value;
    }, 'Other — tell us more if you would like'));
  }
  function optionList(p, options, group) {
    const container = document.createElement('div');
    const list = document.createElement('div'); list.className = 'options' + (options.every(v => v.length < 27) ? ' short' : '');
    if (!group) list.setAttribute('aria-labelledby', 'question-title');
    options.forEach((v, i) => {
      const label = document.createElement('label'); label.className = 'choice';
      const input = document.createElement('input'); input.type = p.kind === 'multi' ? 'checkbox' : 'radio';
      input.name = group ? p.id + '-' + group : p.id; input.value = v;
      input.dataset.question = p.id; if (group) input.dataset.group = group;
      input.checked = group ? answer(p.id)?.groups?.[group] === v : !!answer(p.id)?.choices?.includes(v);
      input.addEventListener('change', e => choicesChanged(p, group, v, input.checked, e));
      label.append(input, text('span', v)); list.append(label);
    });
    container.append(list);
    if (options.includes('Other')) {
      const other = document.createElement('div'); other.id = 'other-' + p.id + '-' + (group || 'other'); container.append(other);
    }
    return container;
  }
  function render() {
    activeVideo = null;
    const page = $('page'), nav = $('navigation'), foot = $('footer');
    page.replaceChildren(); nav.replaceChildren(); foot.replaceChildren();
    const p = currentPage() || schema.pages[0], list = visiblePages();
    $('section').textContent = p.section ? `Part ${p.section} of 4 · ${schema.sections[p.section]}` : 'Welcome';
    const pos = list.findIndex(x => x.id === p.id);
    $('progress').style.width = `${Math.round(100 * pos / Math.max(1, list.length - 1))}%`;
    if (terminal(state)) {
      page.className = 'complete';
      const pending = events.length || inflight;
      page.append(text('h1', pending ? 'Saving your survey…' : 'Thank you for sharing your experiences.'));
      page.append(text('p', pending ? 'Please keep this tab open until saving is complete.' : 'Your responses have been saved. You can now close this tab.'));
      setStatus(); resize(); return;
    }
    page.className = '';
    if (pausedView || state.status === 'paused') {
      page.append(text('h1', 'Your survey is paused'), text('p', 'Take as much time as you need. To return later, choose “Return to your survey” and enter your participant ID.'));
      const pending = events.length || inflight;
      page.append(text('p', pending ? 'Please wait a moment before closing this tab.' : 'You can now close this tab and return later using your participant ID.'));
      nav.append(button('Continue survey', () => {pausedView = false; state.status = 'active'; record('resume', state.page); render(); flush();}, 'primary'));
      setStatus(); resize(); return;
    }
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
    (p.paragraphs || []).forEach((s, i) => page.append(text('p', s, p.id === 'intro' && i === 3 ? 'notice' : '')));
    if (p.item) page.append(text('p', p.item, 'item'));
    if (p.help) page.append(text('p', p.help, 'help'));
    if (['single', 'multi'].includes(p.kind)) {
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
    }
    if (['single', 'multi', 'group', 'text'].includes(p.kind)) page.append(button('Clear answer', () => {
      const previous = answer(p.id) || null; delete state.answers[p.id];
      record('clear_answer', p.id, {previous}); prune(); render();
    }, 'small'));
    if (pos > 0) nav.append(button('Back', () => go(list[pos - 1].id, 'back')));
    if (p.kind === 'finish') {
      const submit = button('Submit survey', () => finish('submitted'), 'primary'); submit.id = 'submit-survey'; nav.append(submit);
    } else {
      if (p.kind !== 'info') nav.append(button(p.kind === 'video' ? 'Skip demonstration' : 'Skip', skip));
      nav.append(button(p.next_label || 'Next', () => {
        if (p.kind === 'video') {
          if (!args.demo_data) return;
          state.answers[p.id] = {status: 'answered', choices: ['watched']};
          record('demo_confirmed', p.id);
        } else if (['single','multi','text','group'].includes(p.kind) && !answer(p.id)) {
          state.answers[p.id] = {status:'unanswered'};
          record('unanswered_next',p.id);
        }
        next();
      }, 'primary'));
      if (p.kind === 'video') nav.lastChild.disabled = !args.demo_data;
    }
    foot.append(button('Save and take a break', () => {
      state.status = 'paused'; pausedView = true; record('pause', state.page); render(); flush();
    }));
    foot.append(button('Finish early', () => {
      page.replaceChildren(text('h1', 'Finish the survey now?'), text('p', 'Your responses so far will be saved. You can also keep going or take a break.'));
      nav.replaceChildren(button('Keep going', render),button('Finish and save', () => finish('ended'), 'primary'));
      foot.replaceChildren(); resize();
    }));
    setStatus(); resize();
  }
  function finish(status) {
    state.status = status; record('survey_' + status, state.page); render(); flush();
  }
  function mountVideo() {
    const host = $('video-host'); if (!host || activeVideo) return;
    host.replaceChildren();
    if (!args.demo_data) {
      host.append(text('p', args.demo_error || 'Loading the demonstration…', 'help')); return;
    }
    activeVideo = document.createElement('video'); activeVideo.controls = true; activeVideo.preload = 'metadata';
    activeVideo.src = 'data:video/mp4;base64,' + args.demo_data;
    activeVideo.setAttribute('aria-label', 'Speech recognition and correction demonstration');
    if (args.demo_captions) {
      const track = document.createElement('track'); track.kind = 'captions'; track.srclang = 'en'; track.label = 'English'; track.default = true;
      track.src = 'data:text/vtt;base64,' + args.demo_captions; activeVideo.append(track);
    }
    for (const kind of ['play','pause','seeked','ended']) activeVideo.addEventListener(kind, e => record('video_' + kind, 'demo_video', {video_seconds: activeVideo.currentTime}, e));
    host.append(activeVideo);
    if (args.demo_transcript) {
      const details = document.createElement('details'); details.append(text('summary','Read the demonstration transcript'),text('p',args.demo_transcript)); host.append(details);
    }
    const nav = $('navigation'); if (nav.lastChild) nav.lastChild.disabled = false;
    resize();
  }
  // Keyboard navigation and activations on choice/navigation controls are also
  // recorded. Text-area keys have their own handler with selection positions.
  for (const kind of ['keydown', 'keyup']) $('survey').addEventListener(kind, e => {
    if (!initialized || terminal(state) || !e.target.matches('input,button,summary')) return;
    const field = e.target.dataset.question || e.target.id || state.page;
    record(kind, field, {key:e.key, code:e.code, repeat:e.repeat, is_composing:e.isComposing,
      ctrl:e.ctrlKey, alt:e.altKey, shift:e.shiftKey, meta:e.metaKey,
      control:e.target.tagName.toLowerCase()}, e);
  });
  $('survey').addEventListener('click', e => {
    if (!initialized || terminal(state)) return;
    const control = e.target.closest('button,input');
    if (control) record('control_click', control.dataset.question || state.page,
      {control:control.tagName.toLowerCase(), label:control.value || control.textContent}, e);
  }, true);
  window.addEventListener('message', e => {
    if (e.source !== window.parent || e.data?.type !== 'streamlit:render') return;
    args = e.data.args;
    if (!initialized) {
      schema = args.schema; state = clone(args.record.state); revision = args.record.revision;
      cacheKey = 'communication-survey:' + args.session_key;
      try {
        const cached = JSON.parse(sessionStorage.getItem(cacheKey) || 'null');
        // A different server revision is authoritative unless it is precisely our pending batch.
        if (cached && (cached.revision === revision || cached.inflight?.batch_id === args.record.batch_id)) {
          ({state, events, inflight, seq, clientId} = cached);
          revision = cached.revision;
          lastSaved = cached.lastSaved || '';
          if (inflight?.batch_id === args.record.batch_id) {
            revision = args.record.revision; inflight = null;
          }
        }
      } catch (_) { storageAvailable = false; }
      initialized = true; render();
      if (!terminal(state)) record('session_open', state.page, {schema_version:schema.version, time_origin_ms: performance.timeOrigin});
      else if (!events.length && !inflight) {
        try {sessionStorage.removeItem(cacheKey);} catch (_) {}
      }
      if (events.length || inflight) flush();
    }
    if (args.ack && args.ack !== lastAck) {
      lastAck = args.ack;
      if (inflight && inflight.batch_id === args.ack) {
        revision = args.revision; inflight = null; saveError = ''; lastSaved = args.saved_at;
        cache(); setStatus();
        if (terminal(state) || state.status === 'paused') render();
        if (terminal(state) && !events.length) {
          try {sessionStorage.removeItem(cacheKey);} catch (_) {}
        } else if (events.length) schedule();
      }
    }
    if (args.error) { saveError = args.error; setStatus(); }
    if (state.page === 'demo_video') mountVideo();
  });
  document.addEventListener('visibilitychange', () => {
    if (!initialized || terminal(state)) return;
    record('visibility_change', state.page, {visibility: document.visibilityState}); flush();
  });
  window.addEventListener('beforeunload', e => {
    if (events.length || inflight) {cache(); flush(); e.preventDefault(); e.returnValue = '';}
  });
  window.addEventListener('pagehide', () => {if (initialized) {cache(); flush();}});
  // Retry the identical in-flight batch; deduplication is performed before server acknowledgement.
  setInterval(() => {if (inflight && Date.now() - lastSent > 12000) flush(); else if (events.length && !inflight) flush();}, 3000);
  new ResizeObserver(resize).observe(document.body);
  bridge('streamlit:componentReady', {apiVersion: 1});
  resize();
})();
