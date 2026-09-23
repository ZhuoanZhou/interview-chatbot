# Fixed communication survey

`streamlit_app.py` runs the fixed survey from `refined_question_list_9.22.2026.docx`.
The previous adaptive app is preserved in `legacy_interview_app.py`. The survey
does not import it, call an LLM, transcribe speech, or record audio.

## Run

```powershell
python -m pip install -r requirements-survey.txt
python -m streamlit run streamlit_app.py
```

Use the existing four Streamlit Secrets: `GDRIVE_FOLDER_ID`, `GDRIVE_CLIENT_ID`,
`GDRIVE_CLIENT_SECRET`, and `GDRIVE_REFRESH_TOKEN`. No OpenAI key is needed.
The survey will not start if these are missing and will not silently switch to
local storage when Drive fails. Existing interview folders are not modified.

For fabricated-data preview without reading secrets or connecting to Drive:

```powershell
$env:SURVEY_PREVIEW = "1"
python -m streamlit run streamlit_app.py
```

Preview writes only to `.survey-preview/`. Unset `SURVEY_PREVIEW` for real Drive
storage. Preview does not fetch the demonstration video. Never deploy preview
mode for study collection.

## Questions and branching

`survey/schema.json` contains all participant questions before “Notes to myself”
in the September 22 guide. Study purposes and researcher instructions are not
shown. Every sourced screen has a zero-based `source_paragraphs` locator, and the
schema records the source document SHA-256. The Word file remains unchanged.

- Each Other choice reveals an optional text area, including the two story fields.
- Both usefulness tables are presented one item per screen with the original six
  rating choices (five usefulness levels and N/A). All five scenarios remain.
- The post-demo section depends on agreeing to and confirming the demo.
  Skipping/unavailable video does not count as watching it.
- Repair outcome, second strategy, stopping, detection cues, and retry questions
  follow the source conditions. The first repair strategy is excluded when asking
  about a different strategy. Changing a parent answer clears inapplicable child
  answers from the final response checkpoint; the event log retains the change.
- All questions are skippable. Back, clear answer, pause/resume, and finish early
  are supported. No response is preselected. Single choices can be cleared.
- Mutually exclusive options such as “Not sure” clear incompatible choices.
- If no story is supplied, its follow-ups and the later specific-partner question
  are skipped. Each rating is shown separately for easier selection.

The scenario introduction is preserved from the supplied document. It describes
the device as showing what the partner understood; this is a hypothetical premise,
not a factual claim about ASR capability. Review this wording before piloting.

## Demonstration

The app reuses the previous app's Drive video ID. Override it with
`DEMO_VIDEO_FILE_ID` in Secrets if needed. Optional `DEMO_TRANSCRIPT` supplies a
readable alternative and `DEMO_CAPTIONS_PATH` points to a server-side WebVTT file.
Configure captions/a transcript if the video does not already have usable captions.
No demonstration transcript or captions have been invented.

## Interaction log

Listeners attach to survey controls only. They do not capture the resume-code
field, other browser tabs, passwords, or other applications. The introduction
discloses that typed/deleted/unsubmitted content and interaction times are saved.
Align this collection with the study's approved participant information before use.

Each event has `event_id`, `client_id`, per-client `sequence`, `page_id`,
`field_id`, `client_utc`, `elapsed_ms` (monotonic browser time), `time_origin_ms`, and, where present,
`browser_event_ms`. Each saved batch adds a server `saved_at` time. Browser clock
times may be inaccurate; use monotonic times within a browser page lifetime and
do not assume clocks across devices agree. A page reload resets monotonic time.

Events include:

- `option_selected` / `option_deselected`, including automatic deselection due to
  a mutually exclusive choice (identified by `source`).
- `keydown` / `keyup` within survey controls, including modifiers and repeated keys;
  text-area events also contain selection positions. These describe keyboard
  actions, not necessarily edits. `control_click` records choice/button activation.
- `before_input` / `text_input`: insertion, deletion, replacement, paste, undo,
  redo, composition and browser/assistive-input changes. `text_input` records
  inserted/deleted text, the resulting value, input type, and UTF-16 offsets.
- Composition start/end, field focus/blur, page views, navigation, skip, clear,
  branch invalidation, pause, resume, finish, and video play/pause/seek/end.

Use `text_input` to reconstruct edits; do not count both keydown and input as two
character edits. Mobile keyboards, dictation, IMEs, paste and assistive tools can
insert multiple characters per input event without exposing individual physical
keystrokes. The app records what the browser actually supplies; it does not invent
per-character times. There is a 10,000-character limit per text area.

## Saving, resuming, and limitations

The browser buffers events and sends immutable batches approximately every 1–3
seconds or on navigation. While a save is in flight, further events stay in a
separate buffer. The server acknowledges only after Drive accepts the batch.
Retries use the same batch ID. Each batch contains an answer checkpoint plus new
events, so prior changes are not overwritten by later edits.

Pending data is temporarily stored in this tab's sessionStorage to survive refresh.
It is cleared after acknowledged completion. This includes deleted text: the
interaction log is more sensitive than final survey answers. Pause is not deletion.
Closing a tab/browser before acknowledgement can lose unsaved events; do not claim
crash-proof or exactly-once capture. A warning and saved indicator communicate this.

A random 256-bit private resume code is required. Drive folder names contain its
SHA-256 hash, not the secret. Participant IDs alone cannot resume surveys. Keep
the resume code private. Use one active tab/device per participant: revision
checks detect sequential stale writes, but Google Drive provides no atomic
compare-and-swap here, so simultaneous multi-device writing is not supported.

The app inherits the configured Drive folder's permissions; it does not create
public sharing permissions. It does not configure retention/deletion, encryption
of exports, institutional consent, or hosting. Those remain deployment duties.

## Export

Run `scripts/export_survey.py` to combine immutable batches into final answers and
a chronological event log. Its output directory is ignored by Git. It exports only
new survey folders, not the older chatbot records. See `--help` for local-preview
and Google Drive options. Skipped, unanswered and inapplicable questions are distinct.

## Rebuild and checks

`scripts/build_survey_schema.py <source.docx>` rebuilds the schema using paragraph
positions in this specific Word version; it is not a general DOCX converter. If the
guide changes, review the mapping and bump the schema version before collecting.
`python -m unittest discover -s tests -p "test_survey*.py"` tests saves and exports.
The frontend browser test uses a separate fake-data harness; it never contacts Drive.
