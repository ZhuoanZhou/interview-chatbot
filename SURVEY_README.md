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
- The survey uses the full available width below Streamlit's header, with the
  participant-ID sidebar initially collapsed (open it from the top-left control).
  Answer choices use two or three columns on wider screens; story fields and
  scenario context sit side by side. Expanded Other fields use spare side space.
  Navigation, clear-answer, and pause/end controls remain
  in a reserved bottom panel and share a row on desktop.
- The question scrollbar and “Show more below” button are removed. Compact
  spacing and wider layouts reduce the need to scroll; on narrow/short screens,
  with enlarged text, or with expanded content, the question area still supports
  touch, mouse-wheel, and keyboard scrolling so no answers are inaccessible.
  Moving between questions resets its scroll. Same-origin hosts also follow
  visual viewport resizing; actual mobile keyboard behavior should be checked on
  participants' target devices before the study.
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

The demo is downloaded on its screen in 8 MB chunks, with transient-error retries,
and cached across participants for one hour (keyed by file ID and Drive credentials).
The first load after a server restart/cache expiry still waits for Drive. Failed
downloads are not cached. The demo uses `st.video()`, as in the earlier chatbot.
Streamlit owns the player, media delivery, seeking, captions, and hosted URL
routing. No video bytes or media URL are sent through the survey component, and
no deployment hostname is hard-coded. No Drive sharing permissions change.
The native player appears above the survey controls; it is paused and hidden
when leaving the demo or taking a break. Preview mode never downloads the demo.

On the supported same-origin host, the component observes only the native player
inside the `survey-demo` container to log play/pause/seek/end events and enable
confirmation once video data is playable. Listener cleanup prevents duplicates
on reruns, and load errors keep skipping available. If hosted cross-origin,
native playback still works, but browser video events cannot be observed and
confirmation relies on the participant's explicit self-report. Check the target
deployment before collecting data.

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

Native video events also include `browser_time_origin_ms`, the host window's
time origin for `browser_event_ms`; other event timestamps retain the survey
frame's time origin. `elapsed_ms` is always measured inside the survey frame.

## Saving, resuming, and limitations

The browser timestamps and buffers interactions locally while participants answer.
It requests a Drive save only on navigation (Next, Back, Skip, and Continue),
Save and take a break, or confirmed submission/ending. Starting a new survey
still creates its initial Drive record. Typing, selections, idle time, tab
visibility changes, and page closing do not trigger uploads.

Each requested save freezes an answer checkpoint and its collected events. Rapid
navigation queues these snapshots while a previous save is in flight; new edits
stay local until the next save action. The server acknowledges only after Drive
accepts the batch. Unacknowledged requested batches retry with the same ID after
12 seconds (checked every 3 seconds) and on refresh. These retries do not include
later, unsaved edits. No timer initiates saves of ordinary drafts.

Pending data is temporarily stored in this tab's sessionStorage to survive refresh.
It is cleared after acknowledged completion. This includes deleted text: the
interaction log is more sensitive than final survey answers. Pause is not deletion.
Refreshing restores a compatible local draft without uploading it automatically.
Closing the tab/browser before a save action or before acknowledgement can lose
unsaved answers and events. Participants should use Save and take a break before
leaving; a browser exit warning is requested when data remains unsaved, but browsers
may suppress it. Do not claim crash-proof or exactly-once capture. Errors remain
visible; pause and completion screens confirm only after all requested saves finish.

A participant ID is the sole resume credential, as requested for participant
simplicity. New IDs use the previous short format, e.g. `P-ABC123`, and are shown
in the left sidebar. Choose “Return to your survey” and enter that ID to resume.
Existing fixed-survey sessions with longer participant IDs can also be resumed
using those IDs: the app finds their original token-based folders and continues
the saved progress without moving or rewriting old batches. Previously issued
long resume codes remain accepted for compatibility. Older adaptive-chatbot
transcripts are not converted into fixed-survey answers.

Keep participant IDs private; anyone with an ID can open that survey. Use one
active tab/device per participant: revision
checks detect sequential stale writes, but Google Drive provides no atomic
compare-and-swap here, so simultaneous multi-device writing is not supported.

The app inherits the configured Drive folder's permissions; it does not create
public sharing permissions. It does not configure retention/deletion, encryption
of exports, institutional consent, or hosting. Those remain deployment duties.

Routine save diagnostics appear only in the browser console. Participants see a
brief message if saving fails, and completion/pause screens confirm when it is
safe to leave. Console diagnostics contain no IDs, answers, or credentials.

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
`tests/survey_layout.cjs` checks the full-width host against a local
`SURVEY_PREVIEW=1` server on port 8511, then checks every question (including
expanded Other fields) in an isolated harness at 1280×720, 1366×768, and
1920×1080. It also checks fixed navigation and reachable overflow on phone and
short-window sizes.
`tests/test_survey_media.py` checks shared caching, invalidation, and retries.
Startup tests check native video rendering and preview isolation.
`tests/survey_frontend.cjs` generates a tiny local clip to check video
loading, failure, logging, and player preservation. For real HTTP media checks,
run `node tests/survey_media_browser.cjs --fixture`, start
`python -m streamlit run tests/survey_media_app.py --server.port 8514 --server.baseUrlPath study`,
then run `node tests/survey_media_browser.cjs`. The fixture runs the actual survey
entry point with an in-memory store and fake credentials, checking native video
playback, byte ranges, video event logging, reruns, navigation/pause, and five
desktop/mobile viewport sizes. These checks never contact Drive.
