# Fixed communication survey

`streamlit_app.py` runs the fixed survey from `refined_question_list_9.29.2026.docx`
(question version `2026-09-29-v1`).
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
in the September 29 guide, with the September 2026 changes to Part 2 and the
post-demonstration exercise described below. Study purposes, researcher notes and
instructions are not shown. Part 3 Q6 is lightly reworded for grammar. Every sourced screen has a zero-based `source_paragraphs` locator, and the
schema records the source document SHA-256. The Word file remains unchanged.

- Each Other choice is last, in a full-width bordered block with an optional text
  area always visible below its label. Typing selects Other automatically;
  deselecting it clears its explanation (the interaction log retains the edits).
  A question can name a different free-text choice with `other_option` (no
  current question uses it).
- Part 3 Q1 and Q7 each appear on one screen, with a rating row per statement:
  five usefulness levels and Not sure. The original per-item answer IDs remain
  unchanged, so existing sessions and exports stay compatible. Older sessions
  resume on the combined question; historical N/A responses are preserved and
  labeled until changed. Clear answer and Skip apply to the whole rating set;
  Next preserves completed rows and records unanswered rows. All five scenarios remain.
- Q7's question and situations use the researcher's revised wording (2026-10),
  set in `scripts/build_survey_schema.py` (`SITUATIONS`) rather than taken from
  the Word guide. Answer IDs (`situation_1`–`situation_7`) are unchanged.
- The post-demo section depends on agreeing to and confirming the demo.
  Skipping/unavailable video does not count as watching it.
- The AAC name and detection cues follow the source conditions. The speech-to-text question has no follow-ups (the guide's two are
  left out). Changing a parent answer clears inapplicable
  child answers from the final response checkpoint; the event log retains the change.
- The September 29 guide removed the "recent time" story and its follow-ups, the
  AAC carrying question, and Part 3 Q8, and added "what has made you stop trying"
  (`stop_trying`; "I have not stopped trying" and "Not sure" are exclusive). Part 3
  Q3 (word not in the suggestions) and Q4 with its Re-check retry follow-up were
  later removed as well.
- All questions are skippable. Back, clear answer, pause/resume, and finish early
  are supported. No response is preselected. Single choices can be cleared.
- Mutually exclusive options such as “Not sure” clear incompatible choices.
- The participant-ID side panel opens by default on computers and tablets
  (Streamlit's `auto` setting keeps it collapsed on phones, where it would cover the
  survey). Its show/hide arrow is always visible, not only on hover, and the first
  screen tells participants where the ID is. The survey uses the remaining width.
  Answer choices are listed vertically. If a single column would make the question
  scroll, survey.js switches to two columns (three or four on very wide screens),
  filled top to bottom; phones always use one column. Scenario context sits beside
  its question. Other fields follow the options. The rating tables are unchanged.
  Navigation, clear-answer, and pause/end controls remain
  in a reserved bottom panel and share a row on desktop.
- The question scrollbar and “Show more below” button are removed. Compact
  spacing and wider layouts reduce the need to scroll; on narrow/short screens,
  with enlarged text, or with expanded content, the question area still supports
  touch, mouse-wheel, and keyboard scrolling so no answers are inaccessible.
  Moving between questions resets its scroll. Same-origin hosts also follow
  visual viewport resizing; actual mobile keyboard behavior should be checked on
  participants' target devices before the study.

The scenario introduction is preserved from the supplied document. It describes
the device as showing what the partner understood; this is a hypothetical premise,
not a factual claim about ASR capability. Review this wording before piloting.

## "What would you do first?" and the post-demonstration examples

The Part 2 examples offer one choice: Continue the conversation without correcting
the text · Stop trying to get this message across · Try to help the other person
understand what I meant · Not sure · Do something else (optional details). Under
"Try to help ..." is a follow-up (`ways` in the schema): "If you chose this, which
ways might you try?" (choose all), with "Try another way" taking optional details.
The ways are grayed out and cannot be selected unless "Try to help ..." is chosen;
ticks already made are kept (grayed) if another choice is made, and come back if it
is chosen again. The answer stores `choices`, `ways`, and `other` (`other` for "Do
something else", `ways` for "Try another way"); the export has a `ways` column, to be
read only when the choice is "Try to help ...". These replace the Word guide's list.

Part 2 examples (`s1_action`–`s5_action`) ask only this question. Their follow-ups
(What would you change? / Which words? / Why leave it?) were removed; `retired_pages`
in the schema sends a session saved on a removed screen back to its example. Answers
already given to removed questions, or with the previous choice wording, stay in the
saved records and `responses.json`; removed questions no longer appear as rows in
`answers.csv`.

Part 3 order, for participants who agreed to and confirmed watching the video: the
Part 3 questions (Q1–Q8), then an optional exercise at the end of Part 3. Its intro
(`edit_intro`, "Try the example situations") explains that it includes text editing
and suggested words but not speech-to-text or the "Re-check" function, and asks
"Yes, I’d like to try" / "No, skip the examples". Only "Yes" shows the five examples
as editable transcripts (`e1_edit`–`e5_edit`); No, Skip or no answer goes on to the
closing question. A session saved on the removed question step
(`e1_action`–`e5_action`) resumes on its edit screen.

"Stop the exercise" on each example leaves the exercise: examples not yet answered
from that point are saved as `{status: skipped, stopped: true}`, answered ones are
kept, and an `exercise_stopped` event lists the skipped examples.

Each edit screen shows the context, what they meant to say, and the misrecognized
transcript, pre-filled and editable. Under it:

- Delete all / Reset: empty the text, or restore the original transcript. Each logs
  its own event (`delete_all`, `transcript_reset`) plus a `text_input` event with
  `input_type` `deleteAllButton` / `resetButton`.
- Keep as is / Say it again / Switch to my AAC / Ask for help / Abandon / Not sure:
  record the decision (`decision` event) and move
  straight to the next screen. Going Back shows the chosen button as pressed.

Word candidates: clicking or tapping a word in the transcript shows six suggestions,
three above and three below it, with the word outlined. Choosing one replaces the
word (punctuation kept) and leaves the text box focused; typing still works as
before and closes the suggestions, as do Escape, scrolling and clicking elsewhere.
The candidates are hand-written prototype stimuli, not ASR output, stored in
`survey/word_candidates.json` (six per transcript word, fixed positions) and copied
into the schema by `scripts/build_survey_schema.py`. Current words are matched to the
original transcript with a word-level alignment, so a word the participant retyped
still shows the candidates for the word it replaced; newly inserted words have none.
After a candidate is chosen, clicking it again shows the same six positions with the
original word in the chosen one's place. Events: `candidates_shown` (position, word,
options), `candidate_selected` (from, to, option_index, options), `candidates_closed`
(reason: typing, escape, scroll, outside, other_word, ...), and the replacement itself
as a `text_input` event with `input_type` `candidateSelected`.

The answer stores `decision` (`edited`, `kept`, `say_again`, `switch_aac`,
`ask_help`, `abandoned` or `not_sure`), the final
`text`, the `original` transcript, and `edited` (whether the text differs). Next after
an edit saves `decision: edited`; Next with no edit and no decision (including after
Reset) is `unanswered`; Skip is `skipped`. Typing after choosing a decision changes it
back to `edited`. Keystrokes, input deltas, focus and composition use the same
text-area logging as every other field. These screens are not in the Word guide;
`scripts/build_survey_schema.py` generates them, so a rebuild keeps them.

## Demonstration

The app reuses the previous app's Drive video ID. Override it with
`DEMO_VIDEO_FILE_ID` in Secrets if needed. Optional `DEMO_TRANSCRIPT` supplies a
readable alternative and `DEMO_CAPTIONS_PATH` points to a server-side WebVTT file.
Configure captions/a transcript if the video does not already have usable captions.
No demonstration transcript or captions have been invented.

The demo starts downloading in a background thread as soon as a participant starts
or resumes the survey (with its own Drive client), in 8 MB chunks with transient-error
retries, and is cached across participants for one hour (keyed by file ID and Drive
credentials). The demo screen waits for a download already in progress rather than
starting another, so it only waits for Drive if the download has not finished yet. Failed
downloads are not cached. The browser receives a short Streamlit media URL rather
than the entire video encoded into each survey message. This restores the old
chatbot's cached media delivery and supports byte-range playback and seeking.
The media URL is registered again on each demo-screen rerun so it stays available;
this uses the media manager from the pinned Streamlit version. No Drive sharing
permissions change. Video interaction logging stays in the survey player, and
confirmation remains disabled until video data is playable. Load errors keep the
skip option available. Preview mode still never downloads the demonstration.
Media paths resolve against the component's external app prefix, preserving
hosting proxy routes as well as a configured base path; no deployment hostname
is hard-coded.

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
The export only combines sessions saved under the current version; to export
surveys saved under `2026-09-22-v1`, use `survey/schema.json` from before the
September 29 update.
`python -m unittest discover -s tests -p "test_survey*.py"` tests saves and exports.
The frontend browser test uses a separate fake-data harness; it never contacts Drive.
`tests/survey_layout.cjs` checks the full-width host against a local
`SURVEY_PREVIEW=1` server on port 8511, then checks every question (including
expanded Other fields) in an isolated harness at 1280×720, 1366×768, and
1920×1080. It also checks fixed navigation and reachable overflow on phone and
short-window sizes.
`tests/test_survey_media.py` checks shared caching, invalidation, retries, and media
registration. `tests/survey_frontend.cjs` generates a tiny local clip to check video
loading, failure, logging, and player preservation. For real HTTP media checks,
run `node tests/survey_media_browser.cjs --fixture`, start
`python -m streamlit run tests/survey_media_app.py --server.port 8514 --server.baseUrlPath study`,
then run `node tests/survey_media_browser.cjs`. These checks never contact Drive.
`node tests/survey_media_routing.cjs` checks root, base-path, proxy-prefix, and
combined routing with a playable generated clip and origin-only referrers.
