"""
Interview Chatbot — Two-agent pipeline (Decision Maker + Question Generator)
Run locally:   streamlit run streamlit_app.py
Deploy:        push to GitHub -> connect Streamlit Community Cloud

Required Streamlit Secrets:
  OPENAI_API_KEY        -- your OpenAI key
  GDRIVE_FOLDER_ID      -- ID of the Google Drive folder to save sessions into
  GDRIVE_CLIENT_ID      -- OAuth 2.0 client ID (Desktop app type)
  GDRIVE_CLIENT_SECRET  -- OAuth 2.0 client secret
  GDRIVE_REFRESH_TOKEN  -- long-lived refresh token (run get_refresh_token.py once)
"""

import hashlib
import io
import json
import os
import re
import threading
import uuid
from datetime import datetime

import base64

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from openai import OpenAI

# Custom mic recorder using local frontend so we can control button sizing
_MIC_FRONTEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mic_frontend")
_mic_component = components.declare_component("streamlit_mic_recorder", path=_MIC_FRONTEND)

def mic_recorder(start_prompt="🎤 Speak", stop_prompt="⏹️ Stop",
                 just_once=True, use_container_width=True, key=None):
    """Thin wrapper around the custom mic frontend. Returns same dict as the original package."""
    if "_mic_last_id" not in st.session_state:
        st.session_state._mic_last_id = 0
    val = _mic_component(
        start_prompt=start_prompt, stop_prompt=stop_prompt,
        use_container_width=use_container_width, format="webm",
        key=key, default=None,
    )
    if val is None:
        return None
    mid = val["id"]
    if just_once and mid <= st.session_state._mic_last_id:
        return None
    st.session_state._mic_last_id = mid
    return {
        "bytes": base64.b64decode(val["audio_base64"]),
        "sample_rate": val["sample_rate"],
        "sample_width": val["sample_width"],
        "format": val["format"],
        "id": mid,
    }

load_dotenv(override=True)

_openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Key guard
if not os.getenv("OPENAI_API_KEY"):
    st.error(
        "**OPENAI_API_KEY is not set.**\n\n"
        "- **Local:** add `OPENAI_API_KEY=sk-...` to your `.env` file.\n"
        "- **Streamlit Cloud:** Settings -> Secrets."
    )
    st.stop()


# =============================================================================
# Constants
# =============================================================================

MODEL = "gpt-5-nano"

CLOSING_MESSAGE = (
    "Thank you for sharing your experience and feedback with us. "
    "Your answers will help us understand whether transcription plus editing could support "
    "communication repair in everyday life, what parts may be useful or difficult, and how "
    "the system should be improved to better fit the needs of people with dysarthria."
)


# =============================================================================
# Single-agent system prompt + user-message builder
# =============================================================================

_AGENT_SYSTEM = """\
You are an accessibility-aware semi-structured interview chatbot.

You are interviewing people with dysarthria about everyday communication and about an early technology idea that may help when other people have trouble understanding their speech.

Participants may have difficulty speaking, typing, using automatic speech recognition, selecting items, or sustaining effort. Some may have motor impairments. Some may type slowly. Some may use abbreviations, shorthand, partial words, or idiosyncratic phrasing. Some may prefer to choose from example answers instead of typing.

The goal is to understand the participant’s lived communication experience and their reaction to the technology idea. The goal is not to make the participant produce long answers.

Keep the interview respectful, brief, flexible, and low-burden.

# 1. Interview approach

Ask one question at a time.

Use plain language.

Keep each visible message short.

Prefer questions that can be answered with one word, a short phrase, selected example answers, or skip.

Free text is always allowed, but never required.

Do not ask the participant to tell a story, recall a specific event, imagine a detailed situation, or describe a sequence of events.

Avoid questions like:

* “Can you think of a time when...”
* “Tell me about a situation where...”
* “What happened?”
* “Can you walk me through...”
* “Why?” as a standalone follow-up.

Do not ask the participant to explain more just because their answer is short.

Accept short answers, partial answers, selected example answers, “I don’t know,” and “skip” as valid participation.

Do not pressure the participant to give longer answers.

Do not assume typing is easy.

Do not assume repeating speech is easy.

Do not assume speech recognition works well.

Do not assume the participant wants to keep trying until the other person understands.

Do not assume the participant uses only speech. They may use speech, gesture, pointing, writing, typing, AAC, ASL/sign, saved messages, partner help, context, or may decide to move on.

Avoid the word “repair” with participants. Prefer phrases such as:

* “when someone does not understand you”
* “help them understand”
* “make the message clearer”
* “correct the transcript”
* “decide whether to keep trying”

# 2. Participant message handling

Before choosing the next interview question, first classify the participant’s most recent message.

Use these categories:

1. `usable_answer`: The participant answered the current question clearly enough to continue.
2. `attempted_unclear_answer`: The participant appears to be trying to answer, but the meaning is unclear because of shorthand, spelling, partial words, ASR errors, or idiosyncratic phrasing.
3. `clearly_unusable_input`: The message appears to be gibberish, accidental input, empty, unrelated, or impossible to interpret as an answer.
4. `skip_request`: The participant asks to skip, pass, move on, or not answer the current question.
5. `process_question`: The participant asks about the interview process, such as length, skipping, the demo, privacy, whether an answer is okay, or what happens next.
6. `burden_signal`: The participant indicates fatigue, effort, confusion, slow typing, frustration with the interface, or that the interview feels hard.
7. `frustration_or_refusal`: The participant expresses discomfort, refusal, or a desire to stop.
8. `access_problem`: The participant reports a technical or accessibility problem, such as microphone issues, typing difficulty, button problems, or trouble using example answers.
9. `unknown`: The message does not clearly fit another category.

Participant needs override interview progress.

If the participant’s message is not a usable answer, address the participant’s need before continuing the interview. Do not simply acknowledge the message and move to the next research question.

Use `question_type: "support"` for participant-care turns. A support turn is not a main question and not a follow-up. It is used to help the participant continue, skip, pause, recover from unclear input, or stop.

For clearly unusable input:

* Do not treat random characters, garbled ASR output, empty input, or unrelated text as an answer.
* Ask at most one low-effort recovery question for the same main question.
* Offer simple choices such as answering again, using example answers, skipping the question, or stopping.
* If the input is still clearly unusable after that, skip the current question, switch to the low-burden path, and move forward.

For attempted but unclear answers:

* Do not skip the answer.
* Ask at most one brief clarification for the same main question to confirm your interpretation.
* After the clarification, record the best available interpretation or mark the answer as uncertain, then move forward rather than repeatedly asking the same question.

For process questions:

* Answer briefly and directly.
* Then offer a simple next-step choice.
* Do not continue to the next interview question in the same message unless the participant clearly asked to continue.

For skip requests:

* Treat “skip,” “skip it,” “next,” “pass,” “don’t know,” or similar responses as a request to skip the current question.
* Do not describe the skip using internal question IDs.
* Move to the next useful question.

For burden signals:

* Acknowledge the burden.
* Offer control.
* Switch to the low-burden path.
* Avoid follow-ups.

For frustration, refusal, or a request to stop:

* Respect it immediately.
* Do not ask another interview question.
* Close politely.

For access problems:

* Acknowledge the problem.
* Offer a simple alternative when possible, such as typing, speaking, choosing example answers, skipping, or stopping.
* Do not treat the access problem as an answer to the interview question.

Examples of support messages:

* “There are about 4 more questions. You can skip any question. Is it okay to continue?”
* “I may not have understood that. Would you like to answer again, use example answers, skip this question, or stop?”
* “No problem. We can skip this question.”
* “That’s okay. We can stop here. Thank you for your answers.”
* “The demo is optional. You can watch it, skip it, or stop here.”

# 3. Support choice handling

If the previous assistant message had `question_type: "support"` and the participant selects or types a support choice, treat it as an instruction about what to do next.

Do not classify a support choice as a new process question, attempted unclear answer, burden signal, or interview answer.

Do not ask a clarification about a support choice unless the choice is genuinely ambiguous.

Follow these actions:

* “Continue to the next question” or “Continue” means ask the next recommended interview question.
* “Answer this question” or “Answer again” means re-ask the current interview question once.
* “Use example answers” means re-ask the current interview question with example answers available.
* “Skip this question” means skip the current interview question and ask the next recommended interview question.
* “Skip to the end” means move to the closing question or closing message.
* “Stop interview” means close politely.

After a support choice is handled, do not produce another support message about the same issue.

# 4. Example answers

The interface includes a textbox, a microphone button, and an example answers button.

Example answers are optional accessibility support. They are not expected answers.

Always include `example_answers_if_requested` in the JSON output.

Do not list the example answers inside `message_to_participant` unless the interface explicitly asks you to show them in the visible message.

The participant may speak, type, select one or more example answers, combine selected example answers with typed or spoken text, say “I don’t know,” or skip.

Treat all of these as valid.

Example answers should be easy to choose from, but should not feel like a forced list of options.

Usually provide 4–6 substantive example answers, plus “Other” and “Skip.”

Use “None of these” when appropriate.

# 5. Overall interview flow

Do not ask every question in the guide automatically.

Use the smallest set of questions needed to cover:

1. who the participant communicates with,
2. what they do when someone does not understand them,
3. what is hardest,
4. what other people can do that helps,
5. reaction to the demo,
6. what, if anything, seems useful,
7. what, if anything, seems not useful or concerning,
8. easiest way to correct or clarify if the system guesses wrong,
9. final advice for designers,
10. anything important not asked.

Default path:
A1, A2, A3, A4, DemoConsent, B1, B2-useful, B2-concern, B3, B4, C1.

Low-burden path:
A1, A2, A3, DemoConsent, B1, B2-useful, B2-concern, B4, C1.

If the demo is skipped:
A1, A2, A3, A4 if burden allows, DemoConsent, B4-general, C1.

Target length:

* Default path: 8–10 main questions total, including closing.
* Low-burden path: 6–8 main questions total, including closing.
* Follow-ups: 0–2 total preferred, 3 maximum.

The default action is to move forward to the next useful main question, unless the participant’s message requires a support response or clarification first.

# 6. Interview guide

Use this guide flexibly. Skip questions that have already been answered. Ask optional questions only when burden is low and the topic has not already been covered.

## A1. People they communicate with

Main question:
“Who do you communicate with most often?”

Example answers:

* Family
* Friends
* Caregivers or support workers
* Doctors or health workers
* People at work or school
* Store or service workers
* Other
* Skip

Research purpose:
Understand the participant’s everyday communication context.

Possible follow-up only if truly useful and burden is low:
“Who is easiest to communicate with?”

Follow-up example answers:

* Family
* Friends
* Caregivers or support workers
* Doctors or health workers
* People who know me well
* No one is easy
* Other
* Skip

## A2. Current ways of helping someone understand

Main question:
“When someone does not understand you, what do you usually do?”

Example answers:

* Say it again
* Say it differently
* Gesture or point
* Type or write
* Use AAC, sign, or another device
* Ask someone else to help
* Let it go
* Other
* Skip

Research purpose:
Learn the participant’s own communication strategies without assuming that repeating speech is the main strategy.

Possible follow-up only if truly useful and burden is low:
“Do you usually use one way, or more than one?”

Follow-up example answers:

* One way
* More than one
* It depends
* I am not sure
* Other
* Skip

## A3. What is hardest

Main question:
“What is usually hardest when someone does not understand you?”

Example answers:

* Repeating myself
* Saying it another way
* Typing or using a device
* Feeling rushed
* The other person gets impatient
* Losing what I wanted to say
* Nothing is especially hard
* Other
* Skip

Research purpose:
Identify burdens that a new technology should reduce, not add to.

Possible follow-up only if truly useful and burden is low:
“Which one is hardest?”

Follow-up example answers:

* Repeating myself
* Saying it another way
* Typing or using a device
* Feeling rushed
* Other person gets impatient
* Losing my thought
* Other
* Skip

## A4. What other people can do that helps

Main question:
“What can other people do that helps you be understood?”

Example answers:

* Be patient
* Wait longer
* Ask yes/no questions
* Guess from context
* Watch my gestures
* Read what I type or show
* Move to a quieter place
* Nothing helps much
* Other
* Skip

Research purpose:
Understand listener-side and environment-side supports.

Possible follow-up only if truly useful and burden is low:
“What is most helpful?”

Follow-up example answers:

* Patience
* Waiting
* Yes/no questions
* Guessing from context
* Watching gestures
* Reading what I type or show
* Other
* Skip

## A5. Optional: When it is harder

Ask only if this has not already been covered and participant burden is low.

Main question:
“When is it harder for people to understand you?”

Example answers:

* With strangers
* In noisy places
* When people are rushed
* When I am tired
* On the phone or video call
* In groups
* It is about the same
* Other
* Skip

Research purpose:
Understand variation by listener, setting, fatigue, urgency, and communication channel.

## A6. Optional: Desired support before demo

Ask only if there is room before the demo and the participant has not already expressed this need.

Main question:
“What help would matter most in conversation?”

Example answers:

* Less repeating
* Easier typing
* Word choices
* Saved messages
* Help for the other person
* Help in noisy places
* I do not want technology help
* I am not sure
* Other
* Skip

Research purpose:
Elicit participant-centered needs before showing the prototype.

Do not ask a follow-up unless the answer is unclear and important.

# 7. Demo handling

Do not assume the participant has seen the demo just because they agreed to watch it.

Use `DEMO_STATUS` and interview history.

DEMO_STATUS values:

* `not_shown`: the demo has not been shown yet.
* `permission_requested`: the chatbot has asked whether the participant is ready to watch.
* `ready_to_show`: the participant has agreed and the interface should show the demo next.
* `shown`: the participant has watched the demo.
* `skipped`: the participant skipped the demo or was not sure.

If the next step is the demo and `DEMO_STATUS` is `not_shown`, ask:

Question ID:
DemoConsent

Message:
“Next, we would like to show a short demo video of an early idea. Is now an okay time to watch it?”

Example answers:

* Yes
* Skip the demo
* I’m not sure

Question type:
transition

If the participant answers “Yes” to DemoConsent and the demo has not yet been shown, return a transition output with:

* `question_id: "DemoShow"`
* `question_type: "transition"`
* `state_update.demo_action: "show_demo"`

Message:
“Great — please watch the short demo now. After that, we will ask a few questions.”

Example answers:

* Done
* Skip
* I need help

Do not ask B1 until `DEMO_STATUS` is `shown`.

If the participant selects “Skip the demo” or “I’m not sure,” set `state_update.demo_action: "skip_demo"` and skip Section B reaction questions. Move to B4-general, then C1.

# 8. Reaction to demo

Ask this section only after `DEMO_STATUS` is `shown`.

## B1. First reaction

Main question:
“What is your first reaction to the demo?”

Example answers:

* I like it
* I partly like it
* I do not like it
* Interesting, but I am not sure
* Seems too much work
* Not useful for me
* Other
* Skip

Research purpose:
Capture initial reaction without assuming the idea is good.

After B1, do not branch based only on whether the participant likes or dislikes the idea.

A participant who likes the idea may still have concerns. A participant who dislikes the idea may still see one useful part.

Therefore, after B1, ask B2-useful first, then B2-concern.

Do not ask “why?” as a follow-up. B2-useful and B2-concern are enough.

## B2-useful. What seems useful, if anything

Main question:
“What, if anything, seems useful?”

Example answers:

* Transcript
* Word choices
* Less repeating
* Helps the other person
* Gives me control
* Could save time
* Nothing seems useful
* I am not sure
* Other
* Skip

Research purpose:
Understand possible perceived benefits without forcing a positive reaction.

Possible follow-up only if truly useful and burden is low:
“Which part seems most useful?”

Follow-up example answers:

* Transcript
* Word choices
* Less repeating
* Helps the other person
* Control
* Saving time
* None
* Other
* Skip

## B2-concern. What seems not useful or concerning, if anything

Main question:
“What, if anything, seems not useful or concerning?”

Example answers:

* Too slow
* Too much effort
* Transcript may be wrong
* Hard to choose options
* Typing is hard
* Other person may not wait
* I have better ways now
* Nothing concerns me
* I am not sure
* Other
* Skip

Research purpose:
Understand concerns, disliked parts, and possible barriers without assuming the participant dislikes the idea.

Possible follow-up only if truly useful and burden is low:
“Which concern matters most?”

Follow-up example answers:

* Too slow
* Too much effort
* Wrong transcript
* Hard to choose
* Typing is hard
* Other person may not wait
* None
* Other
* Skip

## B3. Easiest correction or clarification

Ask after B2-concern unless the participant is showing high burden. If burden is high, skip to B4.

Main question:
“If the system guessed wrong, what would be easiest?”

Example answers:

* Pick the right word
* Pick from a few choices
* Tap the wrong word
* Type a short fix
* Use a saved phrase
* Gesture or point
* Let the other person help
* Do not fix it
* Other
* Skip

Research purpose:
Identify low-effort correction options without assuming that speaking again, typing, or detailed editing is easy.

Possible follow-up only if truly useful and burden is low:
“Which would take the least effort?”

Follow-up example answers:

* Pick the right word
* Pick from choices
* Tap the wrong word
* Type a short fix
* Use a saved phrase
* Gesture or point
* Other person helps
* Skip

## B4. What designers should remember

Main question after demo:
“What should the people making this remember?”

Example answers:

* Keep it low effort
* Do not assume typing is easy
* Do not assume speaking again works
* Support gesture, AAC, or sign
* Make it work in real conversations
* Let the other person help
* Give me control
* Other
* Skip

Research purpose:
Elicit participant-centered design implications.

Possible follow-up only if truly useful and burden is low:
“What is most important?”

Follow-up example answers:

* Low effort
* Typing is not easy
* Speaking again may not work
* Support gesture, AAC, or sign
* Real conversations
* Other person can help
* Control
* Skip

## B4-general. If demo is skipped

If the demo is skipped, do not ask reaction questions about the demo.

Instead ask:

Main question:
“What should people making communication technology remember?”

Example answers:

* Keep it low effort
* Do not assume typing is easy
* Do not assume speaking again works
* Support gesture, AAC, or sign
* Make it work in real conversations
* Let the other person help
* Give me control
* Other
* Skip

Question type:
main

Then move to C1.

# 9. Closing

## C1. Anything missing

Main question:
“Is there anything important we did not ask?”

Example answers:

* Yes
* No
* I’m not sure
* Other
* Skip

Research purpose:
Allow participant-led concerns or insights not anticipated by the guide.

Possible follow-up only if the participant selects “Yes” or clearly indicates there is something else:
“What else should we know?”

Follow-up example answers:

* Something about communication
* Something about the technology
* Something about access or effort
* Something about privacy
* Something else
* Skip

After C1 is answered, close the interview.

Closing message:
“Thank you. Your answers are very helpful.”

Question type:
closing

# 10. Follow-up rules

A follow-up is any question that asks for more detail about the participant’s immediately previous answer.

The default is no follow-up.

Ask a follow-up only if all of the following are true:

1. The answer raises an important design-relevant issue, or clarification is necessary.
2. The answer cannot be adequately captured without one more question.
3. No follow-up has already been asked after the current main question.
4. Fewer than 3 follow-ups have been asked in the whole interview.
5. The participant does not appear tired, frustrated, or burdened.

Never ask more than one follow-up after the same main question.

Never ask two follow-ups in a row.

Never ask a follow-up only because the answer is short.

If choosing between a follow-up and the next main question, choose the next main question.

Clarifications and support turns do not count as follow-ups, but they should still be brief and limited.

B2-useful and B2-concern are main questions, not follow-ups. They should both be asked after B1 unless participant burden is very high or the topic was already clearly answered.

# 11. Managing participant burden

Watch for signs that the participant may want a lower-burden interview:

* very short answers,
* repeated skips,
* “I don’t know,”
* frustration,
* long pauses,
* difficulty typing,
* difficulty using speech recognition,
* comments about being tired,
* repeated use of example answers only,
* unclear or incomplete text that suggests effort.

When burden seems high:

* switch to the low-burden path,
* ask no follow-ups unless absolutely necessary,
* use simpler wording,
* move toward the demo or closing,
* do not say the participant is doing badly.

If the participant gives two very short answers in a row, skips once, says “I don’t know,” or burden notes suggest fatigue, switch to the low-burden path unless the interview is already near closing.

If burden is very high after B1, still try to ask both B2-useful and B2-concern because they capture different information. However, keep them short and do not ask follow-ups.

# 12. Acknowledgment and clarification

Begin `message_to_participant` with a brief natural acknowledgment when appropriate.

Examples:

* “Got it — family and friends.”
* “Thanks — typing can be hard.”
* “No problem, we can skip that.”

If the participant skipped, say “No problem” and move on.

If the participant says “I don’t know,” say “That’s okay” and move on.

If there is no previous participant answer, do not invent an acknowledgment.

Participants may use abbreviations, shorthand, partial words, or idiosyncratic phrasing.

Clarify only when the participant appears to be trying to answer and the meaning is uncertain and important for recording the answer or choosing the next question.

Do not clarify obvious shorthand when the meaning is clear from context.

Examples that usually do not need clarification:

* “fam” = family
* “frnds” = friends
* “doc” = doctor
* “ppl” = people, if context is clear
* “typing hard” = typing is hard

Examples that may need clarification:

* an unfamiliar abbreviation,
* a word that could refer to multiple communication methods,
* a phrase that changes which question should be asked next,
* a statement where the meaning is unclear and important.

When clarification is needed:

* ask only the clarification,
* do not ask the next interview question in the same message,
* use `question_type: "clarification"`,
* use the same `question_id` as the question being clarified, with suffix `_clarification`,
* `example_answers_if_requested` must be exactly:

  * “Yes”
  * “No, I meant something else”

Clarification message format:
“It sounds like you mean [brief interpretation]. Is that right?”

Ask at most one clarification for the same main question. After the participant confirms, corrects, or remains unclear, record the best available interpretation or mark the answer as uncertain, then move forward.

# 13. Avoiding repetition

Use the interview history to avoid asking about topics already answered.

If the participant already answered a later topic, mark that topic as covered and move to the next useful topic.

If a planned question would repeat information already given, skip it.

If the participant gives an answer that covers several topics, do not ask those topics again unless clarification is important.

For Section B:

* Do not skip B2-useful only because the participant disliked the demo.
* Do not skip B2-concern only because the participant liked the demo.
* Skip either question only if the participant has already clearly answered that exact topic.

# 14. Opening message

The opening message has already been displayed by the interface before the interview started.

Do not repeat it.

The participant has already been told:

* this is not a test,
* there are no right or wrong answers,
* short answers are fine,
* they can skip any question,
* they can answer by speaking, typing, choosing example answers, or using a mix,
* they can press the example answers button to see possible answers.

# 15. Runtime inputs

The system provides these inputs each turn.

INTERVIEW_HISTORY:
A compact record of the interview so far. Include:

* question IDs already asked,
* message shown to participant,
* selected example answers,
* typed or spoken free text,
* whether the question was main, follow-up, clarification, transition, support, or closing.

Example:
[
{
"question_id": "A2",
"question_type": "main",
"message_to_participant": "When someone does not understand you, what do you usually do?",
"participant_response": {
"selected_example_answers": ["Gesture or point"],
"free_text": "sometimes type"
}
}
]

INTERVIEW_STATE:
A compact state object. If not provided, infer conservatively from INTERVIEW_HISTORY.

Recommended fields:
{
"covered_topics": [],
"total_main_questions_asked": 0,
"total_followups_asked": 0,
"last_main_question_id": null,
"followup_asked_after_last_main": false,
"clarification_asked_after_last_main": false,
"unclear_recovery_asked_after_last_main": false,
"burden_level": "low | medium | high | unknown",
"path": "default | low_burden",
"next_recommended_question_id": null
}

DEMO_STATUS:
One of:

* "not_shown"
* "permission_requested"
* "ready_to_show"
* "shown"
* "skipped"

PARTICIPANT_BURDEN_NOTES:
Any observed signs of burden, fatigue, frustration, slow typing, repeated skipping, preference for example answers, difficulty using speech recognition, or other access needs.

# 16. Task each turn

Generate the next interview message.

First check whether the previous assistant message had `question_type: "support"` and whether the participant’s most recent message is a support choice. If yes, execute that support choice. If not, classify the participant’s most recent message using `participant_message_type`.

Then decide whether to:

1. continue with the next interview question,
2. ask a clarification,
3. provide a support response,
4. transition to the demo,
5. skip ahead,
6. or close the interview.

Use the participant’s previous answers to avoid repetition.

Prefer moving forward over asking for more detail, but participant needs override interview progress.

Choose the next question based on:

1. whether a support choice should be executed,
2. participant message type,
3. interview history,
4. covered topics,
5. participant burden,
6. follow-up limits,
7. clarification and unclear-input recovery limits,
8. demo status,
9. Section B sequencing rules.

If the last answer was an attempted but unclear answer and clarification is important, ask one brief clarification.

If the last input was clearly unusable, ask one low-effort recovery question or skip the current question if recovery has already been tried.

If the participant has a process question, burden signal, access problem, refusal, or request to stop, address that before continuing.

Return only JSON.

# 17. Output format

Use this format:

{
"question_id": "...",
"message_to_participant": "...",
"example_answers_if_requested": [
{"label": "..."}
],
"question_type": "main | follow_up | clarification | transition | support | closing",
"participant_message_type": "usable_answer | attempted_unclear_answer | clearly_unusable_input | skip_request | process_question | burden_signal | frustration_or_refusal | access_problem | unknown",
"state_update": {
"mark_covered": [],
"followup_used": false,
"clarification_used": false,
"unclear_recovery_used": false,
"demo_action": "none | show_demo | skip_demo",
"path": "default | low_burden",
"recommended_next": null
}
}

# 18. Output rules

Return only valid JSON.

Do not include internal reasoning.

Do not show question IDs such as “A1” or “B2-useful” to the participant.

Question IDs and section labels are internal metadata only. Never include labels such as A1, A2, B1, B2-useful, B2-concern, C1, DemoConsent, or DemoShow in `message_to_participant`; use natural wording such as “this question,” “the demo,” or simply ask the next question directly.

The participant should see only `message_to_participant`.

`example_answers_if_requested` must not be empty.

Do not include example answers inside `message_to_participant` unless the interface explicitly asks for visible example answers.

For support:

* use `question_type: "support"`,
* address the participant’s immediate need,
* do not ask a normal interview question in the same message unless the participant clearly asked to continue,
* use support-specific example answers when offering choices,
* usually set `state_update.path` to `"low_burden"` when the support turn is caused by burden, frustration, access difficulty, or repeated clearly unusable input.

For clarification:

* use `question_type: "clarification"`,
* do not ask the next interview question in the same message,
* `example_answers_if_requested` must be exactly:
  [
  {"label": "Yes"},
  {"label": "No, I meant something else"}
  ]

For demo show transition:

* use `question_id: "DemoShow"`,
* use `question_type: "transition"`,
* set `state_update.demo_action: "show_demo"`,
* do not ask a reaction question yet.

For skipped demo:

* set `state_update.demo_action: "skip_demo"`,
* skip Section B reaction questions,
* move to B4-general.

For closing:

* after the participant answers C1 or its one allowed follow-up, output the closing message:
  “Thank you. Your answers are very helpful.”
* use `question_type: "closing"`.

# 19. Decision defaults

If the participant responds to a support message with a support choice:

* execute the choice directly,
* do not reclassify it as a new interview answer or process question,
* do not ask another support question about the same issue.

If the participant gives a clear short answer:

* acknowledge it briefly,
* move to the next main question.

If the participant gives a long or rich answer:

* mark any covered topics,
* ask at most one follow-up only if it is important and allowed,
* otherwise move forward.

If the participant gives an attempted but unclear answer:

* ask one brief clarification if needed,
* then record the best available interpretation or mark uncertain and move forward.

If the participant gives clearly unusable input:

* ask one low-effort recovery question,
* if clearly unusable input repeats, skip the current question and move forward on the low-burden path.

If the participant skips:

* say “No problem,”
* move forward.

If the participant says “I don’t know”:

* say “That’s okay,”
* move forward.

If the participant seems tired or burdened:

* switch to the low-burden path,
* avoid follow-ups,
* move toward the demo or closing.

If the participant asks a process question:

* answer it briefly,
* offer a simple next-step choice,
* do not continue with the next research question in the same message unless the participant clearly asked to continue.

If the participant asks to stop:

* stop the interview and close politely.

If the participant asks for example answers:

* keep the same question if appropriate,
* provide example answers through `example_answers_if_requested`,
* do not treat asking for example answers as a failure to answer.

If the participant answers a later topic early:

* mark that topic as covered,
* do not repeat it.

If choosing between a follow-up and the next main question:

* choose the next main question.

For Section B:

* After B1, ask B2-useful.
* After B2-useful, ask B2-concern.
* After B2-concern, ask B3 unless burden is high.
* Do not treat B2-useful or B2-concern as follow-ups.
* Do not infer that liking the demo means there are no concerns.
* Do not infer that disliking the demo means there are no useful parts.
"""


# =============================================================================
# OpenAI helpers
# =============================================================================

def _call_llm_json(system_prompt, user_prompt, label="agent"):
    """Call the LLM and return a parsed JSON dict. Appends raw log to session state."""
    raw_text = None
    try:
        resp = _openai_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        raw_text = resp.choices[0].message.content
        result = json.loads(raw_text)
    except Exception:
        resp = _openai_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw_text = resp.choices[0].message.content or ""
        m = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if m:
            result = json.loads(m.group())
        else:
            raise ValueError(f"LLM did not return valid JSON. Raw: {raw_text[:300]}")

    if "agent_logs" in st.session_state:
        st.session_state.agent_logs.append({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "label": label,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": raw_text,
            "parsed_response": result,
        })

    return result


def _format_chat_for_prompt(chat):
    lines = []
    for msg in chat:
        if msg.get("role") == "video":
            lines.append("[Demo video was shown to the participant here]")
        elif msg["role"] == "assistant":
            lines.append(f"Interviewer: {msg['content']}")
            if msg.get("options"):
                opts = " | ".join(o["label"] for o in msg["options"])
                lines.append(f"[Options shown: {opts}]")
        elif msg["role"] == "user":
            lines.append(f"Participant: {msg['content']}")
    return "\n".join(lines) if lines else "(No conversation yet)"


# =============================================================================
# Agent turn
# =============================================================================

def _build_interview_history(chat):
    """Build the compact INTERVIEW_HISTORY list the single agent expects."""
    history = []
    i = 0
    msgs = [m for m in chat if m.get("role") in ("assistant", "user", "video")]
    while i < len(msgs):
        msg = msgs[i]
        if msg.get("role") == "assistant":
            entry = {
                "question_id": msg.get("question_id", ""),
                "question_type": msg.get("question_type", ""),
                "message_to_participant": msg.get("content", ""),
                "participant_response": None,
            }
            # Look for the immediately following user message
            if i + 1 < len(msgs) and msgs[i + 1].get("role") == "user":
                i += 1
                user_msg = msgs[i]
                # Read structured fields stored at submission time
                selected = user_msg.get("selected_suggestions", [])
                free = user_msg.get("free_text", user_msg.get("content", ""))
                entry["participant_response"] = {
                    "selected_example_answers": selected,
                    "free_text": free,
                }
            history.append(entry)
        i += 1
    return history


# Regex to detect internal question IDs leaked into participant-facing text
_QID_PATTERN = re.compile(
    r'\b(A[1-4]|B[1-4](?:-useful|-concern|-general)?|C1|DemoConsent|DemoShow)\b'
)


_A1_RESULT = {
    "question_id": "A1",
    "message_to_participant": "Who do you communicate with most often?",
    "example_answers_if_requested": [
        {"label": "Family"},
        {"label": "Friends"},
        {"label": "Caregivers or support workers"},
        {"label": "Doctors or health workers"},
        {"label": "People at work or school"},
        {"label": "Store or service workers"},
        {"label": "Other"},
        {"label": "Skip"},
    ],
    "question_type": "main",
}


def run_agent_turn():
    chat = st.session_state.chat
    demo_status = st.session_state.get("demo_status", "not_shown")

    # First question is always A1 — no LLM call needed
    if not any(m.get("role") == "user" for m in chat):
        result = _A1_RESULT.copy()
    else:
        # Count skips / short answers for burden notes
        skip_count = sum(
            1 for m in chat
            if m.get("role") == "user" and m.get("content", "").lower().strip() in ("skip", "i don't know", "")
        )
        burden_notes = f"{skip_count} skip(s) or empty answers so far." if skip_count else "No signs of high burden observed."

        history = _build_interview_history(chat)
        interview_state = st.session_state.get("interview_state", {})
        user_prompt = (
            f"INTERVIEW_HISTORY:\n{json.dumps(history, indent=2)}\n\n"
            f"INTERVIEW_STATE:\n{json.dumps(interview_state, indent=2)}\n\n"
            f"DEMO_STATUS:\n{demo_status}\n\n"
            f"PARTICIPANT_BURDEN_NOTES:\n{burden_notes}"
        )

        result = _call_llm_json(_AGENT_SYSTEM, user_prompt, label="agent")

        # Fallback: retry if message_to_participant leaks an internal question ID
        for _retry in range(2):
            leaked = _QID_PATTERN.findall(result.get("message_to_participant", ""))
            if not leaked:
                break
            retry_prompt = (
                user_prompt
                + f"\n\nCORRECTION REQUIRED: Your previous response included the internal ID(s) {leaked} "
                "inside `message_to_participant`. Question IDs must never appear in the participant-facing message. "
                "Rewrite `message_to_participant` using natural wording only (e.g. 'this question', 'the demo', "
                "or ask directly). Return the full corrected JSON."
            )
            result = _call_llm_json(_AGENT_SYSTEM, retry_prompt, label="agent_retry")

        # Fallback: retry if example_answers_if_requested is missing or empty
        for _retry in range(2):
            if result.get("example_answers_if_requested"):
                break
            retry_prompt = (
                user_prompt
                + "\n\nCORRECTION REQUIRED: Your previous response had an empty or missing "
                "`example_answers_if_requested`. This field must always contain at least one option. "
                "Return the full corrected JSON with a non-empty `example_answers_if_requested` list."
            )
            result = _call_llm_json(_AGENT_SYSTEM, retry_prompt, label="agent_retry")

    # Detect end-of-interview — closing type always ends after showing the message
    q_type = result.get("question_type", "")
    q_id = result.get("question_id", "")
    if q_type == "closing":
        # Show the closing message if present, then mark ended on next rerun
        if not result.get("message_to_participant", "").strip():
            st.session_state.interview_ended = True
            return False, None

    # Demo action from state_update (replaces old heuristic)
    demo_action = result.get("state_update", {}).get("demo_action", "none")
    show_video = False
    if demo_action == "show_demo" and demo_status != "shown":
        show_video = True
        st.session_state.demo_status = "shown"
    elif demo_action == "skip_demo":
        st.session_state.demo_status = "skipped"

    # Persist state_update fields into interview_state
    istate = st.session_state.get("interview_state", {})
    su = result.get("state_update", {})
    for topic in su.get("mark_covered", []):
        if topic not in istate.get("covered_topics", []):
            istate.setdefault("covered_topics", []).append(topic)
    if su.get("followup_used"):
        istate["total_followups_asked"] = istate.get("total_followups_asked", 0) + 1
        istate["followup_asked_after_last_main"] = True
    if su.get("clarification_used"):
        istate["clarification_asked_after_last_main"] = True
    if su.get("unclear_recovery_used"):
        istate["unclear_recovery_asked_after_last_main"] = True
    if su.get("path"):
        istate["path"] = su["path"]
        if su["path"] == "low_burden":
            istate["burden_level"] = "high"
    if su.get("recommended_next") is not None:
        istate["next_recommended_question_id"] = su["recommended_next"]
    q_type = result.get("question_type", "")
    if q_type == "main":
        istate["total_main_questions_asked"] = istate.get("total_main_questions_asked", 0) + 1
        istate["last_main_question_id"] = result.get("question_id")
        istate["followup_asked_after_last_main"] = False
        istate["clarification_asked_after_last_main"] = False
        istate["unclear_recovery_asked_after_last_main"] = False
    st.session_state.interview_state = istate

    # Normalise output to the fields the UI expects
    result["question_text"] = result.get("message_to_participant", "")
    options = result.get("example_answers_if_requested", [])
    # Fallback: if it's a yes/no question with no options, provide them
    question_lower = result["question_text"].lower()
    if not options and any(phrase in question_lower for phrase in ("okay time", "is now", "would you like", "are you ready")):
        options = [
            {"label": "Yes"},
            {"label": "No"},
            {"label": "Maybe later"},
        ]
    result["options"] = options
    result["answer_mode"] = "multiple_choice"

    return show_video, result


# =============================================================================
# Google Drive helpers
# =============================================================================

def _get_drive_config():
    try:
        return {
            "folder_id": st.secrets.get("GDRIVE_FOLDER_ID", ""),
            "client_id": st.secrets.get("GDRIVE_CLIENT_ID", ""),
            "client_secret": st.secrets.get("GDRIVE_CLIENT_SECRET", ""),
            "refresh_token": st.secrets.get("GDRIVE_REFRESH_TOKEN", ""),
        }
    except Exception:
        return {"folder_id": "", "client_id": "", "client_secret": "", "refresh_token": ""}


def _make_service(config):
    from googleapiclient.discovery import build
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    creds = Credentials(
        token=None,
        refresh_token=config["refresh_token"],
        client_id=config["client_id"],
        client_secret=config["client_secret"],
        token_uri="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    creds.refresh(Request())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _get_or_create_folder(name, parent_id, svc):
    q = (
        f"name='{name}' and '{parent_id}' in parents "
        "and mimeType='application/vnd.google-apps.folder' and trashed=false"
    )
    results = svc.files().list(q=q, fields="files(id)").execute().get("files", [])
    if results:
        return results[0]["id"]
    return svc.files().create(
        body={"name": name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent_id]},
        fields="id",
    ).execute()["id"]


def _upsert_bytes(name, data, folder_id, svc):
    from googleapiclient.http import MediaIoBaseUpload
    q = f"name='{name}' and '{folder_id}' in parents and trashed=false"
    existing = svc.files().list(q=q, fields="files(id)").execute().get("files", [])
    media = MediaIoBaseUpload(io.BytesIO(data), mimetype="application/octet-stream")
    if existing:
        svc.files().update(fileId=existing[0]["id"], media_body=media).execute()
    else:
        svc.files().create(
            body={"name": name, "parents": [folder_id]},
            media_body=media,
        ).execute()


def _download_bytes(file_id, svc):
    from googleapiclient.http import MediaIoBaseDownload
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, svc.files().get_media(fileId=file_id))
    done = False
    while not done:
        _, done = dl.next_chunk()
    return buf.getvalue()


def _update_participants_log(user_id, root_folder_id, svc):
    try:
        q = f"name='participants_log.json' and '{root_folder_id}' in parents and trashed=false"
        existing = svc.files().list(q=q, fields="files(id)").execute().get("files", [])
        if existing:
            data = json.loads(_download_bytes(existing[0]["id"], svc).decode("utf-8"))
        else:
            data = {}
        if user_id not in data:
            data[user_id] = {
                "first_seen": datetime.utcnow().isoformat() + "Z",
                "last_seen": datetime.utcnow().isoformat() + "Z",
                "status": "in_progress",
                "turns": 0,
            }
        else:
            data[user_id]["last_seen"] = datetime.utcnow().isoformat() + "Z"
            data[user_id]["turns"] = data[user_id].get("turns", 0) + 1
        _upsert_bytes(
            "participants_log.json",
            json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
            root_folder_id, svc,
        )
    except Exception:
        pass


def _do_save(user_id, chat, agent_logs, config):
    if not config.get("folder_id") or not config.get("refresh_token"):
        missing = [k for k in ("folder_id", "refresh_token") if not config.get(k)]
        raise RuntimeError(f"Drive not configured — missing secrets: {', '.join(missing)}")
    svc = _make_service(config)
    root = config["folder_id"]
    pfolder = _get_or_create_folder(f"participant_{user_id}", root, svc)
    _upsert_bytes(
        "chat_history.json",
        json.dumps(chat, ensure_ascii=False, indent=2).encode("utf-8"),
        pfolder, svc,
    )
    if agent_logs:
        _upsert_bytes(
            "agent_logs.json",
            json.dumps(agent_logs, ensure_ascii=False, indent=2).encode("utf-8"),
            pfolder, svc,
        )
    _update_participants_log(user_id, root, svc)
    return True, "Saved."


_drive_errors = []

def save_async(user_id, chat, agent_logs, config):
    def _run():
        try:
            _do_save(user_id, chat, agent_logs, config)
        except Exception as e:
            _drive_errors.append(str(e))
    threading.Thread(target=_run, daemon=True).start()


def save_sync(user_id, chat, agent_logs, config):
    try:
        return _do_save(user_id, chat, agent_logs, config)
    except Exception as e:
        return False, str(e)


def _save_audio_async(user_id, question_id, audio_bytes, transcript, config):
    """Save a .webm recording and update audio_log.json in Drive (async)."""
    def _run():
        try:
            if not config.get("folder_id") or not config.get("refresh_token"):
                return
            svc = _make_service(config)
            root = config["folder_id"]
            pfolder = _get_or_create_folder(f"participant_{user_id}", root, svc)
            afolder = _get_or_create_folder("audio", pfolder, svc)

            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            filename = f"{question_id}_{ts}.webm"

            # Upload the audio file
            from googleapiclient.http import MediaIoBaseUpload
            media = MediaIoBaseUpload(
                io.BytesIO(audio_bytes), mimetype="audio/webm"
            )
            svc.files().create(
                body={"name": filename, "parents": [afolder]},
                media_body=media,
            ).execute()

            # Update audio_log.json
            q = f"name='audio_log.json' and '{pfolder}' in parents and trashed=false"
            existing = svc.files().list(q=q, fields="files(id)").execute().get("files", [])
            log = json.loads(_download_bytes(existing[0]["id"], svc).decode("utf-8")) if existing else []
            log.append({
                "timestamp": ts,
                "question_id": question_id,
                "filename": filename,
                "transcript": transcript,
            })
            _upsert_bytes(
                "audio_log.json",
                json.dumps(log, ensure_ascii=False, indent=2).encode("utf-8"),
                pfolder, svc,
            )
        except Exception as e:
            _drive_errors.append(f"Audio save error: {e}")
    threading.Thread(target=_run, daemon=True).start()


def restore_from_drive(participant_id, config):
    try:
        if not config.get("folder_id") or not config.get("refresh_token"):
            return [], False
        svc = _make_service(config)
        root = config["folder_id"]
        q = (
            f"name='participant_{participant_id}' and '{root}' in parents "
            "and mimeType='application/vnd.google-apps.folder' and trashed=false"
        )
        folders = svc.files().list(q=q, fields="files(id)").execute().get("files", [])
        if not folders:
            return [], False
        pfolder = folders[0]["id"]
        files = {
            f["name"]: f["id"]
            for f in svc.files().list(
                q=f"'{pfolder}' in parents and trashed=false",
                fields="files(id, name)",
            ).execute().get("files", [])
        }
        chat = []
        if "chat_history.json" in files:
            chat = json.loads(_download_bytes(files["chat_history.json"], svc).decode("utf-8"))
        return chat, bool(chat)
    except Exception:
        return [], False



# =============================================================================
# Demo video
# =============================================================================

@st.cache_data(show_spinner=False)
def _load_demo_video_bytes():
    try:
        from googleapiclient.http import MediaIoBaseDownload
        config = _get_drive_config()
        service = _make_service(config)
        file_id = "1FCfzZslMnuyQAPhcZoiACrx0sWaYskxV"
        request = service.files().get_media(fileId=file_id)
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, request, chunksize=8 * 1024 * 1024)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return buf.getvalue()
    except Exception:
        return None


# =============================================================================
# Whisper transcription
# =============================================================================

def _transcribe(audio_bytes):
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=("recording.wav", io.BytesIO(audio_bytes), "audio/wav"),
        )
        return result.text.strip()
    except Exception:
        return ""


# =============================================================================
# Page setup & state init
# =============================================================================

st.set_page_config(page_title="Interview", page_icon="mic", layout="wide")
st.title("Interview")

if "phase" not in st.session_state:
    st.session_state.update(
        phase="id_entry",
        user_id=None,
        chat=[],
        waiting=False,
        drive_config=None,
        session_saved=False,
        last_audio_hash=None,
        user_draft="",
        demo_status="not_shown",
        interview_ended=False,
        form_generation=0,
        agent_logs=[],
        interview_state={
            "covered_topics": [],
            "total_main_questions_asked": 0,
            "total_followups_asked": 0,
            "last_main_question_id": None,
            "followup_asked_after_last_main": False,
            "clarification_asked_after_last_main": False,
            "unclear_recovery_asked_after_last_main": False,
            "burden_level": "unknown",
            "path": "default",
            "next_recommended_question_id": None,
        },
    )


st.markdown("""
<style>
html, body, [class*="css"], .stMarkdown, .stChatMessage { font-size: 20px !important; }
div[data-testid="stChatMessage"] p { font-size: 1.05rem !important; line-height: 1.7 !important; }
div[data-testid="stTextArea"] textarea {
    min-height: 80px !important; font-size: 1.1rem !important;
    line-height: 1.7 !important; border-radius: 14px !important;
    padding: 14px 18px !important; resize: none !important;
}
div[data-testid="stButton"] button[kind="primary"],
div[data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"] {
    font-size: 1rem !important;
    border-radius: 8px !important; width: 100% !important;
}
[data-testid="stColumn"] div[data-testid="stButton"] button[kind="primary"],
[data-testid="stColumn"] div[data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"] {
    height: 100px !important;
}
/* Keep Speak + text area on the same line; pin mic column to fixed width */
div[data-testid="stHorizontalBlock"]:has(iframe),
div[data-testid="stColumns"]:has(iframe) {
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: stretch !important;
}
div[data-testid="stHorizontalBlock"]:has(iframe) > div[data-testid="stColumn"]:first-child,
div[data-testid="stColumns"]:has(iframe) > div[data-testid="stColumn"]:first-child {
    flex: 0 0 110px !important;
    min-width: 110px !important;
    max-width: 110px !important;
}
div[data-testid="stHorizontalBlock"]:has(iframe) > div[data-testid="stColumn"]:last-child,
div[data-testid="stColumns"]:has(iframe) > div[data-testid="stColumn"]:last-child {
    flex: 1 1 auto !important;
    min-width: 0 !important;
}
[data-testid="stColumn"] iframe {
    height: 100px !important; min-height: 100px !important;
    width: 100% !important;
}
/* Multiple Choice Options toggle button */
div[data-testid="stButton"] button[kind="secondary"] {
    height: 100px !important;
    width: 100% !important;
    font-size: 1rem !important;
    border-radius: 8px !important;
    background-color: #f0f4ff !important;
    color: #1a237e !important;
}
/* Option grid cards (overrides above due to higher specificity) */
div[data-testid="stColumn"] div[data-testid="stButton"] button[kind="secondary"] {
    min-height: 90px !important; height: auto !important;
    white-space: normal !important; word-break: break-word !important;
    border-radius: 12px !important; font-size: 1rem !important;
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Phase: participant ID entry
# =============================================================================

if st.session_state.phase == "id_entry":

    st.markdown("### Welcome")
    st.info(
        "After clicking **Start**, you will be given a **Participant ID**.  \n"
        "Please **write it down** -- you will need it to continue the interview "
        "later if you close the browser or need a break."
    )

    tab_new, tab_return = st.tabs(["New participant", "Returning participant"])

    with tab_new:
        st.markdown("Click the button to begin a new interview session.")
        if st.button("Start interview ->", type="primary", key="btn_new"):
            user_id = "P-" + uuid.uuid4().hex[:6].upper()
            cfg = _get_drive_config()
            st.session_state.update(user_id=user_id, drive_config=cfg, phase="intro")
            st.rerun()

    with tab_return:
        st.markdown("Enter the Participant ID you received when you started.")
        pid_input = st.text_input("Participant ID (e.g. P-ABC123):", key="pid_input")
        if st.button("Resume interview ->", key="btn_return"):
            pid = pid_input.strip().upper()
            if not pid:
                st.warning("Please enter your Participant ID.")
            else:
                cfg = _get_drive_config()
                with st.spinner(f"Looking up session for {pid}..."):
                    chat, found = restore_from_drive(pid, cfg)
                if found:
                    video_shown = any(m.get("role") == "video" for m in chat)
                    # If the session ended mid-turn (last message is from user,
                    # agent never responded), resume in waiting state so the
                    # agent fires immediately — this also handles the case where
                    # the participant answered "yes" to the demo consent question
                    # but the video was never shown.
                    last_role = chat[-1].get("role") if chat else None
                    resume_waiting = last_role == "user"
                    st.session_state.update(
                        user_id=pid, drive_config=cfg, chat=chat,
                        demo_status="shown" if video_shown else "not_shown",
                        waiting=resume_waiting, phase="active",
                    )
                    st.rerun()
                else:
                    st.error(
                        f"No session found for **{pid}**.  \n"
                        "Please double-check your ID and try again.  \n"
                        "If you have not started before, use the **New participant** tab."
                    )

    st.stop()


# =============================================================================
# Phase: intro
# =============================================================================

if st.session_state.phase == "intro":

    INTRO_TEXT = (
        "Thank you for meeting with us.\n\n"
        "We are interested in your everyday experiences communicating with other people, "
        "especially times when someone has trouble understanding you.\n\n"
        "Later, we will show you a short demo of an early technology idea and ask what you think about it.\n\n"
        "This is not a test of you. We are learning from your experience.\n\n"
        "There are no right or wrong answers. Short answers are fine. You can skip any question.\n\n"
        "You can answer by speaking, typing, choosing suggested answers, or using a mix of these.\n\n"
        "If helpful, you can press the suggestions button to see possible answers."
    )

    st.markdown(INTRO_TEXT)
    st.markdown("")

    if st.button("Continue to interview ->", type="primary", key="btn_intro_continue"):
        st.session_state.chat = []
        st.session_state.waiting = True
        st.session_state.phase = "active"
        st.rerun()

    st.stop()


# =============================================================================
# Phase: active interview
# =============================================================================

user_id = st.session_state.user_id
cfg = st.session_state.drive_config

if "_pending_draft" in st.session_state:
    st.session_state.user_draft = st.session_state.pop("_pending_draft")

with st.sidebar:
    st.markdown("### Your Participant ID")
    st.code(user_id, language=None)
    st.caption(
        "Keep this ID safe. If you need to leave and continue later, "
        "use the **Returning participant** tab on the start screen and enter this ID."
    )
    if _drive_errors:
        st.error(f"⚠️ Drive save error: {_drive_errors[-1]}")


# Render chat history
_first_assistant_seen = False
for msg in st.session_state.chat:
    if msg.get("role") == "video":
        st.markdown("#### Demo Video")
        st.markdown("<p style='font-size:18px; color:black;'>Please watch the short demo video below before answering the next question.</p>", unsafe_allow_html=True)
        _video_bytes = _load_demo_video_bytes()
        if _video_bytes:
            _, vid_col, _ = st.columns([1, 5, 1])
            with vid_col:
                st.video(_video_bytes, format="video/mp4")
        else:
            st.info("Video unavailable -- please ask the researcher to share the demo link.")
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.write(msg["content"])
            if not _first_assistant_seen:
                st.caption(
                    'You can respond in whatever way works best for you: '
                    'click "Speak" to use speech-to-text, '
                    'click the button below to see multiple-choice options and select one or more, '
                    'type your own answer, or use any combination of these.'
                )
                _first_assistant_seen = True
    elif msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])

# State machine
if st.session_state.waiting:
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                show_video, result = run_agent_turn()
            except Exception as e:
                st.session_state.waiting = False
                st.error(f"Something went wrong: {e}")
                st.stop()

    if show_video:
        st.session_state.chat.append({"role": "video"})
    if result:
        st.session_state.chat.append({
            "role": "assistant",
            "content": result["question_text"],
            "question_id": result.get("question_id", ""),
            "question_type": result.get("question_type", ""),
            "participant_message_type": result.get("participant_message_type", ""),
            "answer_mode": result.get("answer_mode", "multiple_choice"),
            "options": result.get("options", []),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })

    st.session_state.waiting = False
    save_async(user_id, st.session_state.chat, st.session_state.agent_logs, cfg)
    st.rerun()

elif st.session_state.get("interview_ended"):
    with st.chat_message("assistant"):
        st.write(CLOSING_MESSAGE)
    st.success("The interview has ended. Thank you for your time!")
    if not st.session_state.session_saved:
        with st.spinner("Saving your session to Google Drive..."):
            ok, save_msg = save_sync(user_id, st.session_state.chat, st.session_state.agent_logs, cfg)
        st.session_state.session_saved = True
        if ok:
            st.info(f"Session saved. Your Participant ID was **`{user_id}`**.")
        else:
            st.caption(f"(Note: auto-save encountered an issue: {save_msg})")

else:
    current_q_msg = None
    for msg in reversed(st.session_state.chat):
        if msg.get("role") == "assistant":
            current_q_msg = msg
            break

    gen = st.session_state.form_generation
    draft_key = f"user_draft_{gen}"

    # Apply any pending pre-fill (from audio transcription) before widgets render
    if "_prefill" in st.session_state:
        st.session_state[draft_key] = st.session_state.pop("_prefill")

    # ── Interactive options (metadata only — rendered below input row) ──────────
    answer_mode = current_q_msg.get("answer_mode", "multiple_choice") if current_q_msg else "multiple_choice"
    options = current_q_msg.get("options", []) if current_q_msg else []
    q_key = current_q_msg.get("question_id", "q") if current_q_msg else "q"

    # ── Speak | Text area | Send ────────────────────────────────────────────
    mic_col, right_col = st.columns([1, 11])

    with mic_col:
        audio = mic_recorder(
            start_prompt="🎤  Speak",
            stop_prompt="⏹️  Stop",
            just_once=True,
            use_container_width=True,
            key="mic",
        )

    with right_col:
        with st.form(key=f"response_form_{gen}", clear_on_submit=True):
            text_col, send_col = st.columns([9, 2])
            with text_col:
                typed = st.text_area(
                    "response",
                    key=draft_key,
                    height=100,
                    placeholder="Type your response here, or click 🎤 Speak to record...",
                    label_visibility="collapsed",
                )
            with send_col:
                send_clicked = st.form_submit_button("Send →", type="primary", use_container_width=True)

    # Enter key sends (Shift+Enter = newline)
    components.html("""
    <script>
    (function() {
        function attach() {
            var ta = window.parent.document.querySelector('textarea[aria-label="response"]');
            if (!ta || ta._enterBound) return;
            ta._enterBound = true;
            ta.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    var btns = window.parent.document.querySelectorAll('button');
                    for (var i = 0; i < btns.length; i++) {
                        if (btns[i].innerText.trim().startsWith('Send')) {
                            btns[i].click(); break;
                        }
                    }
                }
            });
        }
        attach();
        new MutationObserver(attach).observe(window.parent.document.body, {childList:true, subtree:true});
    })();
    </script>
    """, height=0)

    # ── Suggested answers (hidden until toggled) ──────────────────────────────────────────
    if options:
        show_key = f"show_opts_{gen}_{q_key}"
        if show_key not in st.session_state:
            st.session_state[show_key] = False

        if not st.session_state[show_key]:
            if st.button("Multiple Choice Options", key=f"show_opts_btn_{gen}_{q_key}"):
                st.session_state[show_key] = True
                st.rerun()
        else:
            if answer_mode in ("multiple_choice", "ranking"):
                grid_cols = st.columns(4)
                _opt_toggled = False
                for i, opt in enumerate(options):
                    sel_key = f"mopt_{gen}_{q_key}_{i}"
                    if sel_key not in st.session_state:
                        st.session_state[sel_key] = False
                    is_sel = st.session_state[sel_key]
                    label = f"\u2713  {opt['label']}" if is_sel else opt["label"]
                    with grid_cols[i % 4]:
                        if st.button(label, key=f"mbtn_{gen}_{q_key}_{i}",
                                     type="primary" if is_sel else "secondary",
                                     use_container_width=True):
                            st.session_state[sel_key] = not is_sel
                            _opt_toggled = True
                if _opt_toggled:
                    st.rerun()

            elif answer_mode == "yes_no_plus_optional_text":
                st.markdown("**Choose one (you can add details below):**")
                n_cols = min(3, len(options))
                cols = st.columns(n_cols)
                for i, opt in enumerate(options):
                    with cols[i % n_cols]:
                        if st.button(opt["label"], key=f"ynopt_{gen}_{q_key}_{i}",
                                     use_container_width=True):
                            st.session_state[draft_key] = opt["label"]

    if send_clicked:
        typed_text = (typed or st.session_state.get(draft_key) or "").strip()

        selected = []
        if answer_mode in ("multiple_choice", "ranking"):
            selected = [
                opt["label"]
                for i, opt in enumerate(options)
                if st.session_state.get(f"mopt_{gen}_{q_key}_{i}")
            ]

        parts = []
        if selected:
            parts.append("; ".join(selected))
        if typed_text:
            parts.append(typed_text)
        answer = ". ".join(parts) if parts else None

        if answer:
            st.session_state.form_generation += 1
            st.session_state.chat.append({
                "role": "user",
                "content": answer,
                "selected_suggestions": selected,
                "free_text": typed_text,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            })
            st.session_state.waiting = True
            st.rerun()
        else:
            st.warning("Please type a response or choose an option before sending.")

    elif audio:
        audio_bytes = audio["bytes"]
        audio_hash = hashlib.md5(audio_bytes).hexdigest()
        if audio_hash != st.session_state.last_audio_hash:
            st.session_state.last_audio_hash = audio_hash
            with st.spinner("Transcribing..."):
                transcript = _transcribe(audio_bytes)
            if transcript:
                st.session_state._prefill = transcript
                _save_audio_async(
                    user_id,
                    q_key,
                    audio_bytes,
                    transcript,
                    cfg,
                )
                st.rerun()
