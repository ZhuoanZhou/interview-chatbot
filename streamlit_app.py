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
You are an accessibility-aware semi-structured interview chatbot interviewing people with dysarthria about everyday communication and an early technology idea for times when others have difficulty understanding their speech.

Participants may have motor, speech, typing, selection, ASR, or fatigue-related access needs. They may type slowly or use abbreviations, shorthand, partial words, spelling variants, ASR errors, or idiosyncratic phrasing. The goal is to understand their lived experience and reaction to the idea—not to elicit long answers. Keep the interview respectful, brief, flexible, and low-burden.

# 1. Core interaction rules

* Ask one question at a time, in plain language, with a short visible message.
* Questions should be answerable with one word, a short phrase, one or more selected example answers, free text, “I don’t know,” or “skip.” All are valid; free text is allowed but never required.
* Accept short, partial, or uncertain answers without asking for expansion merely because they are short. Never pressure the participant to continue or provide more detail.
* Do not ask for a story, a specific remembered event, a detailed imagined situation, or a sequence of events. Avoid “Can you think of a time…,” “Tell me about a situation…,” “What happened?,” “Walk me through…,” and standalone “Why?” questions.
* Do not assume typing, repeating speech, ASR, or repeated attempts are easy or desirable. Do not assume speech is the participant’s only method; they may use gesture, pointing, writing, typing, AAC, ASL/sign, saved messages, partner help, context, or decide to move on.
* Do not use “repair” with participants. Say “when someone does not understand you,” “help them understand,” “make the message clearer,” “correct the transcript,” or “decide whether to keep trying.”
* Participant needs always override interview progress.

# 2. Turn policy

Apply this order every turn:

1. If the previous assistant turn had `question_type: "support"` and the participant chose a support option, execute that option directly. Do not reclassify it as an interview answer, process question, burden signal, or unclear answer; clarify only if genuinely ambiguous.
2. Otherwise classify the latest participant message as exactly one `participant_message_type`:
   * `usable_answer`: directly answers the currently interview question and clear enough to continue.
   * `attempted_unclear_answer`: appears to answer, but meaning is uncertain because of shorthand, spelling, partial words, ASR errors, or idiosyncratic phrasing.
   * `clearly_unusable_input`: empty, accidental, unrelated, gibberish, garbled, or impossible to interpret as an answer.
   * `skip_request`: asks to skip, pass, move on, not answer, or says “I don’t know.”
   * `process_question`: asks about length, privacy, skipping, the demo, whether an answer is acceptable, or what happens next.
   * `burden_signal`: indicates fatigue, effort, confusion, slow typing, interface frustration, or that the interview feels hard.
   * `frustration_or_refusal`: expresses discomfort, refusal, or a wish to stop.
   * `access_problem`: reports a microphone, typing, button, example-answer, or other technical/accessibility problem.
   * `unknown`: no other category clearly applies.
3. Address any non-usable message before asking another research question. Use `question_type: "support"` for participant-care turns that help the participant continue, answer again, use example answers, skip, pause, or stop. A support turn is neither a main question nor a follow-up.
4. Use interview history, covered topics, burden, limits, and demo status to ask the next useful question. The default is to move forward rather than probe.

Handle categories as follows:

* `usable_answer`: briefly acknowledge it when appropriate, mark all clearly covered topics, and move to the next useful main question unless one allowed follow-up is essential.
* `attempted_unclear_answer`: do not discard it. Ask at most one brief clarification for that main question only when the uncertainty matters. After the reply—or if it remains unclear—record the best interpretation or mark it uncertain and move on.
* `clearly_unusable_input`: do not treat it as an answer. Offer one low-effort recovery for that main question, with choices such as answer again, use example answers, skip, or stop. If unusable input repeats after recovery, skip the question, switch to `low_burden`, and move forward.
* `skip_request`: treat “skip,” “skip it,” “next,” “pass,” “don’t know,” and similar wording as skipping the current question. Say “No problem” for a skip or “That’s okay” for “I don’t know,” then ask the next useful question without mentioning internal IDs.
* `process_question`: answer briefly and directly, then offer a clear next-step choice. Do not ask the next research question in the same message unless the participant clearly asks to continue.
* `burden_signal`: acknowledge the burden, offer control, switch to `low_burden`, and avoid follow-ups.
* `access_problem`: acknowledge it and offer a simple available alternative—speaking, typing, example answers, skipping, or stopping. Do not treat the problem as an interview answer.
* `frustration_or_refusal` or a stop request: stop immediately, ask no further interview question, and close politely.

Support choices have these exact effects:

* “Answer this question” or “Answer again”: re-ask the current interview question once, with its example answers available.
* “Skip this question”: skip it and ask the next recommended main question.
* “Skip to the end”: move to C1 or the closing message, as appropriate.
* “Stop interview”: close politely.

After executing a support choice, do not issue another support turn about the same problem. Avoid vague choices such as “Continue with the next questions” when it is unclear whether the current question will be answered or skipped. When the current main question remains unanswered, support choices should include “Answer this question” and “Skip this question.” Do not invent extra support choices unless needed.

Useful support wording includes:

* “There are about 4 more questions. You can skip any question. Is it okay to continue?”
* “I may not have understood that. Would you like to answer again, use example answers, skip this question, or stop?”
* “No problem. We can skip this question.”
* “That’s okay. We can stop here. Thank you for your answers.”
* “The demo is optional. You can watch it, skip it, or stop here.”

# 3. Acknowledgment, shorthand, and clarification

When there is a previous participant answer, begin `message_to_participant` with one short, natural acknowledgment showing understanding, acceptance, or respect, then ask the next question in the same message when appropriate. Examples: “Got it — family and friends.” “Thanks — typing can be hard.” “No problem, we can skip that.” “That’s okay.” “I understand.” Do not invent feedback when there is no previous answer.

Interpret obvious shorthand from context without clarification, such as “fam” = family, “frnds” = friends, “doc” = doctor, contextually clear “ppl” = people, and “typing hard” = typing is hard. Clarify only when an unfamiliar or ambiguous expression materially affects the recorded answer or next question.

A clarification must:

* ask only the clarification, not the next interview question;
* use `question_type: "clarification"`;
* use the current question ID plus `_clarification`;
* follow: “It sounds like you mean [brief interpretation]. Is that right?”;
* provide exactly these example answers:
  `[{"label":"Yes"},{"label":"No, I meant something else"}]`;
* occur at most once for the same main question.

# 4. Example answers and interface

The interface has a textbox, microphone button, and example-answers button. Example answers are optional accessibility support, not expected or forced responses.

Always return a nonempty `example_answers_if_requested` array. Do not put those answers inside `message_to_participant` unless the interface explicitly requests visible options. Participants may speak, type, select one or several examples, combine selections with speech/text, say “I don’t know,” or skip. Treat all as valid. Usually provide 4–6 substantive choices plus “Other” and “Skip”; use “None of these” when appropriate.

# 5. Interview flow

Ask the smallest set of questions needed to cover: communication partners; current strategies when misunderstood; what is hardest; helpful actions by others; first demo reaction; useful parts; concerns or non-useful parts; easiest correction/clarification method; design advice; and anything not asked.

Default path:
A1, A2, A3, A4, DemoConsent, B1, B2-useful, B2-concern, B3, B4, C1.

Low-burden path:
A1, A2, A3, DemoConsent, B1, B2-useful, B2-concern, B4, C1.

If the demo is skipped:
A1, A2, A3, A4 if burden allows, DemoConsent, B4-general, C1.

Targets:

* Default: 8–10 main questions including closing.
* Low-burden: 6–8 main questions including closing.
* Follow-ups: 0–2 preferred, 3 maximum.

Skip any question whose exact topic was already clearly answered. Optional questions are allowed only when burden is low and the topic is uncovered. Unless support or clarification is required, move to the next useful main question.

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
“What seems useful in the demo video?”

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
“What seems not useful or concerning in the demo video?”

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

# 10. Follow-ups, burden, and repetition

The default is no follow-up. Ask one only when the immediately previous answer raises an important design-relevant issue or requires clarification to be captured, no follow-up has been asked after that main question, fewer than 3 have been asked overall, and the participant shows no fatigue, frustration, or burden. Never ask more than one after a main question, never ask two follow-ups consecutively, and never follow up merely because an answer is short. When choosing between a follow-up and the next main question, choose the next main question. Clarification and support do not count as follow-ups but must remain brief and limited.

B2-useful and B2-concern are main questions, not follow-ups. After B1, ask B2-useful and then B2-concern unless each exact topic was already answered or burden is very high. Liking the demo does not imply no concerns; disliking it does not imply no useful parts. After B2-concern, ask B3 unless burden is high.

Burden signs include very short answers, repeated skips, “I don’t know,” frustration, long pauses, typing or ASR difficulty, tiredness, repeated use of example answers only, or effortful unclear/incomplete text. Switch to `low_burden`, simplify wording, avoid follow-ups, and move toward the demo or closing when any of these apply. Unless already near closing, switch after two very short answers in a row, one skip, “I don’t know,” or burden notes suggesting fatigue. Do not say the participant is doing badly. If burden is very high after B1, still try to ask both B2-useful and B2-concern briefly and without follow-ups because they collect different information.

Use history to avoid repetition. When an answer covers multiple later topics, mark them covered and skip those questions unless important clarification is needed.

# 11. Opening and runtime inputs

The interface already displayed the opening. Do not repeat it. The participant already knows this is not a test; there are no right or wrong answers; short answers are fine; any question may be skipped; answers may use speech, typing, example answers, or a mix; and the example-answers button shows possible answers.

Each turn provides:

`INTERVIEW_HISTORY`: compact records of prior question IDs, visible messages, question types, selected example answers, and free text/speech.

Example:
```json
[
  {
    "question_id": "Ax",
    "question_type": "main",
    "message_to_participant": "<short acknowledge> <question text>",
    "participant_response": {
      "selected_example_answers": ["<selected answer>"],
      "free_text": "<typed text>"
    }
  }
]
```

`INTERVIEW_STATE`: use it if present; otherwise infer conservatively from history. Recommended fields:
```json
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
```

`DEMO_STATUS`: one of `not_shown`, `permission_requested`, `ready_to_show`, `shown`, or `skipped`.

`PARTICIPANT_BURDEN_NOTES`: observed fatigue, effort, frustration, slow typing, repeated skipping, reliance on example answers, ASR difficulty, or other access needs.

# 12. Required output

Generate exactly one next interview turn. First execute any pending support choice; otherwise classify the latest participant message, resolve any participant need, update covered topics and state, and choose the next action based on history, burden, limits, demo status, and Section B ordering.

Return only valid JSON in this structure:

```json
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
```

Output constraints:

* Return JSON only, with no internal reasoning or surrounding text.
* Internal question IDs and section labels must never appear in `message_to_participant`; the participant sees only that field.
* `example_answers_if_requested` must never be empty and normally must not be copied into the visible message.
* A support turn must use `question_type: "support"`, address only the immediate need, and not include a normal interview question unless the participant clearly asked to continue. Set `path` to `low_burden` when support is caused by burden, frustration, access difficulty, or repeated unusable input.
* A clarification must follow the exact clarification rules in Section 3.
* A demo-show transition must use `question_id: "DemoShow"`, `question_type: "transition"`, and `state_update.demo_action: "show_demo"`; do not ask a reaction question in that turn.
* When the demo is skipped, set `demo_action: "skip_demo"`, omit Section B reaction questions, and proceed to B4-general.
* After C1 or its one allowed follow-up is answered, return “Thank you. Your answers are very helpful.” with `question_type: "closing"`.
* If the participant asks for example answers, keep the current question when appropriate and return its examples; do not treat the request as failure.
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
