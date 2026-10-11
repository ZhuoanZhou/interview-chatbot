"""Rebuild the fixed survey from the September 29 Word guide (read-only source).

Usage: python scripts/build_survey_schema.py path/to/refined_question_list_9.29.2026.docx
"""
import hashlib
import json
import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('refined_question_list_9.29.2026.docx')
with ZipFile(SOURCE) as z:
    root = ET.fromstring(z.read('word/document.xml'))
ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
# Read the current visible text, excluding tracked-deletion text nodes.
paragraphs = [''.join(t.text or '' for t in p.findall('.//w:t', ns))
              for p in root.findall('.//w:body//w:p', ns)]

def txt(n):
    return paragraphs[n].strip().rstrip(',')

def opts(*indices):
    # Drop list dashes and blank-line markers such as "Other: ___".
    return [re.sub(r':\s*_+$', '', txt(i).removeprefix('- ')) for i in indices]

def has(q, *values):
    return {'question': q, 'values': list(values)}

def any_of(*conditions):
    return {'any': list(conditions)}

def all_of(*conditions):
    return {'all': list(conditions)}

pages = []
def page(id, section, title, kind='single', options=None, source=None, **kw):
    result = dict(id=id, section=section, title=title, kind=kind, **kw)
    if options is not None:
        result['options'] = options
    if source is not None:
        result['source_paragraphs'] = source if isinstance(source, list) else [source]
    pages.append(result)
    return result

people = opts(53,54,55,56,57)
places = opts(60,61,62,63,64,65)
# Introduction reworded in 2026-10 (based on guide paragraphs 45-46): bold headings
# and a bullet list. "How to come back" explains the participant-ID panel.
page('intro',0,'Your communication experiences', 'info', source=[45,46], blocks=[
    {'heading':'Why we’re asking','paragraphs':[
        'We want to understand what helps when someone has trouble understanding your speech. '
        'We’ll also ask about speech-to-text tools for conversations, messages, and writing.']},
    {'heading':'Taking the survey','bullets':[
        'You can choose answers instead of typing. When you write, a few words are enough.',
        'You can skip questions, take a break and return, or end the survey early.',
        'There are no right or wrong answers.']},
    {'heading':'Time','paragraphs':['Please allow up to 60 minutes.']},
    {'heading':'What we record','paragraphs':[
        'We record your choices and changes, including text you type or delete and the time of each action. '
        'This happens only inside this survey. Deleted or unsubmitted text may remain in the research interaction log.']},
    {'heading':'How to come back','paragraphs':[
        'Your participant ID is in the side panel on the left. Use the arrow button at the top left to show or hide it. '
        'Keep your ID so you can return later.']}],
     next_label='Start survey')
page('people',1,txt(51),'multi',people,51)
page('places',1,txt(58),'multi',places,58)
page('asr',1,txt(68),options=['Yes, I use it now','Yes, I tried but stopped','No, never tried','Not sure'],source=[68,69])
# The guide's two follow-ups (paragraphs 72 and 74) are left out at the researcher's request.
page('aac',1,txt(78),options=['Yes','No','Other'],source=[78,79])
page('aac_name',1,'What is it?','text',source=80,when=has('aac','Yes'))
page('text_input',1,txt(83),'multi',opts(*range(85,92)),83,exclusive=['I do not enter text'])
# The UI already says "Choose all that apply", so only the second sentence of 95 is shown.
page('stop_trying',1,txt(94),'multi',[o.rstrip('.') for o in opts(*range(96,104))],[94,95],
     help=txt(95).removeprefix('Select all that apply. '),exclusive=['I have not stopped trying','Not sure'])
page('detect',1,txt(105),options=opts(*range(107,111)),source=105)
page('detect_cues',1,'What helps you tell?','multi',opts(*range(113,118)),111,when=has('detect','Usually','Sometimes'))
page('pretended',1,txt(120),options=['Yes','No','Not sure'],source=120)
page('tell_when',1,txt(124),options=opts(*range(126,131)),source=124)
# Part 2 (Example situations: intro, "What would you do first?" with follow-up
# ways, and the break before the demonstration) was removed in 2026-10 so that
# participants reach the demonstration sooner. Its five examples (guide paragraphs
# 139-162) are now used only in the exercise at the end of Part 2.
# Six hand-written word candidates per transcript word (prototype stimuli, not ASR
# output), shown around a clicked word on the post-demonstration edit screens.
CANDIDATES=json.loads((Path(__file__).resolve().parents[1]/'survey'/'word_candidates.json').read_text(encoding='utf-8'))
scenarios=[]
for start in [139,144,149,154,159]:
    scenarios.append((start,{'title':txt(start),'situation':txt(start+1),
                             'meant':txt(start+2).split(':',1)[1].strip(),
                             'shown':txt(start+3).split(':',1)[1].strip()}))
page('demo_consent',2,'Would you like to watch the demonstration?',options=['Yes','No'],source=[179,181,182],paragraphs=[txt(181),txt(182)])
demo=has('demo_consent','Yes')
page('demo_video',2,'Demonstration','video',when=demo,source=179,
     help='You can pause the video, watch it again, or skip it.',next_label='I have watched the demonstration')
watched=all_of(demo,has('demo_video','watched'))
ratings=['Very useful','Somewhat useful','Neutral','Not very useful','Not useful at all','Not sure']
for number,i in enumerate(range(188,199,2),1):
    page(f'feature_{number}',2,txt(184).removeprefix('Q1. '),options=ratings,source=[184,i],
         item=txt(i),group='Q1 · Parts of the system',rating_group='features',when=watched)
page('candidates_compare',2,txt(201).removeprefix('Q2. '),options=opts(*range(203,208)),source=201,when=watched)
# Q3 (word not in the suggestions) and Q4 with its Re-check retry follow-up
# (guide paragraphs 209-238) were removed in 2026-10 at the researcher's request.
page('difficulty',2,txt(240).removeprefix('Q5. '),'multi',opts(*range(242,252)),240,when=watched,
     exclusive=['Nothing seems difficult','I cannot judge without trying it'])
# Q6 wording smoothed for participants (guide paragraph 253).
page('look_when',2,'Imagine you are using the tool shown in the demo video during a conversation. When would you look at the text?',
     'multi',opts(*range(255,263)),253,when=watched,
     exclusive=['I would not want to look at the text','I would only use this for messages or writing','Not sure'])
# Q7 question and situations reworded in 2026-10 at the researcher's request
# (guide paragraphs 264 and 268-280; situations 2 and 6 unchanged).
SITUATIONS=['You need to communicate something important, and a misunderstanding would matter.',
            'You are talking with someone unfamiliar with your speech.',
            'You are talking with someone in a public place, such as a shop or service counter.',
            'The conversation is moving quickly, or several people are talking.',
            'You are tired, or speaking or typing is harder than usual.',
            'You are talking with a familiar person who usually understands you.',
            'You have tried repeating or saying it differently, but the other person still does not understand.']
for number,(i,item) in enumerate(zip(range(268,281,2),SITUATIONS),1):
    page(f'situation_{number}',2,'How useful would using the tool shown in the demonstration video be for you in each situation?',
         options=ratings,source=[264,i],item=item,group='Q7 · Situations',rating_group='situations',when=watched)
# At the end of Part 2, the exercise: the five examples as editable transcripts
# (added 2026-09; not in the Word guide). Since 2026-10 it is no longer optional:
# the intro is an information screen, and "Stop the exercise" was removed (each
# example can still be skipped). Word candidates, Delete all/Reset and the
# decision buttons are in survey.js.
page('edit_intro',2,'Try the example situations','info',when=watched,next_label='Start the examples',paragraphs=[
    'Next are five example situations. Each one shows what you meant to say and a transcript of what you said. '
    'Change the transcript the way you would fix it, or choose another response.',
    'This exercise includes text editing and suggested words. It does not include speech-to-text or the “Re-check” function shown in the video. '
    'You will work with prepared transcripts rather than speaking, and the system will not generate a new transcript after a correction.'])
for number,(start,scenario) in enumerate(scenarios,1):
    counter=f'Example {number} of {len(scenarios)}'
    page(f'e{number}_edit',2,scenario['title'],'edit',source=list(range(start,start+4)),when=watched,
         situation=scenario['situation'],meant=scenario['meant'],transcript=scenario['shown'].strip('“”"'),
         candidates=CANDIDATES[f'e{number}_edit'],
         counter=counter,next_label='Next example' if number<len(scenarios) else 'Next')
page('closing',3,txt(302).removeprefix('Q1. '),'text',source=302,help='A few words are enough. You can also leave this blank.')
page('finish',3,'Ready to finish?','finish',paragraphs=[
    'You can go back to change an answer, or submit your survey now.',
    'Thank you for sharing your experiences and ideas.'])
schema={'version':'2026-10-11-v1','title':'Communication experiences survey',
        'source':SOURCE.name,'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'sections':['Welcome','Your experiences','After the demonstration','Closing'],
        'pages':pages,
        # Removed screens -> where a session saved on one of them resumes.
        'retired_pages':{**{f's{n}_{kind}':'demo_consent' for n in range(1,6) for kind in ('action','extent','words','leave')},
                         'scenarios_intro':'demo_consent','break':'demo_consent',
                         **{f'e{n}_action':f'e{n}_edit' for n in range(1,6)}}}
target=Path(__file__).resolve().parents[1]/'survey'/'schema.json'
target.parent.mkdir(exist_ok=True)
target.write_text(json.dumps(schema,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'Built {len(pages)} screens from {SOURCE.name}; researcher notes excluded.')
