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
page('intro',0,'Your communication experiences', 'info', source=[45,46],
     paragraphs=[txt(45),txt(46),'Please allow up to 60 minutes. You can pause and return later.',
     'We record your survey selections and changes, including text you type or delete and the time of each action. This happens only inside this survey. Deleted or unsubmitted text may remain in the research interaction log.'],
     next_label='Start survey')
page('people',1,txt(51),'multi',people,51)
page('places',1,txt(58),'multi',places,58)
page('asr',1,txt(68),options=['Yes, I use it now','Yes, I tried but stopped','No, never tried','Not sure'],source=[68,69])
page('asr_stopped',1,txt(72),'text',source=72,when=has('asr','Yes, I tried but stopped'))
page('asr_uses',1,txt(74),'text',source=74,when=has('asr','Yes, I use it now'))
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
# Intro reworded in 2026-09 (replaces the guide's paragraph 137).
page('scenarios_intro',2,'Example situations','info',source=137,paragraphs=[
     'Imagine you are talking to someone. For these examples, suppose a device could show you the words the other person understood.',
     'You will see what you meant to say and what the other person understood. Choose what you would do first in each situation. You can skip any example.'])
# "What would you do first?" choices, regrouped in 2026-09 (meeting with Slobodan);
# they replace the guide's flat list. Each group is shown on its own line.
ACTION_GROUPS=[{'label':'Continue the conversation','options':['Accept the text and continue the conversation']},
               {'label':'Use my voice again','options':['Say it again the same way','Say it in a different way']},
               {'label':'Use text','options':['Change parts of the text','Type a new message']},
               {'label':'Use another way to communicate','options':['Use my AAC device','Use gestures or signs']},
               {'label':'Stop trying','options':['Stop trying to get this message across']},
               {'label':'','options':['Not sure']}]
ACTION_OPTIONS=[o for g in ACTION_GROUPS[:-1] for o in g['options']]+['Other','Not sure']
# Six hand-written word candidates per transcript word (prototype stimuli, not ASR
# output), shown around a clicked word on the post-demonstration edit screens.
CANDIDATES=json.loads((Path(__file__).resolve().parents[1]/'survey'/'word_candidates.json').read_text(encoding='utf-8'))
scenarios=[]
for number, start in enumerate([139,144,149,154,159],1):
    prefix=f's{number}'
    scenario={'title':txt(start),'situation':txt(start+1),
              'meant':txt(start+2).split(':',1)[1].strip(),
              'shown':txt(start+3).split(':',1)[1].strip()}
    scenarios.append((start,scenario))
    page(prefix+'_action',2,'What would you do first in this situation?',options=ACTION_OPTIONS,source=[start,165],
         scenario=scenario,option_groups=ACTION_GROUPS)
    # The guide's follow-ups (what/which words to change, why leave it) were removed
    # in 2026-09: the editable post-demonstration examples now cover them.
page('break',2,'Take a break if you would like','info',paragraphs=[
    'Next is a short demonstration. You can pause here and come back when you are ready.'],next_label='Continue to demonstration')
page('demo_consent',3,'Would you like to watch the demonstration?',options=['Yes','No'],source=[179,181,182],paragraphs=[txt(181),txt(182)])
demo=has('demo_consent','Yes')
page('demo_video',3,'Demonstration','video',when=demo,source=179,
     help='You can pause the video, watch it again, or skip it.',next_label='I have watched the demonstration')
watched=all_of(demo,has('demo_video','watched'))
ratings=['Very useful','Somewhat useful','Neutral','Not very useful','Not useful at all','Not sure']
for number,i in enumerate(range(188,199,2),1):
    page(f'feature_{number}',3,txt(184).removeprefix('Q1. '),options=ratings,source=[184,i],
         item=txt(i),group='Q1 · Parts of the system',rating_group='features',when=watched)
page('candidates_compare',3,txt(201).removeprefix('Q2. '),options=opts(*range(203,208)),source=201,when=watched)
page('candidate_missing',3,txt(209).removeprefix('Q3. '),'multi',opts(*range(211,217)),209,when=watched,exclusive=['Not sure'])
recheck=opts(*range(220,230))
page('failed_repair',3,txt(218).removeprefix('Q4. '),options=recheck,source=218,when=watched)
# "It depends on something else: ___" is this question's free-text choice.
page('retry_count',3,txt(231),options=opts(*range(233,239)),source=[230,231],
     when=all_of(watched,has('failed_repair',recheck[0])),other_option='It depends on something else')
page('difficulty',3,txt(240).removeprefix('Q5. '),'multi',opts(*range(242,252)),240,when=watched,
     exclusive=['Nothing seems difficult','I cannot judge without trying it'])
# Q6 wording smoothed for participants (guide paragraph 253).
page('look_when',3,'Imagine you are using the tool shown in the demo video during a conversation. When would you look at the text?',
     'multi',opts(*range(255,263)),253,when=watched,
     exclusive=['I would not want to look at the text','I would only use this for messages or writing','Not sure'])
for number,i in enumerate(range(268,281,2),1):
    page(f'situation_{number}',3,txt(264).removeprefix('Q7. '),options=ratings,source=[264,i],
         item=txt(i),group='Q7 · Situations',rating_group='situations',when=watched)
TRY_YES='Yes, I’d like to try'
# At the end of Part 3, an optional exercise: the same five examples as editable
# transcripts (added 2026-09; not in the Word guide), only after "Yes" on the intro.
# Word candidates, Delete all/Reset and the decision buttons are in survey.js.
page('edit_intro',3,'Try the example situations',options=[TRY_YES,'No, skip the examples'],when=watched,paragraphs=[
    'Would you like to try an exercise with the five example situations you saw earlier? You can change the example text or choose another response.',
    'This exercise includes text editing and suggested words. It does not include speech-to-text or the “Re-check” function shown in the video. '
    'You will work with prepared transcripts rather than speaking, and the system will not generate a new transcript after a correction.',
    'This is optional. You can skip any example or stop the exercise at any time.'])
for number,(start,scenario) in enumerate(scenarios,1):
    counter=f'Example {number} of {len(scenarios)}'
    page(f'e{number}_edit',3,scenario['title'],'edit',source=list(range(start,start+4)),when=all_of(watched,has('edit_intro',TRY_YES)),
         situation=scenario['situation'],meant=scenario['meant'],transcript=scenario['shown'].strip('“”"'),
         candidates=CANDIDATES[f'e{number}_edit'],
         counter=counter,next_label='Next example' if number<len(scenarios) else 'Next')
page('closing',4,txt(302).removeprefix('Q1. '),'text',source=302,help='A few words are enough. You can also leave this blank.')
page('finish',4,'Ready to finish?','finish',paragraphs=[
    'You can go back to change an answer, or submit your survey now.',
    'Thank you for sharing your experiences and ideas.'])
schema={'version':'2026-09-29-v1','title':'Communication experiences survey',
        'source':SOURCE.name,'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'sections':['Welcome','Your experiences','Example situations','After the demonstration','Closing'],
        'pages':pages,
        # Removed screens -> where a session saved on one of them resumes.
        'retired_pages':{**{f's{n}_{kind}':f's{n}_action' for n in range(1,6) for kind in ('extent','words','leave')},
                         **{f'e{n}_action':f'e{n}_edit' for n in range(1,6)}}}
target=Path(__file__).resolve().parents[1]/'survey'/'schema.json'
target.parent.mkdir(exist_ok=True)
target.write_text(json.dumps(schema,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'Built {len(pages)} screens from {SOURCE.name}; researcher notes excluded.')
