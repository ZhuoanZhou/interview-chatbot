"""Rebuild the fixed survey from the September 22 Word guide (read-only source)."""
import hashlib
import json
import sys
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    'E:/Research_projects/ACC/draft/interview_plans/refined_question_list_9.22.2026.docx')
with ZipFile(SOURCE) as z:
    root = ET.fromstring(z.read('word/document.xml'))
ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
# Read the current visible text, excluding tracked-deletion text nodes.
paragraphs = [''.join(t.text or '' for t in p.findall('.//w:t', ns))
              for p in root.findall('.//w:body//w:p', ns)]

def txt(n):
    return paragraphs[n].strip().rstrip(',')

def opts(*indices):
    return [txt(i).replace('Other: ____', 'Other').replace('Other: ___', 'Other') for i in indices]

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
strategies = opts(*range(101,110))
page('intro',0,'Your communication experiences', 'info', source=[45,46],
     paragraphs=[txt(45),txt(46),'Please allow up to 60 minutes. You can pause and return later.',
     'We record your survey selections and changes, including text you type or delete and the time of each action. This happens only inside this survey. Deleted or unsubmitted text may remain in the research interaction log.'],
     next_label='Start survey')
page('people',1,txt(51),'multi',people,51)
page('places',1,txt(58),'multi',places,58)
page('asr',1,txt(66),options=['Yes, I use it now','Yes, I tried but stopped','No, never tried','Not sure'],source=66)
page('asr_stopped',1,txt(70),'text',source=70,when=has('asr','Yes, I tried but stopped'))
page('asr_uses',1,txt(72),'text',source=72,when=has('asr','Yes, I use it now'))
page('aac',1,txt(73),options=['Yes','No','Other'],source=73)
page('aac_name',1,'What is it?','text',source=75,when=has('aac','Yes'))
page('aac_carry',1,'How did you carry or hold it?','multi',txt(78).split(' / '),76,when=has('aac','Yes'))
page('text_input',1,txt(79),'multi',opts(*range(81,89)),79,exclusive=['I do not enter text'])
page('story',1,txt(89),'group',source=[89,91,92],fields=[
    {'id':'person','title':'Who were you talking with?','options':people},
    {'id':'place','title':'Where were you?','options':places}],
    help='Choose one in each group. If no example comes to mind, choose Skip.')
story = {'answered':'story'}
page('first_repair',1,'What did you try first to help them understand?',options=strategies,source=99,when=story)
repair = has('first_repair',*strategies[:6], 'Other')
page('understood',1,'Did they understand what you meant after that?',options=['Yes','Partly','No','Maybe'],source=111,when=all_of(story,repair))
page('next_repair',1,'What did you do next?',options=opts(*range(115,120)),source=113,when=all_of(story,repair,has('understood','Partly','No')))
page('different_repair',1,'What did you use?',options=strategies,source=120,
     when=all_of(story,has('next_repair','Used a different method')),exclude_selected_from='first_repair')
page('stop_reason',1,'What made you stop?','multi',opts(*range(134,141)),132,when=all_of(story,any_of(
    has('first_repair','Stopped trying to communicate the message'),has('next_repair','Stopped trying'),
    has('different_repair','Stopped trying to communicate the message'))),exclusive=['Not sure'])
page('partner_action',1,txt(141),'multi',opts(*range(143,150)),141,when=story,exclusive=['Do not remember'])
page('detect',1,txt(150),options=opts(*range(152,156)),source=150)
page('detect_cues',1,'What helps you tell?','multi',opts(*range(158,164)),156,when=has('detect','Usually','Sometimes'))
page('pretended',1,txt(164),options=['Yes','No','Not sure'],source=164)
page('tell_when',1,txt(166),options=opts(*range(168,173)),source=166)
page('scenarios_intro',2,'Example situations','info',source=177,paragraphs=[txt(177),
     'For each example, choose what you would do. You can skip any example.'])
for number, start in enumerate([179,184,189,194,199],1):
    prefix=f's{number}'
    scenario={'title':txt(start),'situation':txt(start+1),
              'meant':txt(start+2).split(':',1)[1].strip(),
              'shown':txt(start+3).split(':',1)[1].strip()}
    page(prefix+'_action',2,'What would you do first in this situation?',options=opts(*range(207,217)),source=[start,205],scenario=scenario)
    edit=has(prefix+'_action','Change the text and show it')
    page(prefix+'_extent',2,'What would you change?',options=opts(*range(220,224)),source=218,when=edit,scenario=scenario)
    page(prefix+'_words',2,txt(224),'text',source=224,when=edit,scenario=scenario)
    page(prefix+'_leave',2,'Why would you leave it as it is?','multi',opts(*range(228,233)),226,
         when=has(prefix+'_action','Continue with the text as it is'),scenario=scenario,exclusive=['Not sure'])
page('break',2,'Take a break if you would like','info',paragraphs=[
    'Next is a short demonstration. You can pause here and come back when you are ready.'],next_label='Continue to demonstration')
page('demo_consent',3,'Would you like to watch the demonstration?',options=['Yes','No'],source=[235,237,238],paragraphs=[txt(237),txt(238)])
demo=has('demo_consent','Yes')
page('demo_video',3,'Demonstration','video',when=demo,source=235,
     help='You can pause the video, watch it again, or skip it.',next_label='I have watched the demonstration')
watched=all_of(demo,has('demo_video','watched'))
ratings=['Very useful','Somewhat useful','Neutral','Not very useful','Not useful at all','Not sure']
for number,i in enumerate(range(244,255,2),1):
    page(f'feature_{number}',3,'How useful do you think each part of the system would be?',options=ratings,source=[240,i],
         item=txt(i),group='Q1 · Parts of the system',rating_group='features',when=watched)
page('candidates_compare',3,txt(257).removeprefix('Q2. '),options=opts(*range(259,265)),source=257,when=watched)
page('candidate_missing',3,txt(266).removeprefix('Q3. '),'multi',opts(*range(268,274)),266,when=watched,exclusive=['Not sure'])
page('failed_repair',3,txt(275).removeprefix('Q4. '),options=opts(*range(277,287)),source=275,when=watched)
page('retry_count',3,txt(288),options=opts(*range(290,295)),source=288,
     when=all_of(watched,has('failed_repair','Change another word and try a new version again')))
page('difficulty',3,txt(296).removeprefix('Q5. '),'multi',opts(*range(298,308)),296,when=watched,
     exclusive=['Nothing seems difficult','I cannot judge without trying it'])
page('look_when',3,txt(309).removeprefix('Q6. '),'multi',opts(*range(311,319)),309,when=watched,
     exclusive=['I would not want to look at the text','I would only use this for messages or writing','Not sure'])
for number,i in enumerate(range(324,337,2),1):
    page(f'situation_{number}',3,'How useful do you think the process could be in each situation?',options=ratings,source=[320,i],
         item=txt(i),group='Q7 · Situations',rating_group='situations',when=watched)
page('partner_expected',3,txt(339).removeprefix('Q8. '),'multi',opts(*range(341,348)),339,
     when=all_of(watched,story),exclusive=['Not sure'])
page('closing',4,txt(351).removeprefix('Q1. '),'text',source=351,help='A few words are enough. You can also leave this blank.')
page('finish',4,'Ready to finish?','finish',paragraphs=[
    'You can go back to change an answer, or submit your survey now.',
    'Thank you for sharing your experiences and ideas.'])
schema={'version':'2026-09-22-v1','title':'Communication experiences survey',
        'source':SOURCE.name,'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'sections':['Welcome','Your experiences','Example situations','After the demonstration','Closing'],
        'pages':pages}
target=Path(__file__).resolve().parents[1]/'survey'/'schema.json'
target.parent.mkdir(exist_ok=True)
target.write_text(json.dumps(schema,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'Built {len(pages)} screens from {SOURCE.name}; researcher notes excluded.')
