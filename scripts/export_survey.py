"""Export fixed-survey batches; never prints credentials or participant answers."""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from survey.storage import DriveStore


def applies(condition, answers):
    if not condition:
        return True
    if 'all' in condition:
        return all(applies(c,answers) for c in condition['all'])
    if 'any' in condition:
        return any(applies(c,answers) for c in condition['any'])
    if 'answered' in condition:
        return answers.get(condition['answered'],{}).get('status')=='answered'
    return any(v in condition['values'] for v in answers.get(condition['question'],{}).get('choices',[]))


def combine(records, schema):
    records=sorted(records,key=lambda r:(r['revision'],r.get('saved_at','')))
    if not records:
        raise ValueError('No saved batches.')
    versions={r['schema_version'] for r in records}
    if versions != {schema['version']}:
        raise ValueError('Schema version does not match the saved survey.')
    revisions=[r['revision'] for r in records]
    if len(set(revisions)) != len(revisions):
        raise ValueError('Conflicting revisions: review simultaneous-session saves before exporting.')
    last=records[-1]
    answers=last['state']['answers']
    rows=[]
    for p in schema['pages']:
        if p['kind'] in ('info','finish'):
            continue
        a=answers.get(p['id'],{})
        rows.append({'participant_id':last['participant_id'],'schema_version':last['schema_version'],
          'question_id':p['id'],'section':p['section'],'question':p['title'],'item':p.get('item',''),
          'status':a.get('status','not_reached') if applies(p.get('when'),answers) else 'not_applicable',
          'choices':json.dumps(a.get('choices',[]),ensure_ascii=False),
          'groups':json.dumps(a.get('groups',{}),ensure_ascii=False),
          'text':a.get('text',''),'other_text':json.dumps(a.get('other',{}),ensure_ascii=False)})
    events=[]
    seen=set()
    for r in records:
        for event in r.get('events',[]):
            if event['event_id'] not in seen:
                seen.add(event['event_id'])
                events.append({**event,'participant_id':last['participant_id'],
                               'batch_revision':r['revision'],'server_saved_at':r.get('saved_at')})
    return last,rows,events


def write_export(records,schema,root):
    last,rows,events=combine(records,schema)
    target=Path(root)/last['participant_id']
    target.mkdir(parents=True,exist_ok=True)
    (target/'responses.json').write_text(json.dumps(last['state'],ensure_ascii=False,indent=2),encoding='utf-8')
    with (target/'answers.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader()
        # Neutralize spreadsheet formulas in free text; exact content is retained in JSON.
        for row in rows:
            writer.writerow({k:("'"+v if isinstance(v,str) and v.startswith(('=','+','-','@','\t','\r')) else v) for k,v in row.items()})
    (target/'events.jsonl').write_text(''.join(json.dumps(e,ensure_ascii=False)+'\n' for e in events),encoding='utf-8')
    (target/'metadata.json').write_text(json.dumps({k:v for k,v in last.items() if k not in ('state','events')},indent=2),encoding='utf-8')


def list_files(service,query):
    token=None
    while True:
        page=service.files().list(q=query,pageToken=token,pageSize=1000,fields='nextPageToken,files(id,name)').execute()
        yield from page.get('files',[])
        token=page.get('nextPageToken')
        if not token:
            break


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--local',type=Path,help='Local preview directory, e.g. .survey-preview')
    source.add_argument('--secrets',type=Path,help='Streamlit secrets TOML for read-only Google Drive export')
    parser.add_argument('--out',type=Path,default=Path('survey_exports'))
    options=parser.parse_args()
    schema=json.loads((Path(__file__).resolve().parents[1]/'survey/schema.json').read_text(encoding='utf-8'))
    count=0
    if options.local:
        groups=([json.loads(f.read_text(encoding='utf-8')) for f in sorted(folder.glob('*.json'))]
                for folder in options.local.glob('survey_*') if folder.is_dir())
    else:
        import tomllib
        with options.secrets.open('rb') as f:
            config=tomllib.load(f)
        store=DriveStore(config)
        folders=list_files(store.service,f"'{store.root}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false")
        groups=([store._read(f['id']) for f in list_files(store.service,
                f"'{folder['id']}' in parents and mimeType='application/json' and trashed=false")]
                for folder in folders if folder['name'].startswith('survey_'))
    for records in groups:
        if records:
            write_export(records,schema,options.out);count+=1
    print(f'Exported {count} survey session(s). Treat exports as sensitive research data.')


if __name__=='__main__':
    main()
