"""Immutable save batches. Each batch contains events and an answer checkpoint.

The same batch ID is safe to retry after an uncertain network response. Resume
secrets are hashed before being used in folder names and are never in records.
"""
import hashlib
import io
import json
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def new_token():
    return secrets.token_urlsafe(32)


def folder_name(token):
    if not isinstance(token, str) or not re.fullmatch(r'[A-Za-z0-9_-]{43}', token):
        raise ValueError('Invalid resume code.')
    return 'survey_' + hashlib.sha256(token.encode()).hexdigest()


def new_record(version):
    return {'participant_id':'P-' + secrets.token_hex(6).upper(),
            'schema_version':version, 'created_at':utc_now(), 'revision':0,
            'state':{'page':'intro','answers':{},'status':'active'}, 'events':[]}


def make_record(previous, packet):
    if previous['state']['status'] in ('submitted','ended'):
        raise ValueError('This survey has already ended.')
    if packet.get('base_revision') != previous['revision']:
        raise ValueError('This survey was saved in another session. Resume it again before continuing.')
    if not re.fullmatch(r'[a-f0-9-]{36}', packet.get('batch_id','')):
        raise ValueError('Invalid save batch.')
    events = packet.get('events')
    state = packet.get('state')
    if not isinstance(events,list) or not isinstance(state,dict):
        raise ValueError('Invalid survey data.')
    if len(events) > 10000 or len(json.dumps(packet)) > 8_000_000:
        raise ValueError('Save batch too large. Please contact the researcher.')
    if state.get('status') not in ('active','paused','submitted','ended'):
        raise ValueError('Invalid survey status.')
    return {**previous, 'revision':previous['revision']+1, 'batch_id':packet['batch_id'],
            'saved_at':utc_now(), 'state':state, 'events':events}


class LocalStore:
    """Explicit preview/test mode only; never a silent fallback for Drive errors."""
    def __init__(self, root):
        self.root = Path(root)

    def create(self, token, version):
        folder = self.root / folder_name(token)
        folder.mkdir(parents=True, exist_ok=False)
        record = new_record(version)
        (folder/'000000000_initial.json').write_text(json.dumps(record),encoding='utf-8')
        return record

    def load(self, token):
        files = sorted((self.root / folder_name(token)).glob('*.json'))
        if not files:
            raise ValueError('No survey found for this resume code.')
        return json.loads(files[-1].read_text(encoding='utf-8'))

    def save(self, token, packet):
        folder = self.root / folder_name(token)
        matches = list(folder.glob('*_' + packet.get('batch_id','invalid') + '.json'))
        if matches:
            return json.loads(matches[0].read_text(encoding='utf-8'))
        result = make_record(self.load(token),packet)
        dest = folder/f"{result['revision']:09d}_{result['batch_id']}.json"
        temp = dest.with_suffix('.tmp')
        temp.write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
        temp.replace(dest)
        return result


class DriveStore:
    def __init__(self, config):
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        creds = Credentials(token=None, refresh_token=config['GDRIVE_REFRESH_TOKEN'],
                            client_id=config['GDRIVE_CLIENT_ID'],client_secret=config['GDRIVE_CLIENT_SECRET'],
                            token_uri='https://oauth2.googleapis.com/token')
        self.service = build('drive','v3',credentials=creds,cache_discovery=False)
        self.root = config['GDRIVE_FOLDER_ID']
        if not re.fullmatch(r'[A-Za-z0-9_-]+',self.root):
            raise ValueError('Invalid Google Drive folder configuration.')

    def _folder(self,token):
        name=folder_name(token)
        files=self.service.files().list(q=f"name='{name}' and '{self.root}' in parents and trashed=false",
                                         fields='files(id)',pageSize=2).execute().get('files',[])
        if len(files)!=1:
            raise ValueError('No unique survey found for this resume code.')
        return files[0]['id']

    def _read(self,id):
        data=self.service.files().get_media(fileId=id).execute()
        return json.loads(data.decode('utf-8'))

    def _write(self,folder,name,record):
        from googleapiclient.http import MediaIoBaseUpload
        data=json.dumps(record,ensure_ascii=False).encode('utf-8')
        self.service.files().create(body={'name':name,'parents':[folder]},
            media_body=MediaIoBaseUpload(io.BytesIO(data),mimetype='application/json'),fields='id').execute()

    def _latest(self,folder):
        files=self.service.files().list(q=f"'{folder}' in parents and trashed=false and mimeType='application/json'",
            orderBy='name desc',pageSize=1,fields='files(id)').execute().get('files',[])
        if not files:
            raise ValueError('No saved survey was found.')
        return self._read(files[0]['id'])

    def create(self,token,version):
        folder=self.service.files().create(body={'name':folder_name(token),
            'mimeType':'application/vnd.google-apps.folder','parents':[self.root]},fields='id').execute()['id']
        record=new_record(version)
        self._write(folder,'000000000_initial.json',record)
        return record

    def load(self,token):
        return self._latest(self._folder(token))

    def save(self,token,packet):
        batch_id=packet.get('batch_id','')
        if not re.fullmatch(r'[a-f0-9-]{36}',batch_id):
            raise ValueError('Invalid batch ID.')
        folder=self._folder(token)
        name=f"{int(packet['base_revision'])+1:09d}_{batch_id}.json"
        existing=self.service.files().list(q=f"'{folder}' in parents and name='{name}' and trashed=false",
                                           fields='files(id)',pageSize=1).execute().get('files',[])
        if existing:
            return self._read(existing[0]['id'])
        record=make_record(self._latest(folder),packet)
        self._write(folder,name,record)
        return record

    def demo_bytes(self,file_id):
        if not re.fullmatch(r'[A-Za-z0-9_-]+',file_id):
            raise ValueError('Invalid demo file ID.')
        return self.service.files().get_media(fileId=file_id).execute()
