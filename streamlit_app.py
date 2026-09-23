"""Fixed survey. Run with streamlit run streamlit_app.py. No AI API calls."""
import base64
import hashlib
import json
import os
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from survey.storage import DriveStore, LocalStore, normalize_access, new_participant_id

ROOT = Path(__file__).resolve().parent
SCHEMA = json.loads((ROOT / 'survey/schema.json').read_text(encoding='utf-8'))
PREVIEW = os.environ.get('SURVEY_PREVIEW') == '1'
KEYS = ['GDRIVE_FOLDER_ID','GDRIVE_CLIENT_ID','GDRIVE_CLIENT_SECRET','GDRIVE_REFRESH_TOKEN']
DEFAULT_DEMO_ID = '1FCfzZslMnuyQAPhcZoiACrx0sWaYskxV'
survey_component = components.declare_component('fixed_communication_survey', path=str(ROOT/'survey/frontend'))
st.set_page_config(page_title='Communication experiences survey',page_icon='💬',layout='centered',initial_sidebar_state='collapsed')
st.markdown('''<style>
 [data-testid="stAppViewContainer"]{background:#f5f8f8}
 .block-container{padding-top:1rem;max-width:960px}
 header[data-testid="stHeader"]{background:transparent}
 </style>''',unsafe_allow_html=True)


def settings():
    if PREVIEW:
        return {}
    try:
        return {key:str(st.secrets.get(key,'')) for key in KEYS + ['DEMO_VIDEO_FILE_ID','DEMO_TRANSCRIPT','DEMO_CAPTIONS_PATH']}
    except FileNotFoundError:
        return {}


config = settings()
if PREVIEW:
    st.info('Preview only — uses local test files and makes no Google Drive or AI calls.')
elif not all(config.get(key) for key in KEYS):
    st.error('Survey storage is not configured. Please contact the researcher.')
    st.caption('Researcher: configure the four GDRIVE settings in Streamlit Secrets. See SURVEY_README.md.')
    st.stop()


def store():
    if '_survey_store' not in st.session_state:
        st.session_state._survey_store = LocalStore(ROOT/'.survey-preview') if PREVIEW else DriveStore(config)
    return st.session_state._survey_store


if 'survey_record' not in st.session_state:
    st.title('Communication experiences')
    st.write('A survey about communication and speech recognition. Short answers are welcome.')
    st.caption('You can skip questions, take a break, and return later using your participant ID.')
    start, resume = st.tabs(['Start a survey','Return to your survey'])
    with start:
        if st.button('Start a new survey',type='primary'):
            token = new_participant_id()
            try:
                record = store().create(token,SCHEMA['version'])
            except Exception:
                st.error('We could not start a saved session. Please try again or contact the researcher.')
            else:
                st.session_state.update(survey_token=token,survey_record=record)
                st.rerun()
    with resume:
        with st.form('resume_survey'):
            token = st.text_input('Participant ID',placeholder='P-ABC123',help='Enter the participant ID you received when you started.')
            submitted = st.form_submit_button('Resume survey')
        if submitted:
            try:
                token = normalize_access(token)
                record = store().load(token)
                if record['schema_version'] != SCHEMA['version']:
                    st.error('This survey uses a different question version. Please contact the researcher.')
                    st.stop()
            except Exception:
                st.error('We could not open this survey. Check your participant ID or contact the researcher.')
            else:
                st.session_state.update(survey_token=token.strip(),survey_record=record)
                st.rerun()
    st.stop()

token = st.session_state.survey_token
record = st.session_state.survey_record
st.markdown('**Your participant ID**')
st.code(record['participant_id'],language=None)
st.caption('Keep this ID to return later. Choose “Return to your survey” on the start screen and enter it. Keep it private.')

# Retrieve only the existing demonstration, only on its screen. Preview does not
# read secrets or fetch video from Drive.
demo_error = ''
if record['state']['page'] == 'demo_video' and not st.session_state.get('survey_demo'):
    if PREVIEW:
        demo_error = 'The video is not loaded in local preview. You can skip the demonstration.'
    else:
        try:
            video = store().demo_bytes(config.get('DEMO_VIDEO_FILE_ID') or DEFAULT_DEMO_ID)
            st.session_state.survey_demo = base64.b64encode(video).decode('ascii')
        except Exception:
            demo_error = 'The video is unavailable. You can skip it or contact the researcher.'

captions = ''
if config.get('DEMO_CAPTIONS_PATH') and record['state']['page']=='demo_video':
    try:
        captions=base64.b64encode(Path(config['DEMO_CAPTIONS_PATH']).read_bytes()).decode('ascii')
    except OSError:
        pass

packet = survey_component(
    schema=SCHEMA,record=record,session_key=hashlib.sha256(token.encode()).hexdigest(),
    ack=st.session_state.get('survey_ack'),revision=record['revision'],saved_at=record.get('saved_at'),
    error=st.session_state.get('survey_error',''),preview=PREVIEW,
    demo_data=st.session_state.get('survey_demo','') if record['state']['page']=='demo_video' else '',
    demo_error=demo_error,demo_transcript=config.get('DEMO_TRANSCRIPT',''),demo_captions=captions,
    key='survey_'+record['participant_id'],default=None)

if packet and packet.get('attempt') != st.session_state.get('survey_attempt'):
    st.session_state.survey_attempt = packet.get('attempt')
    try:
        saved = store().save(token,packet)
    except ValueError as error:
        st.session_state.survey_error = str(error)
    except Exception:
        st.session_state.survey_error = 'The storage service could not be reached.'
    else:
        st.session_state.update(survey_record=saved,survey_ack=packet['batch_id'],survey_error='')
    st.rerun()
