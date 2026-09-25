"""Fixed survey. Run with streamlit run streamlit_app.py. No AI API calls."""
import base64
import faulthandler
import hashlib
import importlib
import json
import os
from pathlib import Path

# Keep native crash traces in server logs (no response values or local variables).
faulthandler.enable()
# Streamlit reruns scripts on new threads. Avoid Arrow's bundled allocator for
# this workload, including when Arrow was imported before this script ran.
# https://github.com/apache/arrow/issues/50471
os.environ['ARROW_DEFAULT_MEMORY_POOL'] = 'system'
import pyarrow as pa
pa.set_memory_pool(pa.system_memory_pool())

import streamlit as st
import streamlit.components.v1 as components
import survey.storage as _storage

# Cloud can rerun this entry point during a Git update while retaining imported
# modules from the previous release. Refresh only when its required API is stale.
if not all(hasattr(_storage, name) for name in ('normalize_access', 'new_participant_id')):
    importlib.invalidate_caches()
    importlib.reload(_storage)
from survey.storage import DriveStore, LocalStore, normalize_access, new_participant_id
from survey.media import credential_scope, load_demo, demo_url

ROOT = Path(__file__).resolve().parent
SCHEMA = json.loads((ROOT / 'survey/schema.json').read_text(encoding='utf-8'))
PREVIEW = os.environ.get('SURVEY_PREVIEW') == '1'
KEYS = ['GDRIVE_FOLDER_ID','GDRIVE_CLIENT_ID','GDRIVE_CLIENT_SECRET','GDRIVE_REFRESH_TOKEN']
DEFAULT_DEMO_ID = '1FCfzZslMnuyQAPhcZoiACrx0sWaYskxV'
survey_component = components.declare_component('fixed_communication_survey', path=str(ROOT/'survey/frontend'))
st.set_page_config(page_title='Communication experiences survey',page_icon='💬',layout='wide',initial_sidebar_state='collapsed')
st.markdown('''<style>
 [data-testid="stAppViewContainer"]{background:#f5f8f8}
 .block-container{padding-top:1rem;max-width:none;padding-left:1rem;padding-right:1rem}
 header[data-testid="stHeader"]{background:transparent}
 /* Keep the component below Streamlit's header and inside the visible screen. */
 [data-testid="stMain"]:has(iframe[title*="fixed_communication_survey"]){overflow:hidden}
 [data-testid="stMainBlockContainer"]:has(iframe[title*="fixed_communication_survey"]){padding-top:3.5rem;padding-bottom:.5rem}
 [data-testid="stMainBlockContainer"]:has(iframe[title*="fixed_communication_survey"]) > [data-testid="stVerticalBlock"]{gap:0}
 iframe[title*="fixed_communication_survey"]{display:block;height:calc(100vh - 4rem)!important;height:calc(100dvh - 4rem)!important}
 /* Match the native start/resume controls to the survey component's text size. */
 [data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
 [data-testid="stMain"] [data-testid="stWidgetLabel"] p{font-size:20px;line-height:1.5}
 [data-testid="stMain"] [data-testid="stCaptionContainer"] p{font-size:18px;color:#49626b}
 [data-testid="stMain"] [data-testid="stButton"] button,
 [data-testid="stMain"] [data-testid="stFormSubmitButton"] button{
   min-height:56px;padding:12px 24px;font-size:20px;border-radius:9px}
 [data-testid="stMain"] [data-testid="stTextInput"] input{
   min-height:56px;padding:12px 16px;font-size:20px}
 [data-testid="stMain"] [data-testid="stTextInputRootElement"]{min-height:56px;height:auto}
 [data-testid="stMain"] [data-testid="stTabs"] [role="tab"]{
   min-height:60px;height:auto;padding:12px 16px;white-space:normal}
 [data-testid="stMain"] [data-testid="stTabs"] [role="tab"] p{
   font-size:20px;white-space:normal;text-align:center}
 [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{font-size:18px}
 [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p{font-size:16px;line-height:1.5}
 [data-testid="stSidebar"] [data-testid="stCode"] code{font-size:18px}
 @media(max-width:520px){
   [data-testid="stMain"] [data-testid="stTabs"] [role="tab"]{flex:1;min-width:0;padding:10px 8px}
 }
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
    with st.sidebar:
        st.info('Preview only — uses local test files and makes no Google Drive or AI calls.')
elif not all(config.get(key) for key in KEYS):
    st.error('Survey storage is not configured. Please contact the researcher.')
    st.caption('Researcher: configure the four GDRIVE settings in Streamlit Secrets. See SURVEY_README.md.')
    st.stop()


def store():
    expected_type = LocalStore if PREVIEW else DriveStore
    if not isinstance(st.session_state.get('_survey_store'), expected_type):
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
with st.sidebar:
    st.markdown('**Your participant ID**')
    st.code(record['participant_id'],language=None)
    st.caption('Keep this ID to return later. Choose “Return to your survey” on the start screen and enter it. Keep it private.')

# Retrieve only the existing demonstration, only on its screen. Preview does not
# read secrets or fetch video from Drive.
demo_error = ''
video_url = ''
if record['state']['page'] == 'demo_video':
    if PREVIEW:
        demo_error = 'The video is not loaded in local preview. You can skip the demonstration.'
    else:
        try:
            video = load_demo(config.get('DEMO_VIDEO_FILE_ID') or DEFAULT_DEMO_ID,
                              credential_scope(config), store())
            video_url = demo_url(video)
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
    demo_url=video_url,
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
