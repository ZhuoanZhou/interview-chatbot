import json
import re
import tempfile
import unittest
import uuid
from unittest.mock import MagicMock
from pathlib import Path
from survey.storage import DriveStore, LocalStore, folder_name, new_token, new_participant_id, normalize_access
from scripts.export_survey import combine

ROOT=Path(__file__).resolve().parents[1]
SCHEMA=json.loads((ROOT/'survey/schema.json').read_text(encoding='utf-8'))


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store=LocalStore(self.temp.name)
        self.token=new_token()
        self.initial=self.store.create(self.token,SCHEMA['version'])

    def packet(self,revision=0,status='active'):
        return {'batch_id':str(uuid.uuid4()),'base_revision':revision,
          'state':{'page':'places','status':status,'answers':{'people':{'choices':['Other'],'other':{'other':'friend'},'status':'answered'}}},
          'events':[{'event_id':str(uuid.uuid4()),'type':'text_input','inserted':'a','deleted':'','client_utc':'2026-09-23T12:00:00.000Z'}]}

    def test_retry_ack_loss_does_not_duplicate_events(self):
        packet=self.packet()
        first=self.store.save(self.token,packet)
        retry=self.store.save(self.token,packet)
        self.assertEqual(first,retry)
        self.assertEqual(len(list((Path(self.temp.name)/folder_name(self.token)).glob('*.json'))),2)
        self.assertEqual(self.store.load(self.token)['revision'],1)

    def test_stale_revision_is_rejected_without_overwriting(self):
        first=self.store.save(self.token,self.packet())
        with self.assertRaisesRegex(ValueError,'another session'):
            self.store.save(self.token,self.packet())
        self.assertEqual(self.store.load(self.token),first)

    def test_complete_is_read_only_but_retry_is_safe(self):
        packet=self.packet(status='submitted')
        self.store.save(self.token,packet)
        self.assertEqual(self.store.save(self.token,packet)['revision'],1)
        with self.assertRaisesRegex(ValueError,'already ended'):
            self.store.save(self.token,self.packet(1))

    def test_old_survey_resumes_by_participant_id_and_keeps_progress(self):
        first=self.store.save(self.token,self.packet())
        pid=self.initial['participant_id']
        self.assertEqual(self.store.load('  '+pid.lower()+'  '),first)
        next_record=self.store.save(pid,self.packet(1))
        self.assertEqual(self.store.load(self.token),next_record)
        self.assertEqual(len(list(Path(self.temp.name).glob('survey_*'))),1)
        self.assertNotIn(self.token,folder_name(self.token))
        with self.assertRaises(ValueError):
            self.store.load('../outside')

    def test_new_survey_uses_short_participant_id(self):
        pid=new_participant_id()
        self.assertRegex(pid,r'^P-[A-F0-9]{6}$')
        created=self.store.create(pid,SCHEMA['version'])
        self.assertEqual(created['participant_id'],pid)
        self.assertEqual(normalize_access(' '+pid.lower()+' '),pid)
        saved=self.store.save(pid,self.packet())
        self.assertEqual(self.store.load(pid),saved)
        with self.assertRaises(FileExistsError):
            self.store.create(pid,SCHEMA['version'])

    def test_drive_finds_existing_hashed_folder_by_participant_id(self):
        drive=object.__new__(DriveStore)
        drive.service=MagicMock();drive.root='root';drive._folder_cache={}
        drive.service.files.return_value.list.return_value.execute.side_effect=[
            {'files':[]},
            {'files':[{'id':'older_folder','name':folder_name(self.token)}]},
            {'files':[{'id':'initial_record'}]}]
        drive._read=MagicMock(return_value=self.initial)
        pid=self.initial['participant_id']
        self.assertEqual(drive._folder(pid),'older_folder')
        self.assertEqual(drive._folder(pid),'older_folder')
        self.assertEqual(drive.service.files.return_value.list.call_count,3)

    def test_export_retains_edits_but_uses_last_answers(self):
        first=self.store.save(self.token,self.packet())
        next_packet=self.packet(1)
        next_packet['state']['answers']={'people':{'status':'skipped'}}
        second=self.store.save(self.token,next_packet)
        _,rows,events=combine([second,self.initial,first],SCHEMA)
        self.assertEqual(len(events),2)
        self.assertEqual(next(r for r in rows if r['question_id']=='people')['status'],'skipped')
        self.assertEqual(next(r for r in rows if r['question_id']=='aac_name')['status'],'not_applicable')

    def test_schema_keeps_ratings_and_scenarios_excludes_notes(self):
        pages=SCHEMA['pages'];ids=[p['id'] for p in pages]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(sum(i.startswith('feature_') for i in ids),6)
        self.assertEqual(sum(i.startswith('situation_') for i in ids),7)
        self.assertEqual(sum(i.startswith('s') and i.endswith('_action') for i in ids),5)
        self.assertNotIn('Notes to myself',json.dumps(pages))
        self.assertLess(max(i for p in pages for i in p.get('source_paragraphs',[0])),354)

    def test_optional_exercise_ends_part_3(self):
        pages=SCHEMA['pages'];ids=[p['id'] for p in pages];P={p['id']:p for p in pages}
        watched=P['feature_1']['when']
        start=ids.index('edit_intro')
        self.assertEqual(ids[start-1],'situation_7')
        self.assertEqual(ids[start:start+6],['edit_intro']+[f'e{n}_edit' for n in range(1,6)])
        self.assertEqual(ids[start+6],'closing')
        intro=P['edit_intro']
        self.assertEqual((intro['title'],intro['kind'],intro['when']),('Try the example situations','single',watched))
        self.assertEqual(intro['options'],['Yes, I’d like to try','No, skip the examples'])
        self.assertEqual(len(intro['paragraphs']),3)
        self.assertIn('“Re-check” function shown in the video',intro['paragraphs'][1])
        for n in range(1,6):
            edit,part2=P[f'e{n}_edit'],P[f's{n}_action']
            self.assertNotIn(f'e{n}_action',ids)
            self.assertEqual(SCHEMA['retired_pages'][f'e{n}_action'],f'e{n}_edit')
            self.assertEqual(edit['when'],{'all':[watched,{'question':'edit_intro','values':['Yes, I’d like to try']}]})
            self.assertEqual(edit['title'],part2['scenario']['title'])
            self.assertEqual('“'+edit['transcript']+'”',part2['scenario']['shown'])

    def test_questions_follow_the_september_29_guide(self):
        ids=[p['id'] for p in SCHEMA['pages']];P={p['id']:p for p in SCHEMA['pages']}
        self.assertEqual(SCHEMA['version'],'2026-09-29-v1')
        self.assertEqual(SCHEMA['source'],'refined_question_list_9.29.2026.docx')
        for gone in ['asr_stopped','asr_uses','aac_carry','story','first_repair','understood','next_repair','different_repair',
                     'stop_reason','partner_action','partner_expected']:
            self.assertNotIn(gone,ids)
        self.assertEqual(ids[ids.index('text_input')+1:ids.index('text_input')+3],['stop_trying','detect'])
        self.assertEqual(ids[ids.index('asr')+1],'aac')
        self.assertEqual(P['stop_trying']['exclusive'],['I have not stopped trying','Not sure'])
        self.assertEqual(P['text_input']['options'][:2],['Keyboard','A typing aid'])
        self.assertEqual(P['retry_count']['other_option'],'It depends on something else')
        self.assertEqual(P['retry_count']['when']['all'][1]['values'],[P['failed_repair']['options'][0]])
        self.assertTrue(P['look_when']['title'].startswith('Imagine you are using the tool shown in the demo video during a conversation.'))
        for p in SCHEMA['pages']:
            for o in p.get('options',[]):
                self.assertNotRegex(o,r'^- |_{2,}|:\s*$')

    def test_scenario_intro_wording(self):
        page=next(p for p in SCHEMA['pages'] if p['id']=='scenarios_intro')
        self.assertEqual(page['title'],'Example situations')
        self.assertEqual(page['paragraphs'],[
            'Imagine you are talking to someone. For these examples, suppose a device could show you the words the other person understood.',
            'You will see what you meant to say and what the other person understood. Choose what you would do first in each situation. You can skip any example.'])

    def test_every_transcript_word_has_six_candidates(self):
        for page in SCHEMA['pages']:
            if page['kind']!='edit':
                continue
            words=[re.sub(r"^[^\w’']+|[^\w’']+$","",t) for t in page['transcript'].split()]
            self.assertEqual([c['word'] for c in page['candidates']],words)
            for c in page['candidates']:
                self.assertEqual(len(c['options']),6)
                self.assertEqual(len(set(c['options'])),6)
                self.assertNotIn(c['word'],c['options'])

    def test_action_choices_are_grouped_on_separate_lines(self):
        for page in SCHEMA['pages']:
            if not re.fullmatch(r's[1-5]_action',page['id']):
                continue
            groups=page['option_groups']
            self.assertEqual([g['label'] for g in groups],
                ['Continue the conversation','Use my voice again','Use text',
                 'Use another way to communicate','Stop trying',''])
            grouped=[o for g in groups for o in g['options']]
            self.assertEqual(sorted(grouped+['Other']),sorted(page['options']))
            self.assertEqual(len(grouped),len(set(grouped)))

    def test_pre_demo_examples_have_no_follow_ups_and_old_screens_resume(self):
        ids={p['id'] for p in SCHEMA['pages']}
        for n in range(1,6):
            self.assertIn(f's{n}_action',ids)
            for kind in ('extent','words','leave'):
                self.assertNotIn(f's{n}_{kind}',ids)
                self.assertEqual(SCHEMA['retired_pages'][f's{n}_{kind}'],f's{n}_action')
        self.assertTrue(set(SCHEMA['retired_pages'].values())<=ids)

    def test_drive_retry_after_uncertain_upload_reuses_saved_batch(self):
        drive=object.__new__(DriveStore)
        drive.service=MagicMock()
        drive._folder=MagicMock(return_value='test_folder')
        drive._latest=MagicMock(return_value=self.initial)
        drive._write=MagicMock()
        drive.service.files.return_value.list.return_value.execute.return_value={'files':[]}
        packet=self.packet()
        first=drive.save(self.token,packet)
        drive._write.assert_called_once()
        drive.service.files.return_value.list.return_value.execute.return_value={'files':[{'id':'already_saved'}]}
        drive._read=MagicMock(return_value=first)
        self.assertEqual(drive.save(self.token,packet),first)
        drive._write.assert_called_once()

    def test_drive_write_failure_is_not_an_acknowledgement(self):
        drive=object.__new__(DriveStore)
        drive.service=MagicMock()
        drive._folder=MagicMock(return_value='test_folder')
        drive._latest=MagicMock(return_value=self.initial)
        drive._write=MagicMock(side_effect=OSError('simulated network loss'))
        drive.service.files.return_value.list.return_value.execute.return_value={'files':[]}
        with self.assertRaises(OSError):
            drive.save(self.token,self.packet())


if __name__=='__main__':
    unittest.main()
