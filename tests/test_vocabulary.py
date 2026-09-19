"""User outcomes, migration boundaries and transactional dictionary writes."""
from __future__ import annotations

import argparse
import copy
import io
import json
from contextlib import redirect_stdout

import pytest

from omavoi import config, modes, names, paths
from omavoi import vocabulary as v
from omavoi.commands import vocabulary as api
from omavoi.commands import words


def cfg_for(*entries):
    cfg = config.defaults()
    cfg['dictionary'] = {'schema_version': 2, 'revision': 0, 'entries': list(entries)}
    return cfg


@pytest.mark.parametrize(('before', 'after'), [
    ('javascript', 'JavaScript'), ('JAVASCRIPT', 'JavaScript'),
    ('I use javascript.', 'I use JavaScript.'),
    ('JavaScriptCore javascript_utils myjavascript', 'JavaScriptCore javascript_utils myjavascript'),
    ('用javascript开发', '用JavaScript开发'),
])
def test_capitalization_without_sound_matching(before, after):
    cfg = cfg_for(v.word('JavaScript'))
    assert v.Index(cfg).apply(before)[0] == after


@pytest.mark.parametrize(('term', 'text', 'expected'), [
    ('Visual Studio Code', 'use visual studio code', 'use Visual Studio Code'),
    ('C++', 'c++ c++17 c++_api', 'C++ c++17 c++_api'),
    ('.NET', '.net asp.net .network', '.NET asp.net .network'),
    ('Café', 'cafe\u0301!', 'Café!'),
    ('Straße', 'STRASSE!', 'Straße!'),
])
def test_whole_phrases_symbols_and_unicode_offsets(term, text, expected):
    assert v.Index(cfg_for(v.word(term))).apply(text)[0] == expected


def test_known_chinese_errors_and_protection():
    cfg = cfg_for(v.word('林晓彤', aliases=['林小童', '林晓童']))
    assert v.Index(cfg).apply('联系林小童和林晓童。')[0] == '联系林晓彤和林晓彤。'
    assert v.Index(cfg).apply('林晓彤')[1] == []


def test_name_only_does_not_enable_homophone_changes():
    cfg = cfg_for(v.word('林晓彤'))
    assert v.Index(cfg).apply('林小童')[0] == '林小童'
    assert v.Index(cfg).seed_text() == '林晓彤'


def test_phonetic_ties_and_correct_words_stay_untouched():
    one = v.word('李文渊', phonetic={'enabled': True, 'method': 'auto'})
    two = v.word('李文源', phonetic={'enabled': True, 'method': 'auto'})
    assert v.Index(cfg_for(one)).apply('李文远')[0] == '李文渊'
    both = v.Index(cfg_for(one, two))
    assert both.apply('李文远')[0] == '李文远'
    assert both.apply('李文源')[0] == '李文源'


@pytest.mark.parametrize('entries', [
    [v.word('A', aliases=['B']), v.word('B', aliases=['C'])],
    [v.word('A', aliases=['B']), v.word('B', aliases=['A'])],
    [v.word('Alpha', aliases=['wrong']), v.word('Beta', aliases=['WRONG'])],
    [v.word('iOS'), v.word('IOS')],
    [v.word('Alpha', aliases=['wrong']), v.word('wrong phrase')],
])
def test_conflicts_and_chains_are_rejected(entries):
    with pytest.raises(v.InvalidWord):
        v.validate(entries)


def test_disjoint_scopes_allow_different_meanings():
    entries = [v.word('Alpha', aliases=['wrong'], modes=['one']),
               v.word('Beta', aliases=['wrong'], modes=['two'])]
    v.validate(entries, {'one', 'two'})
    cfg = cfg_for(*entries)
    assert v.Index(cfg, 'one').apply('wrong')[0] == 'Alpha'
    assert v.Index(cfg, 'two').apply('wrong')[0] == 'Beta'


def test_off_means_no_hints_or_corrections():
    cfg = cfg_for(v.word('JavaScript', aliases=['java script']))
    index = v.Index(cfg, rules={'vocabulary': False})
    assert index.seed_text() == ''
    assert index.apply('java script javascript')[0] == 'java script javascript'
    cfg['dictionary']['entries'][0]['enabled'] = False
    assert v.Index(cfg).seed_text() == ''
    assert v.Index(cfg).apply('javascript')[0] == 'javascript'


def test_legacy_gates_are_preserved():
    cfg = cfg_for(v.word('JavaScript'))
    cfg['post']['enabled'] = False
    index = v.Index(cfg, rules={'names': True, 'dictionary': True})
    assert index.seed_text() == 'JavaScript'
    assert index.apply('javascript')[0] == 'javascript'
    # An explicit unified mode setting replaces the old mixed state.
    assert v.Index(cfg, rules={'vocabulary': True}).apply('javascript')[0] == 'JavaScript'


def test_migration_plan_is_pure_and_default_corrections_preserved():
    cfg = config.defaults()
    before = copy.deepcopy(cfg)
    data, issues = v.migration(cfg)
    assert not issues
    assert cfg == before
    assert len(data['entries']) == 9
    assert not any(e['recognition_hint'] for e in data['entries'])
    cfg['dictionary'] = data
    from omavoi.post.rules import apply_dictionary
    for source, target in before['dictionary']['rules'].items():
        for text in [source, source.upper(), f'I use {source}.', f'用{source}开发']:
            assert v.Index(cfg).apply(text)[0] == apply_dictionary(text, before['dictionary']['rules'])[0]
    assert v.migration(cfg) == (data, [])


def test_complex_migration_is_blocked_not_silently_changed():
    cfg = config.defaults()
    cfg['dictionary']['rules'] = {'A': 'B', 'B': 'C'}
    assert v.migration(cfg)[1]
    cfg['dictionary']['rules'] = {'wrong': 'Name'}
    cfg['dictionary']['names'] = [{'name': 'Name', 'modes': ['default']}]
    assert v.migration(cfg)[1]


def test_migrated_name_keeps_its_original_settings():
    cfg = config.defaults()
    cfg['dictionary']['rules'] = {}
    cfg['dictionary']['names'] = [{'name': 'JavaScript', 'seed': False, 'enabled': True,
                                  'group': 'work', 'modes': ['default']}]
    data, issues = v.migration(cfg)
    assert not issues
    e = data['entries'][0]
    assert not e['normalize_case'] and not e['recognition_hint']
    assert e['phonetic']['enabled'] and e['modes'] == ['default']
    assert e['legacy']['group'] == 'work'


def test_second_pass_stable_for_valid_deterministic_dictionary():
    entries = [v.word('JavaScript', aliases=['java script']), v.word('林晓彤', aliases=['林小童'])]
    v.validate(entries)
    index = v.Index(cfg_for(*entries))
    fragments = ['javascript', 'java script', 'JavaScriptCore', '林小童', '林晓彤', 'x', '.', '_']
    for a in fragments:
        for b in fragments:
            first = index.apply(a + ' ' + b)[0]
            assert index.apply(first)[0] == first


@pytest.fixture
def no_service(monkeypatch):
    monkeypatch.setattr(api.ipc, 'request', lambda *a, **kw: {'ok': False})


def test_save_migrates_once_preserves_backup_and_rejects_stale_write(home, no_service):
    config.write(config.defaults())
    original = paths.config_file().read_bytes()
    first = api.snapshot(config.load())
    assert first['migration_pending']
    result = api.mutate('save', {'etag': first['etag'], 'entry': {'text': 'Visual Studio Code'}})
    assert result['activation'] == 'pending'
    cfg = config.load()
    assert v.modern(cfg) and 'rules' not in cfg['dictionary'] and 'names' not in cfg['dictionary']
    backups = list(paths.config_dir().glob('*.before-vocabulary-*'))
    assert len(backups) == 1 and backups[0].read_bytes() == original
    with pytest.raises(v.InvalidWord, match='changed_elsewhere'):
        api.mutate('save', {'etag': first['etag'], 'entry': {'text': 'Something'}})
    assert len(api.snapshot(cfg)['entries']) == 10
    assert 'Something' not in [e['text'] for e in config.load()['dictionary']['entries']]


def test_delete_restore_and_no_default_resurrection(home, no_service):
    cfg = cfg_for(v.word('JavaScript', aliases=['java script']))
    config.write(cfg)
    e = cfg['dictionary']['entries'][0]
    result = api.mutate('remove', {'etag': api.snapshot(config.load())['etag'], 'id': e['id']})
    assert config.load()['dictionary']['entries'] == []
    restored = api.mutate('restore', {'etag': result['etag'], 'entry': result['removed']})
    assert restored['entries'][0]['id'] == e['id']
    assert restored['entries'][0]['aliases'] == ['java script']


def test_edit_keeps_id_and_updates_all_targets(home, no_service):
    cfg = cfg_for(v.word('OldName', aliases=['misheard', 'wrong word']))
    config.write(cfg)
    e = copy.deepcopy(cfg['dictionary']['entries'][0])
    e['text'] = 'NewName'
    result = api.mutate('save', {'etag': api.snapshot(config.load())['etag'], 'entry': e})
    assert result['entries'][0]['id'] == e['id']
    assert v.Index(config.load()).apply('misheard wrong word')[0] == 'NewName NewName'


def test_failed_validation_leaves_file_unchanged(home, no_service):
    config.write(cfg_for(v.word('JavaScript')))
    before = paths.config_file().read_bytes()
    with pytest.raises(v.InvalidWord):
        api.mutate('save', {'etag': api.snapshot(config.load())['etag'], 'entry': {'text': 'JAVASCRIPT'}})
    assert paths.config_file().read_bytes() == before


def test_preview_uses_text_cleanup_and_never_writes(home):
    cfg = cfg_for(v.word('JavaScript'))
    config.write(cfg)
    before = paths.config_file().read_bytes()
    result = api.preview(cfg, {'text': 'use javascript', 'mode': 'default'})
    from omavoi import post
    mode = modes.resolve(cfg, forced='default')
    text = post.run('use javascript', cfg, post.Context('', '', mode.rules)).text
    assert result['after'] == v.Index(cfg, mode.name, mode.rules).apply(text)[0]
    assert paths.config_file().read_bytes() == before


def test_legacy_commands_share_entries_without_destroying_other_uses(home):
    config.write(cfg_for(v.word('JavaScript', aliases=['java script'])))
    args = argparse.Namespace(action='rm', heard='java script', json=False)
    assert words.cmd_dict(args) == 0
    e = config.load()['dictionary']['entries'][0]
    assert e['recognition_hint'] and not e['aliases']
    args = argparse.Namespace(action='rm', names=['JavaScript'], group='', json=False)
    assert words.cmd_names(args) == 0
    e = config.load()['dictionary']['entries'][0]
    assert e['normalize_case'] and not e['recognition_hint']
    assert names.load(config.load()) == []


def test_structured_errors_keep_stdout_parseable(home, monkeypatch):
    monkeypatch.setattr('sys.stdin', io.StringIO('{broken'))
    args = argparse.Namespace(action='save', json_input=True)
    out = io.StringIO()
    with redirect_stdout(out):
        assert api.cmd_vocabulary(args) == 1
    assert json.loads(out.getvalue())['ok'] is False


def test_post_stage_keeps_spacing_and_punctuation_after_replacement():
    from omavoi import post
    cfg = cfg_for(v.word('Python', aliases=['派森']))
    result = post.run('我用派森。', cfg, post.Context('', '', {'punctuation': 'strip'}, 'default'))
    assert result.text == '我用 Python'
    assert result.word_changes[0]['before'] == '派森'


def test_wrong_mode_cannot_change_text_in_post_stage():
    from omavoi import post
    cfg = cfg_for(v.word('JavaScript', modes=['code']))
    assert post.run('javascript', cfg, post.Context('', '', {}, 'default')).text == 'javascript'
    assert post.run('javascript', cfg, post.Context('', '', {}, 'code')).text == 'JavaScript'


def test_no_migration_while_old_daemon_is_running(home, monkeypatch):
    config.write(config.defaults())
    before = paths.config_file().read_bytes()
    monkeypatch.setattr(api.ipc, 'ping', lambda **kw: {'ok': True})
    with pytest.raises(v.InvalidWord, match='restart_required'):
        api.mutate('save', {'etag': api.snapshot(config.load())['etag'], 'entry': {'text': 'Acme'}})
    assert paths.config_file().read_bytes() == before


def test_unknown_edit_id_cannot_silently_create_a_word(home, no_service):
    config.write(cfg_for())
    with pytest.raises(v.InvalidWord, match='not_found'):
        api.mutate('save', {'etag': api.snapshot(config.load())['etag'], 'entry': v.word('Missing')})


def test_legacy_case_rule_can_be_removed_without_removing_name(home):
    cfg = config.defaults()
    cfg['dictionary']['names'] = ['JavaScript']
    data, issues = v.migration(cfg)
    assert not issues
    cfg['dictionary'] = data
    config.write(cfg)
    assert words.cmd_dict(argparse.Namespace(action='rm', heard='javascript', json=False)) == 0
    e = next(e for e in config.load()['dictionary']['entries'] if e['text'] == 'JavaScript')
    assert e['recognition_hint'] and not e['normalize_case']


def test_pipeline_uses_one_word_stage_before_optional_ai(home, monkeypatch):
    from types import SimpleNamespace

    import numpy as np

    from omavoi import pipeline
    from omavoi.asr.base import Transcript
    from omavoi.window import Window

    cfg = cfg_for(v.word('JavaScript'))
    cfg['history']['enabled'] = False
    cfg['ui']['notify'] = False
    cfg['modes']['default']['steps'] = [{'llm': 'local', 'prompt': 'translate'}]
    prompts = []

    class Backend:
        def transcribe(self, *a, **kw):
            prompts.append(kw['prompt'])
            return Transcript(text='javascript')

    cap = SimpleNamespace(samples=np.zeros(16000, dtype=np.float32), rate=16000,
                          seconds=1, peak_dbfs=-10, rms_dbfs=-15, preroll_seconds=0.1,
                          tail_seconds=0, truncated=False)
    p = pipeline.Pipeline(cfg, Backend())
    monkeypatch.setattr(p, '_finish', lambda entry, *a, **kw: entry)
    seen = []
    def transform(text, mode, entry):
        seen.append(text)
        return 'javascript translated'
    monkeypatch.setattr(p, '_run_steps', transform)
    result = p.process(cap, inject=False, forced_mode='default', window=Window())
    assert prompts == ['JavaScript']
    assert seen == ['JavaScript']
    assert result['rules_text'] == 'JavaScript'
    assert result['text'] == 'javascript translated'
    assert result['names'][0]['reason'] == 'case'


def test_longer_words_are_not_false_conflicts():
    entries = [v.word('Git'), v.word('GitHub'), v.word('JavaScript'), v.word('JavaScriptCore')]
    v.validate(entries)
    assert v.Index(cfg_for(*entries)).apply('git github javascriptcore')[0] == 'Git GitHub JavaScriptCore'


def test_migration_can_be_rolled_back_with_both_versions_backed_up(home, no_service):
    config.write(config.defaults())
    before = paths.config_file().read_bytes()
    result = api.mutate('save', {'etag': api.snapshot(config.load())['etag'], 'entry': {'text': 'Acme'}})
    backup = next(paths.config_dir().glob('*.before-vocabulary-*'))
    api.rollback({'etag': result['etag'], 'backup': backup.name})
    assert paths.config_file().read_bytes() == before
    assert list(paths.config_dir().glob('*.before-rollback-*'))


def test_parallel_editors_cannot_overwrite_each_other(home, no_service):
    from concurrent.futures import ThreadPoolExecutor
    config.write(cfg_for())
    token = api.snapshot(config.load())['etag']
    def attempt(text):
        try:
            api.mutate('save', {'etag': token, 'entry': {'text': text}})
            return 'ok'
        except v.InvalidWord as exc:
            return exc.code
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, ['Alpha', 'Beta']))
    assert sorted(results) == ['changed_elsewhere', 'ok']
    assert len(config.load()['dictionary']['entries']) == 1


def test_other_setting_writers_cannot_overwrite_a_new_dictionary(home, no_service):
    config.write(cfg_for())
    stale = config.load()
    api.mutate('save', {'etag': api.snapshot(stale)['etag'], 'entry': {'text': 'Acme'}})
    stale['ui']['language'] = 'zh'
    with pytest.raises(ValueError, match='changed elsewhere'):
        config.write(stale)
    assert config.load()['dictionary']['entries'][0]['text'] == 'Acme'


def test_new_dictionary_cannot_overwrite_other_settings(home, no_service):
    config.write(cfg_for())
    stale = api.snapshot(config.load())
    config.set_path('ui.language', 'zh')
    with pytest.raises(v.InvalidWord, match='changed_elsewhere'):
        api.mutate('save', {'etag': stale['etag'], 'entry': {'text': 'Acme'}})
    assert config.load()['ui']['language'] == 'zh'
    assert config.load()['dictionary']['entries'] == []
