import hashlib
import json

import pytest

from eeg_agent.domain_knowledge import HierarchicalKnowledge


def isolated(tmp_path):
    return HierarchicalKnowledge({'brain_knowledge_root': tmp_path / 'bundled',
                                  'brain_knowledge_user_root': tmp_path / 'uploaded'})


def test_shared_and_selected_domain_isolation_and_persistence(tmp_path):
    kb = isolated(tmp_path)
    for domain in ('shared', 'vrms', 'fatigue', 'emotion'):
        kb.add_document(domain, domain, 'theta alpha evidence for ' + domain, domain + '.txt')
    selected = kb.search('theta alpha', domains=['fatigue'])
    assert {c['domain'] for c in selected['results']} == {'shared', 'fatigue'}
    assert selected['domains'] == ['shared', 'fatigue']
    assert selected['stages'][0]['candidate_limit'] == 20
    assert selected['stages'][1]['dense_or_cross_encoder'] is False
    restored = isolated(tmp_path)
    assert restored.search('theta alpha', domains=['fatigue']) == selected
    assert restored.summary()['domains']['emotion']['sources'] == 1


def test_uploaded_source_hash_and_stable_citation(tmp_path):
    kb = isolated(tmp_path)
    text = 'A supplied EEG paragraph with alpha evidence.'
    added = kb.add_document('emotion', 'Supplied paper', text, 'https://example.test/paper',
                            doi='10.0000/example', page='4')
    result = kb.search('alpha', domains=['emotion'])['results'][0]
    assert added['source_sha256'] == hashlib.sha256(text.encode()).hexdigest()
    assert result['source_sha256'] == added['source_sha256']
    assert result['doi'] == '10.0000/example' and result['page'] == '4'
    assert result['evidence_type'] == 'user_supplied'
    assert result['content_sha256'] == hashlib.sha256(result['text'].encode()).hexdigest()
    kb.add_document('emotion', 'Supplied paper', text, 'https://example.test/paper')
    assert kb.search('alpha', domains=['emotion'])['results'][0]['id'] == result['id']
    assert kb.summary()['sources'] == 1


def test_no_match_does_not_return_unrelated_evidence(tmp_path):
    kb = isolated(tmp_path)
    kb.add_document('shared', 'EEG', 'alpha power', 'protocol')
    result = kb.search('qzxvnty', domains=[])
    assert result['status'] == 'insufficient' and result['results'] == []
    assert kb.search('', domains=[])['results'] == []
    assert kb.search('alpha', limit=0)['results'] == []


def test_candidates_and_evidence_are_bounded_and_deterministic(tmp_path):
    kb = isolated(tmp_path)
    for i in range(25):
        kb.add_document('fatigue', 'Item ' + str(i), 'theta alpha reference ' + str(i), 'source ' + str(i))
    result = kb.search('theta alpha', domains=['fatigue'], limit=100)
    assert result['stages'][0]['candidates'] == 20
    assert len(result['results']) == 5
    assert kb.search('theta alpha', domains=['fatigue'], limit=100) == result
    json.dumps(result, allow_nan=False)


def test_upload_cannot_escape_domain_or_claim_verified_status(tmp_path):
    kb = isolated(tmp_path)
    for domain in ('../emotion', 'unknown'):
        with pytest.raises(ValueError):
            kb.add_document(domain, 'x', 'y', 'z')
    with pytest.raises(ValueError):
        kb.add_document('emotion', 'x', 'y', 'z', evidence_type='local_protocol')
    with pytest.raises(ValueError):
        kb.add_document('emotion', 'x', 'x' * 200001, 'z')
    with pytest.raises(ValueError):
        kb.search('alpha', domains=['unknown'])


def test_bundled_corpus_carries_paper_pages_and_local_protocols(tmp_path):
    kb = HierarchicalKnowledge({'brain_knowledge_user_root': tmp_path / 'uploaded'})
    summary = kb.summary()
    assert all(summary['domains'][domain]['sources'] > 0 for domain in ('shared', 'vrms', 'fatigue', 'emotion'))
    results = kb.search('Emotion Copilot', domains=['emotion'])['results']
    paper = next(item for item in results if item['evidence_type'] == 'paper_excerpt')
    assert paper['source_sha256'] == '4cd6f424a6f69991091c90ebfaf01218c299764857b1f9d54322f3ad51461a1a'
    assert '14' in paper['page'] and paper['url'].startswith('https://arxiv.org/')
    assert summary['retrieval_method'] == 'two_stage_bm25_lexical_fallback'
