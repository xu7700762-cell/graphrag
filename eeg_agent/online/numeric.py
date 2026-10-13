"""Same eight frozen D2 tools, aggregating cached window features only."""
import json
import numpy as np
from scipy.special import expit, logit

from .features import compact_spectrum, predict_score

SECOND_FAMILIES = ('femba_mean', 'femba_temporal', 'spectral_compact', 'filterbank_csp')


class FrozenNumeric:
    def __init__(self, fold, assets):
        self.fold, self.assets = fold, assets
        self.first, a = assets.bundle(fold, 'first')
        self.second, b = assets.bundle(fold, 'second')
        self.selector, c = assets.bundle(fold, 'selector')
        if self.first['split'] != self.second['split'] or self.first['split'] != self.selector['split']:
            raise ValueError('Numerical model roles differ')
        bank = self.first['tools']['case_retrieval']
        if fold in bank['reference_subjects']:
            raise ValueError('Query subject in frozen estimator support')
        self.reference_features = bank['scaler'].transform(bank['reference_features'])
        self.hashes = {'numeric_first_sha256': a, 'numeric_second_sha256': b,
                       'numeric_selector_sha256': c}
        audit = assets.optional(fold, 'nested_audit')
        self.audit = json.loads(audit.read_text(encoding='utf-8')) if audit else None

    def predict(self, cached, features, mil_probability):
        if len(cached) < 3 or len(features) != len(cached):
            raise ValueError('Numerical path evidence requires at least three matched windows')
        a = np.asarray([row['absolute'] for row in cached], float)
        n = max(1, len(a) // 3)
        spectrum = np.r_[a.mean(axis=0), a.std(axis=0),
                         a[-n:, :120].mean(axis=0) - a[:n, :120].mean(axis=0), np.zeros(120), 0.]
        covs = np.asarray([row['covariance'] for row in cached])
        scales = np.trace(covs, axis1=1, axis2=2) / 30
        mean_cov = (covs / scales[:, None, None]).mean(axis=0)
        eigen, vectors = np.linalg.eigh(mean_cov)
        log_cov = (vectors * np.log(np.maximum(eigen, 1e-10))) @ vectors.T
        ii, jj = np.triu_indices(30)
        spatial = np.r_[log_cov[ii, jj] * np.where(ii == jj, 1., np.sqrt(2.)),
                        np.log(scales).mean(), np.log(scales).std(),
                        np.log(np.diagonal(covs, axis1=1, axis2=2)).mean(axis=0)]
        feature_mean, feature_std = features.mean(axis=0), features.std(axis=0)
        chunks = np.array_split(features, 3)
        matrices = {'femba_features': np.r_[feature_mean, feature_std], 'spectral': spectrum,
                    'spatial_covariance': spatial, 'femba_mean': feature_mean,
                    'femba_temporal': np.r_[feature_mean, feature_std,
                                           chunks[-1].mean(axis=0) - chunks[0].mean(axis=0)],
                    'spectral_compact': compact_spectrum(a),
                    'filterbank_csp': np.asarray([row['filterbank'] for row in cached]).mean(axis=0).ravel()}
        scores, evidence = {}, []
        for family, x in matrices.items():
            bundle = self.second if family in SECOND_FAMILIES else self.first
            score = float(predict_score(bundle['tools'][family]['model'], x[None])[0])
            scores[family] = score
            evidence.append({'tool': family, 'raw_decision_score': score,
                             'predicted_class': 'High' if score >= 0 else 'Low'})
        bank = self.first['tools']['case_retrieval']
        query = bank['scaler'].transform(feature_mean[None])[0]
        distance = np.sqrt(np.mean((self.reference_features - query) ** 2, axis=1))
        k, weighted = bank['config']
        chosen, seen = [], set()
        for index in np.argsort(distance, kind='stable'):
            subject = int(bank['reference_subjects'][index])
            if subject not in seen:
                chosen.append(int(index)); seen.add(subject)
            if len(chosen) == k:
                break
        weights = 1 / np.maximum(distance[chosen], 1e-6) if weighted else np.ones(len(chosen))
        vote = float(np.average(bank['reference_labels'][chosen], weights=weights))
        scores['case_retrieval'] = float(logit(np.clip(vote, 1e-5, 1 - 1e-5)))
        evidence.append({'tool': 'case_retrieval', 'raw_decision_score': scores['case_retrieval'],
                         'predicted_class': 'High' if vote >= .5 else 'Low'})
        combined = [float(logit(np.clip(mil_probability, 1e-6, 1 - 1e-6)))]
        for name in self.selector['feature_names'][1:]:
            score = scores[name]
            if name in SECOND_FAMILIES:
                score = float(logit(np.clip(expit(score), 1e-6, 1 - 1e-6)))
            combined.append(score)
        model, x = self.selector['model'], np.asarray(combined)[None]
        probability = (float(expit(model.decision_function(x))[0]) if hasattr(model, 'decision_function')
                       else float(model.predict_proba(x)[0, 1]))
        profiles = {}
        for item in evidence:
            family = item['tool']
            bundle = self.second if family in SECOND_FAMILIES else self.first
            fitted = bundle['tools'][family]
            calibrator = fitted.get('calibrator')
            item['calibration_usable'] = calibrator is not None
            item['calibrated_high_probability'] = (None if calibrator is None else
                float(calibrator.predict_proba([[item['raw_decision_score']]])[0, 1]))
            report = bundle.get('reports', {}).get(family, {})
            profiles[family] = {key: report[key] for key in
                ('policy', 'policy_correction', 'raw_policy', 'calibrated_policy',
                 'raw_correction', 'calibrated_correction') if key in report}
        case = evidence[-1]
        case.update(descriptive_vote=vote, high_count=int(np.sum(bank['reference_labels'][chosen] == 1)),
                    low_count=int(np.sum(bank['reference_labels'][chosen] == 0)),
                    different_subjects=len(chosen), closest_distance=float(distance[chosen].min()),
                    training_case_context={'schema': 'raw_case_rag_v1', 'available': False,
                                           'reason': 'online_optional_training_case_bank_not_packaged'})
        report = self.selector.get('report', {})
        profiles['corrective_evidence'] = {key: report[key] for key in
                                          ('policy_gated', 'correction', 'margin') if key in report}
        if self.audit is not None:
            profiles['corrective_evidence']['independent_nested_audit'] = {
                key: self.audit[key] for key in ('approved', 'nested_validation', 'correction')}
        return {'raw_combined_score': probability, 'predicted_class': 'High' if probability >= .5 else 'Low',
                'probability_calibrated': False, 'actual_tools': evidence, 'reference_available': False,
                'tool_validation_profiles': profiles}
