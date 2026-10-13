"""Frozen numerical feature definitions and one-time per-window extraction."""
import numpy as np
from scipy import signal, linalg
from scipy.special import logit
from sklearn.base import BaseEstimator, TransformerMixin

FS = 256
BANDS = ((.5, 4), (4, 8), (8, 13), (13, 30))
REGIONS = ((0, 1), tuple(range(2, 9)), tuple(range(9, 14)),
           tuple(range(14, 19)), tuple(range(19, 27)), (27, 28, 29))
FILTERBANK_SOS = tuple(signal.butter(3, [low, high], btype='bandpass', fs=FS, output='sos')
                      for low, high in BANDS)


def spectral_features(x, spectrum=None):
    if spectrum is None:
        frequency, psd = signal.welch(x, fs=FS, nperseg=512, noverlap=256, axis=1)
    else:
        frequency, psd = map(np.asarray, spectrum)
        if psd.shape == (len(frequency), 30):
            psd = psd.T
        if (frequency.shape != (257,) or psd.shape != (30, 257)
                or not np.array_equal(frequency, np.fft.rfftfreq(512, 1 / FS))
                or not np.isfinite(psd).all() or np.any(psd < 0)):
            raise ValueError('Shared spectrum must be a finite 256-Hz, 512-sample Welch PSD')
    power = np.stack([np.trapezoid(psd[:, (frequency >= low) & (frequency < high)],
                                   frequency[(frequency >= low) & (frequency < high)], axis=1)
                      for low, high in BANDS], axis=1)
    power = np.maximum(power, 1e-10)
    relative = power / np.maximum(power.sum(axis=1, keepdims=True), 1e-10)
    return np.concatenate((np.log(power).ravel(), relative.ravel())).astype(np.float32), power


def covariance(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    c = x @ x.T / (x.shape[1] - 1)
    return .9 * c + .1 * max(float(np.trace(c)) / len(c), 1e-10) * np.eye(len(c))


def predict_score(model, values):
    if hasattr(model, 'decision_function'):
        return np.asarray(model.decision_function(values), float)
    return logit(np.clip(model.predict_proba(values)[:, 1], 1e-6, 1 - 1e-6))


def compact_spectrum(absolute, reference=None):
    a = np.asarray(absolute, float)
    logpower, relative = a[:, :120].reshape(-1, 30, 4), a[:, 120:].reshape(-1, 30, 4)

    def regional(values):
        return np.stack([values[:, group, :].mean(axis=1) for group in REGIONS], axis=1)

    logs, rel = regional(logpower), regional(relative)
    n = max(1, len(a) // 3)
    changes = np.zeros(24) if reference is None else regional(
        np.asarray(reference).reshape(-1, 30, 4)).mean(axis=0).ravel()
    centered = logs - logs.mean(axis=2, keepdims=True)
    return np.r_[rel.mean(axis=0).ravel(), rel.std(axis=0).ravel(),
                 centered.mean(axis=0).ravel(),
                 centered[-n:].mean(axis=0).ravel() - centered[:n].mean(axis=0).ravel(),
                 changes, float(reference is not None)]


def filterbank_window(window):
    result = []
    for sos in FILTERBANK_SOS:
        values = signal.sosfilt(sos, np.asarray(window, float), axis=-1)
        values -= values.mean(axis=-1, keepdims=True)
        cov = values @ values.T / (values.shape[-1] - 1)
        cov /= max(float(np.trace(cov)), 1e-10)
        result.append(cov)
    return np.asarray(result)


def filterbank_covariance(windows):
    return np.asarray([filterbank_window(w) for w in windows]).mean(axis=0).ravel()


def cache_window(window, spectrum=None):
    absolute, power = spectral_features(window, spectrum=spectrum)
    return {'absolute': absolute, 'powers': power, 'covariance': covariance(window),
            'filterbank': filterbank_window(window)}


class FilterBankCSP(BaseEstimator, TransformerMixin):
    """Serialization-compatible frozen CSP; inference never calls fit."""
    def __init__(self, components=4, shrinkage=.2):
        self.components, self.shrinkage = components, shrinkage

    def fit(self, x, y):
        values = np.asarray(x).reshape(-1, 4, 30, 30)
        if set(np.unique(y)) != {0, 1}:
            raise ValueError('CSP requires both training classes')
        self.filters_ = []
        for band in range(4):
            matrices = []
            for label in (0, 1):
                c = values[np.asarray(y) == label, band].mean(axis=0)
                matrices.append((1 - self.shrinkage) * c + self.shrinkage * np.trace(c) / 30 * np.eye(30))
            _, vectors = linalg.eigh(matrices[1], matrices[0] + matrices[1])
            n = self.components // 2
            self.filters_.append(vectors[:, list(range(n)) + list(range(30 - n, 30))].T)
        self.filters_ = np.asarray(self.filters_)
        return self

    def transform(self, x):
        values = np.asarray(x).reshape(-1, 4, 30, 30)
        variances = np.einsum('bkc,nbcd,bkd->nbk', self.filters_, values, self.filters_)
        variances = np.maximum(variances, 1e-10)
        return np.log(variances / variances.sum(axis=2, keepdims=True)).reshape(len(values), -1)
