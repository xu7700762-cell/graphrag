from dataclasses import dataclass
import numpy as np
from scipy import signal
from .config import EEG_CHANNELS

def filter_sos():
    return np.vstack([signal.tf2sos(*signal.iirnotch(50,30,fs=1024)),
                      signal.butter(4,[.5,45],btype='bandpass',fs=1024,output='sos'),
                      signal.iirdesign(100,128,.1,80,fs=1024,ftype='ellip',output='sos')])

class CausalFilter:
    def __init__(self, channels, cfg):
        required = EEG_CHANNELS + ['M1','M2','VEOG','HEOG']
        if any(x not in channels for x in required): raise ValueError('Missing required signal channel')
        self.eeg_ix = [channels.index(x) for x in EEG_CHANNELS]
        self.ref_ix = [channels.index('M1'),channels.index('M2')]
        self.eog_ix = [channels.index('VEOG'),channels.index('HEOG')]
        self.sos = filter_sos(); self.cfg = cfg; self.next_sample = None
        self.zi = np.zeros((len(self.sos),2,32)); self.warm_until = int(cfg['warmup_seconds']*1024)

    def process(self, start, raw):
        if self.next_sample is not None and start<self.next_sample: raise ValueError('Out-of-order sample chunk')
        gap = self.next_sample is not None and start != self.next_sample
        if gap or self.next_sample is None:
            self.zi[:] = 0; self.warm_until = start + int(self.cfg['warmup_seconds']*1024)
        self.next_sample = start + len(raw)
        eeg = raw[:,self.eeg_ix] - raw[:,self.ref_ix].mean(axis=1,keepdims=True)
        values = np.column_stack([eeg,raw[:,self.eog_ix]])
        finite = np.isfinite(values).all(axis=1)
        output=np.zeros_like(values); validity=np.zeros(len(values),dtype=bool)
        pointer=0
        while pointer<len(values):
            if not finite[pointer]:
                # Reset at the anomaly, never at the beginning of its containing block.
                self.zi[:]=0; self.warm_until=start+pointer+1+int(self.cfg['warmup_seconds']*1024)
                pointer+=1; continue
            end=pointer+1
            while end<len(values) and finite[end]: end+=1
            output[pointer:end],self.zi=signal.sosfilt(self.sos,values[pointer:end],axis=0,zi=self.zi)
            validity[pointer:end]=np.arange(start+pointer,start+end)>=self.warm_until
            pointer=end
        indices=np.arange(start,start+len(raw)); take=indices%4==0
        return output[take],indices[take],validity[take],raw[:,self.eeg_ix],gap or not finite.all()

@dataclass
class Calibration:
    available: bool = False
    reason: str = 'not completed'
    B: object = None
    mean: object = None
    samples: int = 0
    cutoff_sample: int = 0
    condition: float = 0

    def clean(self, x, e):
        if not self.available: return x.copy()
        return x - (e-self.mean) @ self.B.T

    def summary(self):
        return {'available':self.available,'reason':self.reason,'samples':self.samples,
                'cutoff_sample':self.cutoff_sample,'condition':self.condition}

def fit_calibration(values, cutoff, cfg):
    c = Calibration(samples=len(values),cutoff_sample=cutoff)
    if len(values) < cfg['calibration_min_seconds']*256:
        c.reason = 'insufficient past baseline'; return c
    x,e = values[:,:30],values[:,30:]; e0=e-e.mean(axis=0)
    if not np.isfinite(values).all() or np.any(e.std(axis=0)<.1) or np.any(np.ptp(x,axis=0)>cfg['qc']['peak_to_peak_uv']):
        c.reason='invalid baseline amplitude or EOG variation'; return c
    covariance=e0.T@e0/len(e0); cond=float(np.linalg.cond(covariance))
    c.condition=cond if np.isfinite(cond) else 0
    if not np.isfinite(cond) or cond>1e4:
        c.reason='ill-conditioned EOG baseline'; return c
    ridge=1e-3*np.trace(covariance)/2
    B=np.linalg.solve(covariance+ridge*np.eye(2),e0.T@(x-x.mean(axis=0))/len(e0)).T
    if not np.isfinite(B).all() or np.max(np.abs(B))>50:
        c.reason='unstable regression'; return c
    c.available=True; c.reason='frozen prior unlabeled baseline'; c.B=B; c.mean=e.mean(axis=0)
    return c

def _measurements(values,fs,thresholds):
    safe=np.nan_to_num(values,nan=0,posinf=0,neginf=0)
    p2p=np.ptp(safe,axis=0);std=safe.std(axis=0)
    jumps=float(np.mean(np.abs(np.diff(safe,axis=0))>thresholds['jump_uv'])) if len(safe)>1 else 0.
    f,p=signal.welch(safe,fs=fs,nperseg=min(int(fs),len(safe)),axis=0)
    denominator=np.sum(p[(f>=1)&(f<=min(100,fs/2))])
    line=float(np.sum(p[(f>=48)&(f<=52)])/max(denominator,1e-12))
    return {'peak_to_peak_uv':float(p2p.max()),'flat_channel_fraction':float(np.mean(std<thresholds['flat_std_uv'])),
            'jump_fraction':jumps,'line_power_ratio':line}

def _eog_correlation(x,e):
    a=x-x.mean(axis=0);b=e-e.mean(axis=0)
    corr=a.T@b/np.maximum(np.sqrt(np.sum(a*a,axis=0))[:,None]*np.sqrt(np.sum(b*b,axis=0))[None,:],1e-12)
    return float(np.max(np.abs(corr[:9])))

def quality(x,e,raw,cfg,observed_end,raw_eog=None):
    reasons=[]; thresholds=cfg['qc']
    if not len(raw) or not np.isfinite(raw).all() or not np.isfinite(x).all() or not np.isfinite(e).all(): reasons.append('missing_or_nonfinite')
    safe=np.nan_to_num(raw,nan=0,posinf=0,neginf=0) if len(raw) else np.zeros((1,30))
    p2p=np.ptp(safe,axis=0); std=safe.std(axis=0)
    flat=float(np.mean(std<thresholds['flat_std_uv']))
    jumps=float(np.mean(np.abs(np.diff(safe,axis=0))>thresholds['jump_uv'])) if len(safe)>1 else 0
    # Large electrode DC offsets are not evidence of amplifier saturation.
    # This is a provisional dynamic-range guard, not a calibrated hardware rail.
    if p2p.max()>thresholds['raw_peak_to_peak_uv']: reasons.append('raw_range_exceeded')
    safe_x=np.nan_to_num(x,nan=0,posinf=0,neginf=0);safe_e=np.nan_to_num(e,nan=0,posinf=0,neginf=0)
    processed_p2p=np.ptp(safe_x,axis=0)
    if processed_p2p.max()>thresholds['peak_to_peak_uv']: reasons.append('amplitude')
    if flat>0: reasons.append('flat_required_channel')
    if jumps>.001: reasons.append('abrupt_jump')
    raw_metrics=_measurements(safe,1024,thresholds);processed_metrics=_measurements(safe_x,256,thresholds)
    line=raw_metrics['line_power_ratio']
    a=safe_x-safe_x.mean(axis=0); b=safe_e-safe_e.mean(axis=0)
    corr=a.T@b/np.maximum(np.sqrt(np.sum(a*a,axis=0))[:,None]*np.sqrt(np.sum(b*b,axis=0))[None,:],1e-12)
    raw_metrics['frontal_eog_correlation']=_eog_correlation(safe,np.nan_to_num(raw_eog)) if raw_eog is not None and len(raw_eog)==len(safe) else None
    processed_metrics['frontal_eog_correlation']=float(np.max(np.abs(corr[:9])))
    return {'observed_end_sample':int(observed_end), 'unit':'uV','valid':not reasons,'reasons':reasons,
            'peak_to_peak_uv':float(p2p.max()),'processed_peak_to_peak_uv':float(processed_p2p.max()),
            'flat_channel_fraction':flat,'jump_fraction':jumps,'line_power_ratio':line,
            'line_warning':line>thresholds['line_ratio_warning'],'frontal_eog_correlation':float(np.max(np.abs(corr[:9]))),
            'eog_std_uv':[float(v) for v in safe_e.std(axis=0)],'raw_metrics':raw_metrics,'processed_metrics':processed_metrics}

class StreamPipeline:
    """Arrived-only continuous processing, with half-open complete raw windows."""
    def __init__(self, channels, cfg, on_window, on_path, choose_profile=None, on_baseline=None, on_gap=None, on_calibration=None):
        self.cfg=cfg; self.filter=CausalFilter(channels,cfg); self.on_window=on_window; self.on_path=on_path
        self.on_baseline=on_baseline; self.on_gap=on_gap
        self.on_calibration=on_calibration
        self.choose_profile=choose_profile or (lambda start:'P0'); self.calibration=Calibration()
        self.eog_cleaner=None; self.eog_snapshots={}; self.eog_update_last_raw=None; self.eog_jump_tail=np.zeros(3,dtype=bool)
        self.cal_parts=[]; self.cal_count=0; self.first_task=False; self.active=None; self.last_observed=0
        self.clear_window()

    def clear_window(self):
        self.values=[]; self.indices=[]; self.valid=[]; self.raw_parts=[]; self.raw_starts=[]; self.count=0; self.profile=None
        self.eog_snapshots={}

    def event(self,sample,mark):
        if mark==20:
            if self.active is not None: self.finish(sample,False)
            if not self.first_task:
                values=np.concatenate(self.cal_parts) if self.cal_parts else np.empty((0,32))
                fitter=fit_calibration
                if self.cfg.get('classification_profile','P0')=='P1':
                    from .eog import fit_eog_calibration
                    fitter=fit_eog_calibration
                self.calibration=fitter(values,sample,self.cfg); self.cal_parts=[]; self.first_task=True
                if self.cfg.get('classification_profile','P0')=='P1' and self.calibration.available:
                    from .eog import AdaptiveEOGRegression
                    self.eog_cleaner=AdaptiveEOGRegression(self.calibration,self.cfg)
                if self.on_calibration: self.on_calibration(self.calibration)
            self.active=sample; self.clear_window()
        elif mark==22 and self.active is not None: self.finish(sample,True)

    def finish(self,end,complete):
        self.on_path({'start_sample':self.active,'end_sample':end,'complete':complete,'duration_seconds':(end-self.active)/1024})
        self.active=None; self.clear_window()

    def feed(self,start,raw,arrived_events=()):
        pointer=start
        for sample,mark in arrived_events:
            if not start <= sample <= start+len(raw): raise ValueError('Event not yet arrived')
            if sample<pointer: raise ValueError('Out-of-order event')
            self.segment(pointer,raw[pointer-start:sample-start]); self.event(sample,mark); pointer=sample
        self.segment(pointer,raw[pointer-start:]); self.last_observed=start+len(raw)

    def segment(self,start,raw):
        if not len(raw): return
        required=self.filter.eeg_ix+self.filter.ref_ix+self.filter.eog_ix
        finite=np.isfinite(raw[:,required]).all(axis=1)
        # Splitting exactly at invalid samples preserves valid prefixes across block sizes.
        boundaries=np.flatnonzero(~finite); pointer=0
        for bad in boundaries:
            if bad>pointer: self._finite_segment(start+pointer,raw[pointer:bad])
            self.filter.process(start+int(bad),raw[bad:bad+1])
            if self.eog_cleaner: self.eog_cleaner.reset()
            self.eog_update_last_raw=None; self.eog_jump_tail[:]=False
            self.clear_window()
            if self.on_gap: self.on_gap(start+int(bad),'nonfinite')
            pointer=int(bad)+1
        if pointer<len(raw): self._finite_segment(start+pointer,raw[pointer:])

    def _finite_segment(self,start,raw):
        if self.filter.next_sample is not None and start!=self.filter.next_sample:
            self.clear_window()
            if self.eog_cleaner: self.eog_cleaner.reset()
            self.eog_update_last_raw=None; self.eog_jump_tail[:]=False
            if self.on_gap: self.on_gap(start,'missing_samples')
        out,ix,valid,ref,_=self.filter.process(start,raw)
        if self.eog_cleaner:
            # Update only from arrived, finite, warmed-up, non-jumping samples.
            previous=self.eog_update_last_raw if self.eog_update_last_raw is not None else ref[0]
            jumps=np.max(np.abs(np.diff(np.vstack([previous,ref]),axis=0)),axis=1)>self.cfg['qc']['jump_uv']
            history=np.concatenate([self.eog_jump_tail,jumps])
            guarded=np.convolve(history.astype(np.int8),np.ones(4,dtype=np.int8),mode='valid')>0
            self.eog_jump_tail=history[-3:]
            take=(np.arange(start,start+len(raw))%4)==0
            update_valid=valid & ~guarded[take]
            self.eog_update_last_raw=ref[-1].copy()
            cleaned=np.empty((len(out),30));pointer=0;phase=self.count%1280
            while pointer<len(out):
                length=min(1280-phase,len(out)-pointer);end_pointer=pointer+length
                cleaned[pointer:end_pointer]=self.eog_cleaner.clean_samples(out[pointer:end_pointer,:30],out[pointer:end_pointer,30:32],valid=update_valid[pointer:end_pointer])
                phase=(phase+length)%1280
                if not phase and self.active is not None:
                    self.eog_snapshots[int(ix[end_pointer-1])+4]=self.eog_cleaner.summary()
                pointer=end_pointer
            out=np.column_stack([out,cleaned])
        if self.active is None and self.first_task: return
        if len(out):
            if not self.count: self.profile=self.choose_profile(int(ix[0])) if self.active is not None else 'P0'
            self.values.append(out); self.indices.append(ix); self.valid.append(valid); self.count+=len(out)
        self.raw_parts.append(np.column_stack([ref,raw[:,self.filter.eog_ix]])); self.raw_starts.append(start)
        while self.count>=1280:
            joined=np.concatenate(self.values); times=np.concatenate(self.indices); validity=np.concatenate(self.valid)
            end=int(times[0])+5120
            # A downsample at 5116 does not mean raw samples through 5119 have arrived.
            if start+len(raw)<end: break
            raw_values=np.concatenate(self.raw_parts)
            raw_times=np.concatenate([np.arange(s,s+len(v)) for s,v in zip(self.raw_starts,self.raw_parts)])
            raw_take=(raw_times>=times[0])&(raw_times<end)
            x,e=joined[:1280,:30],joined[:1280,30:32]
            q=quality(x,e,raw_values[raw_take,:30],self.cfg,end,raw_eog=raw_values[raw_take,30:])
            if not validity[:1280].all() or not np.array_equal(times[:1280],times[0]+np.arange(1280)*4) or raw_take.sum()!=5120:
                q['valid']=False; q['reasons'].append('warmup_or_missing')
            w={'start_sample':int(times[0]),'end_sample':end,'path_start_sample':self.active,
               'profile':self.profile,'x':x,'e':e,'quality':q,'calibration':self.calibration}
            if joined.shape[1]==62:
                w['eog_adaptive_x']=joined[:1280,32:62]
                w['eog_adaptive_state']=self.eog_snapshots.pop(end,{})
            if self.active is not None: self.on_window(w)
            elif not self.first_task and q['valid']:
                remaining=int(self.cfg['calibration_max_seconds']*256)-self.cal_count
                if remaining>=1280:
                    self.cal_parts.append(joined[:1280,:32]); self.cal_count+=1280
                    if self.on_baseline: self.on_baseline(w)
            self.values=[joined[1280:]]; self.indices=[times[1280:]]; self.valid=[validity[1280:]]; self.count-=1280
            keep=raw_times>=end
            self.raw_parts=[raw_values[keep]]; self.raw_starts=[int(raw_times[keep][0])] if keep.any() else [end]
            self.profile=self.choose_profile(int(times[1280])) if self.count and self.active is not None else None

    def eof(self):
        if self.active is not None: self.finish(self.last_observed,False)
