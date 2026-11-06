from pathlib import Path
import textwrap
import nbformat as nbf
ROOT=Path(__file__).resolve().parent
M=lambda s:nbf.v4.new_markdown_cell(textwrap.dedent(s).strip())
C=lambda s:nbf.v4.new_code_cell(textwrap.dedent(s).strip())
setup='''
from pathlib import Path
import sys
course_dir = Path.cwd() if (Path.cwd() / 'course_utils.py').exists() else Path.cwd() / 'EEG_from_ground_up'
assert (course_dir / 'course_utils.py').exists(), 'Open this notebook from the course folder or project root.'
sys.path.insert(0, str(course_dir.resolve()))
from course_utils import ROOT, DATA, OUTPUT, SEED, load_run, versions, style, save_table
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from IPython.display import display
%matplotlib inline
style()
rng = np.random.default_rng(SEED)
display(versions().to_frame())
'''
def write(name,title,goal,parts,refs):
    cells=[M(f'# {title}\n\n## Goal\n{goal}\n\n**Scope:** a reproducible teaching analysis of public recordings, not clinical validation. Downloaded data are cached locally; no synthetic fallback is used. Each notebook runs independently.\n\n## Setup\nUse the environment in `README.md`. The default sample is deliberately small; results are descriptive unless explicitly stated otherwise.'),C(setup),M('## Steps')]
    for prose,code in parts:
        cells.append(M(prose))
        if code:cells.append(C(code))
    cells.append(M('## Sources and further reading\nOriginal teaching code; the resources below provide APIs, methodological guidance, and dataset provenance. No tutorial text or full external notebook is reproduced.\n\n'+'\n'.join(f'- [{label}]({url})' for label,url in refs)))
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'eeg-ground-up','display_name':'Python (EEG ground up)','language':'python'}})
    nbf.validate(nb); nbf.write(nb,ROOT/name)
phys=('PhysioNet dataset and protocol','https://physionet.org/content/eegmmidb/1.0.0/')
mne_ref=('MNE tutorials','https://mne.tools/stable/auto_tutorials/index.html')
write('01_EEG_fundamentals.ipynb','01 · EEG fundamentals and data provenance',
'Understand voltage references, samples, channels, annotations, and the distinction between a recording, a run, a trial, and a participant.',[
('''### 1. The measurement model
For C channels and T samples, X has shape C × T. A scalp channel measures a voltage difference, not the activity of one isolated brain region. A conceptual model is X = AS + E, where A mixes neural sources and E contains artifacts and noise. Source recovery is not unique.

PhysioNet uses 64 channels at 160 Hz. Runs 1/2 are eyes open/closed; runs 4/8/12 are left/right **imagery**. Runs 3/7/11 are actual movement. T1/T2 mean something different in the hands/feet runs: never interpret labels without the run protocol.''','''
SUBJECT, RUN = 1, 4
raw = load_run(SUBJECT, RUN)
display(pd.DataFrame([{'subject':SUBJECT,'run':RUN,'channels':len(raw.ch_names),
    'samples':raw.n_times,'sample_rate_Hz':raw.info['sfreq'],
    'recording_length_s':raw.n_times/raw.info['sfreq'],'Nyquist_Hz':raw.info['sfreq']/2}]))
display(pd.DataFrame({'channel':raw.ch_names,'type':raw.get_channel_types()}).head(12))
'''),('''### 2. Read traces in physical units
MNE represents EEG in volts. Multiply by 10⁶ only for a microvolt plot. Offsets below are for readability, not a reference change. Large excursions are observations requiring inspection, not automatically neural events.''','''
picks=['Fp1','C3','Cz','C4','O1']
x=raw.get_data(picks=picks, start=0, stop=int(8*raw.info['sfreq']))*1e6
t=np.arange(x.shape[1])/raw.info['sfreq']
fig,ax=plt.subplots(figsize=(10,4))
for k,ch in enumerate(picks): ax.plot(t,x[k]+k*150,lw=.7,label=ch)
ax.set(xlabel='Time (s)',ylabel='Voltage (µV), channels offset by 150 µV',title='Unfiltered EEG: first 8 seconds')
ax.legend(ncol=5); plt.tight_layout(); plt.show()
raw.plot_sensors(show_names=['Fp1','C3','Cz','C4','O1','Oz','O2'], show=False); plt.show()
'''),('''### 3. Events and labels
An annotation onset is not necessarily the onset of the participant’s internal intention. Here it is the experimental task marker. Preserve these timestamps and the recording identity.''','''
annotations=pd.DataFrame({'onset_s':raw.annotations.onset,'duration_s':raw.annotations.duration,
    'annotation':raw.annotations.description})
display(annotations.head(12))
events,event_id=mne.events_from_annotations(raw,event_id={'T1':1,'T2':2},verbose=False)
trial_table=pd.DataFrame({'subject':SUBJECT,'run':RUN,'sample':events[:,0],
    'onset_s':events[:,0]/raw.info['sfreq'],'label':np.where(events[:,2]==1,'left_hand','right_hand')})
display(save_table(trial_table,'01_trial_inventory.csv'))
'''),('''### 4. Referencing is a linear transformation
Average reference is x′ᵢ(t)=xᵢ(t)−C⁻¹Σⱼxⱼ(t). It removes one spatial degree of freedom. It does not reconstruct the original acquisition reference, which this notebook does not independently establish. Bad channels should be addressed before estimating a common average.''','''
ref=raw.copy().set_eeg_reference('average',projection=False,verbose=False)
assert np.max(np.abs(ref.get_data().mean(axis=0))) < 1e-12
fig,ax=plt.subplots()
for r,label in [(raw,'Original reference'),(ref,'Average reference')]:
    ax.plot(r.times[:800],r.get_data(picks=['C3'])[0,:800]*1e6,label=label,lw=.9)
ax.set(xlabel='Time (s)',ylabel='C3 voltage (µV)',title='Reference choice changes the waveform')
ax.legend(); plt.tight_layout(); plt.show()
'''),('''## Checks
Sample spacing must equal 1/fs; all values must be finite. The manifest records the exact source URL and local SHA-256 for each EDF. These hashes identify your local files; they are not a substitute for an upstream signature.''','''
assert np.isfinite(raw.get_data()).all()
assert np.allclose(np.diff(raw.times),1/raw.info['sfreq'])
assert set(events[:,2])=={1,2}
print('Checks passed. Trial counts:',trial_table.label.value_counts().to_dict())
'''),('''## Next Steps
In Notebook 02, ask whether eyes-open and eyes-closed recordings differ spectrally.

**Exercises:** (1) Calculate the sample index at 2.5 s. (2) Explain why a 160 Hz recording cannot resolve a 100 Hz oscillation unambiguously. (3) Explain why 64 channels need not contain 64 independent signals.

**Answer checks:** 400 samples from the start; Nyquist is 80 Hz; volume conduction, shared reference, and re-referencing create dependence.''',None)], [phys,mne_ref,('OpenNeuro and BIDS metadata','https://docs.openneuro.org/user_guide.html')])
write('02_Spectral_structure.ipynb','02 · Spectral structure, PSD, and scalp maps',
'Estimate spectra correctly, integrate power in a frequency band, and distinguish channel-level descriptions from population evidence.',[
('''### 1. Load paired baseline recordings
Use one participant, runs 1 and 2. This is a within-person, two-recording comparison, not a randomized group experiment. We remove each channel’s temporal mean through Welch’s detrending; no narrow bandpass is applied before inspecting the spectrum.''','''
from scipy.signal import welch
from scipy.integrate import trapezoid
SUBJECT=1
recordings={'Eyes open':load_run(SUBJECT,1),'Eyes closed':load_run(SUBJECT,2)}
recordings={name:r.copy().set_eeg_reference('average',projection=False,verbose=False) for name,r in recordings.items()}
fs=recordings['Eyes open'].info['sfreq']
'''),('''### 2. Welch spectra and spectral resolution
Welch averages modified periodograms. PSD units are V²/Hz; µV²/Hz requires multiplication by 10¹². A 4 s segment has Fourier-bin spacing 0.25 Hz. Zero padding makes a denser frequency grid but does not add independent spectral information. Adjacent overlapping segments are not independent replicates.''','''
spectra={}
fig,ax=plt.subplots()
for label,r in recordings.items():
    f,p=welch(r.get_data(),fs=fs,nperseg=int(4*fs),noverlap=int(2*fs),detrend='constant',axis=-1)
    spectra[label]=(f,p)
    occ=[r.ch_names.index(ch) for ch in ['O1','Oz','O2']]
    ax.semilogy(f,p[occ].mean(axis=0)*1e12,label=label)
ax.set(xlim=(1,45),xlabel='Frequency (Hz)',ylabel='PSD (µV²/Hz)',title='Mean occipital spectrum: O1, Oz, O2')
ax.legend(); plt.tight_layout(); plt.show()
'''),('''### 3. Integrate band power and map its spatial distribution
P[a,b] = ∫ₐᵇ S(f)df. Here alpha is operationally defined as 8–13 Hz, rather than assuming every person has identical boundaries. A topomap interpolates scalp values; it does not locate cortical generators.''','''
rows=[]; alpha={}
for label,(f,p) in spectra.items():
    mask=(f>=8)&(f<=13)
    alpha[label]=trapezoid(p[:,mask],f[mask],axis=-1)*1e12
    for ch,power in zip(recordings[label].ch_names,alpha[label]):
        rows.append({'condition':label,'channel':ch,'alpha_power_uV2':power})
table=pd.DataFrame(rows)
display(save_table(table,'02_alpha_power.csv').query("channel in ['O1','Oz','O2']"))
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
values=[np.log10(alpha[name]) for name in recordings]
vmin,vmax=min(v.min() for v in values),max(v.max() for v in values)
for ax,(name,r),v in zip(axes,recordings.items(),values):
    im,_=mne.viz.plot_topomap(v,r.info,axes=ax,show=False,vlim=(vmin,vmax),cmap='viridis')
    ax.set_title(name)
fig.colorbar(im,ax=axes,label='log₁₀(alpha power / µV²)',shrink=.8); plt.show()
ratio=alpha['Eyes closed']/alpha['Eyes open']
occ=[recordings['Eyes open'].ch_names.index(c) for c in ['O1','Oz','O2']]
print(f'Mean occipital channel-wise closed/open alpha ratio: {ratio[occ].mean():.3f}')
'''),('''### 4. Compare Welch and multitaper estimates
Multitaper averages across orthogonal tapers and trades spectral resolution for reduced variance. The methods need not produce identical curves. Use the same signal, units, and frequency range for comparison.''','''
r=recordings['Eyes closed']
p_mt,f_mt=mne.time_frequency.psd_array_multitaper(r.get_data(picks=['Oz'])[0],sfreq=fs,
    fmin=1,fmax=45,bandwidth=2,normalization='full',verbose=False)
f,p=spectra['Eyes closed']; idx=r.ch_names.index('Oz')
fig,ax=plt.subplots(); ax.semilogy(f,p[idx]*1e12,label='Welch, 4 s segments')
ax.semilogy(f_mt,p_mt*1e12,label='Multitaper, 2 Hz bandwidth')
ax.set(xlim=(1,45),xlabel='Frequency (Hz)',ylabel='PSD (µV²/Hz)',title='Oz: estimator sensitivity')
ax.legend(); plt.tight_layout(); plt.show()
'''),('''## Checks
Integrated PSD should approximately agree with signal variance. Small deviations arise from segment averaging, detrending, and tapering; exact equality is not expected for this Welch estimate.''','''
f,p=spectra['Eyes closed']; r=recordings['Eyes closed']
variance=r.get_data().var(axis=1)
ratio_psd=trapezoid(p,f,axis=-1)/variance
assert np.isfinite(p).all() and (p>=0).all()
print('Median integrated-PSD / full-recording variance:',np.median(ratio_psd))
'''),('''## Next Steps
Notebook 03 inspects how cleaning changes spectra and waveforms.

**Exercises:** repeat with 2 s and 8 s Welch segments. Compare alpha-power sensitivity, not just appearance. Add another participant and keep each participant’s result separate before computing a group summary. An alpha increase is a common expectation, not a result to force through preprocessing.''',None)], [phys,('MNE spectral analysis','https://mne.tools/stable/auto_tutorials/time-freq/20_sensors_time_frequency.html')])
write('03_Cleaning_and_quality_control.ipynb','03 · Cleaning and quality control',
'Inspect artifacts, quantify preprocessing effects, and learn why a cleaner-looking trace is not sufficient evidence of better science.',[
('''### 1. Audit channels before filtering
The diagnostics below are screening measures, not validated artifact labels. A high peak-to-peak value may reflect a transient, and a high robust spread may reflect either noise or physiology. No channel is automatically removed by this table.''','''
from scipy.signal import welch
raw=load_run(1,4)
x=raw.get_data()*1e6
median=np.median(x,axis=1,keepdims=True)
qc=pd.DataFrame({'channel':raw.ch_names,'SD_uV':x.std(axis=1),
    'MAD_uV':np.median(np.abs(x-median),axis=1),'peak_to_peak_uV':np.ptp(x,axis=1),
    'finite_fraction':np.isfinite(x).mean(axis=1)})
display(save_table(qc,'03_channel_qc.csv').sort_values('peak_to_peak_uV',ascending=False).head(10))
ranked=qc.nlargest(12,'SD_uV').sort_values('SD_uV')
fig,ax=plt.subplots(figsize=(9,4)); ax.barh(ranked.channel,ranked.SD_uV)
ax.set(ylabel='Channel',xlabel='Standard deviation (µV)',title='12 channels with greatest raw variability; full table saved'); plt.tight_layout(); plt.show()
'''),('''### 2. Explicit cleaning decisions
A 1–40 Hz band is an example for this oscillatory teaching analysis, not a universal EEG setting. A 1 Hz high-pass is unsuitable for preserving some slow ERP phenomena. Filter continuous runs separately to avoid filtering across recording boundaries. No notch is applied automatically: first establish whether line noise is present and whether it remains inside the analysis band.

Average reference is appropriate to demonstrate here with broad scalp coverage. Manual bad-channel decisions are empty until evidence supports a choice. Interpolation estimates signals from neighbors; it does not recover lost measurements.''','''
BAD_CHANNELS=[]  # Populate only after inspecting raw traces and diagnostics.
clean=raw.copy()
clean.info['bads']=BAD_CHANNELS
if BAD_CHANNELS:
    clean.interpolate_bads(reset_bads=True,verbose=False)
clean.set_eeg_reference('average',projection=False,verbose=False)
clean.filter(1,40,method='fir',phase='zero',verbose=False)
f0,p0=welch(raw.get_data(),fs=raw.info['sfreq'],nperseg=640,axis=-1)
f1,p1=welch(clean.get_data(),fs=clean.info['sfreq'],nperseg=640,axis=-1)
fig,axes=plt.subplots(1,2,figsize=(11,4))
for r,label in [(raw,'Original'),(clean,'Referenced + filtered')]:
    axes[0].plot(r.times[:800],r.get_data(picks=['Fp1'])[0,:800]*1e6,label=label,lw=.8)
axes[0].set(xlabel='Time (s)',ylabel='Fp1 voltage (µV)',title='Waveform comparison'); axes[0].legend()
axes[1].semilogy(f0,np.median(p0,axis=0)*1e12,label='Original')
axes[1].semilogy(f1,np.median(p1,axis=0)*1e12,label='Referenced + filtered')
axes[1].set(xlim=(0,80),xlabel='Frequency (Hz)',ylabel='Median channel PSD (µV²/Hz)',title='Combined processing effect')
axes[1].legend(); plt.tight_layout(); plt.show()
'''),('''### 3. ICA inspection, without unjustified component removal
ICA models linear mixtures of statistically independent components. It is not a classifier of neural versus non-neural signals. We fit a reduced 15-component decomposition for this small demonstration, inspect scalp patterns, and **remove nothing by default**. There is no dedicated EOG recording in this dataset; frontal electrodes are not equivalent to an EOG ground truth.

This fit uses the full run for descriptive inspection only. For prediction, any learned artifact-removal procedure must respect the training/test split. See EEGLAB’s conceptual discussion and MNE’s implementation guidance below.''','''
from mne.preprocessing import ICA
ica=ICA(n_components=15,method='fastica',random_state=SEED,max_iter=500)
ica.fit(clean,decim=4,verbose=False)
ica.plot_components(picks=range(6),nrows=2,ncols=3,show=False); plt.show()
ICA_EXCLUDE=[]  # Set only after reviewing component evidence.
ica.exclude=ICA_EXCLUDE
reconstructed=ica.apply(clean.copy(),verbose=False)
print('ICA iterations:',ica.n_iter_,'; excluded components:',ica.exclude)
print('RMS change (µV):',np.sqrt(np.mean((reconstructed.get_data()-clean.get_data())**2))*1e6)
'''),('''### 4. Epoch rejection and automated repair
Use fixed 0–3 s task epochs. A transparent 150 µV peak-to-peak screen is illustrative, not a sensor-independent standard. Report retention separately by condition to detect unequal rejection. AutoReject provides a learned alternative; here it is fitted on the same epochs solely to demonstrate its QC behavior, not to estimate prediction performance.''','''
events,_=mne.events_from_annotations(clean,event_id={'T1':1,'T2':2},verbose=False)
epochs=mne.Epochs(clean,events,event_id={'left':1,'right':2},tmin=0,tmax=3,
    baseline=None,preload=True,verbose=False)
thresholded=epochs.copy().drop_bad(reject={'eeg':150e-6},verbose=False)
retention=pd.DataFrame([{'condition':k,'before':len(epochs[k]),'after_threshold':len(thresholded[k])} for k in epochs.event_id])
display(save_table(retention,'03_epoch_retention.csv'))
from autoreject import AutoReject
# A small fixed grid limits tutorial compute. Not a comprehensive parameter search.
ar=AutoReject(n_interpolate=[1],consensus=[0.2],cv=3,random_state=SEED,n_jobs=1,verbose=False)
repaired,log=ar.fit_transform(epochs,return_log=True)
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
# AutoReject labels: 0=good, 1=bad, 2=interpolated. Hatched columns mark dropped trials.
colors=['#e5e5e5','#c27524','#246a9c']
fig,ax=plt.subplots(figsize=(10,6))
ax.imshow(log.labels.T,aspect='auto',interpolation='nearest',
    cmap=ListedColormap(colors),norm=BoundaryNorm([-.5,.5,1.5,2.5],3))
for trial in np.flatnonzero(log.bad_epochs):
    ax.axvspan(trial-.5,trial+.5,facecolor='none',edgecolor='black',hatch='//',lw=.7)
ax.set(xticks=np.arange(len(epochs)),yticks=np.arange(0,len(epochs.ch_names),5),
    yticklabels=epochs.ch_names[::5],xlabel='Epoch index',ylabel='Channel',
    title='AutoReject decisions; hatching marks rejected epochs')
ax.legend(handles=[Patch(facecolor=c,label=l) for c,l in zip(colors,['Good','Bad','Interpolated'])],
    loc='upper center',bbox_to_anchor=(.5,1.16),ncol=3)
plt.tight_layout(); plt.show()
print(f'AutoReject retained {len(repaired)} / {len(epochs)} epochs.')
if len(thresholded)==0:
    print('The illustrative 150 µV rule rejects every trial here. This is a QC finding, not a useful cleaned dataset; inspect channels and justify any revised rule.')
'''),('''## Checks
No source file is overwritten. The cleaning choices and retention counts are explicit. ICA nonconvergence, if reported, is a limitation of this demonstration; do not interpret unstable components as identified sources.''','''
assert np.isfinite(clean.get_data()).all()
assert len(thresholded)<=len(epochs)
assert raw.n_times==clean.n_times
assert np.allclose(reconstructed.get_data(),clean.get_data(),atol=1e-12) if not ICA_EXCLUDE else True
print('Original data preserved; no unreviewed ICA component exclusions.')
'''),('''## Next Steps
Notebook 04 uses fresh recordings and explicitly specified preprocessing, not the manually inspected ICA/AutoReject output from this lesson.

**Exercises:** quantify the effect of a 0.1 Hz versus 1 Hz high-pass on a slow signal; vary the rejection threshold and inspect class-specific retention. Explain why rejecting more data can increase bias even if remaining traces look cleaner.''',None)], [phys,('MNE ICA','https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html'),('EEGLAB ICA concepts','https://eeglab.org/tutorials/ConceptsGuide/ICA_background.html'),('AutoReject example','https://autoreject.github.io/stable/auto_examples/plot_auto_repair.html')])
write('04_Time_frequency_analysis.ipynb','04 · Fourier and wavelet time-frequency analysis',
'Compare STFT and Morlet representations of the same EEG and understand temporal resolution, spectral resolution, baseline normalization, and edge effects.',[
('''### 1. Inspect a continuous segment with the STFT
STFT projects a windowed signal onto sinusoids. With a fixed window, the time/frequency tradeoff is fixed. A 1 s window has 1 Hz Fourier-bin spacing. The plot below uses PSD scaling and no boundary padding.''','''
from scipy.signal import spectrogram
raw=load_run(1,4).set_eeg_reference('average',projection=False,verbose=False)
raw.filter(1,40,verbose=False)
fs=raw.info['sfreq']; signal=raw.get_data(picks=['C3'])[0]
f,t,S=spectrogram(signal,fs=fs,nperseg=int(fs),noverlap=int(.9*fs),scaling='density',mode='psd')
fig,ax=plt.subplots(figsize=(10,4)); use=(f>=4)&(f<=35)
im=ax.pcolormesh(t,f[use],10*np.log10(np.maximum(S[use]*1e12,1e-20)),shading='auto',cmap='viridis')
ax.set(xlabel='Time (s)',ylabel='Frequency (Hz)',title='C3: STFT, 1 s window')
fig.colorbar(im,ax=ax,label='PSD (dB re 1 µV²/Hz)'); plt.tight_layout(); plt.show()
'''),('''### 2. Event-aligned Morlet power
Morlet wavelets combine a sinusoid with a Gaussian envelope. With fixed n_cycles, low frequencies use longer temporal support. We use 4 cycles over 8–30 Hz, compute on −2 to +4 s epochs, then display only −1.5 to +3.5 s. This excludes approximately the 5σ wavelet half-support at the lowest frequency (about 0.40 s).

Compute on padded epochs before cropping; padding or cropping does not make zero-phase filtering causal. These are offline analyses.''','''
events,_=mne.events_from_annotations(raw,event_id={'T1':1,'T2':2},verbose=False)
epochs=mne.Epochs(raw,events,event_id={'left':1,'right':2},tmin=-2,tmax=4,
    baseline=None,picks=['C3','C4'],preload=True,verbose=False)
freqs=np.arange(8,31,2); cycles=4
power=mne.time_frequency.tfr_array_morlet(epochs.get_data(),sfreq=fs,freqs=freqs,
    n_cycles=cycles,output='power',decim=2,n_jobs=1,verbose=False)
times=epochs.times[::2]
bmask=(times>=-1.5)&(times<=-.5)
baseline=power[...,bmask].mean(axis=-1,keepdims=True)
db=10*np.log10(np.maximum(power,1e-30)/np.maximum(baseline,1e-30))
print('Power dimensions [trial, channel, frequency, time]:',power.shape)
print('Approximate maximum 5σ half-support (s):',5*cycles/(2*np.pi*freqs.min()))
'''),('''### 3. Compare task, channel, and time
Each trial is normalized to its own baseline, then its log-ratio is averaged. This is not identical to taking the log of average power. Red/blue here indicate increases/decreases in power relative to baseline, not positive/negative voltages. Cue-related effects may contribute; these maps do not isolate intention.''','''
fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True,layout='constrained')
show=(times>=-1.5)&(times<=3.5)
for row,(label,event) in enumerate(epochs.event_id.items()):
    mean=db[epochs.events[:,2]==event].mean(axis=0)
    for col,ch in enumerate(epochs.ch_names):
        ax=axes[row,col]
        im=ax.pcolormesh(times[show],freqs,mean[col][:,show],cmap='RdBu_r',vmin=-5,vmax=5,shading='auto')
        ax.axvline(0,color='black',ls='--',lw=1)
        ax.set(title=f'{label} imagery · {ch}',xlabel='Time from cue (s)',ylabel='Frequency (Hz)')
fig.colorbar(im,ax=axes,label='Mean trial-level power change (dB)',shrink=.8,extend='both'); plt.show()
'''),('''### 4. Compare normalized STFT and Morlet on the same first trial
Normalization permits a qualitative comparison despite different estimator scaling. Use identical baseline and display windows, and recognize that smoothing differs. Neither method is automatically superior.''','''
first=epochs.get_data()[0,0]
f_s,t_s,p_s=spectrogram(first,fs=fs,nperseg=int(fs),noverlap=int(.9*fs),scaling='density')
t_s=t_s+epochs.tmin
b=(t_s>=-1.5)&(t_s<=-.5)
p_s_db=10*np.log10(np.maximum(p_s,1e-30)/np.maximum(p_s[:,b].mean(axis=1,keepdims=True),1e-30))
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
mask=(f_s>=8)&(f_s<=30)
axes[0].pcolormesh(t_s,f_s[mask],p_s_db[mask],shading='auto',cmap='RdBu_r',vmin=-10,vmax=10)
im=axes[1].pcolormesh(times,freqs,db[0,0],shading='auto',cmap='RdBu_r',vmin=-10,vmax=10)
for ax,title in zip(axes,['STFT: 1 s window','Morlet: 4 cycles']):
    ax.set(xlim=(-1.5,3.5),xlabel='Time from cue (s)',ylabel='Frequency (Hz)',title=title)
fig.colorbar(im,ax=axes,label='First-trial power change (dB)',extend='both'); plt.show()
'''),('''## Checks
Baseline mean power is computed before taking logarithms. Consequently, the mean baseline **log-power ratio** need not be exactly zero. Verify dimensions and finite values instead of forcing a zero mean.''','''
assert power.shape[0]==len(epochs) and power.shape[-1]==len(times)
assert np.isfinite(db).all() and (baseline>0).all()
print('Trials by condition:',{k:len(epochs[k]) for k in epochs.event_id})
'''),('''## Next Steps
Notebook 05 summarizes trial-level effects and uncertainty without treating time points as independent participants.

**Exercises:** repeat with 3 and 7 cycles; recalculate the valid edge margin each time. Determine which short-lived patterns persist. Explain why more pixels in a time-frequency plot do not imply more independent observations.''',None)], [phys,('MNE time-frequency tutorial','https://mne.tools/stable/auto_tutorials/time-freq/20_sensors_time_frequency.html'),('MNE ERD/ERS example','https://mne.tools/stable/auto_examples/time_frequency/time_frequency_erds.html')])
write('05_Event_related_activity_and_uncertainty.ipynb','05 · Event-related activity and uncertainty',
'Distinguish evoked voltage from changes in oscillatory power, retain participant/run/trial identity, and estimate uncertainty at the correct level.',[
('''### 1. Build a trial collection with provenance
Use participants 1–3 and imagery runs 4, 8, 12. These are repeated runs, not established separate-day sessions. Each run is filtered independently at 1–40 Hz and average-referenced. This descriptive analysis does not use an adaptive artifact model. Inspect the QC limitations before interpreting effects.

The −2 to +4 s epochs provide wavelet padding. Cue timing anchors the analysis; the internal onset of imagery is not observed.''','''
SUBJECTS=[1,2,3]; RUNS=[4,8,12]
collections=[]
for subject in SUBJECTS:
    for run in RUNS:
        r=load_run(subject,run).set_eeg_reference('average',projection=False,verbose=False)
        r.filter(1,40,verbose=False)
        ev,_=mne.events_from_annotations(r,event_id={'T1':1,'T2':2},verbose=False)
        meta=pd.DataFrame({'subject':subject,'run':run,'trial':np.arange(len(ev)),
            'condition':np.where(ev[:,2]==1,'left','right')})
        e=mne.Epochs(r,ev,event_id={'left':1,'right':2},tmin=-2,tmax=4,baseline=None,
            picks=['C3','C4'],metadata=meta,preload=True,verbose=False)
        collections.append(e)
epochs=mne.concatenate_epochs(collections,verbose=False)
meta=epochs.metadata.reset_index(drop=True)
display(meta.groupby(['subject','run','condition']).size().unstack(fill_value=0))
print('Retained trials:',len(epochs),'Unique participants:',meta.subject.nunique())
'''),('''### 2. Evoked voltage versus induced power
An event-related average retains activity consistently phase-aligned to the event. Averaging voltage can cancel an oscillation whose phase varies across trials; averaging its power does not. The curves below are cue-locked averages, not a validated motor-specific ERP. The 1 Hz high-pass also limits interpretation of slow potentials.

Average within each participant first, then weight participants equally. This avoids giving participants with more retained trials greater group weight.''','''
voltage=epochs.copy().apply_baseline((-.8,-.2)).get_data()*1e6
time=epochs.times
fig,ax=plt.subplots()
for label in ['left','right']:
    curves=np.stack([voltage[(meta.subject==s)&(meta.condition==label),0].mean(axis=0) for s in SUBJECTS])
    ax.plot(time,curves.mean(axis=0),label=f'{label}: equal-participant mean')
ax.axvline(0,color='black',ls='--',lw=1)
ax.set(xlim=(-1,3),xlabel='Time from cue (s)',ylabel='C3 baseline-corrected voltage (µV)',title='Cue-locked averages: 3 participants')
ax.legend(); plt.tight_layout(); plt.show()
'''),('''### 3. Define an effect before estimating it
Use mu-band frequencies 8–12 Hz, baseline −1.5 to −0.5 s, and task interval +0.5 to +2.5 s. These fixed teaching choices are not optimized on significance.

For each trial and channel, define D=10 log₁₀(P_task/P_baseline). Negative D indicates a power decrease. Define lateralization L=D_C3−D_C4 and the participant contrast L_left−L_right. This signed contrast is descriptive; it is not by itself evidence of a particular neural generator.''','''
fs=epochs.info['sfreq']; freqs=np.arange(8,13)
power=mne.time_frequency.tfr_array_morlet(epochs.get_data(),sfreq=fs,freqs=freqs,
    n_cycles=4,output='power',decim=4,n_jobs=1,verbose=False)
t=epochs.times[::4]; base=(t>=-1.5)&(t<=-.5); active=(t>=.5)&(t<=2.5)
pbase=power[...,base].mean(axis=(-1,-2)); ptask=power[...,active].mean(axis=(-1,-2))
effect=10*np.log10(ptask/pbase)
trials=meta.copy(); trials['C3_change_dB']=effect[:,0]; trials['C4_change_dB']=effect[:,1]
trials['lateralization_dB']=effect[:,0]-effect[:,1]
save_table(trials,'05_trial_effects.csv')
# First average trials within run, then runs within participant and condition.
run_effects=trials.groupby(['subject','run','condition']).lateralization_dB.mean()
participant=run_effects.groupby(['subject','condition']).mean().unstack()
participant['left_minus_right_dB']=participant['left']-participant['right']
display(save_table(participant.reset_index(),'05_participant_effects.csv'))
'''),('''### 4. Participant-level uncertainty
Resample participants, not time samples or overlapping windows. With only three participants, a percentile bootstrap has very limited support and may be unstable. The interval is a teaching demonstration, not a reliable population claim. Trial-to-trial variability and between-person variability answer different questions.''','''
d=participant.left_minus_right_dB.to_numpy()
boot=rng.choice(d,size=(5000,len(d)),replace=True).mean(axis=1)
lo,hi=np.percentile(boot,[2.5,97.5])
fig,axes=plt.subplots(1,2,figsize=(10,4))
axes[0].plot(participant.index,d,'o',ms=8)
axes[0].axhline(0,color='black',ls='--'); axes[0].set(xticks=SUBJECTS,xlabel='Participant',ylabel='Left − right lateralization (dB)',title='Independent participant summaries')
axes[1].hist(boot,bins=20,color='#246a9c'); axes[1].axvline(lo,color='black',ls='--'); axes[1].axvline(hi,color='black',ls='--')
axes[1].set(xlabel='Resampled mean contrast (dB)',ylabel='Bootstrap replicates',title='Small-N bootstrap: interpret cautiously')
plt.tight_layout(); plt.show()
print(f'Mean contrast = {d.mean():.3f} dB; illustrative 95% percentile interval [{lo:.3f}, {hi:.3f}], N={len(d)}.')
'''),('''### 5. An exact small-sample null illustration
A sign-flip test assumes independent participant effects symmetric around zero under the null. Enumerate all 2ᴺ signs. With N=3, two-sided inference is extremely coarse. This is an explicit statistical assumption, not something guaranteed by the recording design. Scanning many channels/frequencies/times would additionally require multiplicity control.''','''
from itertools import product
signs=np.array(list(product([-1,1],repeat=len(d))))
null=(signs*d).mean(axis=1)
p_exact=np.mean(np.abs(null)>=abs(d.mean())-1e-12)
print(f'Exact two-sided sign-flip p={p_exact:.3f} from {len(null)} sign patterns.')
'''),('''## Checks''','''
assert len(meta)==len(effect)
assert not meta.duplicated(['subject','run','trial']).any()
assert np.isfinite(effect).all() and len(participant)==len(SUBJECTS)
print('Trial provenance and participant-level aggregation verified.')
'''),('''## Next Steps
Notebook 06 asks a different question: whether unseen runs can be classified. A descriptive difference need not yield useful prediction, and predictive accuracy does not establish mechanism.

**Exercises:** calculate a within-participant interval by resampling whole runs; explain why it cannot substitute for a population interval. Expand the participant list before making population claims. Treat frequency-band changes as sensitivity analyses, not opportunities to choose the smallest p-value.''',None)], [phys,('MNE ERD/ERS','https://mne.tools/stable/auto_examples/time_frequency/time_frequency_erds.html'),('MNE statistical inference','https://mne.tools/stable/auto_tutorials/stats-sensor-space/10_background_stats.html')])
write('06_Decoding_MOABB_pyRiemann_Braindecode.ipynb','06 · Decoding: MNE → MOABB → pyRiemann → Braindecode',
'Compare transparent baselines, spatial filtering, covariance geometry, and a compact neural network using identical held-out runs.',[
('''### 1. MOABB: consistent dataset access and task definition
MOABB standardizes dataset loading, task selection, and evaluation. We use its PhysionetMI loader and LeftRightImagery paradigm, then implement a transparent **held-out-run** evaluation using its metadata. We intentionally do not call ordinary trial-shuffled cross-validation a cross-run test.

Participants 1–3; eight prespecified motor-region channels; fixed 8–30 Hz filtering; task window +0.5 to +3.5 s. This is an offline, two-class, cue-driven problem. It does not test a no-command state or asynchronous control. The anatomical channel subset is fixed before seeing labels or scores.''','''
from moabb.datasets import PhysionetMI
from moabb.paradigms import LeftRightImagery
from sklearn.preprocessing import LabelEncoder
SUBJECTS=[1,2,3]
CHANNELS=['C3','C4','Cz','FC3','FC4','FCz','CP3','CP4']
# Cache source EDFs through MNE before MOABB reads them.
for subject in SUBJECTS:
    for run in [4,8,12]:
        load_run(subject,run)
dataset=PhysionetMI()
dataset.feet_runs=[]  # Exclude unused hands/feet tasks; retain the three left/right runs.
paradigm=LeftRightImagery(fmin=8,fmax=30,tmin=.5,tmax=3.5,channels=CHANNELS)
moabb_epochs,labels,metadata=paradigm.get_data(dataset=dataset,subjects=SUBJECTS,return_epochs=True)
X=moabb_epochs.get_data()*1e6  # Explicit volts -> µV for all models.
encoder=LabelEncoder(); y=encoder.fit_transform(labels)
metadata=metadata.reset_index(drop=True)
metadata['trial_row']=np.arange(len(y))
# MOABB labels these source runs 0/1/2; retain the original EDF run identity too.
run_mapping={str(i):run for i,run in enumerate(dataset.hand_runs)}
metadata['edf_run']=metadata['run'].astype(str).map(run_mapping)
assert metadata['edf_run'].notna().all(), 'Inspect changed MOABB run identifiers before proceeding.'
metadata['label']=labels
metadata['group']=metadata[['subject','session','run']].astype(str).agg('/'.join,axis=1)
display(metadata.groupby(['subject','session','run','label']).size().unstack(fill_value=0))
print('Classes:',encoder.classes_,'; X [trials, channels, samples]:',X.shape)
save_table(metadata,'06_trial_metadata.csv')
'''),('''### 2. Freeze a split before fitting
Within each participant, reserve the last run (ordered by MOABB run ID) for testing and use earlier runs for training. This tests transfer to another run within the recorded session, not to a new participant or day. No hyperparameters are chosen using test scores; deep learning uses a fixed 20-epoch budget. For tuning, add an inner run-grouped validation loop.

All methods see the same arrays and split. Fixed filters act within individual source runs; learned CSP, scaling, and tangent-space references are fitted only on training trials. No overlapping trial crops are created.''','''
splits=[]; split_rows=[]
for subject in SUBJECTS:
    idx=np.flatnonzero(metadata.subject.to_numpy()==subject)
    groups=sorted(metadata.loc[idx,'group'].unique())
    assert len(groups)>=3, 'Need at least three runs for this demonstration.'
    test=idx[metadata.loc[idx,'group'].to_numpy()==groups[-1]]
    train=np.setdiff1d(idx,test)
    assert set(metadata.loc[train,'group']).isdisjoint(metadata.loc[test,'group'])
    assert set(y[train])==set(y[test])=={0,1}
    splits.append((subject,train,test))
    split_rows.extend({'trial_row':int(i),'subject':subject,'split':role} for role,ids in [('train',train),('test',test)] for i in ids)
save_table(pd.DataFrame(split_rows),'06_frozen_split.csv')
display(pd.DataFrame([{'subject':s,'train_trials':len(tr),'test_trials':len(te)} for s,tr,te in splits]))
'''),('''### 3. MNE and pyRiemann baselines
Compare a dummy classifier, log-variance + shrinkage LDA, CSP + shrinkage LDA, covariance minimum-distance-to-mean (MDM), and covariance tangent-space + logistic regression.

CSP learns spatial projections that contrast class variances. A covariance matrix summarizes channel variances and cross-channel co-fluctuation; it discards much temporal ordering. OAS shrinkage helps keep covariance estimates positive definite. Tangent-space features depend on a reference covariance fitted on the training set. Neither covariance classification nor good accuracy establishes connectivity or causality.''','''
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace

def log_variance(data):
    return np.log(np.maximum(data.var(axis=-1),1e-12))

models={
    'Dummy':make_pipeline(FunctionTransformer(log_variance),DummyClassifier(strategy='prior')),
    'Log variance + LDA':make_pipeline(FunctionTransformer(log_variance),StandardScaler(),LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')),
    'CSP + LDA':make_pipeline(CSP(n_components=4,reg='oas',log=True,norm_trace=False),LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')),
    'Riemann MDM':make_pipeline(Covariances(estimator='oas'),MDM(metric='riemann')),
    'Tangent + logistic':make_pipeline(Covariances(estimator='oas'),TangentSpace(metric='riemann'),StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=SEED)),
}
results=[]; predictions=[]
def record(name,subject,test,pred,prob):
    results.append({'model':name,'subject':subject,'test_trials':len(test),
        'balanced_accuracy':balanced_accuracy_score(y[test],pred),'ROC_AUC':roc_auc_score(y[test],prob)})
    predictions.extend({'model':name,'subject':subject,'trial_row':int(i),'true':int(a),'predicted':int(b),'p_class_1':float(p)} for i,a,b,p in zip(test,y[test],pred,prob))

for subject,train,test in splits:
    for name,pipeline in models.items():
        fitted=clone(pipeline).fit(X[train],y[train])
        record(name,subject,test,fitted.predict(X[test]),fitted.predict_proba(X[test])[:,1])
display(pd.DataFrame(results))
'''),('''### 4. Braindecode EEGNet on exactly the same split
EEGNet learns temporal and spatial convolutional features. This is a small fixed-budget comparison, not a tuned deep-learning benchmark. Twenty epochs and one seed do not establish a neural network’s potential or superiority.

Scale each channel using training-only mean and standard deviation. Use CPU execution and a fixed seed for a reproducible tutorial. Keep the test set out of normalization, training, early stopping, and architecture selection. We use Braindecode’s model with an explicit PyTorch training loop so every update is visible.''','''
import torch
from torch.utils.data import TensorDataset,DataLoader
from braindecode.models import EEGNet
import random
TRAIN_EPOCHS=20
BATCH_SIZE=16
history=[]
torch.set_num_threads(2)
for subject,train,test in splits:
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    mu=X[train].mean(axis=(0,2),keepdims=True)
    sigma=np.maximum(X[train].std(axis=(0,2),keepdims=True),1e-6)
    xtrain=((X[train]-mu)/sigma).astype('float32')
    xtest=((X[test]-mu)/sigma).astype('float32')
    net=EEGNet(n_chans=X.shape[1],n_outputs=2,n_times=X.shape[2],
        sfreq=moabb_epochs.info['sfreq'],F1=4,D=2,F2=8,drop_prob=.25)
    optimizer=torch.optim.Adam(net.parameters(),lr=1e-3)
    loss_fn=torch.nn.CrossEntropyLoss()
    generator=torch.Generator().manual_seed(SEED)
    loader=DataLoader(TensorDataset(torch.from_numpy(xtrain),torch.tensor(y[train],dtype=torch.long)),
        batch_size=BATCH_SIZE,shuffle=True,generator=generator)
    for epoch in range(TRAIN_EPOCHS):
        net.train(); total_loss=0
        for xb,yb in loader:
            optimizer.zero_grad(); logits=net(xb); loss=loss_fn(logits,yb)
            loss.backward(); optimizer.step(); total_loss+=loss.item()*len(yb)
        history.append({'subject':subject,'epoch':epoch+1,'training_loss':total_loss/len(train)})
    net.eval()
    with torch.no_grad(): probs=net(torch.from_numpy(xtest)).softmax(dim=1).numpy()
    record('EEGNet (20 epochs)',subject,test,probs.argmax(axis=1),probs[:,1])
print('Trainable parameters in final EEGNet:',sum(p.numel() for p in net.parameters() if p.requires_grad))
history=pd.DataFrame(history)
fig,ax=plt.subplots()
for s,g in history.groupby('subject'): ax.plot(g.epoch,g.training_loss,label=f'Participant {s}')
ax.set(xlabel='Training epoch',ylabel='Cross-entropy loss',title='Training curves; test data never used for stopping')
ax.legend(); plt.tight_layout(); plt.show()
save_table(history,'06_EEGNet_training_history.csv')
'''),('''### 5. Compare results without overclaiming
Balanced accuracy is the mean class recall; its binary chance reference is 0.5. AUC measures ranking, not probability calibration. Show every participant rather than only a grand mean. Test sets are small; no population superiority claim is justified.

The confusion matrix pools held-out predictions for description. Participant-level mean performance gives each participant equal weight. These quantities need not agree when trial counts differ.''','''
results=pd.DataFrame(results); predictions=pd.DataFrame(predictions)
save_table(results,'06_model_results.csv'); save_table(predictions,'06_predictions.csv')
display(results.pivot(index='model',columns='subject',values='balanced_accuracy').round(3))
fig,ax=plt.subplots(figsize=(10,5))
order=list(models)+['EEGNet (20 epochs)']
for k,s in enumerate(SUBJECTS):
    vals=results[results.subject==s].set_index('model').loc[order,'balanced_accuracy']
    ax.scatter(np.arange(len(order))+(k-1)*.1,vals,label=f'Participant {s}',marker=['o','s','^'][k])
ax.axhline(.5,color='black',ls='--',label='Chance reference')
ax.set(xticks=np.arange(len(order)),xticklabels=order,ylim=(0,1.03),ylabel='Held-out-run balanced accuracy',title='Identical test trials for all methods')
ax.tick_params(axis='x',labelrotation=20); ax.legend(ncol=2); plt.tight_layout(); plt.show()
p=predictions[predictions.model=='EEGNet (20 epochs)']
ConfusionMatrixDisplay(confusion_matrix(p.true,p.predicted,labels=[0,1]),display_labels=encoder.classes_).plot(cmap='Blues',colorbar=False)
plt.title('EEGNet: pooled held-out counts'); plt.tight_layout(); plt.show()
'''),('''## Checks
Verify split separation, consistent test trial IDs across models, finite predictions, and complete scores. These checks establish implementation consistency, not scientific generalization.''','''
assert np.isfinite(X).all()
assert len(results)==len(SUBJECTS)*(len(models)+1)
for subject,train,test in splits:
    for model in order:
        rows=predictions[(predictions.subject==subject)&(predictions.model==model)]
        assert set(rows.trial_row)==set(test)
assert predictions.p_class_1.between(0,1).all()
print('All models evaluated on identical held-out trials; results saved.')
'''),('''## Next Steps
1. Add participants, then run a **cross-subject** evaluation with MOABB’s `CrossSubjectEvaluation` and prespecified models. Do not label this notebook’s within-person results cross-subject evidence.
2. For separate-day transfer, select a documented multi-session dataset, such as BNCI2014_001, and use `CrossSessionEvaluation`. Verify task labels, channel count, and protocol before comparing datasets.
3. Add nested run-grouped model selection, multiple random seeds, and paired participant-level uncertainty. Keep a final test set untouched.
4. Add wavelet-derived or dynamical-systems features only as additional pipelines under the same split. Include ablations and suitable surrogate/null controls.
5. For real-time work, replace offline zero-phase filtering with a validated causal pipeline; measure latency and false activations in a no-command state.

**Exercises:** implement leave-one-run-out evaluation; compare average performance across outer folds without tuning on them. Explain why a deep model can fit training data while failing on another run. Ask whether performance comes from neural activity, cue responses, or residual artifacts.''',None)], [phys,('MOABB getting started','https://moabb.neurotechx.com/docs/auto_examples/tutorials/tutorial_0_plot_getting_started.html'),('MOABB PhysionetMI','https://moabb.neurotechx.com/docs/generated/moabb.datasets.PhysionetMI.html'),('MNE CSP','https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html'),('pyRiemann tangent space','https://pyriemann.readthedocs.io/en/latest/generated/pyriemann.tangentspace.TangentSpace.html'),('Braindecode EEGNet','https://braindecode.org/stable/generated/braindecode.models.EEGNet.html'),('BNCI datasets','https://bnci-horizon-2020.eu/database/data-sets')])
print('Created six notebooks.')

# Keep the additional mathematical exposition in the generated notebooks.
import runpy
runpy.run_path(str(ROOT / "enrich_explanations.py"), run_name="__main__")
