from pathlib import Path
import nbformat
root=Path(__file__).resolve().parent
additions={
'01':('### 1. The measurement model',r'''

**Objects used throughout the course:** `Raw` stores continuous channels × samples plus annotations; `Epochs` stores trials × channels × samples plus metadata; `Evoked` stores an average over selected epochs. Preserve the unaveraged trials whenever uncertainty or decoding matters.

At sampling rate $f_s$, sample $n$ corresponds to $t_n=n/f_s$. The Nyquist frequency is $f_s/2$. A file’s length in samples divided by $f_s$ is its nominal duration; its last sample occurs one sample interval earlier. Electrode coordinates describe sensor positions, not an inverse solution for brain sources.
'''),
'02':('### 2. Welch spectra and spectral resolution',r'''

For a tapered segment of $N$ samples, a two-sided periodogram uses $|\mathrm{FFT}(wx)|^2/(f_s\sum_n w_n^2)$. A one-sided estimate doubles interior positive-frequency bins, while preserving DC and Nyquist conventions. Welch then averages across segments. `scipy.signal.welch` handles this normalization.

**Power versus amplitude:** multiplying voltage by 10 multiplies power by 100. Therefore voltage uses $20\log_{10}$ for amplitude ratios, while power uses $10\log_{10}$. Absolute band power integrates the PSD; relative band power divides by power in a declared reference band. Peaks sit on an aperiodic background, so a band-power change does not necessarily mean a stronger narrowband oscillator.
'''),
'03':('### 3. ICA inspection, without unjustified component removal',r'''

If $X\approx AS$, ICA estimates a demixing matrix $W$ such that $S\approx WX$ has approximately independent components. Independence is a modeling assumption, and component order or sign has no inherent physiological meaning. Removing a component projects out its contribution across all channels. Do not infer “eye blink” solely from a large frontal weight.
'''),
'04':('### 2. Event-aligned Morlet power',r'''

A schematic Morlet wavelet is $\psi_f(t)=K e^{i2\pi ft}e^{-t^2/(2\sigma_t^2)}$, with $\sigma_t=n_{cycles}/(2\pi f)$ and an optional correction for zero mean. Convolution yields a complex coefficient; squared magnitude gives power and its angle gives phase. Power loses phase information. More cycles sharpen frequency localization while spreading activity over time.
'''),
'05':('### 4. Participant-level uncertainty',r'''

The unit of replication follows the question. For an individual’s trial-to-trial reliability, trials or blocks may be relevant. For inference to a population, independently sampled participants are the relevant units. Thousands of correlated time samples do not turn three participants into a large population sample. The bootstrap here weights the three observed participants equally and does not establish representativeness.
'''),
'06':('### 3. MNE and pyRiemann baselines',r'''

For a centered trial matrix $X\in\mathbb{R}^{C\times T}$, the sample covariance is $\hat\Sigma=XX^\top/(T-1)$. Shrinkage combines it with a scaled identity: $\Sigma_\lambda=(1-\lambda)\hat\Sigma+\lambda\,\mathrm{tr}(\hat\Sigma)I/C$; the estimator determines $\lambda$ from data.

CSP solves a class-variance problem such as $\Sigma_1w=\lambda(\Sigma_1+\Sigma_2)w$. The learned filters must be fitted inside the training split. Spatial filters are not automatically source-localization maps.

For symmetric positive-definite covariance matrices $A,B$, the affine-invariant distance is $d(A,B)=\|\log(A^{-1/2}BA^{-1/2})\|_F$. MDM assigns the nearest class mean under this geometry. Tangent-space classification maps covariances around a training-derived reference and applies an ordinary classifier. Scaling and regularization still matter.
''')}
for prefix,(anchor,extra) in additions.items():
    p=next(root.glob(prefix+'_*.ipynb'));nb=nbformat.read(p,4)
    for c in nb.cells:
        if c.cell_type=='markdown' and c.source.startswith(anchor):
            if extra.strip() not in c.source:c.source+='\n'+extra.strip()
    nbformat.write(nb,p)
print('Added mathematical explanations without changing computations.')
