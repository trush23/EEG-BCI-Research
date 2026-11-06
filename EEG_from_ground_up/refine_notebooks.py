from pathlib import Path
import nbformat
root=Path(__file__).resolve().parent
# Apply the same small presentation corrections to the source builder for reproducibility.
replacements={
"raw.plot_sensors(show_names=False, show=False)":"raw.plot_sensors(show_names=['Fp1','C3','Cz','C4','O1','Oz','O2'], show=False)",
"fig,ax=plt.subplots(figsize=(10,3)); ax.bar(qc.channel,qc.SD_uV)\nax.set(xlabel='Channel',ylabel='Standard deviation (µV)',title='Raw channel variability')\nax.tick_params(axis='x',labelrotation=90,labelsize=7)":"ranked=qc.nlargest(12,'SD_uV').sort_values('SD_uV')\nfig,ax=plt.subplots(figsize=(9,4)); ax.barh(ranked.channel,ranked.SD_uV)\nax.set(ylabel='Channel',xlabel='Standard deviation (µV)',title='12 channels with greatest raw variability; full table saved')",
"ica.plot_components(picks=range(6),show=False)":"ica.plot_components(picks=range(6),nrows=2,ncols=3,show=False)",
"log.plot('horizontal',show=False); plt.show()":"fig=log.plot('horizontal',show=False)\nfig.set_size_inches(10,6)\nfig.axes[0].set_aspect('auto')\nfig.axes[0].set_title('AutoReject decisions by channel and epoch')\nplt.tight_layout(); plt.show()",
"print(f'AutoReject retained {len(repaired)} / {len(epochs)} epochs.')":"print(f'AutoReject retained {len(repaired)} / {len(epochs)} epochs.')\nif len(thresholded)==0:\n    print('The illustrative 150 µV rule rejects every trial here. This is a QC finding, not a useful cleaned dataset; inspect channels and justify any revised rule.')",
"fig.colorbar(im,ax=axes,label='Mean trial-level power change (dB)',shrink=.8)":"fig.colorbar(im,ax=axes,label='Mean trial-level power change (dB)',shrink=.8,extend='both')",
"fig.colorbar(im,ax=axes,label='First-trial power change (dB)')":"fig.colorbar(im,ax=axes,label='First-trial power change (dB)',extend='both')",
}
for path in root.glob('0*.ipynb'):
    nb=nbformat.read(path,4); changed=False
    for cell in nb.cells:
        for old,new in replacements.items():
            if old in cell.source:
                cell.source=cell.source.replace(old,new); changed=True
    if changed: nbformat.write(nb,path)
p=root/'build_notebooks.py'; s=p.read_text()
for old,new in replacements.items(): s=s.replace(old,new)
p.write_text(s)
