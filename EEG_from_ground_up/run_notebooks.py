"""Execute with this environment's Python; export HTML and figures for inspection."""
from pathlib import Path
import sys, os, time, json, base64
ROOT=Path(__file__).resolve().parent
for variable,relative in [('IPYTHONDIR','.cache/ipython'),('JUPYTER_RUNTIME_DIR','.cache/jupyter'),('MPLCONFIGDIR','.cache/matplotlib'),('NUMBA_CACHE_DIR','.cache/numba')]:
    path=ROOT/relative; path.mkdir(parents=True,exist_ok=True); os.environ[variable]=str(path)
os.environ['MNE_DATA']=str(ROOT/'data')
os.environ['MNE_DATASETS_EEGBCI_PATH']=str(ROOT/'data')
os.environ['MNE_DONTWRITE_HOME']='true'
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager
previews=ROOT/'previews'; previews.mkdir(exist_ok=True)
figures=previews/'figures'; figures.mkdir(exist_ok=True)
summary=[]
paths=sorted(ROOT.glob('0*.ipynb'))
if len(sys.argv)>1: paths=[p for p in paths if any(p.name.startswith(prefix) for prefix in sys.argv[1:])]
for path in paths:
    nb=nbformat.read(path,as_version=4)
    km=KernelManager(kernel_name='python3')
    km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    def progress(cell, cell_index, **kwargs):
        print(f'  Cell {cell_index + 1} complete', flush=True)
    client=NotebookClient(nb,km=km,timeout=600,resources={'metadata':{'path':str(ROOT)}},
        on_cell_executed=progress)
    start=time.time(); print('START',path.name,flush=True)
    error=None
    try:
        client.execute()
        nbformat.validate(nb)
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
        print(error,flush=True)
    finally:
        nbformat.write(nb,path)
        if km.has_kernel: km.shutdown_kernel(now=True)
    html,_=HTMLExporter().from_notebook_node(nb)
    (previews/(path.stem+'.html')).write_text(html)
    count=0
    for cell in nb.cells:
        for out in cell.get('outputs',[]):
            if 'image/png' in out.get('data',{}):
                count+=1
                (figures/f'{path.stem}_{count:02d}.png').write_bytes(base64.b64decode(out.data['image/png']))
    row=dict(notebook=path.name,status='FAILED' if error else 'PASS',seconds=round(time.time()-start,1),figures=count,error=error)
    summary.append(row)
    print(json.dumps({k:v for k,v in row.items() if k!='error'}),flush=True)
    report=ROOT/'outputs'/'execution_status.json'
    existing=json.loads(report.read_text()) if report.exists() else []
    existing=[r for r in existing if r['notebook']!=path.name]+[row]
    report.write_text(json.dumps(sorted(existing,key=lambda r:r['notebook']),indent=2))
print('Finished',flush=True)
