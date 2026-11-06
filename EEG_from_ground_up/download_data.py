from course_utils import load_run, DATA, OUTPUT
for subject in [1,2,3]:
    for run in ([1,2,4,8,12] if subject==1 else [4,8,12]):
        r=load_run(subject,run)
        print(f'Cached subject {subject}, run {run}: {r.n_times} samples',flush=True)
print('Hand-imagery recordings cached for both MNE and MOABB.',flush=True)
from pathlib import Path
import hashlib, json
inventory=[]
for path in sorted(DATA.rglob('*.edf')):
    inventory.append({'file':str(path.relative_to(DATA)), 'bytes':path.stat().st_size,
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'source':f'https://physionet.org/files/eegmmidb/1.0.0/{path.name[:4]}/{path.name}'})
(OUTPUT/'download_inventory.json').write_text(json.dumps(inventory,indent=2))
