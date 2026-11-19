#!/usr/bin/env python3
import argparse, csv, hashlib, json
from pathlib import Path
import torch

p=argparse.ArgumentParser(); p.add_argument('--work-root', action='append', required=True); p.add_argument('--output-dir', required=True); a=p.parse_args()
rows=[]
for root in a.work_root:
    for path in Path(root).glob('seed_*/*/iter_14000.pth'):
        payload=torch.load(str(path), map_location='cpu')
        meta=payload.get('meta', {}) if isinstance(payload, dict) else {}
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        rel=path.relative_to(root).parts
        rows.append(dict(root=str(Path(root).resolve()), seed=rel[0].replace('seed_',''), model=rel[1],
                         checkpoint=str(path.resolve()), sha256=digest, iteration=meta.get('iter',''),
                         mmseg_version=meta.get('mmseg_version',''), exp_name=meta.get('exp_name',''),
                         embedded_config=meta.get('config','')))
out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
fields=list(rows[0]) if rows else ['root','seed','model','checkpoint','sha256','iteration','mmseg_version','exp_name','embedded_config']
with (out/'checkpoint_manifest.tsv').open('w',newline='',encoding='utf-8') as h:
    w=csv.DictWriter(h,fieldnames=fields,delimiter='\t'); w.writeheader(); w.writerows(rows)
(out/'checkpoint_manifest.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
print('checkpoints=',len(rows))

