"""Frozen MedSeq-Grounder target tool; reuse Qwen source observations exactly."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import run_crossmodal_correspondence_v2 as core

SOURCE=core.RUN
core.RUN=core.ROOT/'runs/medsg-correspondence-specialist-dev-v1'
core.MODEL='/home/dbw/models/MedSeq-Grounder'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','evaluate']);p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    if a.stage=='prepare':
        core.prepare()
        m=json.loads((core.RUN/'manifest.json').read_text())
        prior=json.loads((SOURCE/'manifest.json').read_text())
        assert [(r['id'],r['hashes']) for r in m['selected']]==[(r['id'],r['hashes']) for r in prior['selected']]
        m.update(experiment=core.RUN.name,source_observer_model=prior['model'],source_observation_origin=str(SOURCE),
                 target_model_revision='7c18e944897e8fdb86e93dcf919ad78517330df4',
                 wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),observation_sha256={})
        for r in m['selected']:
            src=SOURCE/'observations'/f"{r['id']}.json";dest=core.RUN/'observations'/src.name
            dest.parent.mkdir(exist_ok=True);shutil.copyfile(src,dest)
            m['observation_sha256'][r['id']]=hashlib.sha256(src.read_bytes()).hexdigest()
        core.write(core.RUN/'manifest.json',m)
    elif a.stage=='run':
        m=json.loads((core.RUN/'manifest.json').read_text())
        assert m['wrapper_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        for r in m['selected']:
            p=core.RUN/'observations'/f"{r['id']}.json"
            assert hashlib.sha256(p.read_bytes()).hexdigest()==m['observation_sha256'][r['id']]
        core.run(a.limit)
    else:core.evaluate()
