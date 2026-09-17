"""Quilt backend for MERIT's existing native-generation expert contract.

Reuse the existing generator request and EvidenceItem construction, changing
only the inference backend and its truthful adapter provenance.
"""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess

from merit_feddg.experts.native_qwen import QwenVqaCapabilityExpert


class IsolatedQuiltBackend:
    def __init__(self, model_id, output, gpu_uuid):
        self.model_id,self.output,self.gpu_uuid=model_id,Path(output),gpu_uuid

    def generate_with_usage(self,image,prompt,max_new_tokens):
        request={'image':str(image),'prompt':prompt,'max_new_tokens':max_new_tokens,
                 'checkpoint':self.model_id,'gpu_uuid':self.gpu_uuid,
                 'image_sha256':hashlib.sha256(Path(image).read_bytes()).hexdigest()}
        key=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
        target=self.output/(key+'.json')
        if not target.exists():
            subprocess.run(['/home/dbw/.runtime/quilt-env/bin/python',
                            str(Path(__file__).with_name('native_quilt_infer.py')),
                            '--output',str(target)],input=json.dumps(request),text=True,
                           check=True,timeout=300)
        v=json.loads(target.read_text())
        assert v['request']==request
        return v['output']


class NativeQuiltExpert(QwenVqaCapabilityExpert):
    def __init__(self,model_id,expert_id,scope,output,gpu_uuid,max_new_tokens=96):
        super().__init__(model_id,expert_id,scope,max_new_tokens=max_new_tokens)
        self.probe=IsolatedQuiltBackend(model_id,output,gpu_uuid)

    def infer(self,request):
        result=super().infer(request)
        return replace(result,items=tuple(replace(item,provenance={
            **item.provenance,'adapter':'native_quilt_vqa'}) for item in result.items))


def build(model_id,**kwargs):
    return NativeQuiltExpert(model_id,**kwargs)
