"""Execute frozen MedCPT requests on the machine containing the pinned PubMed KB."""
import json
import sys
from dataclasses import asdict
from pathlib import Path

from merit_feddg.capabilities import CapabilityRequest, validate_result
from merit_feddg.experts.medcpt_retriever import MedCPTRetrievalExpert
from merit_feddg.open_study import atomic_json

requests = json.loads(Path(sys.argv[1]).read_text())
root = Path(sys.argv[2])
expert = MedCPTRetrievalExpert(
    model_id='/root/merit-medcpt-artifacts/models/ncbi--MedCPT-Query-Encoder',
    expert_id='medcpt_pubmed',
    kb_manifest='/root/merit-medcpt-artifacts/medcpt-pubmed/manifest.json',
    cross_encoder_path='/root/merit-medcpt-artifacts/models/ncbi--MedCPT-Cross-Encoder',
    device='cuda', candidate_k=32, top_k=3, max_query_tokens=64, max_pair_tokens=512,
)
for index, row in enumerate(requests):
    dest = root / row['relative_path']
    if dest.exists():
        previous = json.loads(dest.read_text())
        if previous['identity'] != row['identity']:
            raise ValueError('Existing retrieval belongs to another frozen cache')
        continue
    request = CapabilityRequest(**row['request'])
    result = validate_result(expert.infer(request), 'medcpt_pubmed', request)
    atomic_json(dest, {'identity': row['identity'], 'output': asdict(result)})
    if index % 50 == 0:
        print('RETRIEVAL', index + 1, len(requests), flush=True)
print('COMPLETE', len(requests), flush=True)
