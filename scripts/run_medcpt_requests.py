"""Execute frozen MedCPT requests on the machine containing the pinned PubMed KB."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from merit_feddg.capabilities import CapabilityRequest, validate_result
from merit_feddg.experts.medcpt_retriever import MedCPTRetrievalExpert
from merit_feddg.open_study import atomic_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('requests', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--model-root', type=Path, default=Path('/root/merit-medcpt-artifacts/models'))
parser.add_argument('--kb-root', type=Path, default=Path('/root/merit-medcpt-artifacts/medcpt-pubmed'))
args = parser.parse_args()
requests = json.loads(args.requests.read_text())
root = args.output
expert = MedCPTRetrievalExpert(
    model_id=str(args.model_root / 'ncbi--MedCPT-Query-Encoder'),
    expert_id='medcpt_pubmed',
    kb_manifest=str(args.kb_root / 'manifest.json'),
    cross_encoder_path=str(args.model_root / 'ncbi--MedCPT-Cross-Encoder'),
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
