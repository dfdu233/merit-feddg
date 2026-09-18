"""Download only missing pinned public inference resources; never install packages.

Weights remain outside git. Existing incompatible files are rejected, not replaced.
HF_ENDPOINT may point at a mirror; publisher SHA-256 remains authoritative.
"""
import argparse
import hashlib
import json
from pathlib import Path

RESOURCES = {
    'monet': ('chanwkim/monet', '1d1efd0b8d61bde82cde22d2e93c28d73441f29e',
              ['*.json', '*.txt', 'model.safetensors', 'README.md']),
    'flair': ('jusiro2/FLAIR', '5f6bdd0a068353dc41a896ba3abdd7c0f6d35938',
              ['config.json', 'model.safetensors', 'README.md']),
    'unimed': ('UzairK/unimed-clip-vit-b16', '84a67cb0331b511b630136e4878beaa5a3fdc657',
               ['unimed-clip-vit-b16.pt', 'README.md']),
    'clinicalbert': ('emilyalsentzer/Bio_ClinicalBERT', 'd5892b39a4adaed74b92212a44081509db72f87b',
                     ['config.json', 'vocab.txt', 'pytorch_model.bin', 'LICENSE', 'README.md']),
    'biomedbert': ('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract',
                   'd673b8835373c6fa116d6d8006b33d48734e305d',
                   ['config.json', 'vocab.txt', 'tokenizer_config.json', 'pytorch_model.bin',
                    'LICENSE.md', 'README.md']),
}


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    import fnmatch

    from huggingface_hub import HfApi, snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resource', choices=[*RESOURCES, 'all'], default='all')
    parser.add_argument('--root', type=Path, default=Path('artifacts/models/coverage-v1'))
    args = parser.parse_args()
    names = list(RESOURCES) if args.resource == 'all' else [args.resource]
    api = HfApi()
    for name in names:
        repo, revision, patterns = RESOURCES[name]
        # Always obtain publisher metadata independently of a download mirror.
        info = HfApi(endpoint='https://huggingface.co').model_info(
            repo, revision=revision, files_metadata=True)
        expected = {x.rfilename: x.lfs.sha256 for x in info.siblings
                    if x.lfs and any(fnmatch.fnmatch(x.rfilename, p) for p in patterns)}
        target = args.root / name
        for filename, digest in expected.items():
            existing = target / filename
            if existing.exists() and sha256(existing) != digest:
                raise ValueError(f'existing resource hash mismatch: {existing}; not overwritten')
        snapshot_download(repo, revision=revision, allow_patterns=patterns,
                          local_dir=target, max_workers=2, endpoint=api.endpoint)
        for filename, digest in expected.items():
            if sha256(target / filename) != digest:
                raise ValueError(f'publisher SHA mismatch: {name}/{filename}')
        audit = {'repo_id': repo, 'revision': revision, 'publisher_sha256': expected,
                 'files_sha256': {str(p.relative_to(target)): sha256(p)
                                  for p in target.iterdir() if p.is_file()
                                  and p.name != 'resource.json'},
                 'weight_integrity_verified': True, 'training_performed': False,
                 'inference_verified': False}
        report = target / 'resource.json'
        if report.exists():
            if json.loads(report.read_text()) != audit:
                raise ValueError('existing resource identity differs; not overwritten')
        else:
            with report.open('x') as stream:
                json.dump(audit, stream, indent=2)
        print(f'VERIFIED {name} {revision}', flush=True)


if __name__ == '__main__':
    main()
