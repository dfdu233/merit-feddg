"""Check/download MedCPT models and print real PubMed knowledge-base commands."""
from __future__ import annotations

import argparse
from pathlib import Path

MODELS = {
    "query": "ncbi/MedCPT-Query-Encoder",
    "article": "ncbi/MedCPT-Article-Encoder",
    "cross": "ncbi/MedCPT-Cross-Encoder",
}


def local_path(root, model_id):
    return Path(root) / model_id.replace("/", "--")


def print_commands(args):
    print("# MedCPT model downloads")
    for model_id in MODELS.values():
        target = local_path(args.models_root, model_id)
        print(f"hf download {model_id} --local-dir {target}")
    print()
    print("# PubMed 2026 production baseline (real NLM/NCBI citations + abstracts)")
    print(f"mkdir -p {args.pubmed_dir}")
    for index in range(1, args.baseline_files + 1):
        name = f"pubmed26n{index:04d}.xml.gz"
        print(
            "wget -c "
            f"https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/{name} "
            f"-P {args.pubmed_dir}"
        )
    print()
    print("# Build a pilot KB first; remove --limit only after retrieval validation")
    print(
        "python scripts/build_medcpt_kb.py "
        f"--pubmed-dir {args.pubmed_dir} "
        f"--article-encoder {local_path(args.models_root, MODELS['article'])} "
        f"--output {args.kb_dir} --limit 100000"
    )
    print()
    print("# Faster alternative: official NCBI MedCPT precomputed PubMed chunk")
    precomputed = Path(args.precomputed_dir)
    print(f"mkdir -p {precomputed}")
    chunk = args.precomputed_chunk
    for prefix, suffix in (
        ("embeds_chunk_", ".npy"),
        ("pmids_chunk_", ".json"),
        ("pubmed_chunk_", ".json"),
    ):
        filename = f"{prefix}{chunk}{suffix}"
        print(
            "wget -c "
            f"https://ftp.ncbi.nlm.nih.gov/pub/lu/MedCPT/pubmed_embeddings/{filename} "
            f"-P {precomputed}"
        )
    print(
        "python scripts/import_medcpt_pubmed.py "
        f"--source {precomputed} --chunks {chunk} "
        f"--output {args.kb_dir}"
    )


def download_models(args):
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Install/use the existing research environment with huggingface_hub, "
            "or run with --print-commands."
        ) from exc
    for model_id in MODELS.values():
        target = local_path(args.models_root, model_id)
        if target.is_dir() and any(target.iterdir()):
            print(f"present: {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {model_id} -> {target}", flush=True)
        snapshot_download(repo_id=model_id, local_dir=target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", default="artifacts/models")
    parser.add_argument("--pubmed-dir", default="artifacts/knowledge/pubmed-2026-baseline")
    parser.add_argument("--kb-dir", default="artifacts/knowledge/medcpt-pubmed")
    parser.add_argument("--baseline-files", type=int, default=1)
    parser.add_argument(
        "--precomputed-dir",
        default="artifacts/knowledge/medcpt-precomputed-pubmed",
    )
    parser.add_argument("--precomputed-chunk", type=int, default=36)
    parser.add_argument("--print-commands", action="store_true")
    parser.add_argument("--download-models", action="store_true")
    args = parser.parse_args()
    if args.baseline_files < 1 or args.precomputed_chunk < 0:
        parser.error("--baseline-files must be positive and --precomputed-chunk nonnegative")

    missing = []
    for model_id in MODELS.values():
        path = local_path(args.models_root, model_id)
        if not path.is_dir() or not any(path.iterdir()):
            missing.append((model_id, path))
    kb = Path(args.kb_dir) / "manifest.json"
    print("MedCPT asset audit:")
    for model_id, path in missing:
        print(f"  MISSING model: {model_id} -> {path}")
    if not missing:
        print("  models: present")
    print(f"  knowledge base: {'present' if kb.is_file() else 'MISSING'} -> {kb}")

    if args.download_models:
        download_models(args)
    if args.print_commands or (missing and not args.download_models) or not kb.is_file():
        print()
        print_commands(args)


if __name__ == "__main__":
    main()
