"""Explicit remote experiment entry point without reinstalling the LLaVA stack."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default="configs/llava_med_capabilities.yaml")
    result.add_argument("--generalist", choices=["llava", "openmed"], default="llava")
    result.add_argument("--study", choices=["capabilities", "value", "diagnose"], default="capabilities")
    result.add_argument("--value-stage", choices=["all", "source", "evaluate"], default="all")
    result.add_argument("--evidence-profile", choices=["legacy", "scoped"], default=None,
                        help="opt-in presentation/subrequest ablation, not a proven better method")
    result.add_argument("--source-manifest")
    result.add_argument("--target-manifest")
    result.add_argument("--references")
    result.add_argument("--model-path")
    result.add_argument("--llava-source")
    result.add_argument("--vision-tower-path")
    result.add_argument("--artifacts", default="artifacts")
    result.add_argument("--output", default="runs/native-v08")
    result.add_argument("--dataset", choices=["pathvqa", "vqarad", "both"], default="both")
    result.add_argument("--source-per-group", type=int, default=16)
    result.add_argument("--target-limit", type=int, default=16)
    result.add_argument("--seed", type=int, default=17)
    result.add_argument("--mirror", choices=["global", "cn"], default="global")
    result.add_argument("--retrieval-answers", choices=["on", "off"], default=None)
    result.add_argument("--chexagent", choices=["auto", "on", "off"], default="auto")
    result.add_argument("--skip-download", action="store_true")
    result.add_argument("--install-deps", action="store_true")
    result.add_argument("--check-only", action="store_true")
    result.add_argument("--prepare-only", action="store_true")
    return result


def experiment_config(args):
    from .generalist_factory import resolve_generalist_spec
    from .io import load_yaml

    config = load_yaml(args.config)
    if args.generalist == "openmed":
        config["generalist"] = {
            "backend": "qwen", "id": "OpenMed/Qwen2.5-3B-MedVL",
            "revision": "6370f5828f259d64a850d7bad0b423770583a66c",
            "dtype": "bfloat16", "device_map": "auto",
        }
    for option, key in (("model_path", "checkpoint_path"), ("llava_source", "source_path"),
                        ("vision_tower_path", "vision_tower_path")):
        value = getattr(args, option)
        if value:
            config["generalist"][key] = str(Path(value).expanduser().resolve())
    for spec in config["experts"].values():
        path = spec.get("checkpoint_path")
        if path and Path(path).parts[0] == "artifacts":
            spec["checkpoint_path"] = str(
                Path(args.artifacts).resolve().joinpath(*Path(path).parts[1:])
            )
        if args.retrieval_answers is not None and spec.get("adapter") == "source_retrieval":
            spec["include_source_answers"] = args.retrieval_answers == "on"
    chexagent = config["experts"].get("chexagent_description")
    if args.chexagent == "off":
        config["experts"].pop("chexagent_description", None)
    elif args.chexagent == "on":
        if chexagent is None:
            raise ValueError("--chexagent on requires chexagent_description in the config")
        # Explicit opt-in turns a missing local checkpoint into an error instead
        # of silently excluding the optional expert.
        chexagent["optional"] = False
    config["generalist"] = resolve_generalist_spec(config["generalist"])
    if args.study in {"value", "diagnose"}:
        if config["generalist"].get("backend") == "llava_med":
            config["generalist"].setdefault("deterministic_image_padding", True)
        # Additive profile: the saved v0.8 configuration/entry point stays intact.
        config.setdefault("capability_value", {
            "generation": {"max_new_tokens": 96, "block_tokens": 16, "max_expert_calls": 2,
                           "max_decisions": 4, "controller_tokens": 48,
                           "max_evidence_chars": 1200, "visual_views": 1},
            "encoder": {"expert": "source_cases", "dimensions": 16, "seed": 17},
            "collection": {"collect_continuations": True,
                           "verify_block_none": True,
                           "pair_first_tools": ["cxr_anatomy", "conch_tissue", "source_cases"],
                           "max_pair_first_tools": 2},
            "fit": {"ridge": 1.0, "min_cases_per_domain": 8, "min_domains": 2,
                    "residual_quantile": 0.9, "cost_weight": 0.0},
            "quality": {"name": "token_f1"}, "single_tools": True,
        })
        if args.evidence_profile is not None:
            config["capability_value"].setdefault("generation", {}).update(
                evidence_style="scoped" if args.evidence_profile == "scoped" else "native",
                request_style="need" if args.evidence_profile == "scoped" else "question",
                visual_views=0 if args.evidence_profile == "scoped" else 1,
                evidence_top_k=2, retrieval_answer_context=False,
            )
        if args.study == "diagnose":
            default_pairs = [["cxr_anatomy", "chexagent_description"],
                             ["conch_tissue", "source_cases"]]
            config.setdefault("capability_diagnostics", {
                "pairs": [pair for pair in default_pairs if all(name in config["experts"] for name in pair)],
                "continuations": True,
            })
            # CLI --chexagent off intentionally removes the optional pair.
            if args.chexagent == "off":
                config["capability_diagnostics"]["pairs"] = [
                    pair for pair in config["capability_diagnostics"]["pairs"]
                    if "chexagent_description" not in pair
                ]
    return config


def prepare_biomed_text_config(artifacts, *, offline=False):
    """Pin the small text architecture config, without downloading a BERT model."""
    from huggingface_hub import hf_hub_download

    from .open_study import atomic_json

    repository = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    revision = "d673b8835373c6fa116d6d8006b33d48734e305d"
    directory = Path(artifacts).resolve() / (
        "models/microsoft--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/.cache/merit-text-config"
    )
    target, marker = directory / "config.json", directory / "dependency.json"
    if target.is_file() and marker.is_file():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if saved.get("revision") == revision and saved.get("sha256") == digest:
            return saved
        raise RuntimeError("BiomedBERT architecture config changed; refusing stale dependency reuse")
    path = hf_hub_download(repository, "config.json", revision=revision,
                           local_dir=directory, local_files_only=offline)
    recorded = {"id": repository, "revision": revision, "path": str(Path(path).resolve()),
                "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                "weights_downloaded": False}
    atomic_json(marker, recorded)
    return recorded


def preflight(config, artifacts):
    """Load package APIs/config only, not the 7B weights. No implicit downloads."""
    from .generalist_factory import generalist_provenance

    report = {"python": sys.executable, "versions": {}}
    if sys.version_info < (3, 10):  # noqa: UP036 - executable reused from an external environment
        raise RuntimeError("This experiment runner requires Python >=3.10")
    for name in ("torch", "transformers", "open_clip_torch", "torchxrayvision"):
        try:
            report["versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Missing {name}; use --install-deps for optional tools only") from exc
    if tuple(int(x) for x in report["versions"]["open_clip_torch"].split(".")[:2]) < (3, 2):
        raise RuntimeError(
            "Local BiomedCLIP needs open_clip_torch>=3.2. Upgrade that package explicitly "
            "with --no-deps in a compatible environment; do not upgrade torch/transformers."
        )
    if importlib.util.find_spec("conch") is None:
        raise RuntimeError(
            "CONCH Python package missing in this interpreter. Reuse its installed source via "
            "PYTHONPATH or install the official CONCH package with --no-deps; see docs/V08_LLAVA.md."
        )
    if "chexagent_description" in config.get("experts", {}):
        missing = [
            name for name in ("albumentations", "qudida")
            if importlib.util.find_spec(name) is None
        ]
        if missing:
            raise RuntimeError(
                "CheXagent requires missing local preprocessing packages: "
                + ", ".join(missing)
                + ". Install the pinned optional dependencies before an offline run."
            )
    if config["generalist"].get("backend") == "llava_med":
        from .llava_generalist import _llava_runtime

        _llava_runtime(config["generalist"].get("source_path"))
    else:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "OpenMed needs a Qwen2.5-VL-compatible environment. Select its existing Python "
                "with --python, not the older LLaVA huatuo stack. Do not upgrade huatuo for this comparison."
            ) from exc
    report["generalist"] = generalist_provenance(config["generalist"], artifacts)
    report["weights_loaded"] = False
    return report


def main(argv=None):
    args = parser().parse_args(argv)
    if min(args.source_per_group, args.target_limit) < 1:
        raise ValueError("sample limits must be positive")
    manifests = (args.source_manifest, args.target_manifest, args.references)
    if any(manifests) and not all(manifests):
        raise ValueError("custom data requires --source-manifest, --target-manifest and --references")
    if args.study != "value" and args.value_stage != "all":
        raise ValueError("--value-stage requires --study value")
    if args.study == "capabilities" and args.evidence_profile is not None:
        raise ValueError("--evidence-profile requires --study value or diagnose")
    # Set endpoints before importing huggingface_hub. No credential is logged.
    os.environ["HF_ENDPOINT"] = (
        "https://hf-mirror.com" if args.mirror == "cn" else "https://huggingface.co"
    )
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    from .capability_setup import (
        dependency_report,
        install_missing_dependencies,
        prepare_capabilities,
    )

    # Establish compatibility before weight downloads. Missing optional packages
    # may be installed explicitly, never with automatic core-stack upgrades.
    if args.install_deps and not args.check_only:
        installed = install_missing_dependencies(dependency_report(), mirror=args.mirror)
        print(json.dumps({"installed_missing_dependencies": installed}), flush=True)
        importlib.invalidate_caches()
    config = experiment_config(args)
    print(json.dumps(preflight(config, args.artifacts), indent=2), flush=True)
    setup_report = prepare_capabilities(
        args.artifacts, mirror=args.mirror, check_only=args.check_only or args.skip_download
    )
    print(json.dumps(setup_report, indent=2), flush=True)
    if not setup_report["ready_weights"]:
        raise RuntimeError("Small expert weights incomplete; rerun without --skip-download")
    if args.check_only:
        return
    from .assets import download_profile

    if not args.skip_download:
        print("Reusing/download-resuming only small tools and two real datasets; no generalist download.",
              flush=True)
        plan = download_profile("capability-small", args.artifacts, include_gated=True)
        print(json.dumps(plan, indent=2), flush=True)
        if plan["failed"]:
            raise RuntimeError("Incomplete asset download; see artifacts/download-report.json")
    config["runtime_dependencies"] = {
        "biomedbert_text_config": prepare_biomed_text_config(args.artifacts, offline=args.skip_download)
    }
    import yaml

    from .capability_study import run_capability_study
    from .multimodal_data import prepare_multimodal_vqa
    from .open_study import atomic_json, fingerprint

    datasets = ("pathvqa", "vqarad") if args.dataset == "both" else (args.dataset,)
    root = Path(args.output).resolve()
    cohort_key = f"{args.dataset}-source{args.source_per_group}-target{args.target_limit}-seed{args.seed}"
    data_root = root / "data" / cohort_key
    if all(manifests):
        manifest_paths = [Path(path).expanduser().resolve() for path in manifests]
        data = {"custom_manifests": [str(path) for path in manifest_paths],
                "warning": "domain provenance must be supplied honestly by the dataset owner"}
    else:
        data = prepare_multimodal_vqa(
            args.artifacts, data_root, args.source_per_group, args.target_limit, datasets, args.seed
        )
        manifest_paths = [data_root / name for name in ("source.jsonl", "target.jsonl", "references.json")]
    root.mkdir(parents=True, exist_ok=True)
    settings = root / f"config-{args.generalist}-{fingerprint(config)[:12]}.yaml"
    settings.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    atomic_json(root / "prepared.json", {"config": str(settings), "data": data})
    print(json.dumps({"config": str(settings), "data": data}, indent=2), flush=True)
    if args.prepare_only:
        return
    # Common inference files are label-free; references enter only evaluation/source calibration.
    if args.study in {"value", "diagnose"}:
        from .capability_value_study import run_value_study

        result = run_value_study(*manifest_paths, settings, args.artifacts, root / args.generalist,
                                 stage="diagnose" if args.study == "diagnose" else args.value_stage)
    else:
        result = run_capability_study(*manifest_paths, settings, args.artifacts, root / args.generalist)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
