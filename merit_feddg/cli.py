from __future__ import annotations

import argparse
import json
from pathlib import Path

from .assets import asset_plan, download_profile, verify_assets
from .doctor import diagnostics
from .extract import extract_manifest
from .guided_generate import guided_generate_case
from .io import load_records, load_yaml, save_json
from .manifest import audit_domain_split, build_folder_manifest
from .med_defer_study import run_med_defer_records, run_med_defer_study
from .prepare import prepare_public_suite
from .runner import aggregate_repetitions, compare_records, make_oracle_records
from .simulation import simulate_records

ASSET_PROFILES = ["smoke", "open-small", "medical-small", "research-2d"]


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def command_simulate(args: argparse.Namespace) -> None:
    config = load_yaml(args.config)
    repetitions = int(config["evaluation"].get("repetitions", 1))
    reports = []
    for repetition in range(repetitions):
        records = simulate_records(config, repetition=repetition)
        directory = Path(args.output) / f"seed-{repetition}"
        reports.append(compare_records(records, config, directory, repetition=repetition))
    aggregate = aggregate_repetitions(reports)
    save_json(Path(args.output) / "aggregate.json", aggregate)
    _print(aggregate)


def command_compare(args: argparse.Namespace) -> None:
    config = load_yaml(args.config)
    report = compare_records(load_records(args.input), config, args.output)
    _print(report["metrics"])


def command_med_defer_simulate(args: argparse.Namespace) -> None:
    config = load_yaml(args.config)
    report = run_med_defer_study(config, repetition=args.repetition)
    save_json(args.output, report)
    _print({key: value for key, value in report.items() if key != "traces"})


def command_med_defer_compare(args: argparse.Namespace) -> None:
    config = load_yaml(args.config)
    report = run_med_defer_records(load_records(args.input), config)
    report["execution_mode"] = "cached-evidence counterfactual"
    save_json(args.output, report)
    _print({key: value for key, value in report.items() if key != "traces"})


def command_guided_generate(args: argparse.Namespace) -> None:
    with Path(args.case).open("r", encoding="utf-8") as handle:
        case = json.load(handle)
    report = guided_generate_case(
        case,
        load_yaml(args.model_config),
        load_yaml(args.compare_config),
        load_records(args.source_cache),
        artifact_root=args.artifacts,
        max_new_tokens=args.max_new_tokens,
    )
    save_json(args.output, report)
    _print(report)


def command_download(args: argparse.Namespace) -> None:
    report = download_profile(
        args.profile,
        args.root,
        dry_run=args.dry_run,
        include_gated=args.include_gated,
        force_download=args.force_download,
    )
    _print(report)
    if report["failed"]:
        raise SystemExit(1)


def command_verify_assets(args: argparse.Namespace) -> None:
    report = verify_assets(args.profile, args.root, include_gated=args.include_gated)
    _print(report)
    if not report["ready"]:
        raise SystemExit(1)


def command_extract(args: argparse.Namespace) -> None:
    config = load_yaml(args.config)
    records = extract_manifest(
        args.manifest,
        config,
        args.output,
        artifact_root=args.artifacts,
        limit=args.limit,
        oracle_router=args.oracle_router,
    )
    _print({"records": len(records), "output": str(Path(args.output).resolve())})


def command_prepare_public(args: argparse.Namespace) -> None:
    report = prepare_public_suite(
        args.config,
        args.artifacts,
        args.output,
        limit_per_domain=args.limit_per_domain,
        questions_per_image=args.questions_per_image,
    )
    _print(report)


def command_oracle_cache(args: argparse.Namespace) -> None:
    records = make_oracle_records(load_records(args.input), peak=args.peak)
    from .io import save_records

    save_records(args.output, records)
    _print({"records": len(records), "output": str(Path(args.output).resolve())})


def command_manifest(args: argparse.Namespace) -> None:
    count = build_folder_manifest(args.root, args.output, args.modality, args.domain, args.prompt)
    _print({"records": count, "output": str(Path(args.output).resolve())})


def command_audit(args: argparse.Namespace) -> None:
    _print(audit_domain_split(args.manifest, set(args.held_out)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="merit-feddg")
    commands = parser.add_subparsers(dest="command", required=True)

    simulate = commands.add_parser("simulate", help="run the deterministic end-to-end smoke study")
    simulate.add_argument("--config", required=True)
    simulate.add_argument("--output", required=True)
    simulate.set_defaults(func=command_simulate)

    compare = commands.add_parser("compare", help="compare methods on a cached evidence JSONL")
    compare.add_argument("--input", required=True)
    compare.add_argument("--config", required=True)
    compare.add_argument("--output", required=True)
    compare.set_defaults(func=command_compare)

    med_defer = commands.add_parser(
        "med-defer-simulate",
        help="run sparse claim-level expert deferral with source-only domain trust",
    )
    med_defer.add_argument("--config", default="configs/smoke.yaml")
    med_defer.add_argument("--output", default="runs/med-defer-smoke/result.json")
    med_defer.add_argument("--repetition", type=int, default=0)
    med_defer.set_defaults(func=command_med_defer_simulate)

    med_defer_compare = commands.add_parser(
        "med-defer-compare",
        help="evaluate claim-level deferral on an existing real-model evidence cache",
    )
    med_defer_compare.add_argument("--input", required=True)
    med_defer_compare.add_argument("--config", required=True)
    med_defer_compare.add_argument("--output", required=True)
    med_defer_compare.set_defaults(func=command_med_defer_compare)

    guided = commands.add_parser(
        "guided-generate",
        help="generate one answer with live lazy specialist calls inside decoding",
    )
    guided.add_argument("--case", required=True)
    guided.add_argument("--source-cache", required=True)
    guided.add_argument("--model-config", default="configs/medical_small.yaml")
    guided.add_argument("--compare-config", required=True)
    guided.add_argument("--artifacts", default="artifacts")
    guided.add_argument("--max-new-tokens", type=int, default=96)
    guided.add_argument("--output", required=True)
    guided.set_defaults(func=command_guided_generate)

    download = commands.add_parser("download", help="download registered models and datasets")
    download.add_argument("--profile", choices=ASSET_PROFILES, default="smoke")
    download.add_argument("--root", default="artifacts")
    download.add_argument("--dry-run", action="store_true")
    download.add_argument("--include-gated", action="store_true")
    download.add_argument(
        "--force-download",
        action="store_true",
        help="redownload even when a verified completed snapshot is already present",
    )
    download.set_defaults(func=command_download)

    verify = commands.add_parser("verify-assets", help="verify downloaded snapshot payloads")
    verify.add_argument("--profile", choices=ASSET_PROFILES, default="smoke")
    verify.add_argument("--root", default="artifacts")
    verify.add_argument("--include-gated", action="store_true")
    verify.set_defaults(func=command_verify_assets)

    doctor = commands.add_parser(
        "doctor", help="report server, GPU, disk and Hugging Face readiness"
    )
    doctor.add_argument("--root", default="artifacts")
    doctor.set_defaults(func=lambda args: _print(diagnostics(args.root)))

    plan = commands.add_parser(
        "asset-plan", help="show model, dataset, license and gate requirements"
    )
    plan.add_argument("--profile", choices=ASSET_PROFILES, default="medical-small")
    plan.set_defaults(func=lambda args: _print(asset_plan(args.profile)))

    extract = commands.add_parser(
        "extract", help="extract real-model evidence into a reusable cache"
    )
    extract.add_argument("--manifest", required=True)
    extract.add_argument("--config", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--artifacts")
    extract.add_argument("--limit", type=int, default=0)
    extract.add_argument("--oracle-router", action="store_true")
    extract.set_defaults(func=command_extract)

    prepare = commands.add_parser(
        "prepare-public",
        help="convert downloaded public datasets into a leakage-audited proxy benchmark",
    )
    prepare.add_argument("--config", default="configs/public_benchmarks.yaml")
    prepare.add_argument("--artifacts", default="artifacts")
    prepare.add_argument("--output", default="data/public-benchmark")
    prepare.add_argument("--limit-per-domain", type=int, default=8)
    prepare.add_argument("--questions-per-image", type=int, default=1)
    prepare.set_defaults(func=command_prepare_public)

    oracle = commands.add_parser(
        "oracle-cache", help="derive an oracle-routing cache without rerunning the models"
    )
    oracle.add_argument("--input", required=True)
    oracle.add_argument("--output", required=True)
    oracle.add_argument("--peak", type=float, default=0.98)
    oracle.set_defaults(func=command_oracle_cache)

    manifest = commands.add_parser(
        "make-manifest", help="index a local image folder without copying data"
    )
    manifest.add_argument("--root", required=True)
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--modality", required=True)
    manifest.add_argument("--domain", required=True)
    manifest.add_argument("--prompt", default="Describe the clinically relevant findings.")
    manifest.set_defaults(func=command_manifest)

    audit = commands.add_parser("audit-split", help="check source/target manifests for ID leakage")
    audit.add_argument("--manifest", action="append", required=True)
    audit.add_argument("--held-out", action="append", required=True)
    audit.set_defaults(func=command_audit)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
