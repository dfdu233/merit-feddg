from __future__ import annotations

import argparse
import json
from pathlib import Path

from .assets import asset_plan, download_profile
from .extract import extract_manifest
from .io import load_records, load_yaml, save_json
from .manifest import audit_domain_split, build_folder_manifest
from .runner import aggregate_repetitions, compare_records
from .simulation import simulate_records


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


def command_download(args: argparse.Namespace) -> None:
    report = download_profile(
        args.profile,
        args.root,
        dry_run=args.dry_run,
        include_gated=args.include_gated,
    )
    _print(report)
    if report["failed"]:
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

    download = commands.add_parser("download", help="download registered models and datasets")
    download.add_argument("--profile", choices=["smoke", "open-small", "research-2d"], default="smoke")
    download.add_argument("--root", default="artifacts")
    download.add_argument("--dry-run", action="store_true")
    download.add_argument("--include-gated", action="store_true")
    download.set_defaults(func=command_download)

    plan = commands.add_parser("asset-plan", help="show model, dataset, license and gate requirements")
    plan.add_argument("--profile", choices=["smoke", "open-small", "research-2d"], default="research-2d")
    plan.set_defaults(func=lambda args: _print(asset_plan(args.profile)))

    extract = commands.add_parser("extract", help="extract real-model evidence into a reusable cache")
    extract.add_argument("--manifest", required=True)
    extract.add_argument("--config", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--artifacts")
    extract.add_argument("--limit", type=int, default=0)
    extract.add_argument("--oracle-router", action="store_true")
    extract.set_defaults(func=command_extract)

    manifest = commands.add_parser("make-manifest", help="index a local image folder without copying data")
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
