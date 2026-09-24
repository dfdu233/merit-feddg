#!/usr/bin/env python3
"""Render verified full-TEST rescue–harm figures for the paper."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["svg.hashsalt"] = "researchstudio-vqarad-matched-v1"


ARMS = (
    ("generalist", "Generalist", "#404040", "o"),
    ("joint_all", "Joint", "#d55e00", "s"),
    ("isolated_mean", "Isolated mean", "#0072b2", "^"),
    ("isolated_geomedian", "Isolated geomedian", "#009e73", "D"),
    ("bard", "BARD", "#cc79a7", "P"),
)


def load_full(path, dataset, n):
    report = json.loads(path.read_text())
    if (report.get("status") != "complete_full_test_descriptive"
            or report.get("dataset") != dataset or report.get("n") != n
            or report.get("raw_or_repair") != "raw"
            or set(name for name, _, _, _ in ARMS) - report["methods"].keys()):
        raise ValueError(f"not a verified full raw five-arm {dataset} summary: {path}")
    return report


def load_consensus(path, receiver, dataset, n):
    report = json.loads(path.read_text())
    if (report.get("status") != "complete_full_test_descriptive"
            or report.get("receiver") != receiver
            or report.get("dataset") != dataset or report.get("n") != n
            or report.get("rule") != "strict q=m base-relative normalized-probability consensus; evidence-conditioned adaptation"):
        raise ValueError(f"not the verified complete strict comparator: {path}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("VQA-RAD", "SLAKE"), default="VQA-RAD")
    parser.add_argument("--llava", type=Path, required=True)
    parser.add_argument("--huatuo", type=Path, required=True)
    parser.add_argument("--llava-consensus", type=Path)
    parser.add_argument("--huatuo-consensus", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    n = {"VQA-RAD": 451, "SLAKE": 2094}[args.dataset]
    if bool(args.llava_consensus) != bool(args.huatuo_consensus):
        parser.error("provide both consensus summaries or neither")
    llava, huatuo = load_full(args.llava, args.dataset, n), load_full(args.huatuo, args.dataset, n)
    llava_consensus = (load_consensus(args.llava_consensus, "LLaVA-Med-7B", args.dataset, n)
                       if args.llava_consensus else None)
    huatuo_consensus = (load_consensus(args.huatuo_consensus, "HuatuoGPT-Vision-7B", args.dataset, n)
                        if args.huatuo_consensus else None)

    maxima = [100 * report["methods"][name][metric]
              for report in (llava, huatuo) for name, _, _, _ in ARMS
              for metric in ("harm", "rescue")]
    maxima.extend(100 * report[metric] for report in (llava_consensus, huatuo_consensus)
                  if report for metric in ("harm_vs_generalist", "rescue_vs_generalist"))
    limit = max(maxima) * 1.12 + 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    for ax, report, consensus, title in zip(
            axes, (llava, huatuo), (llava_consensus, huatuo_consensus),
            ("LLaVA-Med-7B", "HuatuoGPT-Vision-7B")):
        ax.plot([0, limit], [0, limit], color="#888888", linestyle="--", linewidth=1,
                label="No net score change")
        for name, label, color, marker in ARMS:
            item = report["methods"][name]
            ax.scatter(100 * item["harm"], 100 * item["rescue"], s=100,
                       marker=marker, color=color, edgecolor="white", linewidth=.7,
                       zorder=3, label=label)
        if consensus:
            ax.scatter(100 * consensus["harm_vs_generalist"],
                       100 * consensus["rescue_vs_generalist"], s=120,
                       marker="X", color="#e69f00", edgecolor="white",
                       linewidth=.7, zorder=3, label="Strict consensus")
        ax.set(title=f"{title} (N={n:,})", xlim=(-.7, limit), ylim=(-.7, limit),
               xlabel="Harm vs generalist (percentage points)")
        ax.title.set_fontsize(11)
        ax.grid(alpha=.18)
    axes[0].set_ylabel("Rescue vs generalist (percentage points)")
    c_handles, c_labels = axes[0].get_legend_handles_labels()
    fig.legend(c_handles, c_labels, loc="lower center", bbox_to_anchor=(.5, .01),
               ncol=4, frameon=False, fontsize=9)
    fig.suptitle(f"{args.dataset} matched-evidence ablation: rescue–harm trade-off",
                 fontsize=14, y=.97)
    fig.subplots_adjust(left=.08, right=.98, top=.85, bottom=.27, wspace=.12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format=args.output.suffix.removeprefix("."))
    if args.output.suffix == ".svg":
        # Matplotlib emits trailing path-coordinate spaces; keep generated SVG
        # diff-clean without changing geometry.
        args.output.write_text("\n".join(line.rstrip() for line in args.output.read_text().splitlines()) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
