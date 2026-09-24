#!/usr/bin/env python3
"""Render the verified full-TEST VQA-RAD rescue–harm figure for the paper."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ARMS = (
    ("generalist", "Generalist", "#404040", "o"),
    ("joint_all", "Joint", "#d55e00", "s"),
    ("isolated_mean", "Isolated mean", "#0072b2", "^"),
    ("isolated_geomedian", "Isolated geomedian", "#009e73", "D"),
    ("bard", "BARD", "#cc79a7", "P"),
)


def load_full(path):
    report = json.loads(path.read_text())
    if (report.get("status") != "complete_full_test_descriptive"
            or report.get("dataset") != "VQA-RAD" or report.get("n") != 451
            or report.get("raw_or_repair") != "raw"
            or set(name for name, _, _, _ in ARMS) - report["methods"].keys()):
        raise ValueError(f"not a verified full raw five-arm VQA-RAD summary: {path}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llava", type=Path, required=True)
    parser.add_argument("--huatuo", type=Path, required=True)
    parser.add_argument("--llava-consensus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    llava, huatuo = load_full(args.llava), load_full(args.huatuo)
    consensus = json.loads(args.llava_consensus.read_text())
    if (consensus.get("status") != "complete_full_test_descriptive"
            or consensus.get("receiver") != "LLaVA-Med-7B"
            or consensus.get("dataset") != "VQA-RAD" or consensus.get("n") != 451
            or consensus.get("rule") != "strict q=m base-relative normalized-probability consensus; evidence-conditioned adaptation"):
        raise ValueError("consensus is not the verified complete strict comparator")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    for ax, report, title in zip(axes, (llava, huatuo),
                                 ("LLaVA-Med-7B", "HuatuoGPT-Vision-7B")):
        ax.plot([0, 30], [0, 30], color="#888888", linestyle="--", linewidth=1,
                label="No net score change")
        for name, label, color, marker in ARMS:
            item = report["methods"][name]
            ax.scatter(100 * item["harm"], 100 * item["rescue"], s=100,
                       marker=marker, color=color, edgecolor="white", linewidth=.7,
                       zorder=3, label=label)
        if title == "LLaVA-Med-7B":
            ax.scatter(100 * consensus["harm_vs_generalist"],
                       100 * consensus["rescue_vs_generalist"], s=120,
                       marker="X", color="#e69f00", edgecolor="white",
                       linewidth=.7, zorder=3, label="Strict consensus")
        ax.set(title=f"{title} (N=451)", xlim=(-.7, 29), ylim=(-.7, 29),
               xlabel="Harm vs generalist (percentage points)")
        ax.title.set_fontsize(11)
        ax.grid(alpha=.18)
    axes[0].set_ylabel("Rescue vs generalist (percentage points)")
    c_handles, c_labels = axes[0].get_legend_handles_labels()
    fig.legend(c_handles, c_labels, loc="lower center", bbox_to_anchor=(.5, .01),
               ncol=4, frameon=False, fontsize=9)
    fig.suptitle("VQA-RAD matched-evidence ablation: rescue–harm trade-off",
                 fontsize=14, y=.97)
    fig.subplots_adjust(left=.08, right=.98, top=.85, bottom=.27, wspace=.12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format=args.output.suffix.removeprefix("."))
    print(args.output)


if __name__ == "__main__":
    main()
