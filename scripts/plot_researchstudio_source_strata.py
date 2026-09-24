#!/usr/bin/env python3
"""Plot the verified SLAKE BARD-versus-geomedian source-count decomposition."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["svg.hashsalt"] = "researchstudio-slake-source-strata-v1"


def load(path, receiver):
    report = json.loads(path.read_text())
    if (report.get("status") != "complete_full_test_descriptive"
            or report.get("dataset") != "SLAKE" or report.get("n") != 2094
            or report.get("receiver") != receiver or report.get("raw_or_repair") != "raw"
            or report.get("matched_evidence_rows") != 2094):
        raise ValueError(f"not a verified full raw matched-evidence SLAKE summary: {path}")
    strata = report["descriptive_strata"]
    if sum(strata[key]["n"] for key in
           ("delivered_groups_1", "delivered_groups_2", "delivered_groups_3+")) != 2094:
        raise ValueError(f"source-count strata do not cover all rows: {path}")
    return report


def difference(stratum):
    return 100 * (stratum["methods"]["bard"]["score"]
                  - stratum["methods"]["isolated_geomedian"]["score"])


def values(report):
    strata = report["descriptive_strata"]
    single = strata["delivered_groups_1"]
    multi = (strata["delivered_groups_2"], strata["delivered_groups_3+"])
    multi_n = sum(item["n"] for item in multi)
    multi_difference = sum(item["n"] * difference(item) for item in multi) / multi_n
    return ([100 * (report["methods"]["bard"]["score"]
                   - report["methods"]["isolated_geomedian"]["score"]),
             difference(single), multi_difference],
            [2094, single["n"], multi_n])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llava", type=Path, required=True)
    parser.add_argument("--huatuo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = (load(args.llava, "LLaVA-Med-7B"),
               load(args.huatuo, "HuatuoGPT-Vision-7B"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharey=True)
    names = ("All", "One source", "Two or more")
    for ax, report, title in zip(axes, reports,
                                 ("LLaVA-Med-7B", "HuatuoGPT-Vision-7B")):
        deltas, counts = values(report)
        colors = ["#009e73" if value >= 0 else "#d55e00" for value in deltas]
        ax.bar(range(3), deltas, color=colors, width=.65)
        ax.axhline(0, color="#555555", linewidth=1)
        ax.set_xticks(range(3), [f"{name}\n(n={count:,})" for name, count in zip(names, counts)])
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        for x, value in enumerate(deltas):
            ax.annotate(f"{value:+.2f}", (x, value),
                        xytext=(0, 5 if value >= 0 else -16),
                        textcoords="offset points", ha="center", fontsize=10)
    axes[0].set_ylabel("BARD minus isolated geomedian (percentage points)")
    axes[0].set_ylim(-8, 18)
    fig.suptitle("SLAKE: overall gain versus source-count mechanism", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .93))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    if args.output.suffix == ".svg":
        args.output.write_text("\n".join(line.rstrip() for line in
                                          args.output.read_text().splitlines()) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
