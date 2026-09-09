"""Source-only teacher forcing for the optional LLaVA-Med native tensor bridge.

Input is JSONL of cached native predictions, never target labels/masks. Expert
inference remains in CapabilityPool; this trainer does not retrain specialists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml

from .capabilities import EvidenceItem
from .generalist_factory import generalist_provenance, load_generalist
from .open_study import fingerprint
from .tensor_bridge import NativeTensorBridge, tensor_projector_context, validate_tensor_backend
from .tensor_evidence import TensorContract


def source_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty source training set")
    seen = set()
    for row in rows:
        if set(row) != {"id", "split", "domain", "image", "prompt", "answer", "evidence"}:
            raise ValueError("training row must contain exactly the documented source fields")
        if (row["split"] != "source" or not row["domain"] or not row["answer"]
                or row["id"] in seen):
            raise ValueError("source-only unique cases with nonempty domain/answer required")
        seen.add(row["id"])
    return rows


def answer_loss(probe, row, packet):
    """Answer labels only; gradients pass through the frozen VLM into the bridge."""
    inputs = probe._inputs(row["image"], row["prompt"])
    ids = probe.tokenizer.encode(row["answer"], add_special_tokens=False)
    if not ids:
        raise ValueError("empty tokenized source answer")
    if probe.tokenizer.eos_token_id is not None:
        ids.append(probe.tokenizer.eos_token_id)
    extra = torch.tensor([ids], dtype=inputs["inputs"].dtype, device=inputs["inputs"].device)
    original = inputs["inputs"]
    inputs["inputs"] = torch.cat((original, extra), dim=1)
    inputs["attention_mask"] = torch.ones_like(inputs["inputs"])
    probe._validate_context(inputs, 0)
    labels = torch.cat((torch.full_like(original, -100), extra), dim=1)
    with tensor_projector_context(probe, packet):
        result = probe.model(input_ids=inputs.pop("inputs"), labels=labels, **inputs,
                             use_cache=False, return_dict=True)
    if not torch.isfinite(result.loss):
        raise ValueError("non-finite source answer loss")
    return result.loss


def train(probe, rows, contract, *, epochs=1, learning_rate=1e-4, seed=0, width=128):
    if type(epochs) is not int or epochs < 1 or not 0 < learning_rate < 1:
        raise ValueError("positive epochs and learning_rate in (0,1) required")
    torch.manual_seed(seed)
    probe.model.eval().requires_grad_(False)
    device = probe.model.get_input_embeddings().weight.device
    bridge = NativeTensorBridge(contract, int(probe.model.config.hidden_size), width=width).to(device)
    validate_tensor_backend(probe, bridge)
    probe.tensor_bridge = bridge
    # Compile all rows first: invalid/unregistered outputs cannot silently become
    # apparently successful baseline-only training examples.
    packets = []
    for row in rows:
        if row["split"] != "source":
            raise ValueError("target examples cannot train the bridge")
        items = tuple(EvidenceItem(**item) for item in row["evidence"])
        packet = probe.tensor_packet(items, row["image"])
        if not len(packet) or packet.rejected:
            raise ValueError(f"source evidence contract failed for {row['id']}: {packet.rejected}")
        packets.append(packet)
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=learning_rate)
    losses = []
    bridge.train()
    for _epoch in range(epochs):
        for index in torch.randperm(len(rows)).tolist():
            optimizer.zero_grad(set_to_none=True)
            loss = answer_loss(probe, rows[index], packets[index])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
    bridge.eval()
    return bridge, losses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generalist", required=True, help="YAML with a generalist mapping")
    parser.add_argument("--contract", required=True, help="JSON concept/scope registry")
    parser.add_argument("--source", required=True, help="source-only JSONL native evidence cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rows = source_rows(args.source)
    contract = TensorContract.from_dict(json.loads(Path(args.contract).read_text(encoding="utf-8")))
    spec = yaml.safe_load(Path(args.generalist).read_text(encoding="utf-8"))["generalist"]
    if spec.get("backend") != "llava_med" or spec.get("tensor_bridge_checkpoint"):
        raise ValueError("train from an explicit llava_med base config without a bridge checkpoint")
    probe = load_generalist(spec)
    bridge, losses = train(probe, rows, contract, epochs=args.epochs,
                           learning_rate=args.learning_rate, seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    bridge.save(output, training_steps=len(losses), source_domains=[r["domain"] for r in rows],
                base_identity=fingerprint(generalist_provenance(spec, None)))
    report = {"training_steps": len(losses), "losses": losses, "seed": args.seed,
              "source_sha256": hashlib.sha256(Path(args.source).read_bytes()).hexdigest(),
              "checkpoint_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "clinical_efficacy_evaluated": False}
    output.with_suffix(".training.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
