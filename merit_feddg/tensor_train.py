"""ARCHIVED implementation for historical negative-result regression tests.

The command-line training entrypoint is retired and is not part of the method.

Input is JSONL of cached native predictions, never target labels/masks. Expert
inference remains in CapabilityPool; this trainer does not retrain specialists.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .capabilities import EvidenceItem
from .tensor_bridge import NativeTensorBridge, tensor_projector_context, validate_tensor_backend


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
    raise SystemExit("Retired training entrypoint. Use scripts/run_vqarad_vector_pipeline.sh "
                     "with the original full test manifest; no source preparation or bridge training.")


if __name__ == "__main__":
    main()
