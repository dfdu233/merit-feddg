"""Prepare or run an isolated, answer-blind request-parser audit. No answer generation."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import yaml

from merit_feddg.request_parse_study import (
    MODES,
    SCHEMA,
    audit_parse,
    catalog_for,
    digest,
    parse_response,
    parser_prompt,
    select_questions,
)

EXPERTS = ("cxr_findings", "cxr_anatomy")
BASE = "de056a80909d0e2d479351b7c8b279326a81ba32"


def save_new(path, obj):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False, allow_nan=False)


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def code_hashes():
    root = Path(__file__).resolve().parents[1]
    names = ("merit_feddg/request_parse_study.py", "scripts/run_request_parse_study.py",
             "merit_feddg/evidence_admission.py", "merit_feddg/capability_contracts.py",
             "merit_feddg/generalist_factory.py", "merit_feddg/llava_generalist.py")
    return {name: digest((root / name).read_text()) for name in names}


def prepare(args):
    if args.manifest is None or args.config is None:
        raise ValueError("prepare requires --manifest and --config")
    config = yaml.safe_load(args.config.read_text())
    specs = {k: config["experts"][k] for k in EXPERTS}
    # Source files may contain additional fields; none are used for selection/prompting.
    # Caller must identify the official TRAIN manifest. This is not a new split.
    rows = read_lines(args.manifest)
    metadata_source = {"kind": "manifest"}
    if args.incumbent_json is not None:
        incumbent = json.loads(args.incumbent_json.read_text())
        if set(incumbent) != {r["id"] for r in rows}:
            raise ValueError("incumbent/manifest ID alignment mismatch")
        # Match the prior routing audit; inspect ONLY pre-existing input modality.
        rows = [{**r, "modality": incumbent[r["id"]]["input_modality"], "task": "open_vqa"}
                for r in rows]
        metadata_source = {"kind": "incumbent_input_modality",
                           "fingerprint": digest(args.incumbent_json.read_text())}
    eligible = [r for r in rows if r["modality"] == "cxr"]
    selected = select_questions(eligible, args.limit)
    if not selected:
        raise ValueError("no CXR TRAIN questions in this manifest")
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = {"schema": SCHEMA, "base_commit": BASE, "specs": specs,
              "generalist": config["generalist"], "questions": selected,
              "selection": "sha256(seed=197,id,question,modality,task); CXR TRAIN only",
              "source_manifest_sha": digest(args.manifest.read_text()),
              "source_count": len(rows), "eligible_count": len(eligible),
              "input_metadata_source": metadata_source,
              "code_hashes": code_hashes(), "max_new_tokens": args.max_new_tokens,
              "split": "train", "production_intervention": False,
              "image_hashes": {r["id"]: r.get("image_sha256") for r in eligible
                               if r["id"] in {x["id"] for x in selected}},
              "grammar_constrained_decoding": False, "model_retries": 0}
    frozen["identity"] = digest(frozen)
    save_new(args.output / "frozen.json", frozen)
    # Blind worksheet: no parser prediction, expert prediction or patient answer.
    with (args.output / "review-template.jsonl").open("x", encoding="utf-8") as stream:
        for row in selected:
            obj = {**row, "reviewed": False, "reviewer": "", "gold_parse": None,
                   "notes": "Question semantics only. No diagnosis/reference labels."}
            stream.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(json.dumps({"prepared": len(selected), "identity": frozen["identity"],
                      "planned_parser_calls": len(selected) * (1 + len(EXPERTS))}))


class TextOnlyLlava:
    """Reuse the frozen language backbone via its official images=None path.

The official llava_mistral.generate uses inputs_embeds and returns new tokens.
Unexpected return layouts stop this audit instead of silently slicing a response.
"""
    def __init__(self, probe, max_new_tokens):
        self.probe, self.max_new_tokens = probe, max_new_tokens

    def __call__(self, prompt):
        probe, torch = self.probe, self.probe.torch
        conversation = probe.runtime.conv_templates[probe.conv_mode].copy()
        conversation.append_message(conversation.roles[0], prompt)
        conversation.append_message(conversation.roles[1], None)
        inputs = probe.tokenizer(conversation.get_prompt(), return_tensors="pt")
        device = probe.model.get_input_embeddings().weight.device
        ids = inputs["input_ids"].to(device)
        mask = inputs["attention_mask"].to(device)
        max_context = int(getattr(probe.model.config, "max_position_embeddings", 4096))
        if ids.shape[-1] + self.max_new_tokens > max_context:
            raise ValueError("parser context would exceed model limit")
        start = time.perf_counter()
        with torch.inference_mode():
            output = probe.model.generate(inputs=ids, images=None, attention_mask=mask,
                do_sample=False, max_new_tokens=self.max_new_tokens, use_cache=True,
                pad_token_id=probe.tokenizer.eos_token_id,
                eos_token_id=probe.tokenizer.eos_token_id)
        token_ids = output[0].tolist()
        if len(token_ids) > self.max_new_tokens:
            raise RuntimeError("unverified text-only generation return layout")
        return {"text": probe.tokenizer.decode(token_ids, skip_special_tokens=True),
                "input_tokens": int(ids.shape[-1]), "output_tokens": len(token_ids),
                "seconds": time.perf_counter() - start,
                "hit_token_budget": len(token_ids) == self.max_new_tokens}


def run(args):
    frozen = json.loads((args.output / "frozen.json").read_text())
    identity = frozen.pop("identity")
    if digest(frozen) != identity or frozen["code_hashes"] != code_hashes():
        raise RuntimeError("frozen protocol/code changed; use a new study directory")
    questions, specs = frozen["questions"], frozen["specs"]
    if not args.execute:
        print(json.dumps({"status": "preflight_only", "questions": len(questions),
                          "planned_calls": len(questions) * (1 + len(specs)),
                          "medical_answers_generated": 0}))
        return
    if not args.gpu_uuid:
        raise ValueError("execute requires explicit --gpu-uuid")
    if (args.output / "predictions.json").exists():
        raise FileExistsError("completed predictions will not be overwritten")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "1"
    import torch

    from merit_feddg.capability_contracts import assess_authority
    from merit_feddg.generalist_factory import generalist_provenance, load_generalist

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one authorized CUDA device")
    actual = str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
    if actual.removeprefix("GPU-").lower() != args.gpu_uuid.removeprefix("GPU-").lower():
        raise RuntimeError("cannot verify authorized GPU UUID")
    model_spec = dict(frozen["generalist"], device_map="cuda:0")
    if model_spec.get("backend") != "llava_med" or not model_spec.get("checkpoint_path"):
        raise ValueError("only an existing local llava_med checkpoint is supported")
    provenance = generalist_provenance(model_spec, "artifacts")
    # Exclusive execution marker: no automatic resume/retry of partially run studies.
    save_new(args.output / "execution.json", {"identity": identity, "gpu_uuid": actual,
             "model": provenance, "torch": torch.__version__, "max_new_tokens": frozen["max_new_tokens"]})
    probe = load_generalist(model_spec)
    probe.model.eval().requires_grad_(False)
    generate = TextOnlyLlava(probe, frozen["max_new_tokens"])
    records = []
    for row in questions:
        prompt = parser_prompt(row["question"])
        shared_output = generate(prompt)
        for expert, spec in specs.items():
            for mode in MODES:
                if mode == "question_first":
                    output, used_prompt = shared_output, prompt
                else:
                    used_prompt = parser_prompt(row["question"], catalog=catalog_for(spec))
                    output = generate(used_prompt)
                parsed = parse_response(output["text"], row["question"])
                record = {"id": row["id"], "question": row["question"], "expert": expert,
                          "mode": mode, "output": output, "prompt_sha": digest(used_prompt),
                          "parsed": parsed, "audit": audit_parse(parsed, row, expert, spec),
                          "old_regex": assess_authority(row["question"], spec,
                                        spec["capabilities"][0], expert_id=expert),
                          "question_first_call_shared_between_experts": mode == "question_first"}
                records.append(record)
        # Save each case independently so a crash does not hide incurred costs.
        save_new(args.output / (digest(row["id"]) + ".json"), records[-2 * len(specs):])
        print(row["id"], "parser calls complete", flush=True)
    save_new(args.output / "predictions.json", {"identity": identity, "records": records,
             "actual_parser_calls": len(questions) * (1 + len(specs)),
             "medical_accuracy_evaluated": False, "production_intervention": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--incumbent-json", type=Path, help="Optional aligned prior input_modality metadata; never used for answer selection")
    parser.add_argument("--config", type=Path, default=Path("configs/llava_med_capabilities.yaml"))
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--gpu-uuid")
    args = parser.parse_args()
    if not 64 <= args.max_new_tokens <= 1024:
        parser.error("parser token budget must be 64..1024")
    (prepare if args.action == "prepare" else run)(args)


if __name__ == "__main__":
    main()
